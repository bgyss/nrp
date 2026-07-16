// Shared WGSL generator for the exported TorchNRP format (T4 exporter blobs).
//
// Extracted verbatim from bench_t4.mjs (T4) so the G2 demo can reuse it; default
// options generate exactly the T4 shader. Options beyond the default:
//   outputActivation: "softplus" (default) | "linear"  — linear is for signed
//     residual proxies (G1), whose head has no softplus.
//   demo: true  — adds first-hit positions (@5) and a link mask (@6) plus
//     light_rgb / controls uniforms, and applies the two E8 production controls
//     after the emission multiply: per-layer light linking (zero the masked
//     pixels when controls.x > 0.5) and a first-hit linear-distance artist
//     attenuation curve (weight max(0, 1 - controls.y * distance)).
//   composite: "write" (default) | "add_masked" | "add"  — add_masked accumulates
//     into the output buffer on pixels where the mask (@5) is set and leaves other
//     pixels untouched (the G1 residual compositing pass); add accumulates
//     unconditionally on every pixel, no mask binding (H4's N-light rig
//     compositing: one "write" dispatch for the first active light, then "add"
//     for each subsequent one, all sharing the same output buffer).
//   rigLight: true — adds a `light_rgb: vec4<f32>` uniform and multiplies the raw
//     MLP output by it before storing (Eq. 3's emission multiply), same convention
//     as `demo` but without the link-mask/attenuation machinery H4 doesn't need.
//     Mutually exclusive with `demo` (which already does its own light_rgb multiply
//     plus the two E8 controls).
export function buildShader(manifest, opts = {}) {
  const outputActivation = opts.outputActivation || "softplus";
  const demo = !!opts.demo;
  const rigLight = !!opts.rigLight;
  const composite = opts.composite || "write";
  if (demo && composite !== "write") throw new Error("demo and add_masked/add are exclusive");
  if (demo && rigLight) throw new Error("demo and rigLight are exclusive");
  if (rigLight && composite === "add_masked") {
    throw new Error("rigLight uses add, not add_masked (no region mask)");
  }
  const dims = manifest.mlp_dims;
  const enc = manifest.encoding;
  const inDim = dims[0];
  const auxDim = manifest.aux_dim;
  const lightDim = manifest.light_param_dim;
  let encodingCode = "";
  let encDim = 0;
  if (enc) {
    encDim = enc.levels.length * enc.features_per_level;
    const mask = enc.table_size - 1;
    for (let l = 0; l < enc.levels.length; l++) {
      const { resolution: res, dense, offset_floats: off, entries } = enc.levels[l];
      const idx = dense
        ? (ix, iy) => `((${iy} * ${res + 1}u + ${ix}) % ${entries}u)`
        : (ix, iy) => `((${ix} ^ ((${iy} * ${enc.hash_prime}u) & ${mask}u)) % ${entries}u)`;
      encodingCode += `
  {
    let pos = xy * ${res}.0;
    let p0 = clamp(vec2<u32>(floor(pos)), vec2<u32>(0u), vec2<u32>(${res}u));
    let frac = pos - vec2<f32>(p0);
    let x0 = p0.x; let y0 = p0.y;
    let x1 = min(x0 + 1u, ${res}u); let y1 = min(y0 + 1u, ${res}u);
    let w00 = (1.0 - frac.x) * (1.0 - frac.y);
    let w10 = frac.x * (1.0 - frac.y);
    let w01 = (1.0 - frac.x) * frac.y;
    let w11 = frac.x * frac.y;
    for (var f: u32 = 0u; f < ${enc.features_per_level}u; f = f + 1u) {
      buf[${l * enc.features_per_level}u + f] =
          tables[${off}u + ${idx("x0", "y0")} * ${enc.features_per_level}u + f] * w00
        + tables[${off}u + ${idx("x1", "y0")} * ${enc.features_per_level}u + f] * w10
        + tables[${off}u + ${idx("x0", "y1")} * ${enc.features_per_level}u + f] * w01
        + tables[${off}u + ${idx("x1", "y1")} * ${enc.features_per_level}u + f] * w11;
    }
  }`;
    }
  } else {
    encDim = 2;
    encodingCode = `
  buf[0] = xy.x; buf[1] = xy.y;`;
  }
  if (encDim + auxDim + lightDim !== inDim) {
    throw new Error(`input dim mismatch: ${encDim}+${auxDim}+${lightDim} != ${inDim}`);
  }
  const maxDim = Math.max(...dims);
  // MLP with register-resident accumulators. Two earlier variants were ~4x too slow:
  // per-output vec4 dot loops (dynamically indexed function-space arrays spill to
  // device memory) and workgroup-shared weight tiles alone (same spill problem).
  // This version transposes the weights (see repackMlp) so the inner statements are
  // fully unrolled with *constant* array indices — acc[] promotes to registers —
  // while each input activation is read exactly once per output block and weights
  // stream through a <=16 KiB workgroup-shared tile.
  const pad4 = (n) => Math.ceil(n / 4);
  const OTILE = 16; // output vec4s accumulated in registers per block (64 floats)
  // Input vec4s per cooperative weight-tile load: 8 when the layer's padded input
  // width allows it (the T4 kitchen proxy always does), otherwise the largest
  // divisor of din4 <= 8 so small toy/residual models (G1 panel) also compile.
  const tileFactor = (din4) => {
    for (let t = Math.min(8, din4); t >= 1; t--) if (din4 % t === 0) return t;
    return 1;
  };
  const din4Max = Math.max(...dims.slice(0, -1).map(pad4));
  const wtileVec4s = Math.max(
    ...dims.slice(0, -1).map((din, l) => {
      const dout4 = pad4(dims[l + 1]);
      return tileFactor(pad4(din)) * 4 * Math.min(OTILE, dout4);
    })
  );
  if (wtileVec4s * 16 > 16384) throw new Error("weight tile exceeds 16 KiB workgroup storage");
  let mlpCode = `
  var act: array<vec4<f32>, ${din4Max}>;
  for (var i: u32 = 0u; i < ${inDim}u; i = i + 1u) { act[i / 4u][i % 4u] = buf[i]; }`;
  let off4 = 0;
  for (let l = 0; l < dims.length - 1; l++) {
    const [din, dout] = [dims[l], dims[l + 1]];
    const din4 = pad4(din);
    const dout4 = pad4(dout);
    const TI = tileFactor(din4);
    const wOff = off4; // weights: [i4][c][o4] vec4s, o4-major innermost
    const bOff = off4 + din4 * 4 * dout4;
    for (let o0 = 0; o0 < dout4; o0 += OTILE) {
      const ob = Math.min(OTILE, dout4 - o0); // output vec4s in this block
      const tileN = TI * 4 * ob;
      mlpCode += `
  {`;
      for (let o = 0; o < ob; o++) mlpCode += `
    var acc${o}: vec4<f32> = mlp[${bOff + o0 + o}u];`;
      mlpCode += `
    for (var t: u32 = 0u; t < ${din4 / TI}u; t = t + 1u) {
      workgroupBarrier();
      for (var k: u32 = lid; k < ${tileN}u; k = k + 64u) {
        let i4 = k / ${4 * ob}u; let rest = k % ${4 * ob}u;
        wtile[k] = mlp[${wOff}u + (t * ${TI}u + i4) * ${4 * dout4}u + (rest / ${ob}u) * ${dout4}u + ${o0}u + rest % ${ob}u];
      }
      workgroupBarrier();
      for (var ii: u32 = 0u; ii < ${TI}u; ii = ii + 1u) {
        let x = act[t * ${TI}u + ii];
        let wb = ii * ${4 * ob}u;`;
      for (let c = 0; c < 4; c++)
        for (let o = 0; o < ob; o++) mlpCode += `
        acc${o} = acc${o} + wtile[wb + ${c * ob + o}u] * x[${c}];`;
      mlpCode += `
      }
    }`;
      const act_ = (e) => (l < dims.length - 2 ? `max(${e}, vec4<f32>(0.0))` : e);
      for (let o = 0; o < ob; o++) mlpCode += `
    out${l % 2}[${o0 + o}u] = ${act_(`acc${o}`)};`;
      mlpCode += `
  }`;
    }
    mlpCode += `
  ${l < dims.length - 2 ? `for (var i: u32 = 0u; i < ${dout4}u; i = i + 1u) { act[i] = out${l % 2}[i]; }` : ""}`;
    off4 = bOff + dout4;
  }
  const outArr = `out${(dims.length - 2) % 2}`;
  const wtileDecl = `var<workgroup> wtile: array<vec4<f32>, ${wtileVec4s}>;
`;
  const outDecls = `
  var out0: array<vec4<f32>, ${Math.max(...dims.slice(1).map(pad4))}>;
  var out1: array<vec4<f32>, ${Math.max(...dims.slice(1).map(pad4))}>;`;
  const head = (i) =>
    outputActivation === "softplus" ? `softplus(${outArr}[0][${i}])` : `${outArr}[0][${i}]`;
  const demoStruct = demo
    ? `, light_rgb: vec4<f32>, controls: vec4<f32>,`
    : rigLight
      ? `, light_rgb: vec4<f32>,`
      : `,`;
  const demoBindings = demo
    ? `
@group(0) @binding(5) var<storage, read> positions: array<f32>;
@group(0) @binding(6) var<storage, read> linkMask: array<f32>;`
    : composite === "add_masked"
      ? `
@group(0) @binding(5) var<storage, read> regionMask: array<f32>;`
      : rigLight
        ? `
@group(0) @binding(5) var<storage, read> lightParams: array<f32>;`
        : "";
  // rigLight sources light params from the storage buffer above (binding 5), not
  // the small `params.light: vec4<f32>` uniform slot -- unlike sphere-only T4/G2/
  // G1 (light_param_dim<=4, fits a vec4), a mixed rig's quad/textured_quad lights
  // need up to 8 + texture-size*3 floats (H3/H5's textured_quad proxies), far past
  // what a uniform vec4 can hold.
  const lightParamsFillCode = rigLight
    ? `
  for (var i: u32 = 0u; i < ${lightDim}u; i = i + 1u) { buf[${encDim + auxDim}u + i] = lightParams[i]; }`
    : `
  buf[${encDim + auxDim}u] = params.light.x;
  buf[${encDim + auxDim + 1}u] = params.light.y;
  buf[${encDim + auxDim + 2}u] = params.light.z;
  buf[${encDim + auxDim + 3}u] = params.light.w;`;
  let storeCode;
  if (demo) {
    storeCode = `
  if (gid.x < params.n_pixels) {
    var c = vec3<f32>(${head(0)}, ${head(1)}, ${head(2)}) * params.light_rgb.xyz;
    if (params.controls.x > 0.5 && linkMask[sy * ${manifest.resolution[0]}u + sx] > 0.5) {
      c = vec3<f32>(0.0);
    }
    let srcp = (sy * ${manifest.resolution[0]}u + sx) * 3u;
    let p = vec3<f32>(positions[srcp], positions[srcp + 1u], positions[srcp + 2u]);
    let w = max(0.0, 1.0 - params.controls.y * distance(p, params.light.xyz));
    c = c * w;
    outputs[pixel * 3u] = c.x;
    outputs[pixel * 3u + 1u] = c.y;
    outputs[pixel * 3u + 2u] = c.z;
  }`;
  } else if (composite === "add_masked") {
    storeCode = `
  if (gid.x < params.n_pixels && regionMask[sy * ${manifest.resolution[0]}u + sx] > 0.5) {
    outputs[pixel * 3u] = outputs[pixel * 3u] + ${head(0)};
    outputs[pixel * 3u + 1u] = outputs[pixel * 3u + 1u] + ${head(1)};
    outputs[pixel * 3u + 2u] = outputs[pixel * 3u + 2u] + ${head(2)};
  }`;
  } else if (rigLight) {
    const c = `vec3<f32>(${head(0)}, ${head(1)}, ${head(2)}) * params.light_rgb.xyz`;
    storeCode =
      composite === "add"
        ? `
  if (gid.x < params.n_pixels) {
    let c = ${c};
    outputs[pixel * 3u] = outputs[pixel * 3u] + c.x;
    outputs[pixel * 3u + 1u] = outputs[pixel * 3u + 1u] + c.y;
    outputs[pixel * 3u + 2u] = outputs[pixel * 3u + 2u] + c.z;
  }`
        : `
  if (gid.x < params.n_pixels) {
    let c = ${c};
    outputs[pixel * 3u] = c.x;
    outputs[pixel * 3u + 1u] = c.y;
    outputs[pixel * 3u + 2u] = c.z;
  }`;
  } else {
    storeCode = `
  if (gid.x < params.n_pixels) {
    outputs[pixel * 3u] = ${head(0)};
    outputs[pixel * 3u + 1u] = ${head(1)};
    outputs[pixel * 3u + 2u] = ${head(2)};
  }`;
  }
  return `
struct Params { n_pixels: u32, stride: u32, side: u32, pad: u32, light: vec4<f32>${demoStruct} };
@group(0) @binding(0) var<storage, read> pixels: array<f32>;
@group(0) @binding(1) var<storage, read> tables: array<f32>;
@group(0) @binding(2) var<storage, read> mlp: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;${demoBindings}
fn softplus(x: f32) -> f32 { return select(log(1.0 + exp(x)), x, x > 20.0); }
${wtileDecl}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(local_invocation_index) lid: u32) {
  // No early return: workgroupBarrier() requires uniform control flow, so
  // out-of-range threads compute garbage and skip only the final store.
  let pixel = min(gid.x, params.n_pixels - 1u);
  // Strided subsampling of the full-res G-buffer for the sub-512 latency points.
  let sx = (pixel % params.side) * params.stride;
  let sy = (pixel / params.side) * params.stride;
  let src = (sy * ${manifest.resolution[0]}u + sx) * ${2 + auxDim}u;
  let xy = vec2<f32>(pixels[src], pixels[src + 1u]);
  var buf: array<f32, ${maxDim}>;
${encodingCode}
  for (var i: u32 = 0u; i < ${auxDim}u; i = i + 1u) { buf[${encDim}u + i] = pixels[src + 2u + i]; }
${lightParamsFillCode}
${outDecls}
${mlpCode}${storeCode}
}
`;
}

// Repack the exporter's flat MLP layout (per layer: row-major weights, then bias)
// into the shader's transposed vec4 layout: per layer, weights as [i4][c][o4] vec4s
// (vec4 component j = W[o4*4+j][i4*4+c], zero-padded), then ceil(dout/4) bias vec4s.
export function repackMlp(flat, dims) {
  const out = [];
  let off = 0;
  for (let l = 0; l < dims.length - 1; l++) {
    const [din, dout] = [dims[l], dims[l + 1]];
    const din4 = Math.ceil(din / 4);
    const dout4 = Math.ceil(dout / 4);
    for (let i4 = 0; i4 < din4; i4++) {
      for (let c = 0; c < 4; c++) {
        const i = i4 * 4 + c;
        for (let o4 = 0; o4 < dout4; o4++) {
          for (let j = 0; j < 4; j++) {
            const o = o4 * 4 + j;
            out.push(i < din && o < dout ? flat[off + o * din + i] : 0);
          }
        }
      }
    }
    off += din * dout;
    for (let o = 0; o < dout4 * 4; o++) out.push(o < dout ? flat[off + o] : 0);
    off += dout;
  }
  return out;
}
