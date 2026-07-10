// G2 summit demo: the exported T1 kitchen proxy (hashgrid in WGSL, shader_gen.mjs)
// relit live in the browser with animated lights (E1 keyframe format) and two E8
// production controls (per-layer light linking, first-hit artist attenuation),
// plus — G1's deliverable — a toy-scale moving-object panel compositing the frozen
// base proxy with per-frame signed residuals.
//
// The page exposes window.__demo for the scripted trace runner (demo_g2.mjs).
import { buildShader, repackMlp } from "../shader_gen.mjs";

const EXPORT = "/out/g2-demo/export";
const EXPORT_G1 = "/out/g2-demo/export-g1";

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
async function fetchF32(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return new Float32Array(await r.arrayBuffer());
}

// Linear keyframe interpolation, matching nrp.torch_backend.animate semantics.
export function lightAt(keyframes, t) {
  const kf = [...keyframes].sort((a, b) => a.time - b.time);
  if (t <= kf[0].time) return { ...kf[0].light };
  if (t >= kf[kf.length - 1].time) return { ...kf[kf.length - 1].light };
  let a = kf[0], b = kf[kf.length - 1];
  for (let i = 0; i + 1 < kf.length; i++) {
    if (kf[i].time <= t && t <= kf[i + 1].time) { a = kf[i]; b = kf[i + 1]; break; }
  }
  const alpha = (t - a.time) / (b.time - a.time);
  const lerp = (x, y) => (1 - alpha) * x + alpha * y;
  return {
    type: a.light.type,
    center: a.light.center.map((v, i) => lerp(v, b.light.center[i])),
    radius: lerp(a.light.radius, b.light.radius),
    rgb: a.light.rgb.map((v, i) => lerp(v, b.light.rgb[i])),
  };
}

const BLIT_WGSL = `
@group(0) @binding(0) var<storage, read> img: array<f32>;
@group(0) @binding(1) var<uniform> dims: vec4<u32>;
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  var p = array<vec2<f32>, 3>(vec2(-1.0, -3.0), vec2(-1.0, 1.0), vec2(3.0, 1.0));
  return vec4<f32>(p[vi], 0.0, 1.0);
}
fn tone(x: f32) -> f32 { let c = max(x, 0.0); return pow(c / (1.0 + c), 1.0 / 2.2); }
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
  let ix = min(u32(pos.x), dims.x - 1u);
  let iy = min(u32(pos.y), dims.y - 1u);
  let i = (iy * dims.x + ix) * 3u;
  return vec4<f32>(tone(img[i]), tone(img[i + 1u]), tone(img[i + 2u]), 1.0);
}
`;

function mkBuf(device, data, usage) {
  const buf = device.createBuffer({ size: Math.max(16, data.length * 4), usage: usage | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buf, 0, data instanceof Float32Array ? data : Float32Array.from(data));
  return buf;
}

function computeBGL(device, { demo = false, masked = false } = {}) {
  const entries = [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
  ];
  if (demo) {
    entries.push(
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    );
  } else if (masked) {
    entries.push({ binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } });
  }
  return device.createBindGroupLayout({ entries });
}

function blitSetup(device, canvas, imgBuf, width, height, format) {
  const ctx = canvas.getContext("webgpu");
  ctx.configure({ device, format, alphaMode: "opaque" });
  const module = device.createShaderModule({ code: BLIT_WGSL });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    vertex: { module, entryPoint: "vs" },
    fragment: { module, entryPoint: "fs", targets: [{ format }] },
  });
  const dims = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dims, 0, Uint32Array.from([width, height, 0, 0]));
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: imgBuf } },
      { binding: 1, resource: { buffer: dims } },
    ],
  });
  return { ctx, pipeline, bindGroup };
}

function encodeBlit(encoder, blit) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: blit.ctx.getCurrentTexture().createView(),
      loadOp: "clear",
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      storeOp: "store",
    }],
  });
  pass.setPipeline(blit.pipeline);
  pass.setBindGroup(0, blit.bindGroup);
  pass.draw(3);
  pass.end();
}

async function init() {
  const statsEl = document.getElementById("stats");
  if (!navigator.gpu) { statsEl.textContent = "WebGPU unavailable"; throw new Error("no WebGPU"); }
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();
  const format = navigator.gpu.getPreferredCanvasFormat();

  const manifest = await fetchJson(`${EXPORT}/manifest.json`);
  const trace = await fetchJson("/webgpu/demo/trace.json");
  const [pixels, tables, mlpFlat, positions, linkMask] = await Promise.all([
    fetchF32(`${EXPORT}/pixels.bin`),
    manifest.encoding ? fetchF32(`${EXPORT}/tables.bin`) : Promise.resolve(new Float32Array([0])),
    fetchF32(`${EXPORT}/mlp.bin`),
    fetchF32(`${EXPORT}/positions.bin`),
    fetchF32(`${EXPORT}/link_mask.bin`),
  ]);
  const [width, height] = manifest.resolution;
  const nPixels = width * height;

  const shader = buildShader(manifest, { demo: true });
  const bgl = computeBGL(device, { demo: true });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
  });
  const outputsBuf = device.createBuffer({
    size: nPixels * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({ size: nPixels * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const paramsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: mkBuf(device, pixels, GPUBufferUsage.STORAGE) } },
      { binding: 1, resource: { buffer: mkBuf(device, tables, GPUBufferUsage.STORAGE) } },
      { binding: 2, resource: { buffer: mkBuf(device, Float32Array.from(repackMlp(Array.from(mlpFlat), manifest.mlp_dims)), GPUBufferUsage.STORAGE) } },
      { binding: 3, resource: { buffer: outputsBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
      { binding: 5, resource: { buffer: mkBuf(device, positions, GPUBufferUsage.STORAGE) } },
      { binding: 6, resource: { buffer: mkBuf(device, linkMask, GPUBufferUsage.STORAGE) } },
    ],
  });
  const kitchenBlit = blitSetup(device, document.getElementById("kitchen"), outputsBuf, width, height, format);

  // --- G1 moving-object panel (optional: only if the export exists) ---
  let g1 = null;
  try {
    const m = await fetchJson(`${EXPORT_G1}/manifest.json`);
    const [gw, gh] = m.resolution;
    const gn = gw * gh;
    const light = m.light;
    const baseShader = buildShader(
      { mlp_dims: m.base.mlp_dims, encoding: null, aux_dim: 7, light_param_dim: 4, resolution: m.resolution },
      {},
    );
    const resShader = buildShader(
      { mlp_dims: m.residual.mlp_dims, encoding: null, aux_dim: 7, light_param_dim: 4, resolution: m.resolution },
      { outputActivation: "linear", composite: "add_masked" },
    );
    const baseBGL = computeBGL(device, {});
    const resBGL = computeBGL(device, { masked: true });
    const basePipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [baseBGL] }),
      compute: { module: device.createShaderModule({ code: baseShader }), entryPoint: "main" },
    });
    const resPipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [resBGL] }),
      compute: { module: device.createShaderModule({ code: resShader }), entryPoint: "main" },
    });
    const gParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    {
      const u = new ArrayBuffer(32);
      new Uint32Array(u, 0, 4).set([gn, 1, gw, 0]);
      new Float32Array(u, 16, 4).set([...light.center, light.radius]);
      device.queue.writeBuffer(gParams, 0, u);
    }
    const gOut = device.createBuffer({ size: gn * 3 * 4, usage: GPUBufferUsage.STORAGE });
    const gRef = device.createBuffer({ size: gn * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const baseMlp = mkBuf(device, Float32Array.from(repackMlp(Array.from(await fetchF32(`${EXPORT_G1}/base_mlp.bin`)), m.base.mlp_dims)), GPUBufferUsage.STORAGE);
    const dummyTables = mkBuf(device, new Float32Array([0]), GPUBufferUsage.STORAGE);
    const frames = [];
    for (let f = 0; f < m.frames; f++) {
      const s = String(f).padStart(4, "0");
      const [px, resMlp, mask, ref] = await Promise.all([
        fetchF32(`${EXPORT_G1}/pixels_${s}.bin`),
        fetchF32(`${EXPORT_G1}/residual_mlp_${s}.bin`),
        fetchF32(`${EXPORT_G1}/region_mask_${s}.bin`),
        fetchF32(`${EXPORT_G1}/reference_${s}.bin`),
      ]);
      const pxBuf = mkBuf(device, px, GPUBufferUsage.STORAGE);
      frames.push({
        baseBG: device.createBindGroup({
          layout: baseBGL,
          entries: [
            { binding: 0, resource: { buffer: pxBuf } },
            { binding: 1, resource: { buffer: dummyTables } },
            { binding: 2, resource: { buffer: baseMlp } },
            { binding: 3, resource: { buffer: gOut } },
            { binding: 4, resource: { buffer: gParams } },
          ],
        }),
        resBG: device.createBindGroup({
          layout: resBGL,
          entries: [
            { binding: 0, resource: { buffer: pxBuf } },
            { binding: 1, resource: { buffer: dummyTables } },
            { binding: 2, resource: { buffer: mkBuf(device, Float32Array.from(repackMlp(Array.from(resMlp), m.residual.mlp_dims)), GPUBufferUsage.STORAGE) } },
            { binding: 3, resource: { buffer: gOut } },
            { binding: 4, resource: { buffer: gParams } },
            { binding: 5, resource: { buffer: mkBuf(device, mask, GPUBufferUsage.STORAGE) } },
          ],
        }),
        ref,
      });
    }
    g1 = {
      manifest: m, gn, gw, frames, basePipe, resPipe, gOut, gRef, frame: 0,
      blit: blitSetup(device, document.getElementById("g1canvas"), gOut, gw, gh, format),
      refBlit: blitSetup(device, document.getElementById("g1ref"), gRef, gw, gh, format),
    };
    device.queue.writeBuffer(gRef, 0, frames[0].ref);
    document.getElementById("g1panel").style.display = "block";
    document.getElementById("g1frame").max = String(m.frames - 1);
  } catch (e) {
    console.log("G1 panel disabled:", e.message);
  }

  // --- state + frame rendering ---
  const state = {
    t: 0,
    playing: true,
    driven: false,
    controls: { rgb: [1, 1, 1], attenuation_k: 0, link: false },
    frameTimes: [],
  };

  function currentLight() {
    return lightAt(trace.keyframes, state.t / trace.duration_s);
  }

  async function renderFrame() {
    const light = currentLight();
    const t0 = performance.now();
    const u = new ArrayBuffer(64);
    new Uint32Array(u, 0, 4).set([nPixels, 1, width, 0]);
    new Float32Array(u, 16, 4).set([...light.center, light.radius]);
    new Float32Array(u, 32, 4).set([
      light.rgb[0] * state.controls.rgb[0],
      light.rgb[1] * state.controls.rgb[1],
      light.rgb[2] * state.controls.rgb[2],
      0,
    ]);
    new Float32Array(u, 48, 4).set([state.controls.link ? 1 : 0, state.controls.attenuation_k, 0, 0]);
    device.queue.writeBuffer(paramsBuf, 0, u);
    const encoder = device.createCommandEncoder();
    const cp = encoder.beginComputePass();
    cp.setPipeline(pipeline);
    cp.setBindGroup(0, bindGroup);
    cp.dispatchWorkgroups(Math.ceil(nPixels / 64));
    if (g1) {
      cp.setPipeline(g1.basePipe);
      cp.setBindGroup(0, g1.frames[g1.frame].baseBG);
      cp.dispatchWorkgroups(Math.ceil(g1.gn / 64));
      cp.setPipeline(g1.resPipe);
      cp.setBindGroup(0, g1.frames[g1.frame].resBG);
      cp.dispatchWorkgroups(Math.ceil(g1.gn / 64));
    }
    cp.end();
    encodeBlit(encoder, kitchenBlit);
    if (g1) { encodeBlit(encoder, g1.blit); encodeBlit(encoder, g1.refBlit); }
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const ms = performance.now() - t0;
    state.frameTimes.push(ms);
    if (state.frameTimes.length > 5000) state.frameTimes.shift();
    return ms;
  }

  async function readFrame() {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(outputsBuf, 0, readBuf, 0, nPixels * 3 * 4);
    device.queue.submit([encoder.finish()]);
    await readBuf.mapAsync(GPUMapMode.READ);
    const bytes = new Uint8Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();
    let s = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(s);
  }

  // --- UI wiring ---
  const $ = (id) => document.getElementById(id);
  $("time").addEventListener("input", () => { state.t = Number($("time").value); state.playing = false; $("play").textContent = "Play"; });
  $("play").addEventListener("click", () => { state.playing = !state.playing; $("play").textContent = state.playing ? "Pause" : "Play"; });
  $("link").addEventListener("change", () => { state.controls.link = $("link").checked; });
  $("atten").addEventListener("input", () => { state.controls.attenuation_k = Number($("atten").value); $("klabel").textContent = Number($("atten").value).toFixed(2); });
  for (const c of ["r", "g", "b"]) {
    $(`rgb_${c}`).addEventListener("input", () => {
      state.controls.rgb = [Number($("rgb_r").value), Number($("rgb_g").value), Number($("rgb_b").value)];
    });
  }
  if (g1) {
    $("g1frame").addEventListener("input", () => {
      g1.frame = Number($("g1frame").value);
      $("g1label").textContent = String(g1.frame);
      device.queue.writeBuffer(g1.gRef, 0, g1.frames[g1.frame].ref);
    });
  }

  function stats(times) {
    if (!times.length) return { mean_ms: 0, p95_ms: 0, count: 0 };
    const sorted = [...times].sort((a, b) => a - b);
    const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
    return {
      mean_ms: times.reduce((a, b) => a + b, 0) / times.length,
      p50_ms: q(0.5),
      p95_ms: q(0.95),
      min_ms: sorted[0],
      max_ms: sorted[sorted.length - 1],
      count: times.length,
    };
  }

  let last = performance.now();
  let rendering = false;
  async function loop(now) {
    if (state.driven) return; // the trace runner drives frames explicitly
    if (state.playing) {
      state.t = (state.t + (now - last) / 1000) % trace.duration_s;
      $("time").value = String(state.t);
    }
    last = now;
    $("tlabel").textContent = state.t.toFixed(2);
    if (!rendering) {
      rendering = true;
      await renderFrame();
      rendering = false;
      const recent = stats(state.frameTimes.slice(-120));
      statsEl.textContent =
        `frame ${recent.mean_ms.toFixed(1)} ms mean / ${recent.p95_ms.toFixed(1)} ms p95 ` +
        `(${(1000 / recent.p95_ms).toFixed(0)} fps p95, last 120) — ` +
        `${adapter.info?.vendor || "?"} ${adapter.info?.architecture || ""}`;
    }
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  window.__demo = {
    ready: true,
    adapter: { vendor: adapter.info?.vendor || null, architecture: adapter.info?.architecture || null },
    resolution: [width, height],
    setDriven(d) {
      state.driven = d;
      state.playing = !d;
      if (!d) { last = performance.now(); requestAnimationFrame(loop); }
    },
    setTime(t) { state.t = t; document.getElementById("time").value = String(t); document.getElementById("tlabel").textContent = t.toFixed(2); },
    setControls(c) { state.controls = { ...state.controls, ...c }; },
    setG1Frame(i) { if (g1) { g1.frame = i; device.queue.writeBuffer(g1.gRef, 0, g1.frames[i].ref); } },
    hasG1: !!g1,
    lightAt(t) { const l = lightAt(trace.keyframes, t / trace.duration_s); return l; },
    renderFrame,
    readFrame,
    frameStats() { return stats(state.frameTimes); },
    trace,
  };
}

init().catch((e) => {
  document.getElementById("stats").textContent = `init failed: ${e.message}`;
  window.__demo = { ready: false, error: String(e) };
  throw e;
});
