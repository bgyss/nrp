// H4 rig compositor, page-side (loaded as an ES module by h4.html). Fetches its
// own data via fetch() -- an earlier version of bench_h4.mjs shipped the same
// blobs through page.evaluate()'s argument channel instead, and a ~100 MB combined
// payload reproducibly killed the whole Chrome process before any page code ran
// ("Target page, context or browser has been closed", confirmed via a
// binary-search probe: a 95 MB plain array alone as a single page.evaluate
// argument reproduces the crash). fetch() against a local static server (same
// pattern as demo/main.mjs) has no such limit -- this is that fix.
//
// Compositing is a two-stage pipeline exploiting Eq. 1/Eq. 3 structure (the GPU
// analog of H2's CPU optimizer hoist): each light's MLP dispatch writes its *raw*
// (pre-rgb) output into a per-light slice of one contributions buffer, and a
// cheap composite pass sums rgb_l * raw_l over active lights. Raw outputs are
// constant under rgb slider edits, so an rgb nudge re-runs only the composite
// pass (no MLP dispatch at all); a light-shape param edit marks just that light
// dirty (one MLP dispatch + composite). The first version re-dispatched all 8
// full MLP passes (~21 ms each) on every nudge -- 170 ms p95 against a 100 ms
// target -- purely because the loop didn't distinguish "rgb changed" from
// "raw contribution changed".
//
// Exposes `window.__h4` for the Playwright driver (bench_h4.mjs): `ready`,
// `error`, `manifest`, and methods that only ever cross the page.evaluate
// boundary with small JSON (never a pixel buffer).
import { buildShader, repackMlp } from "./shader_gen.mjs";

const EXPORT = "/out/h4-rig-export";

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

function mkBuf(device, data, usage) {
  const buf = device.createBuffer({
    size: Math.max(16, data.length * 4),
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data instanceof Float32Array ? data : Float32Array.from(data));
  return buf;
}

// Composite pass: outputs[p] = sum over lights of weight_l.xyz * contrib[l][p],
// weight_l.w > 0.5 gates active (solo/mute) lights. One thread per pixel.
function compositeShader(nLights) {
  return `
struct CompositeParams { n_pixels: u32, n_lights: u32, pad0: u32, pad1: u32, weights: array<vec4<f32>, ${nLights}> };
@group(0) @binding(0) var<storage, read> contrib: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(2) var<uniform> params: CompositeParams;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= params.n_pixels) { return; }
  let base = gid.x * 3u;
  var c = vec3<f32>(0.0);
  for (var l: u32 = 0u; l < params.n_lights; l = l + 1u) {
    let w = params.weights[l];
    if (w.w > 0.5) {
      let src = l * params.n_pixels * 3u + base;
      c = c + w.xyz * vec3<f32>(contrib[src], contrib[src + 1u], contrib[src + 2u]);
    }
  }
  outputs[base] = c.x;
  outputs[base + 1u] = c.y;
  outputs[base + 2u] = c.z;
}
`;
}

async function init() {
  const manifest = await fetchJson(`${EXPORT}/manifest.json`);
  const [width, height] = manifest.resolution;
  const nPixels = width * height;

  const adapter = await navigator.gpu.requestAdapter();
  const adapterInfo = adapter.info || {};
  const device = await adapter.requestDevice();

  const pixels = await fetchF32(`${EXPORT}/pixels.bin`);
  const compositeReference = await fetchF32(`${EXPORT}/composite_reference.bin`);

  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    ],
  });
  const pixelsBuf = mkBuf(device, pixels, GPUBufferUsage.STORAGE);
  const sliceBytes = nPixels * 3 * 4; // 512^2*12 = 3 MiB, 256-byte aligned
  const nLights = manifest.lights.length;
  const contribBuf = device.createBuffer({
    size: sliceBytes * nLights,
    usage: GPUBufferUsage.STORAGE,
  });
  const outputsBuf = device.createBuffer({
    size: sliceBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: sliceBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const lights = [];
  for (const [lightIndex, l] of manifest.lights.entries()) {
    const lightManifest = { ...l, resolution: manifest.resolution };
    // Raw contribution pass: rigLight sources light params from the storage
    // buffer; light_rgb stays (1,1,1) forever so the slice holds the pre-rgb
    // output and the composite pass owns the Eq. 3 multiply.
    const shader = buildShader(lightManifest, { rigLight: true, composite: "write" });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: device.createShaderModule({ code: shader }), entryPoint: "main" },
    });
    const tables = l.files.tables ? await fetchF32(`${EXPORT}/${l.files.tables}`) : new Float32Array([0]);
    const mlpRaw = await fetchF32(`${EXPORT}/${l.files.mlp}`);
    const tablesBuf = mkBuf(device, tables, GPUBufferUsage.STORAGE);
    const mlpBuf = mkBuf(device, Float32Array.from(repackMlp(Array.from(mlpRaw), l.mlp_dims)), GPUBufferUsage.STORAGE);
    const paramsBuf = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 4).set([nPixels, 1, width, 0]);
    new Float32Array(u, 32, 4).set([1, 1, 1, 0]);
    device.queue.writeBuffer(paramsBuf, 0, u);
    const lightParamsBuf = mkBuf(device, Float32Array.from(l.light_params), GPUBufferUsage.STORAGE);
    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: pixelsBuf } },
        { binding: 1, resource: { buffer: tablesBuf } },
        { binding: 2, resource: { buffer: mlpBuf } },
        { binding: 3, resource: { buffer: contribBuf, offset: lightIndex * sliceBytes, size: sliceBytes } },
        { binding: 4, resource: { buffer: paramsBuf } },
        { binding: 5, resource: { buffer: lightParamsBuf } },
      ],
    });
    const rgb = l.rgb || [1, 1, 1];
    lights.push({
      name: l.name,
      pipeline,
      bindGroup,
      lightParamsBuf,
      lightParams: Array.from(l.light_params),
      rgb: [...rgb],
      dirty: true, // raw contribution not yet computed
    });
  }

  const compositeBgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const compositePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [compositeBgl] }),
    compute: {
      module: device.createShaderModule({ code: compositeShader(nLights) }),
      entryPoint: "main",
    },
  });
  const compositeParamsBuf = device.createBuffer({
    size: 16 + nLights * 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  const compositeBindGroup = device.createBindGroup({
    layout: compositeBgl,
    entries: [
      { binding: 0, resource: { buffer: contribBuf } },
      { binding: 1, resource: { buffer: outputsBuf } },
      { binding: 2, resource: { buffer: compositeParamsBuf } },
    ],
  });

  function writeCompositeParams(activeNames) {
    const u = new ArrayBuffer(16 + nLights * 16);
    new Uint32Array(u, 0, 4).set([nPixels, nLights, 0, 0]);
    const w = new Float32Array(u, 16, nLights * 4);
    lights.forEach((entry, i) => {
      w.set([entry.rgb[0], entry.rgb[1], entry.rgb[2], activeNames.includes(entry.name) ? 1 : 0], i * 4);
    });
    device.queue.writeBuffer(compositeParamsBuf, 0, u);
  }

  const workgroups = Math.ceil(nPixels / 64);
  const allNames = lights.map((l) => l.name);

  // Dispatches every dirty light's raw-contribution pass, then the composite
  // pass over `activeNames`. rgb edits never mark a light dirty (Eq. 3 hoist);
  // `forceDirty` re-dispatches all active lights regardless (parity/overhead
  // measurements, which quantify the cold full-render cost).
  async function render(activeNames, readBack, forceDirty = false) {
    const t0 = performance.now();
    if (forceDirty) {
      for (const entry of lights) if (activeNames.includes(entry.name)) entry.dirty = true;
    }
    writeCompositeParams(activeNames);
    const encoder = device.createCommandEncoder();
    for (const entry of lights) {
      if (!entry.dirty) continue;
      const pass = encoder.beginComputePass();
      pass.setPipeline(entry.pipeline);
      pass.setBindGroup(0, entry.bindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
      entry.dirty = false;
    }
    {
      const pass = encoder.beginComputePass();
      pass.setPipeline(compositePipeline);
      pass.setBindGroup(0, compositeBindGroup);
      pass.dispatchWorkgroups(workgroups);
      pass.end();
    }
    let result = null;
    if (readBack) {
      encoder.copyBufferToBuffer(outputsBuf, 0, readBuf, 0, sliceBytes);
      device.queue.submit([encoder.finish()]);
      await readBuf.mapAsync(GPUMapMode.READ);
      result = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();
    } else {
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
    return { result, elapsedMs: performance.now() - t0 };
  }

  async function parity() {
    const { result } = await render(allNames, true, true);
    let maxAbsDiff = 0;
    for (let i = 0; i < result.length; i++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(result[i] - compositeReference[i]));
    }
    return maxAbsDiff;
  }

  async function overhead(warmupFrames, timedFrames) {
    const rows = [];
    for (let k = 1; k <= allNames.length; k++) {
      const active = allNames.slice(0, k);
      for (let i = 0; i < warmupFrames; i++) await render(active, false, true);
      let total = 0;
      for (let i = 0; i < timedFrames; i++) {
        const { elapsedMs } = await render(active, false, true);
        total += elapsedMs;
      }
      rows.push({ n_lights: k, ms: total / timedFrames });
    }
    return rows;
  }

  // rgb slider session (V2's methodology): only the composite weights change,
  // so the cached raw contributions are reused -- no MLP dispatch per nudge.
  async function sliderLoop(adjustments) {
    await render(allNames, false); // warmup (also fills any dirty slices)
    const latencyMs = [];
    for (const adj of adjustments) {
      const entry = lights.find((l) => l.name === adj.light);
      entry.rgb = adj.rgb;
      const { elapsedMs } = await render(allNames, false);
      latencyMs.push(elapsedMs);
    }
    return latencyMs;
  }

  // Worst-case slider session: a light *shape* param edit invalidates that
  // light's cached raw output -- one MLP dispatch + composite per nudge.
  async function sliderLoopParamEdit(adjustments) {
    await render(allNames, false); // warmup
    const latencyMs = [];
    for (const adj of adjustments) {
      const entry = lights.find((l) => l.name === adj.light);
      entry.lightParams = adj.light_params;
      device.queue.writeBuffer(entry.lightParamsBuf, 0, Float32Array.from(adj.light_params));
      entry.dirty = true;
      const { elapsedMs } = await render(allNames, false);
      latencyMs.push(elapsedMs);
    }
    return latencyMs;
  }

  window.__h4 = {
    ready: true,
    manifest,
    adapter: { vendor: adapterInfo.vendor || null, architecture: adapterInfo.architecture || null },
    lightNames: allNames,
    lightParamsByName: Object.fromEntries(lights.map((l) => [l.name, [...l.lightParams]])),
    parity,
    overhead,
    sliderLoop,
    sliderLoopParamEdit,
  };
}

init().catch((e) => {
  window.__h4 = { ready: false, error: e.message + "\n" + (e.stack || "") };
});
