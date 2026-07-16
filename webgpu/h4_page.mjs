// H4 rig compositor, page-side (loaded as an ES module by h4.html). Fetches its
// own data via fetch() -- an earlier version of bench_h4.mjs shipped the same
// blobs through page.evaluate()'s argument channel instead, and a ~100 MB
// combined payload reproducibly killed the whole Chrome process before any page
// code ran ("Target page, context or browser has been closed", confirmed via a
// binary-search probe: a 95 MB plain array alone as a single page.evaluate
// argument reproduces the crash). fetch() against a local static server (same
// pattern as demo/main.mjs) has no such limit -- this is that fix.
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
  const outputsBuf = device.createBuffer({
    size: nPixels * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: nPixels * 3 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const lights = [];
  for (const l of manifest.lights) {
    const lightManifest = { ...l, resolution: manifest.resolution };
    const shaderWrite = buildShader(lightManifest, { rigLight: true, composite: "write" });
    const shaderAdd = buildShader(lightManifest, { rigLight: true, composite: "add" });
    const pipeWrite = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: device.createShaderModule({ code: shaderWrite }), entryPoint: "main" },
    });
    const pipeAdd = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: device.createShaderModule({ code: shaderAdd }), entryPoint: "main" },
    });
    const tables = l.files.tables ? await fetchF32(`${EXPORT}/${l.files.tables}`) : new Float32Array([0]);
    const mlpRaw = await fetchF32(`${EXPORT}/${l.files.mlp}`);
    const tablesBuf = mkBuf(device, tables, GPUBufferUsage.STORAGE);
    const mlpBuf = mkBuf(device, Float32Array.from(repackMlp(Array.from(mlpRaw), l.mlp_dims)), GPUBufferUsage.STORAGE);
    const paramsBuf = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const lightParamsBuf = mkBuf(device, Float32Array.from(l.light_params), GPUBufferUsage.STORAGE);
    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: pixelsBuf } },
        { binding: 1, resource: { buffer: tablesBuf } },
        { binding: 2, resource: { buffer: mlpBuf } },
        { binding: 3, resource: { buffer: outputsBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
        { binding: 5, resource: { buffer: lightParamsBuf } },
      ],
    });
    const rgb = l.rgb || [1, 1, 1];
    lights.push({ name: l.name, pipeWrite, pipeAdd, bindGroup, paramsBuf, rgb: [...rgb] });
  }

  function writeParams(entry) {
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 4).set([nPixels, 1, width, 0]);
    new Float32Array(u, 32, 4).set([entry.rgb[0], entry.rgb[1], entry.rgb[2], 0]);
    device.queue.writeBuffer(entry.paramsBuf, 0, u);
  }
  for (const entry of lights) writeParams(entry);

  const workgroups = Math.ceil(nPixels / 64);
  const allNames = lights.map((l) => l.name);

  async function render(activeNames, readBack) {
    const active = lights.filter((e) => activeNames.includes(e.name));
    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    if (active.length === 0) {
      encoder.clearBuffer(outputsBuf);
    } else {
      active.forEach((entry, i) => {
        const pass = encoder.beginComputePass();
        pass.setPipeline(i === 0 ? entry.pipeWrite : entry.pipeAdd);
        pass.setBindGroup(0, entry.bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
      });
    }
    let result = null;
    if (readBack) {
      encoder.copyBufferToBuffer(outputsBuf, 0, readBuf, 0, nPixels * 3 * 4);
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
    const { result } = await render(allNames, true);
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
      for (let i = 0; i < warmupFrames; i++) await render(active, false);
      let total = 0;
      for (let i = 0; i < timedFrames; i++) {
        const { elapsedMs } = await render(active, false);
        total += elapsedMs;
      }
      rows.push({ n_lights: k, ms: total / timedFrames });
    }
    return rows;
  }

  async function sliderLoop(adjustments) {
    await render(allNames, false); // warmup
    const latencyMs = [];
    for (const adj of adjustments) {
      const entry = lights.find((l) => l.name === adj.light);
      entry.rgb = adj.rgb;
      writeParams(entry);
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
    parity,
    overhead,
    sliderLoop,
  };
}

init().catch((e) => {
  window.__h4 = { ready: false, error: e.message + "\n" + (e.stack || "") };
});
