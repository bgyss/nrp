// H4 (docs/hardening-track.md): N-light rig compositing on the T4/G2 WGSL runtime.
//
// V1's slider loop is CPU torch: 950 ms mean per adjustment for the 8-light kitchen
// rig (~111 ms/light, linear in N), ~29x off real time, while T4 proved the
// identical proxy architecture at 20.9 ms/frame at 512^2 in real Chrome. This ports
// N-light rig compositing to that WGSL runtime: one compute dispatch per active
// rig light, each `light_rgb`-multiplied (Eq. 3) and additively composited into one
// shared output buffer (Eq. 1's linearity) -- shader_gen.mjs's new `rigLight`/`add`
// composite mode (H2^H4 commit). Solo/mute is resolved on the JS side (which
// lights get dispatched this frame), not a per-shader uniform.
//
//   node bench_h4.mjs                     # run, write out/h4-rig/report.json
//
// Measures: GPU-vs-CPU composite parity against LightRig.render (all 8 lights
// active); a scripted slider-session latency (mean/p95) at 512^2 nudging rgb for
// each colorable light in turn (mirrors V2's `slider_loop` methodology); and
// per-light marginal cost (soloing the first k lights, k=1..N, linear fit),
// mirroring V1's compositing-overhead-vs-light-count methodology.
import { chromium } from "playwright";
import { buildShader, repackMlp } from "./shader_gen.mjs";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const exportDir = path.join(repoRoot, "out", "h4-rig-export");
const reportPath = path.join(repoRoot, "out", "h4-rig", "report.json");

const PARITY_TOLERANCE = 2e-4; // f32 op-order drift, same basis as T4

// Runs inside the browser page — self-contained (no Node globals available here).
async function runInPage({ lightsData, compositeReference, resolution, adjustments }) {
  const adapter = await navigator.gpu.requestAdapter();
  const adapterInfo = adapter.info || {};
  const device = await adapter.requestDevice();
  const [width, height] = resolution;
  const nPixels = width * height;

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
  const mk = (arr, usage) => {
    const buf = device.createBuffer({
      size: Math.max(16, arr.length * 4),
      usage: usage | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, Float32Array.from(arr));
    return buf;
  };
  const pixelsBuf = mk(lightsData[0].pixels, GPUBufferUsage.STORAGE);
  const outputsBuf = device.createBuffer({
    size: nPixels * 3 * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: nPixels * 3 * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const lights = lightsData.map((l) => {
    const shaderWrite = l.shaderWrite;
    const shaderAdd = l.shaderAdd;
    const pipeWrite = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: device.createShaderModule({ code: shaderWrite }), entryPoint: "main" },
    });
    const pipeAdd = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
      compute: { module: device.createShaderModule({ code: shaderAdd }), entryPoint: "main" },
    });
    const tablesBuf = mk(l.tables.length ? l.tables : [0], GPUBufferUsage.STORAGE);
    const mlpBuf = mk(l.mlp, GPUBufferUsage.STORAGE);
    const paramsBuf = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Light params (up to 8 + texture-size*3 floats for textured_quad) live in a
    // storage buffer, not the small uniform -- see shader_gen.mjs's rigLight note.
    const lightParamsBuf = mk(l.lightParams, GPUBufferUsage.STORAGE);
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
    return { name: l.name, pipeWrite, pipeAdd, bindGroup, paramsBuf, lightParams: l.lightParams, rgb: l.rgb };
  });

  function writeParams(entry) {
    const u = new ArrayBuffer(48);
    new Uint32Array(u, 0, 4).set([nPixels, 1, width, 0]);
    // bytes 16-32 (params.light) are unused by rigLight shaders -- left zero.
    new Float32Array(u, 32, 4).set([entry.rgb[0], entry.rgb[1], entry.rgb[2], 0]);
    device.queue.writeBuffer(entry.paramsBuf, 0, u);
  }
  for (const entry of lights) writeParams(entry);

  const workgroups = Math.ceil(nPixels / 64);

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

  const allNames = lights.map((l) => l.name);

  // Parity: all 8 lights active, GPU composite vs LightRig.render (CPU) reference.
  const { result: parityOut } = await render(allNames, true);
  let maxAbsDiff = 0;
  for (let i = 0; i < parityOut.length; i++) {
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(parityOut[i] - compositeReference[i]));
  }

  // Per-light marginal cost: solo the first k lights (V1's overhead-vs-N methodology).
  const overheadRows = [];
  const warmupFrames = 3;
  const timedFrames = 20;
  for (let k = 1; k <= allNames.length; k++) {
    const active = allNames.slice(0, k);
    for (let i = 0; i < warmupFrames; i++) await render(active, false);
    let total = 0;
    for (let i = 0; i < timedFrames; i++) {
      const { elapsedMs } = await render(active, false);
      total += elapsedMs;
    }
    overheadRows.push({ n_lights: k, ms: total / timedFrames });
  }

  // Scripted slider session: one rgb nudge at a time (V2's slider_loop methodology).
  const sliderLatencyMs = [];
  await render(allNames, false); // warmup, untimed
  const rgbByName = Object.fromEntries(lights.map((l) => [l.name, l.rgb.slice()]));
  for (const adj of adjustments) {
    rgbByName[adj.light] = adj.rgb;
    const entry = lights.find((l) => l.name === adj.light);
    entry.rgb = adj.rgb;
    writeParams(entry);
    const { elapsedMs } = await render(allNames, false);
    sliderLatencyMs.push(elapsedMs);
  }

  return {
    adapter_vendor: adapterInfo.vendor || null,
    adapter_architecture: adapterInfo.architecture || null,
    parity_vs_cpu_max_abs_diff: maxAbsDiff,
    compositing_overhead_ms: overheadRows,
    slider_latency_ms: sliderLatencyMs,
  };
}

function loadF32(dir, name) {
  if (!name) return [];
  const buf = readFileSync(path.join(dir, name));
  return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
}

async function main() {
  if (!existsSync(path.join(exportDir, "manifest.json"))) {
    console.error(`missing ${exportDir}/manifest.json — run: mise run h4-export`);
    process.exit(2);
  }
  const manifest = JSON.parse(readFileSync(path.join(exportDir, "manifest.json"), "utf8"));
  const pixels = loadF32(exportDir, "pixels.bin");
  const compositeReference = loadF32(exportDir, "composite_reference.bin");

  const lightsData = manifest.lights.map((l) => {
    const lightManifest = { ...l, resolution: manifest.resolution };
    return {
      name: l.name,
      pixels,
      tables: loadF32(exportDir, l.files.tables),
      mlp: Array.from(repackMlp(loadF32(exportDir, l.files.mlp), l.mlp_dims)),
      lightParams: l.light_params,
      rgb: l.rgb || [1, 1, 1],
      shaderWrite: buildShader(lightManifest, { rigLight: true, composite: "write" }),
      shaderAdd: buildShader(lightManifest, { rigLight: true, composite: "add" }),
    };
  });

  // Scripted per-light rgb nudges, round-robin across colorable lights (matches
  // examples/v2_art_loop.py::default_adjustments's methodology, fixed seed 0).
  const colorable = manifest.lights.filter((l) => l.rgb).map((l) => l.name);
  const adjustments = [];
  let seed = 0;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 10 && colorable.length; i++) {
    adjustments.push({
      light: colorable[i % colorable.length],
      rgb: [0.3 + rand() * 3.7, 0.3 + rand() * 3.7, 0.3 + rand() * 3.7],
    });
  }

  const pagePath = path.join(__dirname, "blank.html");
  writeFileSync(pagePath, "<!doctype html><title>h4</title>");
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: [
      "--headless=new", "--no-sandbox", "--ignore-gpu-blocklist",
      "--use-angle=metal", "--enable-unsafe-webgpu",
    ],
  });
  const page = await browser.newPage();
  await page.goto(`file://${pagePath}`);
  const result = await page.evaluate(runInPage, {
    lightsData,
    compositeReference,
    resolution: manifest.resolution,
    adjustments,
  });
  await browser.close();

  const stats = (times) => {
    const sorted = [...times].sort((a, b) => a - b);
    const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    return { mean_ms: mean, p50_ms: q(0.5), p95_ms: q(0.95), min_ms: sorted[0], max_ms: sorted[sorted.length - 1] };
  };

  const ns = result.compositing_overhead_ms.map((r) => r.n_lights);
  const mss = result.compositing_overhead_ms.map((r) => r.ms);
  const n = ns.length;
  const sumX = ns.reduce((a, b) => a + b, 0);
  const sumY = mss.reduce((a, b) => a + b, 0);
  const sumXY = ns.reduce((a, x, i) => a + x * mss[i], 0);
  const sumXX = ns.reduce((a, x) => a + x * x, 0);
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  const report = {
    rung: "H4",
    scope: "N-light rig compositing on the T4/G2 WGSL runtime, all lights active",
    backend: "webgpu (Chrome, via Playwright)",
    adapter: { vendor: result.adapter_vendor, architecture: result.adapter_architecture },
    resolution: manifest.resolution,
    n_lights: manifest.lights.length,
    parity_vs_cpu_max_abs_diff: result.parity_vs_cpu_max_abs_diff,
    parity_tolerance: PARITY_TOLERANCE,
    parity_passed: result.parity_vs_cpu_max_abs_diff <= PARITY_TOLERANCE,
    compositing_overhead_ms: {
      rows: result.compositing_overhead_ms,
      fit_slope_ms_per_light: slope,
      fit_intercept_ms: intercept,
    },
    slider_loop: {
      n_adjustments: result.slider_latency_ms.length,
      latency_ms: result.slider_latency_ms,
      ...stats(result.slider_latency_ms),
    },
    v1_cpu_baseline_ms_per_light: 111,
    v2_cpu_baseline_slider_mean_ms: 950,
    meets_p95_100ms: stats(result.slider_latency_ms).p95_ms <= 100,
    meets_p95_33ms_stretch: stats(result.slider_latency_ms).p95_ms <= 33,
  };
  mkdirSync(path.dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(
    `parity ${report.parity_passed ? "PASS" : "FAIL"} (${report.parity_vs_cpu_max_abs_diff.toExponential(2)}), ` +
    `slider p95 ${report.slider_loop.p95_ms.toFixed(2)} ms, ` +
    `marginal cost ${slope.toFixed(3)} ms/light -- wrote ${reportPath}`
  );
}

main();
