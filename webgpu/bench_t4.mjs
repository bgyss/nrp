// T4 (docs/production-track.md): runtime baseline lock.
//
// Runs the *actual* exported T1-scene proxy — hashgrid encoding included, ported to
// WGSL — inside real Chrome (same Playwright setup as bench_browser.mjs, E6), against
// the real scene's exported G-buffer. Produces a frame-time histogram (mean and p95,
// not just mean) at 128/256/512 squared, checks parity against the PyTorch reference
// image, and can freeze or check a committed regression baseline:
//
//   node bench_t4.mjs                     # run, write out/t4-runtime/report.json
//   node bench_t4.mjs --freeze            # additionally write the baseline JSON
//   node bench_t4.mjs --check             # fail (exit 1) if parity breaks, p95
//                                         # regresses beyond the baseline threshold,
//                                         # or the 30 fps p95 floor at 512^2 fails
//
// Per timed frame only the light uniform changes (jittered per frame): the G-buffer
// inputs, weights, and hashgrid tables are resident, exactly the interactive-relight
// access pattern the proxy exists to serve.
import { chromium } from "playwright";
import { buildShader, repackMlp } from "./shader_gen.mjs";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const exportDir = path.join(repoRoot, "out", "t4-runtime", "export");
const reportPath = path.join(repoRoot, "out", "t4-runtime", "report.json");
const baselinePath = path.join(repoRoot, "out", "t4-runtime", "baseline.json");

const PARITY_TOLERANCE = 2e-4; // f32 op-order drift over hashgrid + 128-wide MLP
const REGRESSION_THRESHOLD = 0.3; // fail --check if p95 exceeds baseline p95 by >30%
const FLOOR_P95_MS_512 = 1000 / 30; // the E10/T4 floor: 30 fps sustained at 512^2

// Runs inside the browser page — self-contained.
async function runInPage({ shaderCode, pixels, tables, mlp, reference, manifest, resolutions, warmup, timedRuns }) {
  const adapter = await navigator.gpu.requestAdapter();
  const adapterInfo = adapter.info || {};
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: shaderCode });
  const bgl = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
    compute: { module, entryPoint: "main" },
  });
  const mk = (arr, usage) => {
    const buf = device.createBuffer({ size: arr.length * 4, usage: usage | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buf, 0, Float32Array.from(arr));
    return buf;
  };
  const pixelsBuf = mk(pixels, GPUBufferUsage.STORAGE);
  const tablesBuf = mk(tables.length ? tables : [0], GPUBufferUsage.STORAGE);
  const mlpBuf = mk(mlp, GPUBufferUsage.STORAGE);
  const light = manifest.default_light;
  const fullSide = manifest.resolution[0];

  function makePass(side) {
    const n = side * side;
    const stride = fullSide / side;
    const outputsBuf = device.createBuffer({ size: n * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: n * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const paramsBuf = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: pixelsBuf } },
        { binding: 1, resource: { buffer: tablesBuf } },
        { binding: 2, resource: { buffer: mlpBuf } },
        { binding: 3, resource: { buffer: outputsBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
    });
    return { n, stride, side, outputsBuf, readBuf, paramsBuf, bindGroup };
  }

  async function frame(pass, lightParams, readBack) {
    const t0 = performance.now();
    const u = new ArrayBuffer(32);
    new Uint32Array(u, 0, 4).set([pass.n, pass.stride, pass.side, 0]);
    new Float32Array(u, 16, 4).set(lightParams);
    device.queue.writeBuffer(pass.paramsBuf, 0, u);
    const encoder = device.createCommandEncoder();
    const p = encoder.beginComputePass();
    p.setPipeline(pipeline);
    p.setBindGroup(0, pass.bindGroup);
    p.dispatchWorkgroups(Math.ceil(pass.n / 64));
    p.end();
    if (readBack) encoder.copyBufferToBuffer(pass.outputsBuf, 0, pass.readBuf, 0, pass.n * 3 * 4);
    device.queue.submit([encoder.finish()]);
    let result = null;
    if (readBack) {
      await pass.readBuf.mapAsync(GPUMapMode.READ);
      result = new Float32Array(pass.readBuf.getMappedRange().slice(0));
      pass.readBuf.unmap();
    } else {
      await device.queue.onSubmittedWorkDone();
    }
    return { result, elapsedMs: performance.now() - t0 };
  }

  // Parity at full export resolution against the PyTorch reference.
  const defaultParams = [...light.center, light.radius];
  const parityPass = makePass(fullSide);
  const { result: parityOut } = await frame(parityPass, defaultParams, true);
  let maxAbsDiff = 0;
  for (let i = 0; i < parityOut.length; i++) {
    maxAbsDiff = Math.max(maxAbsDiff, Math.abs(parityOut[i] - reference[i]));
  }

  // Latency sweep: light jittered per frame (interactive-relight access pattern).
  const stats = (times) => {
    const sorted = [...times].sort((a, b) => a - b);
    const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
    const mean = times.reduce((a, b) => a + b, 0) / times.length;
    return { mean_ms: mean, p50_ms: q(0.5), p95_ms: q(0.95), min_ms: sorted[0], max_ms: sorted[sorted.length - 1] };
  };
  const latencyRows = [];
  for (const side of resolutions) {
    const pass = makePass(side);
    for (let i = 0; i < warmup; i++) await frame(pass, defaultParams, false);
    const times = [];
    for (let i = 0; i < timedRuns; i++) {
      const jitter = 0.05 * Math.sin(i * 0.7);
      const lp = [light.center[0] + jitter, light.center[1], light.center[2] - jitter, light.radius];
      const { elapsedMs } = await frame(pass, lp, false);
      times.push(elapsedMs);
    }
    const s = stats(times);
    latencyRows.push({
      resolution: [side, side],
      timed_frames: timedRuns,
      ...s,
      fps_mean: 1000 / s.mean_ms,
      fps_p95: 1000 / s.p95_ms,
      meets_30_fps_p95: 1000 / s.p95_ms >= 30,
      meets_60_fps_p95: 1000 / s.p95_ms >= 60,
      frame_times_ms: times.map((t) => Math.round(t * 1000) / 1000),
    });
  }
  return {
    adapter_vendor: adapterInfo.vendor || null,
    adapter_architecture: adapterInfo.architecture || null,
    parity_vs_pytorch_max_abs_diff: maxAbsDiff,
    parity_resolution: manifest.resolution,
    latency_sweep: latencyRows,
  };
}

function loadF32(name) {
  const buf = readFileSync(path.join(exportDir, name));
  return Array.from(new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4));
}

async function main() {
  const mode = process.argv.includes("--check") ? "check" : process.argv.includes("--freeze") ? "freeze" : "run";
  if (!existsSync(path.join(exportDir, "manifest.json"))) {
    console.error(`missing ${exportDir}/manifest.json — run: mise run t4-export`);
    process.exit(2);
  }
  const manifest = JSON.parse(readFileSync(path.join(exportDir, "manifest.json"), "utf8"));
  const shaderCode = buildShader(manifest);

  const pagePath = path.join(__dirname, "blank.html");
  writeFileSync(pagePath, "<!doctype html><title>t4</title>");
  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--headless=new", "--no-sandbox", "--ignore-gpu-blocklist", "--use-angle=metal", "--enable-unsafe-webgpu"],
  });
  const page = await browser.newPage();
  await page.goto(`file://${pagePath}`);
  const result = await page.evaluate(runInPage, {
    shaderCode,
    pixels: loadF32("pixels.bin"),
    tables: manifest.encoding ? loadF32("tables.bin") : [],
    mlp: repackMlp(loadF32("mlp.bin"), manifest.mlp_dims),
    reference: loadF32("reference.bin"),
    manifest,
    resolutions: [128, 256, 512],
    warmup: 10,
    timedRuns: 200,
  });
  await browser.close();

  const report = {
    rung: "T4",
    scope: "runtime baseline lock: exported T1-scene proxy (hashgrid in WGSL) in real Chrome",
    backend: "webgpu (Chrome, via Playwright)",
    model: manifest.model,
    parameter_count: manifest.parameter_count,
    mlp_dims: manifest.mlp_dims,
    parity_tolerance: PARITY_TOLERANCE,
    ...result,
    notes: [
      "Per timed frame only the light uniform changes (jittered): G-buffer, weights, " +
        "and hashgrid tables stay resident — the interactive-relight access pattern.",
      "Sub-512 resolutions use strided subsampling of the real 512^2 G-buffer.",
      "Timing includes uniform upload, dispatch, and onSubmittedWorkDone (no readback).",
    ],
  };
  mkdirSync(path.dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");

  const summary = {
    parity_vs_pytorch_max_abs_diff: result.parity_vs_pytorch_max_abs_diff,
    latency: result.latency_sweep.map(({ frame_times_ms, ...row }) => row),
  };
  console.log(JSON.stringify(summary, null, 2));

  if (result.parity_vs_pytorch_max_abs_diff > PARITY_TOLERANCE) {
    console.error(`FAIL: parity ${result.parity_vs_pytorch_max_abs_diff} > tolerance ${PARITY_TOLERANCE}`);
    process.exit(1);
  }

  if (mode === "freeze") {
    const baseline = {
      frozen_from: reportPath,
      adapter: `${result.adapter_vendor} / ${result.adapter_architecture}`,
      parity_tolerance: PARITY_TOLERANCE,
      regression_threshold: REGRESSION_THRESHOLD,
      floor_p95_ms_512: FLOOR_P95_MS_512,
      latency: Object.fromEntries(
        result.latency_sweep.map((r) => [r.resolution[0], { mean_ms: r.mean_ms, p95_ms: r.p95_ms }])
      ),
    };
    writeFileSync(baselinePath, JSON.stringify(baseline, null, 2) + "\n");
    console.log(`baseline frozen: ${baselinePath}`);
  }

  if (mode === "check") {
    if (!existsSync(baselinePath)) {
      console.error(`missing baseline ${baselinePath} — run: node bench_t4.mjs --freeze`);
      process.exit(2);
    }
    const baseline = JSON.parse(readFileSync(baselinePath, "utf8"));
    let failed = false;
    for (const row of result.latency_sweep) {
      const res = row.resolution[0];
      const base = baseline.latency[res];
      const limit = base.p95_ms * (1 + baseline.regression_threshold);
      if (row.p95_ms > limit) {
        console.error(`FAIL: ${res}^2 p95 ${row.p95_ms.toFixed(2)} ms > baseline ${base.p95_ms.toFixed(2)} ms +${baseline.regression_threshold * 100}%`);
        failed = true;
      }
      if (res === 512 && row.p95_ms > baseline.floor_p95_ms_512) {
        console.error(`FAIL: 512^2 p95 ${row.p95_ms.toFixed(2)} ms misses the 30 fps floor (${baseline.floor_p95_ms_512.toFixed(2)} ms)`);
        failed = true;
      }
    }
    if (failed) process.exit(1);
    console.log("baseline check passed");
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
