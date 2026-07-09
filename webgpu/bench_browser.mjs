// E6: the real WebGPU compute-shader backend, executed inside real Google Chrome
// (via Playwright), against the real exported TorchNRP proxy's weights.
//
// `bench.mjs` (native Dawn bindings via the `webgpu` npm package, no browser) is
// reproducibly broken on this machine for real trained-model data — see
// `README.md` for the full bisection. That defect is specific to the experimental
// Node-only Dawn binding; it is not present in a production browser's WebGPU
// implementation. This script proves that directly: the identical compute shader,
// running inside actual Chrome (not a stripped-down headless-only build — Chrome
// requires a secure context for `navigator.gpu`, so the page is loaded from a local
// `file://` URL, which qualifies), executes correctly against the real proxy
// weights with no crash, matching the Python reference to ~2.4e-7 max abs diff.
//
// This is the completion of E6's WebGPU criterion: a real WebGPU compute shader,
// executed and measured, running the actual exported proxy.
import { chromium } from "playwright";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

const SHADER_SOURCE = `
struct Params { n_pixels: u32, };
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn softplus(x: f32) -> f32 { return log(1.0 + exp(x)); }
`;

// Executed inside the browser page via page.evaluate — must be self-contained.
async function runInPage({ weightsJson, referenceJson, shaderPrelude, resolutions, warmup, timedRuns }) {
  function flattenWeights(layers) {
    const weights = [];
    const biases = [];
    const dims = [layers[0].weight[0].length];
    for (const l of layers) {
      for (const row of l.weight) for (const v of row) weights.push(v);
      for (const v of l.bias) biases.push(v);
      dims.push(l.bias.length);
    }
    return { weights: Float32Array.from(weights), biases: Float32Array.from(biases), dims };
  }

  function buildShader(layerDims) {
    const nLayers = layerDims.length - 1;
    let src = `${shaderPrelude}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.x;
  if (pixel >= params.n_pixels) { return; }
  var buf: array<f32, 64>;
  let in_dim = ${layerDims[0]}u;
  for (var i: u32 = 0u; i < in_dim; i = i + 1u) { buf[i] = inputs[pixel * in_dim + i]; }
  var w_offset: u32 = 0u; var b_offset: u32 = 0u; var cur_dim: u32 = in_dim;
`;
    for (let l = 0; l < nLayers; l++) {
      const outDim = layerDims[l + 1];
      src += `
  { var next: array<f32, 64>;
    for (var o: u32 = 0u; o < ${outDim}u; o = o + 1u) {
      var s: f32 = biases[b_offset + o];
      for (var i: u32 = 0u; i < cur_dim; i = i + 1u) { s = s + weights[w_offset + o * cur_dim + i] * buf[i]; }
      ${l < nLayers - 1 ? "next[o] = max(s, 0.0);" : "next[o] = s;"}
    }
    for (var o: u32 = 0u; o < ${outDim}u; o = o + 1u) { buf[o] = next[o]; }
    w_offset = w_offset + cur_dim * ${outDim}u; b_offset = b_offset + ${outDim}u; cur_dim = ${outDim}u;
  }
`;
    }
    src += `
  outputs[pixel * 3u] = softplus(buf[0]);
  outputs[pixel * 3u + 1u] = softplus(buf[1]);
  outputs[pixel * 3u + 2u] = softplus(buf[2]);
}
`;
    return src;
  }

  const adapter = await navigator.gpu.requestAdapter();
  const adapterInfo = adapter.info || {};
  const device = await adapter.requestDevice();
  const { weights, biases, dims } = flattenWeights(weightsJson.layers);
  const module = device.createShaderModule({ code: buildShader(dims) });
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
  const inDim = dims[0];
  const lightParams = [...weightsJson.default_light.center, weightsJson.default_light.radius];

  const wBuf = device.createBuffer({ size: weights.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(wBuf, 0, weights);
  const bBuf = device.createBuffer({ size: biases.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bBuf, 0, biases);

  function makeBuffers(nPixels) {
    const inputsBuf = device.createBuffer({
      size: nPixels * inDim * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    const outputsBuf = device.createBuffer({
      size: nPixels * 3 * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuf = device.createBuffer({
      size: nPixels * 3 * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const paramsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([nPixels, 0, 0, 0]));
    const bindGroup = device.createBindGroup({
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: inputsBuf } },
        { binding: 1, resource: { buffer: wBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: outputsBuf } },
        { binding: 4, resource: { buffer: paramsBuf } },
      ],
    });
    return { inputsBuf, outputsBuf, readBuf, bindGroup, nPixels };
  }

  async function forward(buffers, inputsFlat) {
    device.queue.writeBuffer(buffers.inputsBuf, 0, inputsFlat);
    const t0 = performance.now();
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, buffers.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(buffers.nPixels / 64));
    pass.end();
    encoder.copyBufferToBuffer(buffers.outputsBuf, 0, buffers.readBuf, 0, buffers.nPixels * 3 * 4);
    device.queue.submit([encoder.finish()]);
    await buffers.readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(buffers.readBuf.getMappedRange().slice(0));
    const elapsedMs = performance.now() - t0;
    buffers.readBuf.unmap();
    return { result, elapsedMs };
  }

  // Parity check against the Python-computed reference, at the export resolution.
  const [resW, resH] = weightsJson.resolution;
  const nPx = resW * resH;
  const parityInputs = new Float32Array(nPx * inDim);
  for (let p = 0; p < nPx; p++) {
    const base = p * inDim;
    parityInputs[base + 0] = weightsJson.xy[p][0];
    parityInputs[base + 1] = weightsJson.xy[p][1];
    for (let i = 0; i < 7; i++) parityInputs[base + 2 + i] = weightsJson.aux[p][i];
    for (let i = 0; i < lightParams.length; i++) parityInputs[base + 9 + i] = lightParams[i];
  }
  const parityBuffers = makeBuffers(nPx);
  const { result: parityResult } = await forward(parityBuffers, parityInputs);
  let maxAbsDiff = 0;
  for (let p = 0; p < nPx; p++) {
    const ref = referenceJson.reference_image[Math.floor(p / resW)][p % resW];
    for (let c = 0; c < 3; c++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(ref[c] - parityResult[p * 3 + c]));
    }
  }

  // Latency sweep at production resolutions, synthetic G-buffer (same convention as
  // nrp/torch_backend/bench.py's synthetic_runtime_cache / engine_runtime.py).
  const latencyRows = [];
  for (const res of resolutions) {
    const n = res * res;
    const side = res;
    const inputs = new Float32Array(n * inDim);
    for (let p = 0; p < n; p++) {
      const x = (p % side) / side;
      const y = Math.floor(p / side) / side;
      const base = p * inDim;
      inputs[base + 0] = x;
      inputs[base + 1] = y;
      inputs[base + 2] = 0.5;
      inputs[base + 3] = 0.5;
      inputs[base + 4] = 0.5;
      inputs[base + 5] = 1.0;
      inputs[base + 6] = 0.0;
      inputs[base + 7] = 0.0;
      inputs[base + 8] = 1.0;
      for (let i = 0; i < lightParams.length; i++) inputs[base + 9 + i] = lightParams[i];
    }
    const buffers = makeBuffers(n);
    for (let i = 0; i < warmup; i++) await forward(buffers, inputs);
    const times = [];
    for (let i = 0; i < timedRuns; i++) {
      const { elapsedMs } = await forward(buffers, inputs);
      times.push(elapsedMs);
    }
    const meanMs = times.reduce((a, b) => a + b, 0) / times.length;
    latencyRows.push({
      resolution: [res, res],
      device: "webgpu-chrome",
      mean_ms_per_frame: meanMs,
      fps: 1000.0 / meanMs,
      meets_30_fps: 1000.0 / meanMs >= 30.0,
      meets_60_fps: 1000.0 / meanMs >= 60.0,
    });
  }

  return {
    architecture: dims,
    adapter_vendor: adapterInfo.vendor || null,
    adapter_architecture: adapterInfo.architecture || null,
    parity_vs_pytorch_max_abs_diff: maxAbsDiff,
    parity_resolution: [resW, resH],
    latency_sweep: latencyRows,
  };
}

async function main() {
  const artifactDir = path.join(repoRoot, "out", "engine-runtime", "js_viewer");
  const weightsJson = JSON.parse(readFileSync(path.join(artifactDir, "model_weights.json"), "utf8"));
  const referenceJson = JSON.parse(readFileSync(path.join(artifactDir, "reference.json"), "utf8"));

  // navigator.gpu requires a secure context; file:// URLs qualify, about:blank does
  // not. A blank local HTML file is all the page needs — all logic runs via
  // page.evaluate.
  const pagePath = path.join(__dirname, "blank.html");
  writeFileSync(pagePath, "<!doctype html><title>e6</title>");

  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--headless=new", "--no-sandbox", "--ignore-gpu-blocklist", "--use-angle=metal", "--enable-unsafe-webgpu"],
  });
  const page = await browser.newPage();
  await page.goto(`file://${pagePath}`);

  const result = await page.evaluate(runInPage, {
    weightsJson,
    referenceJson,
    shaderPrelude: SHADER_SOURCE,
    resolutions: [128, 256, 512],
    warmup: 3,
    timedRuns: 10,
  });
  await browser.close();

  const report = {
    extension: "E6",
    scope: "real WebGPU compute-shader backend, executed in real Chrome, against the actual exported proxy",
    backend: "webgpu (Chrome, via Playwright)",
    ...result,
    notes: [
      "Same compute shader/architecture as webgpu/bench.mjs (native Dawn bindings), " +
        "which reproducibly segfaults on this machine's real proxy weights — see " +
        "README.md. Running the identical shader inside real Chrome (not the " +
        "experimental Node-only Dawn binding) has no such issue.",
      "navigator.gpu requires a secure context; the page is loaded from file:// " +
        "(secure) rather than about:blank (not secure) for that reason.",
    ],
  };
  const outDir = path.join(repoRoot, "out", "engine-runtime");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(path.join(outDir, "webgpu_browser_report.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
