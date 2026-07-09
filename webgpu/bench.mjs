// E6: a genuine WebGPU compute-shader backend for a trained TorchNRP sphere-light
// proxy. Runs on native Dawn bindings via the `webgpu` npm package — no browser
// required, so this executes and is measured the same way `nrp/torch_backend/bench.py`
// measures CPU/MPS. The compute shader replicates the exact forward pass exported by
// `examples/export_js_viewer.py` (Linear(13,32) -> ReLU -> Linear(32,32) -> ReLU ->
// Linear(32,3) -> softplus), the same architecture already parity-checked against
// PyTorch in JS (`tests/test_export_js_viewer.py`). This closes the remaining E6 gap:
// the JS/canvas viewer was a step toward a portable backend; this is an actual
// compute-shader backend, executed and measured.
import { create, globals } from "webgpu";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

Object.assign(globalThis, globals);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

function buildShader(layerDims) {
  // layerDims e.g. [13, 32, 32, 3]: input dim, hidden.., output dim.
  const nLayers = layerDims.length - 1;
  let src = `
struct Params {
  n_pixels: u32,
};

@group(0) @binding(0) var<storage, read> inputs: array<f32>; // n_pixels * ${layerDims[0]}
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>; // n_pixels * 3
@group(0) @binding(4) var<uniform> params: Params;

fn softplus(x: f32) -> f32 {
  return log(1.0 + exp(x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let pixel = gid.x;
  if (pixel >= params.n_pixels) {
    return;
  }
  var buf: array<f32, 64>;
  let in_dim = ${layerDims[0]}u;
  for (var i: u32 = 0u; i < in_dim; i = i + 1u) {
    buf[i] = inputs[pixel * in_dim + i];
  }
  var w_offset: u32 = 0u;
  var b_offset: u32 = 0u;
  var cur_dim: u32 = in_dim;
`;
  for (let l = 0; l < nLayers; l++) {
    const outDim = layerDims[l + 1];
    src += `
  {
    var next: array<f32, 64>;
    for (var o: u32 = 0u; o < ${outDim}u; o = o + 1u) {
      var s: f32 = biases[b_offset + o];
      for (var i: u32 = 0u; i < cur_dim; i = i + 1u) {
        s = s + weights[w_offset + o * cur_dim + i] * buf[i];
      }
      ${l < nLayers - 1 ? "next[o] = max(s, 0.0);" : "next[o] = s;"}
    }
    for (var o: u32 = 0u; o < ${outDim}u; o = o + 1u) {
      buf[o] = next[o];
    }
    w_offset = w_offset + cur_dim * ${outDim}u;
    b_offset = b_offset + ${outDim}u;
    cur_dim = ${outDim}u;
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

function flattenWeights(layers) {
  const weights = [];
  const biases = [];
  const dims = [layers[0].weight[0].length];
  for (const layer of layers) {
    for (const row of layer.weight) {
      for (const v of row) weights.push(v);
    }
    for (const v of layer.bias) biases.push(v);
    dims.push(layer.bias.length);
  }
  return { weights: Float32Array.from(weights), biases: Float32Array.from(biases), dims };
}

async function makeDevice() {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("no WebGPU adapter available");
  const device = await adapter.requestDevice();
  return { gpu, adapter, device };
}

/** Allocate a fixed set of GPU buffers + bind group for one (nPixels) size, reused
 * across repeated forward passes. Creating and destroying buffers every call was
 * measured to segfault the native Dawn bindings after a handful of iterations
 * (a binding-library lifetime bug, not a WebGPU API issue) — reusing buffers avoids
 * the churn entirely and is the standard/recommended usage pattern anyway. */
function makeForwardBuffers(device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, nPixels, inDim) {
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
  const paramsBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([nPixels, 0, 0, 0]));
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: inputsBuf } },
      { binding: 1, resource: { buffer: weightsBuf } },
      { binding: 2, resource: { buffer: biasesBuf } },
      { binding: 3, resource: { buffer: outputsBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ],
  });
  return { inputsBuf, outputsBuf, readBuf, paramsBuf, bindGroup, nPixels };
}

async function runForward(device, pipeline, buffers, inputsFlat) {
  const { inputsBuf, outputsBuf, readBuf, bindGroup, nPixels } = buffers;
  device.queue.writeBuffer(inputsBuf, 0, inputsFlat);

  const t0 = performance.now();
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(nPixels / 64));
  pass.end();
  encoder.copyBufferToBuffer(outputsBuf, 0, readBuf, 0, nPixels * 3 * 4);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const result = new Float32Array(readBuf.getMappedRange().slice(0));
  const elapsedMs = performance.now() - t0;
  readBuf.unmap();
  return { result, elapsedMs };
}

// Chunk size for chunked dispatch. Empirically isolated on this machine: a storage
// buffer with *non-uniform per-invocation* content (varying pixel data — uniform/
// constant-fill buffers never trigger this) reliably segfaults the native Dawn
// bindings once the buffer crosses ~16 KiB (consistent with a page-size-related bug
// in the binding, not a WebGPU API or shader issue — bisected with a standalone
// harness: 256 pixels x 13 floats x 4 bytes = 13,312 B succeeds every time, 320
// pixels = 16,640 B fails every time, and the transition is sharp, not probabilistic
// jitter). 128 pixels/chunk (6,656 B input buffer) leaves a large safety margin.
const CHUNK_PIXELS = 128;

async function runForwardChunked(device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, inputsFlat, nPixels, inDim) {
  const buffers = makeForwardBuffers(device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, CHUNK_PIXELS, inDim);
  const output = new Float32Array(nPixels * 3);
  const t0 = performance.now();
  for (let start = 0; start < nPixels; start += CHUNK_PIXELS) {
    const end = Math.min(start + CHUNK_PIXELS, nPixels);
    const chunkLen = end - start;
    // A fresh, zero-offset copy (not a subarray view) — the crash this chunking
    // scheme works around was only ever reproduced with freshly-allocated,
    // zero-offset typed arrays, so views are avoided here out of caution.
    const chunkInput = Float32Array.from(inputsFlat.subarray(start * inDim, end * inDim));
    const { result } = await runForward(device, pipeline, buffers, chunkInput);
    output.set(result.subarray(0, chunkLen * 3), start * 3);
  }
  const elapsedMs = performance.now() - t0;
  buffers.inputsBuf.destroy();
  buffers.outputsBuf.destroy();
  buffers.readBuf.destroy();
  buffers.paramsBuf.destroy();
  return { result: output, elapsedMs };
}

function syntheticInputs(nPixels, inDim, lightParams) {
  // Matches nrp/torch_backend/engine_runtime.py's synthetic_runtime_cache: constant
  // plausible G-buffer aux, varying pixel xy, fixed light params.
  const arr = new Float32Array(nPixels * inDim);
  const side = Math.round(Math.sqrt(nPixels));
  for (let p = 0; p < nPixels; p++) {
    const x = (p % side) / side;
    const y = Math.floor(p / side) / side;
    const base = p * inDim;
    arr[base + 0] = x;
    arr[base + 1] = y;
    arr[base + 2] = 0.5; // albedo r
    arr[base + 3] = 0.5; // albedo g
    arr[base + 4] = 0.5; // albedo b
    arr[base + 5] = 1.0; // depth
    arr[base + 6] = 0.0; // normal x
    arr[base + 7] = 0.0; // normal y
    arr[base + 8] = 1.0; // normal z
    for (let i = 0; i < lightParams.length; i++) arr[base + 9 + i] = lightParams[i];
  }
  return arr;
}

async function setupModel() {
  const artifactDir = path.join(repoRoot, "out", "engine-runtime", "js_viewer");
  const weightsJson = JSON.parse(readFileSync(path.join(artifactDir, "model_weights.json"), "utf8"));
  const referenceJson = JSON.parse(readFileSync(path.join(artifactDir, "reference.json"), "utf8"));
  const { weights, biases, dims } = flattenWeights(weightsJson.layers);
  const shaderSrc = buildShader(dims);

  const { device } = await makeDevice();
  const shaderModule = device.createShaderModule({ code: shaderSrc });
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
    ],
  });
  const pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    compute: { module: shaderModule, entryPoint: "main" },
  });
  const weightsBuf = device.createBuffer({
    size: weights.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(weightsBuf, 0, weights);
  const biasesBuf = device.createBuffer({
    size: biases.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(biasesBuf, 0, biases);
  const inDim = dims[0];
  const lightParams = [...weightsJson.default_light.center, weightsJson.default_light.radius];
  return {
    device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, dims, inDim, lightParams,
    weightsJson, referenceJson,
  };
}

// Each CLI invocation does exactly one task (parity check, or one resolution's
// latency measurement) and exits immediately after writing its JSON result to
// stdout. Splitting the work into separate process invocations, one GPU-buffer-set
// lifetime each, was the fix for a native-binding segfault: allocating buffers at a
// new, larger size after a previous size had already run (even after destroying the
// old buffers) reliably crashed the `webgpu` package's Dawn bindings on this
// machine — a lifetime bug in the experimental native binding, not a WebGPU API
// misuse (isolated and confirmed via a bisection harness kept in this file's git
// history). One resolution per process sidesteps it entirely and each result is
// still a real, executed, measured WebGPU number.
async function runParity() {
  const { device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, dims, inDim, lightParams, weightsJson, referenceJson } =
    await setupModel();
  const [resW, resH] = weightsJson.resolution;
  const nPx = resW * resH;
  const inputs = new Float32Array(nPx * inDim);
  for (let p = 0; p < nPx; p++) {
    const base = p * inDim;
    inputs[base + 0] = weightsJson.xy[p][0];
    inputs[base + 1] = weightsJson.xy[p][1];
    for (let i = 0; i < 7; i++) inputs[base + 2 + i] = weightsJson.aux[p][i];
    for (let i = 0; i < lightParams.length; i++) inputs[base + 9 + i] = lightParams[i];
  }
  const { result } = await runForwardChunked(
    device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, inputs, nPx, inDim
  );
  let maxAbsDiff = 0;
  for (let p = 0; p < nPx; p++) {
    const ref = referenceJson.reference_image[Math.floor(p / resW)][p % resW];
    for (let c = 0; c < 3; c++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(ref[c] - result[p * 3 + c]));
    }
  }
  return { architecture: dims, parity_vs_pytorch_max_abs_diff: maxAbsDiff, parity_resolution: [resW, resH] };
}

async function runResolution(res) {
  const { device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, inDim, lightParams } = await setupModel();
  const n = res * res;
  const inputs = syntheticInputs(n, inDim, lightParams);
  const warmup = 3;
  const timedRuns = 10;
  for (let i = 0; i < warmup; i++) {
    await runForwardChunked(device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, inputs, n, inDim);
  }
  const times = [];
  for (let i = 0; i < timedRuns; i++) {
    const { elapsedMs } = await runForwardChunked(
      device, pipeline, bindGroupLayout, weightsBuf, biasesBuf, inputs, n, inDim
    );
    times.push(elapsedMs);
  }
  const meanMs = times.reduce((a, b) => a + b, 0) / times.length;
  return {
    resolution: [res, res],
    device: "webgpu",
    mean_ms_per_frame: meanMs,
    fps: 1000.0 / meanMs,
    meets_30_fps: 1000.0 / meanMs >= 30.0,
    meets_60_fps: 1000.0 / meanMs >= 60.0,
  };
}

async function main() {
  const mode = process.argv[2];
  let result;
  if (mode === "parity") {
    result = await runParity();
  } else if (mode === "resolution") {
    result = await runResolution(Number(process.argv[3]));
  } else {
    throw new Error("usage: node bench.mjs parity | node bench.mjs resolution <N>");
  }
  console.log(JSON.stringify(result));
}

main()
  .then(() => process.exit(0)) // native Dawn bindings segfault on normal Node exit
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
