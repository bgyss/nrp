// Smoke test proving the WebGPU compute pipeline itself (bind groups, chunked
// dispatch, compute pipeline, buffer readback) executes correctly end to end on
// this machine, using synthetic weights of the same architecture as the exported
// TorchNRP proxy. See README.md: the *real* exported proxy's weights reproducibly
// crash the native binding on this host (a confirmed third-party defect, not a
// WebGPU API or shader bug) — this smoke test isolates and demonstrates that the
// compute-shader mechanism itself is correct and only the real-weight data path is
// affected.
import { create, globals } from "webgpu";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
Object.assign(globalThis, globals);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

function buildShader(layerDims) {
  const nLayers = layerDims.length - 1;
  let src = `
struct Params { n_pixels: u32, };
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> biases: array<f32>;
@group(0) @binding(3) var<storage, read_write> outputs: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;
fn softplus(x: f32) -> f32 { return log(1.0 + exp(x)); }
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

function cpuForward(dims, weights, biases, input) {
  let buf = input.slice();
  let wOff = 0, bOff = 0;
  for (let l = 0; l < dims.length - 1; l++) {
    const outDim = dims[l + 1];
    const next = new Array(outDim).fill(0);
    for (let o = 0; o < outDim; o++) {
      let s = biases[bOff + o];
      for (let i = 0; i < buf.length; i++) s += weights[wOff + o * buf.length + i] * buf[i];
      next[o] = l < dims.length - 2 ? Math.max(s, 0.0) : s;
    }
    wOff += buf.length * outDim;
    bOff += outDim;
    buf = next;
  }
  return buf.slice(0, 3).map((v) => Math.log(1 + Math.exp(v)));
}

async function main() {
  const dims = [13, 32, 32, 3];
  const rng = () => Math.random() * 1.6 - 0.8; // wide range, same magnitude as a trained proxy
  const nWeights = dims[0] * dims[1] + dims[1] * dims[2] + dims[2] * dims[3];
  const nBiases = dims[1] + dims[2] + dims[3];
  const weights = Array.from({ length: nWeights }, rng);
  const biases = Array.from({ length: nBiases }, rng);

  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  const device = await adapter.requestDevice();
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

  const nPx = 256;
  const inputs = new Float32Array(nPx * dims[0]);
  const cpuInputs = [];
  for (let p = 0; p < nPx; p++) {
    const row = Array.from({ length: dims[0] }, () => Math.random());
    cpuInputs.push(row);
    inputs.set(row, p * dims[0]);
  }
  const wArr = Float32Array.from(weights);
  const bArr = Float32Array.from(biases);

  const inBuf = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(inBuf, 0, inputs);
  const wBuf = device.createBuffer({ size: wArr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(wBuf, 0, wArr);
  const bBuf = device.createBuffer({ size: bArr.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(bBuf, 0, bArr);
  const outBuf = device.createBuffer({ size: nPx * 3 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readBuf = device.createBuffer({ size: nPx * 3 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const paramsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(paramsBuf, 0, new Uint32Array([nPx, 0, 0, 0]));
  const bindGroup = device.createBindGroup({
    layout: bgl,
    entries: [
      { binding: 0, resource: { buffer: inBuf } },
      { binding: 1, resource: { buffer: wBuf } },
      { binding: 2, resource: { buffer: bBuf } },
      { binding: 3, resource: { buffer: outBuf } },
      { binding: 4, resource: { buffer: paramsBuf } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(nPx / 64));
  pass.end();
  encoder.copyBufferToBuffer(outBuf, 0, readBuf, 0, nPx * 3 * 4);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(GPUMapMode.READ);
  const gpuResult = new Float32Array(readBuf.getMappedRange().slice(0));

  let maxAbsDiff = 0;
  for (let p = 0; p < nPx; p++) {
    const cpu = cpuForward(dims, weights, biases, cpuInputs[p]);
    for (let c = 0; c < 3; c++) {
      maxAbsDiff = Math.max(maxAbsDiff, Math.abs(cpu[c] - gpuResult[p * 3 + c]));
    }
  }
  const report = {
    extension: "E6",
    scope: (
      "webgpu compute pipeline mechanism smoke test (synthetic weights, not the " +
      "exported proxy — see webgpu/README.md: real proxy weights reproducibly " +
      "segfault the native binding on this host)"
    ),
    n_pixels: nPx,
    architecture: dims,
    max_abs_diff_vs_cpu_reference: maxAbsDiff,
    passed: maxAbsDiff < 1e-3,
  };
  const outDir = path.join(repoRoot, "out", "engine-runtime");
  mkdirSync(outDir, { recursive: true });
  writeFileSync(path.join(outDir, "webgpu_smoke.json"), JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify(report, null, 2));
}

main().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
