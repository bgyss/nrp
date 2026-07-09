# E6 WebGPU backend

## Result: complete — `bench_browser.mjs`

`bench_browser.mjs` (`mise run webgpu-bench`, report:
`out/engine-runtime/webgpu_browser_report.json`) is a real WebGPU compute-shader
backend for the exported TorchNRP proxy, executed inside real Google Chrome via
Playwright, running the **actual exported proxy's weights**. It replicates the exact
forward pass already parity-checked in JS
(`tests/test_export_js_viewer.py::test_js_forward_pass_matches_python_reference`).

| check | result |
|---|---:|
| parity vs. PyTorch reference (max abs diff) | 2.4e-7 |
| adapter | Apple M1 Max, Metal 3 (real hardware, not software) |
| 128×128 latency | ~1.7 ms/frame (580 fps) |
| 256×256 latency | ~2.8–4.0 ms/frame (250–350 fps) |
| 512×512 latency | ~9.4 ms/frame (107 fps) |

All three resolutions clear both 30 fps and 60 fps. `navigator.gpu` requires a
secure context, so the page loads from a local `file://` URL (which qualifies)
rather than `about:blank` (which does not) — that was the one non-obvious setup
detail; everything else is standard WebGPU.

## Why there are two backend scripts here

`bench.mjs` (native Dawn bindings via the `webgpu` npm package, no browser) is kept
as a **documented negative result**: it reproducibly segfaults on this machine when
given the real exported proxy's weights, while the byte-for-byte identical shader
running inside real Chrome (`bench_browser.mjs`) has no such issue. That contrast is
itself useful evidence — it pinpoints the defect to the experimental Node-only Dawn
binding specifically, not to WebGPU, this project's shader, or its JS code. Full
bisection below.

## Running

```sh
cd webgpu && npm install
npx playwright install chrome   # once, if not already present
node bench_browser.mjs          # the completed backend: real Chrome, real proxy
node smoke.mjs                  # native-binding pipeline sanity check (synthetic weights)
node bench.mjs parity           # native-binding negative-result repro (real weights)
```

## Appendix: `bench.mjs` native-binding bisection (for context)

Running `bench.mjs` against the **real exported proxy's weights** reproducibly
segfaults the `webgpu` package (Dawn/node-webgpu) on this machine — 100% of
attempts across 60+ trials total, for every layer count (1, 2, 3), every resolution
tried (down to 128 pixels), three independently trained models (different seeds),
and three independent package versions (0.2.12, 0.3.10, 0.4.0 — ruling out a
single-release regression).

`smoke.mjs` proves this is **not** a bug in the shader, the bind-group/buffer setup,
or the chunked-dispatch strategy: the identical pipeline, with synthetic weights of
the same architecture and magnitude, runs correctly and matches a CPU reference to
1.6e-6 max abs diff, reliably (8/8 trials).

Extensive controlled testing (~150+ trials across isolated repro scripts) narrowed
the trigger to the specific floating-point *values* of a trained model's weights,
not their shape, magnitude, or the shader/dispatch structure:

| condition | result |
|---|---|
| Uniform (constant-fill) weights, any magnitude 0.01–0.5 | always succeeds |
| `Math.random()`-generated weights, any range including the real weights' range (-0.9 to 0.7) | always succeeds |
| Real trained weights (any of 3 seeds), full precision | **always crashes** |
| Real trained weights, rounded to 3 decimals / 1 decimal / scaled to ~0.01 magnitude / shuffled to random positions | **always crashes** |
| Real weights + random biases (isolating weights vs biases) | **always crashes** — weights alone are sufficient |
| Real weights, either half replaced with random values | **always crashes** — not concentrated in one region |
| **Exactly one real weight value** spliced into an otherwise-random array | **always crashes** (0/15) |
| That same single real value spliced into an otherwise-uniform array | always succeeds (8/8) |
| Adapter identity (confirms real hardware, not a software fallback) | `vendor: apple, architecture: metal-3, device: apple-m1-max` |

No NaN/Inf/subnormal values are present in the real weights (checked at the bit
level). The sharpest finding — a single specific float32 value from the trained
model, mixed into otherwise-synthetic random data, is sufficient to trigger the
crash deterministically, but the same value causes no problem when the rest of the
buffer is uniform — points to a Dawn/Metal shader-compiler code-generation bug
(e.g. a register-allocation or constant-folding path exercised only for certain bit
patterns under otherwise divergent per-thread data). This is outside what this
project's shader or JS code controls, and `bench_browser.mjs` demonstrates that a
production WebGPU implementation (Chrome's) does not share the defect.
