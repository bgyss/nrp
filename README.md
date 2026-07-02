# nrp — Neural Render Proxies, a sample reimplementation

A small, CPU-runnable reimplementation of Sancho et al., *"Neural Render Proxies for
Interactive and Differentiable Lighting"* (Computer Graphics Forum 45(4), EGSR 2026),
written as an educational re-implementation sample, not a production renderer. Every
piece runs end to end on a laptop in minutes, with no GPU required.

The paper's core idea: path tracing decouples into a **light-agnostic path pass**
(SAMPLEPATHS — expensive, needs the scene, done once) and an **emission pass**
(GATHERLIGHT — cheap, needs only the light parameters and cached path segments). A
compact per-light-type MLP then compresses the path cache into a differentiable proxy:
forward evaluation gives interactive relighting, backpropagation gives gradient-based
inverse lighting from image-space objectives.

Two backends share one path-cache/GATHERLIGHT vocabulary:

- **`nrp/` (numpy)** — the dependency-light reference: hand-rolled,
  finite-difference-checked autodiff, sinusoidal positional encoding. Good for reading.
- **`nrp/torch_backend/` (PyTorch)** — the paper's architecture (§4): 2D
  multiresolution hash encoding [MESK22], the paper's exact input set (encoded pixel
  coords + albedo/depth/normal + light shape parameters), the paper's exact loss
  (relative MSE with stop-gradient denominator, ε = 0.01, Eq. 4), pool-of-denoised-images
  training (§4.4), and the paper's inverse-optimization formulation (§5.3: Eq. 5 light
  sum, Reinhard-tonemapped MSE of Eq. 6, logit/inverse-softplus reparameterization,
  mini-batch pixel-fraction SGD of Table 3).

Full documentation lives in [docs/](docs/): [architecture](docs/architecture.md),
[section-by-section paper mapping](docs/paper-mapping.md),
[performance methodology + all measured results](docs/performance.md),
[status report](docs/report-2026-07-01.md), and a
[roadmap of ten goal-prompt-ready improvements](docs/roadmap.md).

## Toolchain (nix + mise + uv)

- **nix** — reproducible tool source of truth: `nix develop` gives python 3.12, uv,
  ruff (direnv users: `.envrc` loads it automatically).
- **mise** — non-nix alternative and task runner: `mise trust && mise install`, then
  `mise run test|lint|fmt|train|train-torch|smoke`.
- **uv** — Python project/venv manager: `uv sync` installs numpy + torch (the only
  required runtime dependencies) and the dev group; `uv run <cmd>` runs inside the
  venv. Optional extras: `uv sync --extra mitsuba --extra oidn` (or `mise run
  sync-all`) adds the Mitsuba 3 exporter and the OIDN denoiser. On macOS the `oidn`
  wheel links against Homebrew's TBB: `brew install tbb`.

## Quickstart

```sh
uv sync

# --- numpy reference backend ---
# Trace the toy Cornell-style scene and train (~4 min CPU). Outputs in out/toy/.
uv run python -m nrp.train --config examples/toy_sphere.json

# --- torch backend (paper architecture) ---
# Reuses the same path cache; trains hashgrid MLP with the denoised image pool.
uv run python -m nrp.torch_backend.train --config examples/toy_sphere_torch.json

# --- Mitsuba 3 scene (needs: uv sync --extra mitsuba --extra oidn) ---
# Export a real Mitsuba scene into the path-cache schema (§4.1), then train on it
# with OIDN-denoised targets (paper-exact §4.4 pipeline).
uv run python -m nrp.mitsuba_exporter --scene builtin:cornell-box \
  --width 48 --height 48 --spp 16 --bounces 4 --out out/mitsuba/path_cache.npz
uv run python -m nrp.torch_backend.train --config examples/mitsuba_cornell_torch.json

# Real academic scene (Mitsuba 3 gallery "Country Kitchen"): download (assets are
# never vendored), export with the drjit wavefront loop, train. Or: mise run
# export-kitchen && mise run train-kitchen.
uv run python examples/scenes/download_scene.py kitchen
uv run python -m nrp.mitsuba_exporter --scene examples/scenes/kitchen/scene.xml \
  --width 128 --height 128 --spp 64 --bounces 4 --out out/kitchen/path_cache.npz
uv run python -m nrp.torch_backend.train --config examples/kitchen_torch.json

# Benchmark proxy inference across resolutions on every available device (cpu/mps/cuda).
uv run python -m nrp.torch_backend.bench --model out/toy-torch/model.pt \
  --resolutions 48 128 256 512 1024 --out out/bench.json

# Consistency check: GATHERLIGHT over the cache vs an independent re-trace.
uv run python -m nrp.compare_reference --cache out/toy/path_cache.npz \
  --light '{"center": [0.5, 0.75, 0.5], "radius": 0.15, "rgb": [15, 14, 12]}' \
  --out out/toy/compare_report.json

# Interactive relighting through the torch proxy (sphere or quad models; lists sum per Eq. 3).
uv run python -m nrp.torch_backend.relight --model out/toy-torch/model.pt \
  --cache out/toy/path_cache.npz \
  --light '{"center": [0.3, 0.6, 0.4], "radius": 0.1, "rgb": [12, 12, 14]}' \
  --out out/toy-torch/relit.npy --bench 100

# Inverse lighting (paper §5.3): recover a hidden light from its rendered image.
uv run python -m nrp.torch_backend.optimize_lights --model out/toy-torch/model.pt \
  --cache out/toy/path_cache.npz \
  --target-light '{"center": [0.6, 0.7, 0.5], "radius": 0.12, "rgb": [10, 9, 8]}' \
  --restarts 4 --pixel-fraction 0.25 --out-dir out/toy-torch/inverse

# Art-directed inverse lighting: paint an objective, then optimize toward it.
uv run python examples/make_art_target.py --cache out/toy/path_cache.npz \
  --out-dir out/toy/art
uv run python -m nrp.torch_backend.optimize_lights --model out/toy-torch/model.pt \
  --cache out/toy/path_cache.npz --target out/toy/art/art_target.npy \
  --mask out/toy/art/art_mask.npy --out-dir out/toy-torch/art-opt
```

The numpy backend keeps its own `nrp.relight` / `nrp.optimize_lights` CLIs with the
same flags (single sphere light, non-tonemapped loss — see deviations below).

Tests and lint:

```sh
uv run python -m unittest discover -s tests    # or: mise run test
uv run ruff check .                            # or: mise run lint
```

## Paper coverage

| Paper section | Status | Where |
|---|---|---|
| §3.1 SAMPLEPATHS / GATHERLIGHT decoupling | **Implemented** | `nrp/path_cache.py`, `nrp/gather_light.py`, traced by `nrp/toy_tracer.py` |
| §3.1 BSDF sampling w/o NEE, throughput Russian roulette | **Implemented** (toy scale) | `nrp/toy_tracer.py` — cosine-weighted Lambertian, no NEE |
| §3.1 volumes / free-flight sampling | **Implemented** (toy scale) | homogeneous medium in `nrp/toy_tracer.py` (free-flight sampling, isotropic phase); GATHERLIGHT unchanged by design — transmittance is implicit in segment lengths; cache schema v2 carries medium metadata |
| §3.2 per-light-type networks, E(v) factored out (Eq. 1–3) | **Implemented** | `torch_backend/model.py` (`light_type` = sphere/quad), `gather_lights` |
| §3.2 sphere lights (4 params) | **Implemented** | `nrp/lights.py` |
| §3.2 / Fig. 13 quad lights (8 params) | **Implemented** | `nrp/lights.py::QuadLight`, `gather_throughput_quad`, quad-conditioned proxy |
| §4 PyTorch implementation | **Implemented** | `nrp/torch_backend/` |
| §4 tiny-cuda-nn + fused Triton GATHERLIGHT kernel | Substituted | plain-PyTorch hashgrid; batched torch gather runs on MPS/CUDA (`torch_backend/gather.py`, all segments in one op — the paper's fused-gather idea at torch-op granularity); numpy gather stays the authoritative reference (`gather_backend` config) |
| §4.2 fp16 / rgb9e5 compressed cache layout | Not implemented | float64 `.npz` (toy caches are MiB, not GiB) |
| §4.3 inputs: hashgrid(px) + albedo/depth/normal + light params | **Implemented** | `torch_backend/encoding.py` (multiresolution hash [MESK22]), `model.py` |
| §4.4 per-pixel random lights + denoised target pool (300 / 2-every-5) | **Implemented** (pool size configurable) | `torch_backend/train.py::ImagePool` |
| §4.4 OIDN denoiser | **Implemented** (optional extra) | `torch_backend/denoise.py::oidn_denoise` (RT filter, HDR, albedo+normal guides); aux-guided joint bilateral as the dependency-free fallback |
| §4.4 segment-based light-position sampling + bbox fallback | **Implemented** | `torch_backend/sampling.py` |
| §4.4 loss: relative MSE, sg(prediction)²+ε denominator, ε=0.01 (Eq. 4) | **Implemented** (torch) | `torch_backend/model.py::relative_mse_loss`, gradient unit-tested |
| §5.3 inverse: Eq. 5 multi-light sum, Reinhard MSE (Eq. 6) | **Implemented** | `torch_backend/optimize_lights.py` |
| §5.3 logit/inv-softplus reparameterization, Adam lr 0.05 × 500 | **Implemented** | `torch_backend/optimize_lights.py::ReparamSphereLights` |
| §5.3 mini-batch pixel-fraction SGD (Table 3) | **Implemented** | `--pixel-fraction` |
| §6.2 art-directed edits with constraint masks | **Implemented** | `--mask`, `--protect`; `examples/make_art_target.py` |
| §6.3 generative targets (Qwen-Image-Edit) | Out of scope | any (H,W,3) `.npy` works as `--target` |
| §6.1 multi-view / compositing-layer NRPs | Not implemented | single fixed camera, single layer |
| §4.1 Mitsuba 3 path-data pass (academic scenes) | **Implemented** (optional extra) | `nrp/mitsuba_exporter.py`: BSDF sampling, no NEE, throughput RR, aux G-buffer; drjit wavefront loop (39–59× over the scalar fallback); XML scenes (gallery scenes via `examples/scenes/download_scene.py`) or builtin cornell box |
| §5 production scenes (Hyperion, 512 spp, RTX 5090) | Out of scope | Hyperion is Disney-internal; any Mitsuba XML scene works via the exporter |

## Measured results (toy scene, laptop CPU)

Scene: hard-coded Cornell-style box + diffuse sphere, fixed pinhole camera, 48×48,
24 spp, 3 bounces (165,888 cached segments, 6.7 MiB compressed). All numbers from real
runs on an Apple Silicon laptop CPU, single process; none are quoted from the paper.

- **GATHERLIGHT consistency (backend-independent):** gathering over the 24-spp cache vs
  an independently re-traced 96-spp reference with inline emission agrees to
  **PSNR 29.3 dB, 0.03% mean radiance**. Both sides are Monte Carlo estimates over
  independent path sets, so this validates the §3.1 decoupling, not a tautology.
- **numpy backend:** 21,635 params, 219 s train, held-out **PSNR 19.97 dB** /
  SMAPE 0.91 (12 fresh lights), 1.6 ms/frame (629 Hz) at 48×48.
- **torch backend (paper architecture):** 62,923 params (hashgrid + 4×128 MLP), **60 s
  train** (3,000 iterations, pool 64 / replace 2 every 5), loss 0.786 → 0.072, held-out
  **PSNR 19.17 dB vs raw GATHERLIGHT** (18.86 dB vs denoised), 10.7 ms/frame (93 Hz)
  at 48×48. Same quality as the numpy backend in 3.6× less training wall-clock.
- **torch inverse recovery (paper §5.3 formulation):** hidden sphere light recovered
  with **center error 0.013, radius error 0.020** (scene is a unit box), rgb within
  ~10% (uniformly low — proxy amplitude bias), 500 steps × 4 restarts at pixel
  fraction 0.25. The paper's reparameterization + tonemapped loss dramatically
  outperforms the numpy backend's naive clipped-Adam recovery (center error 0.44 on
  the same task).
- **Mitsuba cornell box + OIDN (paper-exact data pipeline):** 103,680 segments exported
  in 1.6 s (48×48, 16 spp, 4 bounces, RR); torch proxy trained on OIDN-denoised pool
  targets reaches held-out **PSNR 25.87 dB vs raw GATHERLIGHT** (26.48 dB vs denoised)
  in 48 s — the best-conditioned scene in the repo.
- **Exporter vectorization (roadmap item 1):** the drjit wavefront loop reaches
  **2.4 M seg/s at 48² (39×)** and **3.5 M seg/s at 128² (59×)** over the scalar
  loop (`out/export-bench.json`; both loops statistically equivalent by a fixed-seed
  GATHERLIGHT test).
- **Batched device GATHERLIGHT (roadmap item 3):** `torch_backend/gather.py` gathers
  a light over all cached segments in one weight-and-scatter op — **8.1 ms/image on
  MPS vs 58.6 ms numpy-CPU (7.2×) on a 2.9 M-segment 256² cache** — matching numpy
  at rtol 1e-5 (50 sphere + 50 quad lights, toy + Mitsuba caches). Training runs
  fully device-resident (`device: mps`, `--gather-backend torch`) with held-out PSNR
  within 0.5 dB of the CPU run at equal seed; at toy scale MPS wall-clock is still
  ~15–30% slower (63k-param model under-fills the GPU — documented honestly in
  `docs/performance.md`).
- **Volumetric export (roadmap item 2):** the toy box filled with a homogeneous
  medium (σ_t 2.0, albedo 0.8) trains the same torch architecture to **18.84 dB**
  held-out PSNR — within 0.33 dB of the surface-only baseline — with **zero changes
  to GATHERLIGHT**: free-flight sampling makes transmittance implicit in segment
  lengths (slab-fixture falloff matches analytic exp(−σ_t·d) within 5%,
  `tests/test_volume.py`; measurements in `out/volume-report.json`).
- **Real academic scene (Mitsuba gallery "Country Kitchen"):** 3.27 M segments
  exported at 128×128 / 64 spp in **4.0 s** (129 MB cache); torch proxy (106,085
  params, 430 KB) trains in **126 s** to held-out **PSNR 25.24 dB vs raw
  GATHERLIGHT**, full-frame CPU inference 12 ms (84 Hz) at 128².
- **Inference benchmark** (62,923-param sphere model, `out/bench.json`; Apple Silicon):

  | device | 48² | 128² | 256² | 512² | 1024² |
  |---|---|---|---|---|---|
  | cpu | 170 Hz | 71 Hz | 21 Hz | 9.0 Hz | 2.8 Hz |
  | mps | 481 Hz | 359 Hz | **117 Hz** | **37 Hz** | 8.6 Hz |

  On this laptop's GPU (MPS) the proxy holds the paper's ~30–60 Hz interactive range
  up to 512×512. The paper's RTX 5090 numbers at production resolution remain out of
  reach without CUDA + tiny-cuda-nn, as documented.
- **Interactive-rate statement:** proxy inference is linear in pixel count and
  independent of scene complexity, as the paper argues — but the paper's 30–60 Hz is
  measured on an RTX 5090 at production resolutions. On this CPU the same statement
  holds only up to a few hundred pixels squared. Do not compare the absolute numbers.

## Known deviations from the paper

Documented substitutions, not silent approximations:

- **CPU (or MPS) PyTorch, no tiny-cuda-nn, no Triton kernel** — architecture is
  faithful, absolute performance is not.
- **OIDN is optional** — the paper's denoiser is used when the `oidn` extra is
  installed (`"denoise": {"method": "oidn"}`); the dependency-free default remains the
  aux-guided joint bilateral filter (same guidance signal, weaker prior).
- **Softplus output head** — the paper doesn't specify its head; softplus keeps
  contributions positive.
- **The Mitsuba exporter records paths from Python, not inside the renderer** — the
  default drjit wavefront loop (`llvm_ad_rgb`/`metal_ad_rgb`, auto-detected) advances
  all paths one bounce per kernel launch and pulls per-bounce results to numpy; a pure
  scalar loop (`--mode scalar`, no JIT requirements) remains the fallback and
  semantics reference. No volumes are exported (surface interactions only).
- **numpy backend diverges further by design** (sinusoidal encoding, target-normalized
  loss, extra derived geometric inputs) — it is the readable finite-difference-checked
  reference, not the paper replica; the torch backend is the paper replica.

## Next steps

Roadmap item 1 (vectorized Mitsuba export + a real academic scene) is done — see
`docs/performance.md`, as are item 2 (volumetric path export — homogeneous medium in
the toy tracer, schema v2) and item 3 (batched device GATHERLIGHT + MPS training,
`torch_backend/gather.py`). Remaining candidate improvements —
multi-light/quad inverse (Table 3), compressed caches (§4.2), paper-scale training,
multi-view and per-layer NRPs (§6.1), the Fig. 6 image-based baseline, and the
Table 2 ablation suite with SSIM/FLIP — are written up as ready-to-run goal prompts
(each with verification and performance-testing requirements) in
[docs/roadmap.md](docs/roadmap.md).

## Layout

```
nrp/                 numpy reference backend + shared vocabulary (cache, lights, gather)
nrp/mitsuba_exporter.py  Mitsuba 3 scene -> path cache (optional extra; wavefront + scalar loops)
nrp/export_bench.py  exporter throughput benchmark (scalar vs wavefront)
nrp/torch_backend/   paper-architecture backend (hashgrid, pool training, inverse, bench)
examples/            training configs + art-directed target builder
examples/scenes/     gallery-scene download script (assets never committed)
tests/               90 unit tests (geometry, gather, hashgrid, loss gradients, reparam, exporter, OIDN, smokes)
docs/                architecture, paper mapping, performance, status report, roadmap
flake.nix / mise.toml / .envrc   toolchain
```

## License

MIT — see [LICENSE](LICENSE).
