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

## Toolchain (nix + mise + uv)

- **nix** — reproducible tool source of truth: `nix develop` gives python 3.12, uv,
  ruff (direnv users: `.envrc` loads it automatically).
- **mise** — non-nix alternative and task runner: `mise trust && mise install`, then
  `mise run test|lint|fmt|train|train-torch|smoke`.
- **uv** — Python project/venv manager: `uv sync` installs numpy + torch (the only
  runtime dependencies) and the dev group; `uv run <cmd>` runs inside the venv.

## Quickstart

```sh
uv sync

# --- numpy reference backend ---
# Trace the toy Cornell-style scene and train (~4 min CPU). Outputs in out/toy/.
uv run python -m nrp.train --config examples/toy_sphere.json

# --- torch backend (paper architecture) ---
# Reuses the same path cache; trains hashgrid MLP with the denoised image pool.
uv run python -m nrp.torch_backend.train --config examples/toy_sphere_torch.json

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
| §3.1 volumes / free-flight sampling | Not implemented | toy tracer is surface-only |
| §3.2 per-light-type networks, E(v) factored out (Eq. 1–3) | **Implemented** | `torch_backend/model.py` (`light_type` = sphere/quad), `gather_lights` |
| §3.2 sphere lights (4 params) | **Implemented** | `nrp/lights.py` |
| §3.2 / Fig. 13 quad lights (8 params) | **Implemented** | `nrp/lights.py::QuadLight`, `gather_throughput_quad`, quad-conditioned proxy |
| §4 PyTorch implementation | **Implemented** | `nrp/torch_backend/` |
| §4 tiny-cuda-nn + fused Triton GATHERLIGHT kernel | Substituted | plain-PyTorch hashgrid; numpy gather (CPU project — speed not comparable) |
| §4.2 fp16 / rgb9e5 compressed cache layout | Not implemented | float64 `.npz` (toy caches are MiB, not GiB) |
| §4.3 inputs: hashgrid(px) + albedo/depth/normal + light params | **Implemented** | `torch_backend/encoding.py` (multiresolution hash [MESK22]), `model.py` |
| §4.4 per-pixel random lights + denoised target pool (300 / 2-every-5) | **Implemented** (pool size configurable) | `torch_backend/train.py::ImagePool` |
| §4.4 OIDN denoiser | Substituted | aux-guided joint bilateral filter (`torch_backend/denoise.py`) — same guidance, weaker prior |
| §4.4 segment-based light-position sampling + bbox fallback | **Implemented** | `torch_backend/sampling.py` |
| §4.4 loss: relative MSE, sg(prediction)²+ε denominator, ε=0.01 (Eq. 4) | **Implemented** (torch) | `torch_backend/model.py::relative_mse_loss`, gradient unit-tested |
| §5.3 inverse: Eq. 5 multi-light sum, Reinhard MSE (Eq. 6) | **Implemented** | `torch_backend/optimize_lights.py` |
| §5.3 logit/inv-softplus reparameterization, Adam lr 0.05 × 500 | **Implemented** | `torch_backend/optimize_lights.py::ReparamSphereLights` |
| §5.3 mini-batch pixel-fraction SGD (Table 3) | **Implemented** | `--pixel-fraction` |
| §6.2 art-directed edits with constraint masks | **Implemented** | `--mask`, `--protect`; `examples/make_art_target.py` |
| §6.3 generative targets (Qwen-Image-Edit) | Out of scope | any (H,W,3) `.npy` works as `--target` |
| §6.1 multi-view / compositing-layer NRPs | Not implemented | single fixed camera, single layer |
| §5 production scenes (Hyperion/Mitsuba, 512 spp, RTX 5090) | Out of scope | toy Cornell-style box; a Mitsuba 3 exporter is the natural upgrade |

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
- **Interactive-rate statement:** proxy inference is linear in pixel count and
  independent of scene complexity, as the paper argues — but the paper's 30–60 Hz is
  measured on an RTX 5090 at production resolutions. On this CPU the same statement
  holds only up to a few hundred pixels squared. Do not compare the absolute numbers.

## Known deviations from the paper

Documented substitutions, not silent approximations:

- **CPU (or MPS) PyTorch, no tiny-cuda-nn, no Triton kernel** — architecture is
  faithful, absolute performance is not.
- **Joint bilateral denoiser instead of OIDN** — same auxiliary guidance
  (albedo/normal/depth), far weaker prior. Swap in `oidn` bindings behind
  `joint_bilateral_denoise` for parity.
- **Softplus output head** — the paper doesn't specify its head; softplus keeps
  contributions positive.
- **Toy path tracer instead of Hyperion/Mitsuba 3** — Lambertian Cornell-style box
  only; no volumes, no fur/translucency, no production-scale anything.
- **numpy backend diverges further by design** (sinusoidal encoding, target-normalized
  loss, extra derived geometric inputs) — it is the readable finite-difference-checked
  reference, not the paper replica; the torch backend is the paper replica.

## Next steps

1. **Mitsuba 3 scene exporter** — trace a real academic scene (Kitchen/Bedroom/Cornell)
   into the path-cache schema; the paper's quantitative tables become reproducible.
2. **OIDN denoising** — optional dependency behind the existing denoise interface.
3. **GPU/MPS benchmarking + a fused gather** (torch.compile or Triton when CUDA is
   available) to approach the paper's reconstruction timings (Table 1).
4. **Multi-light joint training targets and quad-light inverse optimization** (the
   paper optimizes spheres; quads are forward-only here too).
5. **Compressed cache layout** (fp16 + rgb9e5) for sequence-scale caches (§4.2).
6. **Multi-view NRPs and per-layer compositing** (§6.1).

## Layout

```
nrp/                 numpy reference backend + shared vocabulary (cache, lights, gather)
nrp/torch_backend/   paper-architecture backend (hashgrid, pool training, inverse)
examples/            training configs + art-directed target builder
tests/               62+ unit tests (geometry, gather, hashgrid, loss gradients, reparam, smokes)
flake.nix / mise.toml / .envrc   toolchain
```

## License

MIT — see [LICENSE](LICENSE).
