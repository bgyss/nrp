# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small, CPU-runnable reimplementation of Sancho et al., *"Neural Render Proxies for
Interactive and Differentiable Lighting"* (CGF 45(4), EGSR 2026) — an educational
sample, not a production renderer. The paper's core idea: path tracing decouples into
a light-agnostic path pass (SAMPLEPATHS, expensive, done once) and a cheap emission
pass (GATHERLIGHT, needs only light parameters + cached path segments). A compact
per-light-type MLP compresses the path cache into a differentiable proxy for
interactive relighting and gradient-based inverse lighting.

Two backends share one path-cache/GATHERLIGHT vocabulary:

- **`nrp/` (numpy)** — dependency-light reference: hand-rolled finite-difference-checked
  autodiff, sinusoidal positional encoding. Read this first.
- **`nrp/torch_backend/` (PyTorch)** — the paper's actual architecture: 2D
  multiresolution hash encoding, the paper's exact input set and loss (relative MSE,
  Eq. 4), pool-of-denoised-images training (§4.4), and the paper's inverse-optimization
  formulation (§5.3).

Full docs live in `docs/`: `quickstart.md` (every CLI), `architecture.md` (pipeline
diagram + module-by-module notes), `paper-mapping.md` (section-by-section paper
coverage table), `performance.md`, `status/` (dated status reports), `roadmap.md`.

## Toolchain and commands

Managed by nix (tool source of truth) + mise (task runner) + uv (Python/venv).
`uv sync` installs numpy + torch only; `uv sync --extra mitsuba --extra oidn` (or
`mise run sync-all`) adds the optional Mitsuba 3 exporter and OIDN denoiser. On macOS
the oidn wheel needs `libtbb.12.dylib`, provided by the nix devshell (oneTBB via
`DYLD_FALLBACK_LIBRARY_PATH` in `flake.nix`) — enter via direnv or `nix develop`;
in a bare shell (no direnv) `oidn_available()` returns False and OIDN tests skip.

```sh
uv sync                                          # or: mise run sync

# tests / lint / format
uv run python -m unittest discover -s tests      # or: mise run test
uv run python -m unittest tests.test_gather_light  # single test module
uv run python -m unittest tests.test_gather_light.TestGatherLight.test_sphere  # single test
uv run ruff check .                              # or: mise run lint
uv run ruff format .                             # or: mise run fmt

# numpy backend: trace toy scene + train (~4 min CPU), outputs in out/toy/
uv run python -m nrp.train --config examples/toy_sphere.json         # or: mise run train

# torch backend: reuses same path cache, trains hashgrid MLP with denoised pool
uv run python -m nrp.torch_backend.train --config examples/toy_sphere_torch.json  # or: mise run train-torch

# fast end-to-end smoke test (numpy train + relight + inverse on a tiny trace)
mise run smoke

# Mitsuba export + train (needs the mitsuba/oidn extras)
mise run export-mitsuba && mise run train-mitsuba

# cross-device inference benchmark (cpu/mps/cuda)
uv run python -m nrp.torch_backend.bench --model out/toy-torch/model.pt --out out/bench.json
```

`[tool.uv] package = false` — the repo is not an installed package. Tests import
modules directly via a `sys.path` shim.

## Architecture

```
                    SAMPLEPATHS (once, needs scene, no lights)
  nrp/toy_tracer.py ──────────────┐
  nrp/mitsuba_exporter.py ────────┤──► PathCache (.npz)
                                  │      segments + throughputs + G-buffer aux
                                  ▼
                    GATHERLIGHT (cheap, needs only light params)
  nrp/gather_light.py: per-pixel emission accumulation over cached segments
                                  │
              ┌───────────────────┼──────────────────────┐
              ▼                   ▼                      ▼
   reference images       training targets         re-render check for
   for any light          (optionally denoised)    inverse results
                                  │
                                  ▼
                    NEURAL RENDER PROXY (per light type)
  numpy: nrp/model.py + nrp/train.py       torch: nrp/torch_backend/{model,train}.py
                                  │
              ┌───────────────────┴──────────────────────┐
              ▼                                          ▼
   forward relighting (Eq. 3)                inverse optimization (§5.3)
   nrp/relight.py                            nrp/optimize_lights.py
   nrp/torch_backend/relight.py              nrp/torch_backend/optimize_lights.py
```

**The path cache (`nrp/path_cache.py`)** is the central artifact connecting both
producers to both backends, for a fixed camera and static scene: per-segment origin,
direction, t_max, pre-segment throughput, plus first-hit G-buffer aux (albedo, depth,
normal, position). Two serializations — compressed `.npz` (tracer exports) and a JSON
dict form for tiny hand-authored test caches. `validate()` enforces shapes, index
ranges, positive t_max, unit directions. Schema v2 adds a `schema_version` field and
optional `medium` metadata (`{sigma_t, albedo}`) for caches free-flight sampled
through a homogeneous medium; v1 surface-only caches load unchanged.
`save(path, compressed=True)` writes the paper's §4.2 packed layout (fp16 geometry
+ rgb9e5 shared-exponent throughput, `nrp/rgb9e5.py`); `load` auto-detects the
layout and always returns float64 arrays.

**Lights (`nrp/lights.py`)** are virtual pure emitters that never block or scatter
cached paths, so one cache serves every light configuration — this is what makes the
SAMPLEPATHS/GATHERLIGHT decoupling work. `SphereLight` (4 params) and `QuadLight` (8
params, tangent frame derived from the normal). Vectorized segment-overlap tests drive
GATHERLIGHT; `light_from_dict` dispatches JSON specs.

**GATHERLIGHT (`nrp/gather_light.py`)**: `gather_throughput[_quad]` returns the
per-pixel pre-emission contribution the proxies learn; `gather_lights` sums a list
(exploits linearity of transport, Eq. 1). The numpy version is the authoritative
reference; `nrp/torch_backend/gather.py::TorchPathCache` is the batched device
(CPU/MPS/CUDA) mirror used for pool builds when `gather_backend: torch` (config key
or `--gather-backend`), parity-tested against numpy at rtol 1e-5.

**Producers**: `nrp/toy_tracer.py` is a dependency-free educational tracer
(hard-coded Cornell-style box + diffuse sphere, Lambertian, cosine-weighted sampling)
that also renders the *independent* emissive-inline reference used by
`nrp/compare_reference.py` for the decoupling consistency check. It can fill the box
with a homogeneous medium (`--medium-sigma-t`, `--medium-albedo`): free-flight-sampled
scatter vertices end segments early (isotropic phase, single-scattering-albedo
throughput factor), so lights inside the medium work through plain GATHERLIGHT —
transmittance is implicit in the segment-length distribution. `--layer sphere|box`
(§6.1 compositing) records only paths whose first hit is on that layer's geometry
while still tracing the full scene, so the two layer caches partition the full
cache's segments and their gathers sum exactly to the full-scene gather
(`layer_ownership_mask` gives per-layer pixel ownership; also available as
`trace.layer` in training configs).
`nrp/mitsuba_exporter.py` (extra: `mitsuba`) drives Mitsuba 3 over any scene XML or
`builtin:cornell-box`; emitters in the scene are ignored since this is the
light-agnostic pass. Default is a drjit wavefront loop (`llvm_ad_rgb`/`metal_ad_rgb`,
auto-detected; 39–59× the scalar throughput); `--mode scalar` keeps the pure-Python
reference loop. Gallery scenes are fetched by `examples/scenes/download_scene.py`
(assets never vendored); `nrp/export_bench.py` benchmarks the two loops.

**numpy backend**: `nrp/model.py` is an MLP with hand-rolled, finite-difference-checked
autodiff, sinusoidal positional encoding, and extra derived geometric inputs (first-hit
→ light offset + distance). `nrp/optimize_lights.py` is a deliberately naive
clipped-Adam optimizer, kept as a baseline the torch backend's §5.3 machinery improves
on.

**torch backend (the paper replica)**:
- `encoding.py` — 2D multiresolution hash encoding [MESK22]: per-level dense/hashed
  feature tables, bilinear interpolation, geometric resolution growth.
- `model.py` — `TorchNRP`: hashgrid(px) ⊕ aux(7: albedo+depth+normal) ⊕ light shape
  params → MLP → softplus. `relative_mse_loss` is Eq. 4 exactly (stop-gradient
  prediction in the denominator, ε=0.01); gradient unit-tested against closed form.
- `sampling.py` — §4.4 light-position strategies: uniform-on-recorded-segments
  (implicit importance sampling) or visible-bbox fallback.
- `denoise.py` — `denoise_image` dispatches `"oidn"` (paper's denoiser, extra: `oidn`)
  or `"bilateral"` (dependency-free aux-guided joint bilateral fallback).
- `train.py` — §4.4 pool scheme: `pool.size` denoised GATHERLIGHT images, every
  training pixel samples its target uniformly from the pool, `pool.replace_count`
  images replaced every `pool.replace_every` iterations. Long runs: `lr_schedule:
  cosine` (+ `lr_min`), `checkpoint: {"every": N}` full-state checkpoints with a
  `--resume` flag (bit-exact on CPU), and a fixed dedicated-RNG validation set
  evaluated at each checkpoint (PSNR-vs-iteration curve in the report).
- `optimize_lights.py` — §5.3: Eq. 5 multi-light sum, Reinhard-tonemapped MSE (Eq. 6),
  logit/inverse-softplus reparameterization, pixel-fraction mini-batch SGD, restarts,
  objective/protect masks, and a mandatory GATHERLIGHT re-render so proxy-space and
  physical errors are reported separately.
- `relight_multiview.py` — §6.1 multi-view NRPs: loads N (model, cache) view pairs
  from a `views.json` manifest, applies one light edit across all resident proxies
  (no cache access at edit time), one image per view; `examples/multiview.py`
  (`mise run multiview`) exports the views (exporter `--sensor-index` /
  `--cam-origin` camera override), trains them, and measures latency vs N.
- `composite.py` — §6.1 per-layer compositing: relight one layer's proxy under a new
  light and add the other layer's fixed image (`examples/layers.py` / `mise run
  layers` builds the layer caches, trains per-layer proxies, and writes the demo).
- `bench.py` — cross-device (cpu/mps/cuda) full-frame inference benchmark with warmup
  and proper synchronization.

Training-config JSON shape differs by backend: torch configs use
`iters`/`batch_pixels`/`pool`/`model.encoding`; the numpy config
(`examples/toy_sphere.json`) is epoch-based (`epochs`/`batch_size`/`n_train_lights`),
`hidden` is a list, and `light_bounds` includes `center_min`/`center_max` (uniform box
sampling instead of segment sampling).

## Known deviations from the paper

These are documented substitutions, not silent approximations — see the README's
"Known deviations" and "Paper coverage" tables for the full list. Key ones: CPU/MPS
PyTorch (no tiny-cuda-nn, no Triton fused kernel); OIDN optional (bilateral filter is
the dependency-free default); softplus output head (paper doesn't specify one); the
Mitsuba exporter records paths from Python rather than inside the renderer (drjit
wavefront loop by default, scalar fallback; no volumes); the numpy backend diverges
further by design (sinusoidal encoding,
target-normalized loss) — it's the readable reference, not a paper replica.

## Testing conventions

Statistical assertions compare windowed means, never single minibatch losses. Optional-
dependency tests skip cleanly via `@unittest.skipUnless(HAVE_MITSUBA, ...)` /
`@unittest.skipUnless(oidn_available(), ...)`.
