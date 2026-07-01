# nrp — neural render proxies, a sample reimplementation

A small, CPU-only, numpy-only reimplementation of the core decomposition from
Sancho et al., *"Neural Render Proxies for Interactive and Differentiable Lighting"*
(Computer Graphics Forum 45(4), EGSR 2026). It is written as an educational
re-implementation sample, not a production renderer: every piece is deliberately
readable, dependency-free, and runnable end to end on a laptop CPU in a few minutes.

The pipeline follows the paper's split between transport and emission:

1. **Light-agnostic path cache** (`nrp/path_cache.py`, traced by `nrp/toy_tracer.py`) —
   a minimal Cornell-style-box path tracer records per-segment origins, directions, and
   throughputs plus per-pixel aux buffers (albedo, depth, normal, first-hit position).
2. **GATHERLIGHT** (`nrp/gather_light.py`) — decoupled emission: given the cache and a
   virtual sphere light, every cached segment that crosses the (transparent, purely
   emissive) light accumulates throughput-weighted emission. Relighting without
   re-tracing.
3. **Compact neural proxy** (`nrp/model.py`, `nrp/dataset.py`, `nrp/train.py`) — a small
   MLP (numpy with hand-rolled, finite-difference-checked autodiff) that predicts the
   pre-emission contribution per pixel from G-buffer features + light parameters.
   Sinusoidal positional encoding stands in for the paper's hashgrid.
4. **Interactive relighting** (`nrp/relight.py`) — forward inference through the proxy,
   with an optional benchmark mode.
5. **Differentiable inverse lighting** (`nrp/optimize_lights.py`) — Adam over the light's
   (center, radius, rgb) through the proxy's input gradients, against either a hidden
   true light or a painted art-directed target, with optional protected-region
   constraints and multi-restart (real local minima exist).

## Quickstart

Requires Python ≥ 3.12 and numpy (the only runtime dependency).

```sh
uv sync                     # or: pip install numpy

# Trace the toy scene, build the dataset, train the proxy (~3 min on a laptop CPU).
# The path cache is traced automatically if missing; outputs land in out/toy/.
uv run python -m nrp.train --config examples/toy_sphere.json

# Consistency check: GATHERLIGHT over the cache vs an independent re-trace.
uv run python -m nrp.compare_reference --cache out/toy/path_cache.npz \
  --light '{"center": [0.5, 0.75, 0.5], "radius": 0.15, "rgb": [15, 14, 12]}' \
  --out out/toy/compare_report.json

# Relight through the trained proxy (add --bench 100 for a throughput benchmark).
uv run python -m nrp.relight --model out/toy/model.npz --cache out/toy/path_cache.npz \
  --light '{"center": [0.3, 0.6, 0.4], "radius": 0.1, "rgb": [12, 12, 14]}' \
  --out out/toy/relit.npy

# Inverse lighting: recover a hidden light from its rendered image.
uv run python -m nrp.optimize_lights --model out/toy/model.npz \
  --cache out/toy/path_cache.npz \
  --target-light '{"center": [0.6, 0.7, 0.5], "radius": 0.12, "rgb": [10, 9, 8]}' \
  --restarts 4 --out-dir out/toy/inverse

# Art-directed inverse lighting: paint an objective, then optimize toward it.
uv run python examples/make_art_target.py --cache out/toy/path_cache.npz \
  --out-dir out/toy/art
uv run python -m nrp.optimize_lights --model out/toy/model.npz \
  --cache out/toy/path_cache.npz --target out/toy/art/art_target.npy \
  --mask out/toy/art/art_mask.npy --out-dir out/toy/art-opt
```

Tests and lint:

```sh
uv run python -m unittest discover -s tests
uv run ruff check .
```

## Known deviations from the paper

Documented substitutions, not silent approximations:

- **CPU/numpy, not GPU/Triton.** Interactive rates hold only at toy resolutions
  (roughly 30 Hz at 256×256 on a laptop CPU; the paper's numbers are GPU numbers).
- **Sinusoidal positional encoding instead of the hashgrid** — a performance structure,
  not a correctness requirement at toy scale.
- **Relative-MSE loss normalized by target²+ε**, not the paper's
  stop-gradient(prediction)²+ε — a constant denominator keeps the objective exactly
  finite-difference-checkable against the hand-rolled autodiff.
- **Raw (non-denoised) training targets**, sphere lights only, no MIS/next-event
  estimation, and occlusion is whatever the cached segments encode (a light radius grown
  past a blocker is not re-checked, matching the paper's fixed-transport limitation).
- Not implemented: textured/environment/arbitrary area lights, multi-light joint
  optimization, hashgrid encoding, GPU kernels, out-of-core caches, dynamic scenes.

## Layout

```
nrp/            the package: tracer, path cache, gather, model, train, relight, inverse
examples/       training config + art-directed target builder
tests/          unit tests (geometry, cache round-trip, gather correctness, training smoke)
```

## License

MIT — see [LICENSE](LICENSE).
