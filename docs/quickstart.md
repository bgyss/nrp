# Quickstart

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

# Paper-scale run (8x256 net, 50k iterations, pool 128, cosine LR, checkpoint every
# 1000 with --resume support): export at 128x128 / 64 spp, then train (~25 min MPS).
uv run python -m nrp.mitsuba_exporter --width 128 --height 128 --spp 64 --bounces 4 \
  --out out/mitsuba/path_cache_128_64spp.npz
uv run python -m nrp.torch_backend.train --config examples/mitsuba_cornell_128_torch.json \
  --gather-backend torch --device mps   # add --resume to continue an interrupted run

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

# Perceptual quality gate (production track T3): pass/fail a render against a
# reference at a named tier (preview/draft/final), exit code 0/1 for CI, or
# re-emit an existing report JSON with quality_gate verdicts attached.
uv run python -m nrp.quality.gate images pred.npy ref.npy --tier draft
uv run python -m nrp.quality.gate report out/ablation/report.json --tier preview \
  --psnr-key psnr_db_mean --ssim-key ssim_mean --flip-key flip_mean \
  --out out/ablation/report_gated.json

# WebGPU runtime baseline (production track T4): export the T1-scene proxy
# (hashgrid included) for the browser backend, bench it in real Chrome, and gate
# frame-time regressions against the committed baseline.
mise run t4-export   # examples/export_webgpu_runtime.py -> out/t4-runtime/export/
mise run t4-bench    # webgpu/bench_t4.mjs -> out/t4-runtime/report.json
mise run t4-check    # fails on parity break, >30% p95 regression, or <30 fps p95 at 512^2
```

The numpy backend keeps its own `nrp.relight` / `nrp.optimize_lights` CLIs with the
same flags (single sphere light, non-tonemapped loss — see
[known deviations](paper-mapping.md#known-deviations-summary)).

## Tests and lint

```sh
uv run python -m unittest discover -s tests    # or: mise run test
uv run ruff check .                            # or: mise run lint
```
