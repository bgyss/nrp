---
name: nrp-torch-proxy-workflow
description: Use when changing the PyTorch NRP backend, hash encoding, relative MSE loss, denoised pool training, torch config loading, relighting CLIs, inverse lighting optimization, device handling, checkpoints, or torch backend tests.
---

# NRP Torch Proxy Workflow

## Overview

Work on the paper-replica backend while preserving the repo's reference vocabulary:
path caches and GATHERLIGHT come from the shared layer, while `nrp/torch_backend/`
implements the hashgrid proxy, denoised-pool training, relighting, and inverse
optimization.

## First Files

Read the smallest slice that matches the task:

- `nrp/torch_backend/model.py` for `TorchNRP`, sphere/quad parameter vectors, ablation switches,
  save/load, and `relative_mse_loss`.
- `nrp/torch_backend/encoding.py` for 2D multiresolution hash encoding.
- `nrp/torch_backend/train.py` for config resolution, pool construction/replacement, checkpointing,
  validation, and device placement.
- `nrp/torch_backend/sampling.py` and `nrp/torch_backend/denoise.py` for training targets.
- `nrp/torch_backend/relight.py`, `relight_multiview.py`, `composite.py`, and
  `optimize_lights.py` for user-facing torch workflows.
- `tests/test_torch_backend.py`, `tests/test_training_smoke.py`, `tests/test_checkpoint_resume.py`,
  `tests/test_inverse_quad.py`, and feature-specific tests before broad test runs.

## Backend Rules

- Keep the torch backend compatible with both CPU and accelerator devices when the operation is
  meant to be device-resident. Use explicit `torch.device` handling and synchronize benchmarks.
- Keep config paths resolved relative to the config file, not the current shell directory.
- Preserve the paper input contract for the main torch model: encoded pixel coordinates, aux
  features `albedo + depth + normal`, and light shape parameters.
- Keep `relative_mse_loss` equal to Eq. 4: denominator uses `pred.detach()` and epsilon `0.01`.
- Keep model outputs non-negative unless changing the documented softplus-head deviation.
- Keep pool replacement stochastic but reproducible from the configured seed and checkpoint state.
- For inverse optimization, report both proxy-space objective and physical GATHERLIGHT re-rendered
  error. Rank restarts by the physical re-render when that path exists.
- Treat numpy backend differences as intentional unless the task explicitly asks to align them.

## Change Patterns

- Model or encoding change: add shape, determinism, gradient, save/load, and ablation-flag coverage.
- Training-loop change: add a tiny training smoke test that checks a windowed loss trend, not a
  single final minibatch.
- Device or gather-backend change: check CPU first, then optional MPS/CUDA paths when available;
  keep skips clean when hardware or extras are absent.
- Checkpoint change: verify resume reproduces the uninterrupted trajectory on a tiny deterministic
  run.
- Inverse-optimization change: test reparameterization round trips, bounds, gradient flow, and a
  small recovery fixture if runtime is acceptable.
- CLI change: cover argument parsing or a small end-to-end output artifact rather than only helper
  functions.

## Verification

Use targeted checks while iterating:

```sh
uv run python -m unittest tests.test_torch_backend
uv run python -m unittest tests.test_training_smoke
uv run python -m unittest tests.test_checkpoint_resume
uv run python -m unittest tests.test_torch_gather
uv run python -m unittest tests.test_inverse_quad
```

Use smoke tasks when a change touches user-facing flows:

```sh
mise run smoke
mise run train-torch
uv run python -m nrp.torch_backend.bench --model out/toy-torch/model.pt --out out/bench.json
```

Run the full gate before handing off broad backend changes:

```sh
uv run python -m unittest discover -s tests
uv run ruff check .
```

Update `docs/architecture.md`, `docs/paper-mapping.md`, or `docs/performance.md` only when the
public behavior, paper-fidelity story, or measured result changes.
