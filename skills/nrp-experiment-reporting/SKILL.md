---
name: nrp-experiment-reporting
description: Use when running or designing NRP experiments, benchmarks, roadmap items, paper-replication studies, status reports, performance tables, JSON evidence under out, or docs that must separate measured repo results from paper claims.
---

# NRP Experiment Reporting

## Overview

Keep experiments reproducible and honest: every measured claim should have a command,
a JSON artifact under `out/`, hardware or device context when relevant, and a matching
human-readable update in the docs.

## Start Here

Read the relevant experiment entrypoint before running or editing it:

- `mise.toml` for canonical runnable tasks.
- `docs/roadmap.md` for gated goal prompts and required evidence.
- `docs/performance.md` for methodology, tables, and prior results.
- `docs/paper-mapping.md` when the experiment changes paper coverage or known deviations.
- `examples/*.py` for experiment runners and report JSON shapes.
- The matching config in `examples/*.json` for train or export runs.

## Reporting Contract

- Put machine-readable outputs under `out/`, normally as JSON next to generated models or scenes.
- Include enough config in the report to rerun or audit the result: seed, scene, resolution, spp,
  bounces, model size, iterations, device, gather backend, denoiser, and metric definitions when
  applicable.
- Update `docs/performance.md` for new measurements. Include the exact command or task and avoid
  quoting paper hardware numbers as repo results.
- When a result is negative or mixed, say so directly and keep the evidence. Do not smooth away
  divergences from the paper.
- Keep optional dependencies explicit. Mitsuba and OIDN work should skip cleanly or document the
  extra install path through `uv sync --extra mitsuba --extra oidn` / `mise run sync-all`.
- Never vendor downloaded gallery scene assets. Keep download scripts under `examples/scenes/`.

## Common Commands

Use the repo tasks when they exist:

```sh
mise run bench-export
mise run bench
mise run volume-report
mise run inverse-grid
mise run quad-check
mise run layers
mise run multiview
mise run ablation
```

Use direct CLIs when the task needs custom parameters:

```sh
uv run python -m nrp.compare_reference --cache out/toy/path_cache.npz \
  --light '{"center": [0.5, 0.75, 0.5], "radius": 0.15, "rgb": [15, 14, 12]}' \
  --out out/toy/compare_report.json
uv run python -m nrp.torch_backend.train --config examples/mitsuba_cornell_128_torch.json \
  --gather-backend torch --device mps --resume
```

## Metrics Discipline

- Prefer held-out light configurations that are disjoint from training lights.
- Report PSNR and SMAPE for HDR gather/proxy quality unless the experiment specifically uses
  tonemapped display metrics.
- Use SSIM and LDR-FLIP for display-referred ablations through `nrp/metrics.py`.
- For timing, include warmup where benchmarks already support it, synchronize GPU devices, and
  state the measured device.
- For stochastic training, compare windowed curves or aggregate held-out metrics; do not draw
  conclusions from a single minibatch loss.

## Verification

Before publishing a new experiment result:

```sh
uv run python -m unittest discover -s tests
uv run ruff check .
```

If the experiment is too expensive for the current turn, run the smallest representative smoke,
state what remains unrun, and leave the exact full command and expected output path in the handoff.
