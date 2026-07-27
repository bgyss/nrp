# Representation Track — World-Anchored, Camera-Conditioned NRPs

Phase 7 asks whether the per-view pixel-coordinate proxy can become a representation
of rendered scene data across cameras. The approved design and full R1–R6 ladder are
in `docs/plans/2026-07-17-representation-track-design.md`.

## Status

| rung | title | status | evidence |
|---|---|---|---|
| R1 | World-space encoding at parity | **done — honest negative**: toy passes; Country Kitchen fails the binding 0.5 dB parity gate | `out/r1-worldgrid/report.json`, `docs/performance.md#world-space-encoding-at-parity-representation-track-rung-r1` |
| R2 | One network, N cameras | **blocked by R1 gate; not attempted** | — |
| R3 | Novel-view interpolation | blocked by R1/R2; not attempted | — |
| R4 | Real scene, real scale | blocked by R2; not attempted | — |
| R5 | Camera in the WebGPU runtime | blocked by R4; not attempted | — |
| R6 | Scene4D diagnostic-buffer bridge | blocked by R2; not attempted | — |

R1 implemented `model.spatial_encoding: "world3d"` as a selectable alternative to
the default `"pixel2d"` path. It uses the cache's first-hit world position, normalizes
it with bounds stored in the checkpoint, and evaluates a 3D multiresolution hashgrid
with dense/hashed tables and trilinear interpolation. The existing 2D checkpoint and
inference path remains the default and loads unchanged.

## R1 verdict

The 0.5 dB gate is binding and evaluated per scene against the committed 2D
baselines at the same 3,000 iterations and within 0.5% parameter budget:

| scene | committed pixel2d | world3d CPU | delta | gate |
|---|---:|---:|---:|---|
| toy box, 48² | 19.17 dB | 20.84 dB | +1.67 dB | **pass** |
| Country Kitchen, 128² | 25.24 dB | 21.62 dB | −3.62 dB | **fail** |

The same-run controls are retained because the current kitchen 2D rerun itself
reproduced at only 22.16 dB. They do not relax the binding gate: world3d also misses
that CPU control, 21.62 versus 22.16 dB (−0.544 dB), by 0.044 dB beyond the allowed
loss. MPS reverses the same-run ordering (21.79 versus 21.07 dB), but remains 3.45 dB
below the committed kitchen baseline.

This is therefore an honest negative on R1, not permission to proceed with a
camera-conditioned representation. The likely 3D-collision/locality risk named in
the design remains plausible, but the experiment does not isolate it from optimizer
and runtime drift strongly enough to claim a single root cause. Per the approved
ordering, R2–R6 are not implemented.

## Reproduce and verify

The standard configs are the 2D controls; the matched world-grid configs are
`examples/r1_toy_world3d.json` and `examples/r1_kitchen_world3d.json`.

```sh
UV_CACHE_DIR=.uv-cache uv run python examples/r1_worldgrid.py --devices cpu mps
mise run test
mise run lint
mise run pipeline-audit
```

The experiment command exits nonzero after writing all JSON evidence when the
binding gate fails. That is expected for this recorded negative.
