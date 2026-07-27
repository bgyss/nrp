# Representation Track — World-Anchored, Camera-Conditioned NRPs

Phase 7 asks whether the per-view pixel-coordinate proxy can become a representation
of rendered scene data across cameras. The approved design and full R1–R6 ladder are
in `docs/plans/2026-07-17-representation-track-design.md`.

## Status

| rung | title | status | evidence |
|---|---|---|---|
| R1 | World-space encoding at parity | **done — honest negative**: toy passes; Country Kitchen passes on only 1/3 controlled seeds | `out/r1-worldgrid/report.json`, `out/r1-followup/report.json`, `docs/performance.md#world-space-encoding-at-parity-representation-track-rung-r1` |
| R1 follow-up | Provenance, collision, and tri-plane diagnosis | **done — candidate not promoted**: tri-plane passes on 2/3 seeds, but fails the unchanged per-seed gate | `out/r1-followup/report.json`, `docs/plans/2026-07-27-r1-next-experiments.md` |
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

The 0.5 dB gate is binding and evaluated per scene against a same-run 2D control:
same iterations, seed, pool, denoiser, held-out lights, and within 0.5% parameter
budget.

| scene | same-run pixel2d | world3d CPU | delta | gate |
|---|---:|---:|---:|---|
| toy box, 48², seed 0 | 19.98 dB | 20.84 dB | +0.86 dB | **pass** |
| Country Kitchen, 128², seed 0 | 22.16 dB | 21.62 dB | −0.544 dB | **fail** |

The previously reported −3.62 dB comparison against the historical 25.24 dB
Kitchen artifact was not controlled: that artifact predates the dedicated validation
RNG and output-scale initialization. Re-evaluating it on the fixed validation set
drops it to 23.27 dB. The historical number remains in the JSON as context with
`comparable_for_gate: false`; it is not the gate.

The three-seed controlled follow-up confirms the honest negative:

| seed | pixel2d | world3d | delta | gate |
|---:|---:|---:|---:|---|
| 0 | 22.16 dB | 21.62 dB | −0.544 dB | fail |
| 1 | 24.68 dB | 23.91 dB | −0.771 dB | fail |
| 2 | 22.00 dB | 22.25 dB | +0.247 dB | pass |

Mean delta is −0.356 dB, but the approved gate is per run rather than mean-only, so
1/3 passing seeds is not promotion evidence.

## R1 failure analysis and follow-up

The matched 3D grid touches eight vertices per level and reaches an observed weighted
collision fraction of 86.7%, versus 22.8% for the 2D control. An 88%-larger 3D
diagnostic reduces that to 66.6% but reaches only 21.75 dB on seed 0, still below
22.16 dB for 2D. Capacity alone therefore does not explain or repair the gap.

A selectable world-anchored tri-plane encoder (`"world_triplane"`) distributes the
same budget across XY, XZ, and YZ multiresolution grids: 106,239 parameters versus
106,085 for 2D (+0.15%), with 35.3% observed collision fraction. It is promising but
unstable:

| seed | delta vs paired pixel2d | gate |
|---:|---:|---|
| 0 | +1.359 dB | pass |
| 1 | −0.935 dB | fail |
| 2 | −0.356 dB | pass |

Its mean delta is +0.023 dB, but only 2/3 seeds pass. The candidate is **not
promoted**, the original R1 remains negative, and R2–R6 remain blocked. Output
initialization also changes seed-0 absolute PSNR by more than 2 dB for both
representations without closing their paired gap, so the next campaign treats
optimizer/initialization variance as a first-class factor. The bounded follow-up
ladder and its stop conditions are in
`docs/plans/2026-07-27-r1-next-experiments.md`.

## Reproduce and verify

The standard configs are the 2D controls; the matched world-grid configs are
`examples/r1_toy_world3d.json` and `examples/r1_kitchen_world3d.json`.

```sh
UV_CACHE_DIR=.uv-cache uv run python examples/r1_worldgrid.py --devices cpu mps
UV_CACHE_DIR=.uv-cache uv run python examples/r1_failure_analysis.py --reuse
mise run test
mise run lint
mise run pipeline-audit
```

The experiment command exits nonzero after writing all JSON evidence when the
binding gate fails. That is expected for this recorded negative.
