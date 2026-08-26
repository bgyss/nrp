# Representation Track — World-Anchored, Camera-Conditioned NRPs

Phase 7 asks whether the per-view pixel-coordinate proxy can become a representation
of rendered scene data across cameras. The approved design and full R1–R6 ladder are
in `docs/plans/2026-07-17-representation-track-design.md`.

## Status

| rung | title | status | evidence |
|---|---|---|---|
| R1 | World-space encoding at parity | **done — honest negative**: toy passes; Country Kitchen passes on only 1/3 controlled seeds | `out/r1-worldgrid/report.json`, `out/r1-followup/report.json`, `docs/performance.md#world-space-encoding-at-parity-representation-track-rung-r1` |
| R1 follow-up | Provenance, collision, and tri-plane diagnosis | **done — candidate not promoted**: tri-plane passes on 2/3 seeds, but fails the unchanged per-seed gate | `out/r1-followup/report.json`, `docs/plans/2026-07-27-r1-next-experiments.md` |
| R1A | Five-seed variance decomposition | **done — candidate identified**: target-scale world tri-plane passes all 5 Kitchen seeds | `out/r1a/report.json`, `examples/r1a_variance.py` |
| R1 promotion audit | R1C coordinate robustness + R1E independent scene | **not promoted — R1C fails seed 2 at 90° AABB; R1E Bedroom passes 5/5** | `out/r1-promotion/report.json`, `examples/r1_promotion.py` |
| R2 | One network, N cameras | **implemented pilot — honest negative; promotion blocked by R1 gate** | `out/r2-conditioned/report.json`, `docs/performance.md#r2-one-network-n-cameras` |
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

## R1A variance decomposition

The R1A runner crosses the three matched-budget representations with both output-bias
policies on the 128² Country Kitchen cache: five seeds × three representations × two
policies, for 30 CPU arms. Each seed has one 12-light held-out set generated before
training; the same light specifications and frozen references are shared by all six
arms for that seed. The report stores the light fingerprints, per-light paired PSNR
deltas, seed-level summaries, and percentile paired-bootstrap 95% CIs over the five
seed-level means.

| representation | output-bias policy | seed deltas vs same-policy pixel2d (dB) | mean ± std (dB) | paired 95% CI (dB) | seeds passing −0.5 dB |
|---|---|---|---:|---:|---:|
| world3d | target-scale | −1.262, −0.862, −0.826, −1.001, +0.124 | −0.765 ± 0.470 | [−1.105, −0.291] | 1/5 |
| world3d | framework-default | −1.665, −0.578, −0.741, 0.000, −0.913 | −0.779 ± 0.539 | [−1.262, −0.331] | 1/5 |
| world tri-plane | target-scale | −0.008, +1.016, +1.118, +0.069, +1.961 | +0.831 ± 0.732 | [+0.228, +1.435] | **5/5 — pass** |
| world tri-plane | framework-default | −3.789, −0.727, +0.526, 0.000, +1.242 | −0.550 ± 1.743 | [−2.170, +0.707] | 3/5 |

The target-scale world tri-plane is the only crossed world-policy arm whose every
seed clears the unchanged gate. The framework-default policy does not reproduce that
result: its tri-plane arm fails two seeds, including seed 0. This is evidence of an
initialization/representation interaction, not permission to average away the failed
seeds. R2 remains unattempted and unauthorized; no camera-conditioned experiment was
started from this result.

All arms use 106,085 parameters for pixel2d, 106,345 for world3d, and 106,239 for
world tri-plane, with 3,000 iterations, batch size 4,096, the 64-image pool, CPU
training, and OIDN targets. Mean training throughput was 24.9 it/s for pixel2d,
21.3 it/s for world3d, and 24.3 it/s for world tri-plane; the two bias policies had
the same architecture and comparable throughput.

## R1 promotion audit

The target-scale world-triplane candidate passes the five paired Kitchen seeds in
R1A, but the unchanged promotion rule also requires coordinate robustness and an
independent real scene. The audit uses the same −0.5 dB per-seed threshold and
stops the R1C matrix after its first decisive failure rather than averaging it away.

R1C's 90° Y-axis rotation with AABB normalization produced these paired deltas on
the Country Kitchen cache:

| seed | pixel2d control (dB) | world tri-plane (dB) | delta (dB) | gate |
|---:|---:|---:|---:|---|
| 0 | 18.11 | 18.43 | +0.317 | pass |
| 1 | 19.75 | 20.48 | +0.734 | pass |
| 2 | 15.80 | 14.75 | **−1.045** | **fail** |
| 3 | 21.09 | 20.63 | −0.453 | pass |
| 4 | 24.50 | 25.45 | +0.946 | pass |

Only 4/5 seeds pass. The R1C stop condition therefore fired; the remaining 180°
and percentile-bound variants were not used to manufacture a promotion decision.

R1E independently re-traced the Mitsuba Bedroom gallery scene at 128² / 64 spp /
4 bounces and ran the same five-seed target-scale tri-plane versus pixel2d control:

| seed | delta (dB) | gate |
|---:|---:|---|
| 0 | +0.806 | pass |
| 1 | −0.134 | pass |
| 2 | +0.140 | pass |
| 3 | +1.772 | pass |
| 4 | +0.442 | pass |

R1E passes 5/5, but it cannot compensate for the R1C seed-2 failure. The canonical
audit is `out/r1-promotion/report.json`; its `promoted` field is false and its stop
reason names the failing R1C row. R1 remains unpromoted, so R2's measured quality
negative and the R3–R6 promotion gates remain unchanged.

## R2 implementation and pilot

The R2 machinery is implemented and measured, but the rung is not promoted because
the R1 prerequisite remains unmet. `TorchNRP` now has an opt-in
`camera_conditioned` input: one normalized camera forward direction is broadcast
over each view's pixels and concatenated with a global-bound `world3d` representation.
The camera-aware manifest loader accepts N cache/camera pairs; one shared light pool
renders targets through every cache, while each view gets a dedicated held-out light
set. The shared relighter keeps one checkpoint resident and does not touch path
segments during an edit.

The full pilot used:

```sh
UV_CACHE_DIR=.uv-cache uv run python examples/r2_conditioned.py \
  --out out/r2-conditioned/report.json --n-views 3 --width 48 --height 48 \
  --spp 16 --bounces 4 --iters 3000 --devices cpu --denoise bilateral
```

It ran on macOS 27 arm64 with Python 3.12.11, PyTorch 2.12.1, and Mitsuba 3.9.0.
LLVM/Metal initialization was unavailable, so Mitsuba used the scalar CPU exporter.
The three per-view conditioned deltas against same-light-set 2D baselines were
−1.067, −1.513, and −2.916 dB; all miss R2's ≤1 dB per-view gate. One conditioned
checkpoint is 0.375 MB versus 0.773 MB for the three baseline checkpoints. CPU
all-view proxy edit latency was 8.05 / 17.43 / 25.49 ms for N = 1 / 2 / 3;
the per-view baseline path measured 4.66 / 9.18 / 13.67 ms. Validation light
sets were disjoint for all three views, and the report records those checks directly.

This is an R2 implementation/pilot result, not evidence for novel-view
interpolation or R3–R6 promotion. The R1 candidate promotion gates remain open.

## Reproduce and verify

The standard configs are the 2D controls; the matched world-grid configs are
`examples/r1_toy_world3d.json` and `examples/r1_kitchen_world3d.json`.

```sh
UV_CACHE_DIR=.uv-cache uv run python examples/r1_worldgrid.py --devices cpu mps
UV_CACHE_DIR=.uv-cache uv run python examples/r1_failure_analysis.py --reuse
UV_CACHE_DIR=.uv-cache uv run python examples/r1a_variance.py --seeds 0 1 2 3 4 --denoise-method oidn
UV_CACHE_DIR=.uv-cache uv run python examples/r1_promotion.py --second-cache out/r1-promotion/bedroom_cache.npz --second-scene Bedroom
mise run test
mise run lint
mise run pipeline-audit
```

The experiment command exits nonzero after writing all JSON evidence when the
binding gate fails. That is expected for this recorded negative.
