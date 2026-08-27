# Representation Track — World-Anchored, Camera-Conditioned NRPs

Phase 7 asks whether the per-view pixel-coordinate proxy can become a representation
of rendered scene data across cameras. The approved design and full R1–R6 ladder are
in `docs/plans/2026-07-17-representation-track-design.md`.

## Status

| rung | title | status | evidence |
|---|---|---|---|
| R1 | World-space encoding at parity | **not promoted (redesigned gate) — no arm clears held-out-camera generalization on all seeds; closed as a characterized negative** | `out/r1-encoding-redesign/report.json`, `docs/superpowers/specs/2026-08-26-world-anchored-encoding-redesign-design.md`, `docs/performance.md#world-anchored-encoding-redesign-representation-track-rung-r1` |
| R1 follow-up | Provenance, collision, and tri-plane diagnosis | **done — candidate not promoted**: tri-plane passes on 2/3 seeds, but fails the unchanged per-seed gate | `out/r1-followup/report.json`, `docs/plans/2026-07-27-r1-next-experiments.md` |
| R1A | Five-seed variance decomposition | **candidate only**: target-scale world tri-plane passes all 5 original Kitchen seeds | `out/r1a/report.json`, `examples/r1a_variance.py` |
| R1 promotion audit | R1C coordinate robustness + R1E independent scene | **not promoted — corrected R1C and R1E both contain binding failures** | `out/r1-promotion/report.json`, `examples/r1_promotion.py` |
| R2 | One network, N cameras | **implemented pilot — honest negative; promotion blocked by R1 gate** | `out/r2-conditioned/report.json`, `docs/performance.md#r2-one-network-n-cameras` |
| R3 | Novel-view interpolation | blocked by R1/R2; not attempted | — |
| R4 | Real scene, real scale | blocked by R2; not attempted | — |
| R5 | Camera in the WebGPU runtime | blocked by R4; not attempted. Also blocked by a known limitation: `TorchNRP.load` cannot reload occupancy-allocated arms (`world_sparse`, `world3d` with `allocation: "occupancy"`) outside the R1 runner, which reconstructs occupancy from `config["world_bounds"]`/`config["encoding"]` before loading — WebGPU export has no equivalent path today (see `docs/performance.md`). | — |
| R6 | Scene4D diagnostic-buffer bridge | blocked by R2; not attempted. Also blocked by the same `TorchNRP.load` occupancy-allocated-arm limitation as R5 — see `docs/performance.md`. | — |

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
seeds. R2 was not started from this result; its later camera-conditioned pilot is
recorded below as an honest negative and does not change the R1 prerequisite.

All arms use 106,085 parameters for pixel2d, 106,345 for world3d, and 106,239 for
world tri-plane, with 3,000 iterations, batch size 4,096, the 64-image pool, CPU
training, and OIDN targets. Mean training throughput was 24.9 it/s for pixel2d,
21.3 it/s for world3d, and 24.3 it/s for world tri-plane; the two bias policies had
the same architecture and comparable throughput.

## R1 promotion audit

R1A's target-scale world-triplane candidate passes the five paired Kitchen seeds,
but the corrected promotion audit uses the authoritative NumPy gather backend,
OIDN with pinned one-thread execution, and the unchanged −0.5 dB per-seed gate.
The canonical result is `out/r1-promotion/report.json`; `promoted` is false.

The complete R1C matrix shows that no tested normalization is robust across the
three coordinate frames:

| normalization | 0° pass count | 90° pass count | 180° pass count | worst delta (dB) |
|---|---:|---:|---:|---:|
| AABB | 5/5 | 3/5 | 3/5 | −3.651 |
| 1st–99th percentile | 2/5 | 2/5 | 3/5 | −3.612 |

The initial 90° AABB seed-2 measurement (−1.045 dB) is replaced by a reproducible
−3.651 dB failure when the gather backend is made consistent; the corrected run
also fails 90° AABB seed 3 and 180° AABB seeds 2/3. Percentile normalization adds
out-of-bounds clamping (5.35% of first-hit positions) and fails at least two seeds
in every orientation. These are per-seed failures, not mean-only effects.

R1E independently re-traced the Mitsuba Bedroom gallery scene at 128² / 64 spp /
4 bounces under the same corrected protocol:

| seed | delta (dB) | gate |
|---:|---:|---|
| 0 | +1.526 | pass |
| 1 | +0.811 | pass |
| 2 | +0.142 | pass |
| 3 | +2.060 | pass |
| 4 | **−3.083** | **fail** |

The exact execution correction resolved the earlier backend/process confound, but
it did not rescue the candidate: R1C remains coordinate-sensitive and R1E fails
one independent-scene seed. R1 is therefore not promoted. R2's measured quality
negative and the R3–R6 promotion gates remain unchanged.

## R1 redesign

The three campaigns above (R1, R1A, R1 promotion audit) tuned capacity, tri-plane
allocation, and initialization against a single-view parity gate. The design in
`docs/superpowers/specs/2026-08-26-world-anchored-encoding-redesign-design.md`
stops that tuning loop and re-diagnoses the problem instead.

**The previous R1 negative is now explained, not just repeated.** Measured on the
real 128² Country Kitchen cache, matching *parameter* budgets between `pixel2d`
and `world3d` left `world3d`'s finest level with 4,096 slots against 78,084
distinct queried vertices (0.052 slots per distinct vertex), while `pixel2d` at
the same budget kept 16,384 slots against 16,641 distinct vertices (0.98 slots
per distinct vertex) — a **~19× allocation handicap** awarded by the gate's own
matching rule, not a defect in the 3D representation. The +88%-capacity
diagnostic in the R1 failure analysis above closed only ~2× of a ~19× deficit,
which is why more capacity never rescued the candidate.

**The gate changed from single-view parity to held-out-camera generalization,
and here is why.** At `finest_resolution=128` on a 128² render, `pixel2d`'s
finest level is dense with zero collisions everywhere except level 7, which lands
at exactly one hashgrid vertex per pixel — a free per-pixel lookup table. Asking
a world-anchored encoder to match that at single-view reconstruction asks it to
match a memorizer at the one task a memorizer is unbeatable at, and the one task
where it cannot generalize to any other camera at all. R1's redesigned G1 gate
instead requires beating `pixel2d` by a comparative margin (1.0 dB, unchanged
from the R2 ladder) *and* clearing an absolute 15 dB PSNR floor (new to this
campaign — see provenance below), at held-out cameras never used for training,
across 5 seeds and 3 world rotations, with no averaging.

**Outcome, per arm, against G1/G3/G4 — no averaging away of failures:**

| arm | rows failing G1 (of 60) | seeds passing G3 (of 5) | worst Δ across rotations |
|---|---:|---:|---:|
| `world_sparse` (arm B, primary) | 24 | 0 | −1.51 dB (rotation 90°, seed 4) |
| `world_normal_triplane` (arm C) | 29 | 0 | −2.25 dB (rotation 0°, seed 4) |
| `world3d`, occupancy-allocated (arm A, control) | 30 | 0 | −1.51 dB |

83 rows fail G1 across all three arms (24 + 29 + 30, out of 60 rows per arm —
180 rows total), and all 83 fail for the reason
`below_delta_threshold` — none fails the 15 dB absolute floor, so the floor was
not the binding constraint for any arm. Mean deltas are positive for every arm
(+1.88, +1.40, +1.41 dB respectively), and every arm still fails because G3
requires every one of 5 seeds to pass and none does. **A favorable mean is not
evidence of a pass.**

**`world_sparse` is the strongest arm and the only frame-robust one, while still
not clearing the bar.** Its mean delta is highest (+1.88 dB) and its per-rotation
mean is nearly flat — +2.00 / +1.83 / +1.80 dB at 0°/90°/180° — because its
collision-free exact index has no orientation-dependent hash structure to
degrade. Both hashed arms degrade with rotation instead (triplane: +2.03 / +1.53
/ +0.64 dB; world3d: +2.42 / +0.98 / +0.83 dB). The mandatory G5 decomposition
for `world_sparse` rules out its own sparse-fallback mechanism as the cause of
the shortfall: mean out-of-occupancy query fraction at held-out cameras is only
0.446% (max 1.343%), and out-of-occupancy PSNR (24.44 dB) is *higher* than
in-occupancy PSNR (23.46 dB) — the gap to the gate is uniform, not concentrated
in unseen geometry.

**Gate provenance.** The 1.0 dB comparative margin is the existing R2 convention,
not invented for this campaign. The 15 dB absolute floor is invented for this
campaign, chosen against measured trained-view quality of 19.17–22.16 dB at this
scale so a held-out camera could show genuine degradation without automatically
failing on threshold choice alone; it turned out non-binding, so the outcome is
driven entirely by the inherited 1.0 dB margin against ≈1.9 dB of row-to-row
noise on all three arms.

**Two campaign runs were spent and invalidated before the result above.** Run 1
(~40 minutes, aborted) let each arm fall back to differing encoder-class defaults
— a ~3× parameter spread and three different resolution schedules — so any
arm-to-arm difference was confounded with capacity and schedule, not
representation. Run 2 completed but computed evaluation lights once per seed from
the unrotated cache and reused them at every rotation, so at 90°/180° the light
sat in the wrong place relative to the rotated geometry (a different physical
setup, not a frame change); it produced near-black references at some rows and
per-row deltas up to −18.21 dB, preserved at
`out/r1-encoding-redesign/report-INVALID-unrotated-lights.json` rather than
discarded. Both are recorded so the cost of getting a comparative experiment
wrong is visible, not just the final valid number.

**Per gate spec, no fourth arm, threshold widening, seed drop, or rotation-set
change follows this result.** `promoted: false` and `stop_reason` in
`out/r1-encoding-redesign/report.json` record the stop condition; R2–R6 remain
blocked on R1. Full per-row numbers, the rotation table, and the G2 capacity
accounting are in
`docs/performance.md#world-anchored-encoding-redesign-representation-track-rung-r1`.

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
interpolation or R3–R6 promotion. The R1 candidate promotion gates remain closed.

## Reproduce and verify

The standard configs are the 2D controls; the matched world-grid configs are
`examples/r1_toy_world3d.json` and `examples/r1_kitchen_world3d.json`.

```sh
UV_CACHE_DIR=.uv-cache uv run python examples/r1_worldgrid.py --devices cpu mps
UV_CACHE_DIR=.uv-cache uv run python examples/r1_failure_analysis.py --reuse
UV_CACHE_DIR=.uv-cache uv run python examples/r1a_variance.py --seeds 0 1 2 3 4 --denoise-method oidn
UV_CACHE_DIR=.uv-cache uv run python examples/r1_promotion.py --second-cache out/r1-promotion/bedroom_cache.npz --second-scene Bedroom --gather-backend numpy --denoise-method oidn --workers 2
mise run test
mise run lint
mise run pipeline-audit
```

The Bedroom cache is generated locally rather than vendored. Recreate it with the
existing downloader/exporter before the promotion command:

```sh
UV_CACHE_DIR=.uv-cache uv run python examples/scenes/download_scene.py bedroom
UV_CACHE_DIR=.uv-cache uv run python -m nrp.mitsuba_exporter --scene examples/scenes/bedroom/scene.xml --width 128 --height 128 --spp 64 --bounces 4 --mode scalar --out out/r1-promotion/bedroom_cache.npz
```

The experiment command exits nonzero after writing all JSON evidence when the
binding gate fails. That is expected for this recorded negative.
