# R1 Next Experiments — Stabilize Before Adding Cameras

## Decision carried forward

The original 3D world hashgrid is an honest negative. Across three controlled
Country Kitchen seeds its deltas versus paired 2D controls are −0.544, −0.771, and
+0.247 dB, so only one seed meets the binding −0.5 dB floor. A matched-budget
world tri-plane improves the mean delta to +0.023 dB but is less stable
(+1.359, −0.935, −0.356 dB) and passes only two seeds. Neither representation is
promoted. R2 multi-camera work stays blocked.

This plan does not weaken the 0.5 dB gate. It tries to explain and reduce the
variance before spending on multi-camera caches.

## What the first diagnosis established

1. The historical 25.24 dB 2D artifact is not a gate control. It used a different
   validation-light stream and predates output-scale initialization. On the current
   fixed lights it scores 23.27 dB.
2. Absolute training variance is large: paired 2D controls range from 22.00 to
   24.68 dB across three seeds.
3. The direct 3D grid has much higher observed hash collision pressure (86.7%) than
   2D (22.8%). An 88%-larger 3D grid reduces collisions to 66.6% but does not restore
   parity, so raw capacity is not a sufficient fix.
4. Tri-plane allocation reduces observed collisions to 35.3% and can outperform 2D,
   but the benefit is seed-sensitive.
5. Turning off output-scale initialization raises seed-0 absolute PSNR by 2.35 dB
   for 2D and 2.25 dB for 3D, while their paired delta remains a failure (−0.64 dB).
   Initialization is an interaction to characterize, not an explanation that erases
   R1.

Machine-readable evidence: `out/r1-followup/report.json`.

## Experiment ladder

### R1A — Variance decomposition

Run five paired seeds for `pixel2d`, `world3d`, and `world_triplane`, crossing two
initialization policies:

- target-scale output bias;
- framework-default output bias.

Freeze the validation lights once per seed and share them across every arm. Record
per-light paired PSNR deltas, median, standard deviation, and a paired bootstrap 95%
confidence interval. Do not select a checkpoint by its validation result.

**Question:** is the representation delta stable after separating initialization
from data/pool order?

**Stop:** if no world-anchored arm passes the −0.5 dB floor on all five seeds, do
not start R2.

### R1B — Collision and allocation sweep

Keep total model parameters within ±0.5% of the 2D control and sweep only allocation:

- direct 3D with fewer levels and larger per-level tables;
- direct 3D with level-specific table sizes instead of one shared cap;
- tri-plane with 2, 3, and 4 levels;
- shared-plane versus independent-plane tables;
- concatenation versus learned gated fusion of XY/XZ/YZ features.

Log queried vertices, occupied slots, maximum slot load, and collision fraction per
level. Rank by the five-seed gate, not by seed-0 PSNR.

**Question:** is the instability tied to collisions, the multiresolution schedule,
or the way plane features are fused?

**Stop:** an unmatched-capacity arm may diagnose a mechanism but cannot pass R1.

### R1C — Coordinate robustness

For the best two R1B arms, repeat the five-seed matrix under:

- three fixed rotations of world coordinates about the scene up axis;
- scene-AABB normalization versus robust percentile bounds;
- a small out-of-bounds holdout to test clamping behavior.

Axis-aligned tri-planes can accidentally favor one kitchen orientation. A viable
world representation should not depend on a lucky world frame.

**Gate:** worst-orientation delta must remain at or above −0.5 dB for every seed.

### R1D — Error localization

Extend the current depth-quartile report with:

- surface-normal bins;
- world-space occupancy/collision bins;
- light-to-hit distance bins;
- per-light paired deltas;
- image-space error heatmaps saved alongside numeric summaries.

This rung is diagnostic. It cannot promote a representation, but it should determine
whether failures cluster at depth discontinuities, particular planes, or overloaded
hash slots.

### R1E — Independent scene confirmation

Only after an arm clears R1A–R1C, run the unchanged matched-budget gate on one more
real scene/cache at 128². Use the same five paired seeds and fixed validation
protocol.

**Promotion rule:** all five seeds must clear −0.5 dB on both real scenes. Report
mean and confidence intervals as context, but do not average away an individual
failure.

## Possible way forward

If a world-anchored representation clears R1E, proceed to a redesigned R2 pilot:
two training cameras plus one held-out interpolation camera, explicit camera
metadata, and a comparison against separate per-camera 2D proxies at matched total
parameters. If no arm clears R1E, close this branch as a characterized negative and
investigate surface-attached alternatives (mesh/UV features or sparse point/voxel
features) rather than continuing to tune a camera-conditioned MLP around an unstable
world hash.

The important direction is not “make the best seed look better.” It is to produce a
world-anchored representation whose worst controlled run is reliable enough to
justify the much more expensive multi-camera question.
