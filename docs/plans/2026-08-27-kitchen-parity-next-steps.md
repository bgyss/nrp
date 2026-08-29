# Kitchen Parity Next Steps — Testing the Vertex-Support Hypothesis

> **Outcome (2026-08-27): K1 ran and gives NO SUPPORT for the hypothesis (and is
> underpowered to establish either direction). K2, K3, and K4 are cancelled**
> under this plan's own stop condition — they were conditional on K1 confirming
> the predicted direction, which it did not. Sweeping `world_sparse`'s finest
> resolution over 32/48/64/96/128 x 5 seeds against the fixed committed `pixel2d`
> control gives a Spearman correlation of **+0.20** (p ≈ 0.75) between resolution
> and mean delta where the prediction required a negative one, is not monotonic,
> and passes the unchanged -0.5 dB gate at no setting — but 5 points and p ≈ 0.75
> against per-seed noise comparable to the effect size cannot rule out the
> opposite direction either. K1 additionally found ~1.5 dB run-to-run variation
> at a fixed seed, which exceeds the gate and is not removed by
> single-threading. Results:
> `docs/performance.md#k1-finest-resolution-sweep-does-not-support-the-vertex-support-hypothesis-kitchen-128`,
> `out/r1-kitchen-parity-k1/report.json`, runner `examples/r1_kitchen_k1.py`.
>
> **Re-run (2026-08-28): K1 was re-measured under the equivalence gate at 8 seeds,
> against a deterministic 8-seed `pixel2d` control, and every resolution returns
> `continue`** — the interval for the mean delta straddles −0.5 dB at all five
> settings, so nothing is promoted and nothing is rejected. Spearman is now −0.30
> (permutation p = 0.68), nominally the predicted sign but indistinguishable from
> noise and still not monotonic. K2–K4 remain cancelled: the stop condition
> requires K1 to confirm the direction, and an undecided sweep does not. Deciding a
> single setting would take 19–47 seeds. The run also settles the nondeterminism
> caveat above: the sweep's 128 row is bit-identical, per seed, to the control
> run's `world_sparse` arm trained in a separate process, so the ~1.5 dB
> run-to-run variation is gone under the single-threaded OIDN fix. Results:
> `docs/performance.md#k1-re-run-under-the-equivalence-gate-kitchen-128-8-seeds`,
> `out/r1-kitchen-parity-k1-eq/report.json`,
> `out/r1-parity-kitchen-eq/report.json` (the fixed control).
>
> The plan below is preserved as written, before the result was known.

## Decision carried forward

The fair-allocation parity re-measurement (`docs/performance.md#r1-fair-allocation-parity-re-measurement-toy-64-and-kitchen-128`,
`out/r1-parity/report.json`, `out/r1-parity-kitchen/report.json`) found the
opposite result on two scenes under the identical arms, protocol, and unchanged
−0.5 dB per-seed gate: on toy 64² all three world-anchored arms
(`world_sparse`, `world_normal_triplane`, `world3d`) pass 5/5 seeds against
`pixel2d`; on Country Kitchen 128² none passes, and giving `world_sparse` 5.5×
the control's grid slots with zero hash collisions makes it the *worst*
performer, not the best. This also retracts the earlier claim that the
~19× parameter-matching allocation handicap explained the original Kitchen
negative: `world3d`'s fair-allocation Kitchen mean (−0.355 dB) reproduces the
original handicapped mean (−0.356 dB) almost exactly, so removing the handicap
changed nothing.

This plan does not weaken the −0.5 dB gate and does not propose more capacity
or more tuning of the existing arms as a first move. It proposes a diagnostic
sweep to test a specific hypothesis for *why* Kitchen fails where toy passes,
with an explicit falsifier before any remediation is attempted.

## The hypothesis under test

Measured directly from the caches, finest-level grid-vertex support (pixels
touching each vertex) differs sharply between the two scenes:

| scene | vertices/pixel | median support | vertices touched by ≤1 pixel |
|---|---:|---:|---:|
| toy 64² | 3.35 | 2 px | 33.7% |
| Kitchen 128² | 4.77 | 1 px | 59.1% |

**Hypothesis:** on Kitchen, a majority of finest-level world-space vertices are
constrained by a single observed pixel — an under-determined free parameter
that can memorize its one sample but has nothing forcing it to generalize to a
neighbor. `pixel2d` avoids this by construction: at `finest_resolution=128` on
a 128² render, every finest-level vertex is shared by roughly 4 neighboring
pixels regardless of scene content, because screen-space vertex density is
tied to pixel density. World-space vertex density instead follows scene
geometry, and Kitchen's geometry happens to spread the same pixel budget over
enough distinct surface positions that most vertices go under-supported.

This is currently a hypothesis "consistent with the evidence," not an
established mechanism — it has not been tested by deliberately varying vertex
support and observing the predicted effect on the parity delta. That is what
K1 does.

## Experiment ladder

### K1 — finest-resolution sweep (the decisive test)

Sweep `world_sparse`'s `finest_resolution` on Kitchen — e.g. 32, 48, 64, 96,
128 — holding everything else fixed (base resolution 4, levels, iterations,
pool, denoiser, seeds) and compare each setting against the **same fixed,
already-committed `pixel2d` control at `finest_resolution=128`**
(`out/r1-parity-kitchen/report.json`'s pixel2d arm) — not a `pixel2d` re-run at
the swept resolution, since the point is to test whether *lowering
`world_sparse`'s* resolution closes the gap against the existing screen-space
baseline, not to weaken the baseline too.

At each setting, report both:

1. the parity gate result (per-seed delta vs. the fixed `pixel2d` control,
   5/5-seed pass/fail, mean, std), and
2. the measured finest-level vertex-support distribution for that resolution
   (vertices/pixel, median support, % touched by ≤1 pixel), using the same
   occupancy measurement (`nrp/torch_backend/occupancy.py`) as the table above.

**Prediction:** the parity delta improves monotonically (or close to it) as
`finest_resolution` falls, with the best delta near the resolution where
median vertex support approaches `pixel2d`'s ~4 pixels/vertex.

**Falsifier:** if the delta is flat across the swept resolutions, or gets
worse as resolution falls, the under-determination hypothesis is wrong as
stated, and **K2–K4 below should not be run** — a different mechanism is
responsible for the Kitchen negative and this plan's next step becomes
re-diagnosis, not remediation.

**Scope:** 5 resolutions × 5 seeds ≈ 25 trainings, plus the fixed control
already on hand. Comparable wall-clock cost to the existing 5-seed Kitchen
parity run (`out/r1-parity-kitchen/`) times ~5.

**Stop condition:** K1's result is read and reported — pass or fail on the
prediction — before any of K2–K4 is started, regardless of how promising an
intermediate resolution looks.

### K2 — low-support vertex pruning (conditional on K1)

If K1 confirms the direction, target the mechanism directly rather than
uniformly reducing resolution everywhere: drop vertices whose measured support
falls below a threshold from `world_sparse`'s occupied-vertex index, letting
the coarser levels carry them through the existing "not in the sparse set →
zero features at this level, fall back to coarser levels" path (already
implemented for out-of-occupancy queries at held-out cameras; §"Arm B" in
`docs/superpowers/specs/2026-08-26-world-anchored-encoding-redesign-design.md`).

Sweep the support threshold (e.g. drop vertices with support ≤1, ≤2, ≤3
pixels) and report the parity delta and remaining vertex count at each
setting. This keeps full resolution where the data supports it and only
removes capacity where the data cannot constrain it, rather than trading away
resolution uniformly as K1 does.

**Question:** does pruning specifically the under-supported vertices recover
more of the gap than uniformly lowering `finest_resolution` did in K1, for the
same resulting vertex count?

### K3 — more supervision per vertex (conditional on K1)

Increase the light pool size and/or the number of distinct training lights, so
each finest-level vertex is exposed to more (light, target) pairs before the
model commits to a value for it. This is the cheapest lever to try — no
geometry or index changes — and directly addresses "one observation per
vertex" without touching the encoding.

**Question:** does more supervision alone reduce the memorization gap, or does
the vertex remain under-determined regardless of how many lights it is shown
(because the geometric degree of freedom, not the light count, is the limit)?

### K4 — regularization on the feature tables (conditional on K1, run last)

Add weight decay or a smoothness prior (e.g. penalizing feature-value
differences between spatially adjacent occupied vertices) on the grid feature
tables — the standard remedy for under-determined parameters in an
optimization sense, independent of the specific cause.

**Question:** does regularizing the under-determined vertices directly close
the gap, as a sanity check against K1–K3's more targeted interventions?

This is the least targeted of the four and is ordered last: it treats the
symptom (unconstrained parameters) rather than the diagnosed cause (support
density mismatch with the screen-space control), so a positive K4 result would
need to be reconciled with whatever K1–K3 found before drawing a conclusion
about mechanism.

## Ordering and gating

K1 is the only rung in this plan that can falsify the vertex-support
hypothesis. K2, K3, and K4 are conditional on K1 confirming the predicted
direction and should not be started otherwise — running them against a
falsified hypothesis would produce results with no diagnostic value, just more
tuning of the kind the original R1 redesign explicitly stopped doing.

Machine-readable evidence for K1 (and, if run, K2–K4) should land in
`out/r1-kitchen-parity-k1/report.json` (and `-k2`/`-k3`/`-k4` siblings),
following the existing `out/r1-parity*/report.json` schema and gate
convention, with hardware context and the fixed `pixel2d` control's report
path cited explicitly rather than re-measured.
