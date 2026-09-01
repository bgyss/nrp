# R1 Generalization Gate Re-specification — One Pre-registered Attempt

> **Status: DRAFT, NOT APPROVED. No training may start under this plan until it
> is signed off.** Nothing in it changes a recorded verdict: R1 remains closed as
> a characterized negative and R2–R6 remain blocked unless and until the
> confirmatory run below passes.

## Decision carried forward

Three results bound what this plan may propose, and all three are committed
evidence rather than expectation:

1. **Single-view fair-allocation parity is settled and is no longer the
   question.** At the equivalence gate's n=32 and n=48 looks on Country Kitchen
   128² with 96 held-out lights, `world_sparse` **passes**
   (n=48: mean −0.146 dB, 99.17% CI [−0.402, +0.109]); `world_normal_triplane`
   and `world3d` end **`underpowered`** at the n=48 cap and would need 120 and
   126 seeds to pass if their means hold. See
   `docs/performance.md#r1-kitchen-parity-at-the-n48-cap-the-other-two-arms-end-underpowered-2026-08-31`.
2. **The K-series is cancelled and cannot be revived for this purpose.** K2–K4
   were conditional on K1 confirming that vertex support drives the parity
   delta; K1 is undecided across three readings (ρ = +0.20, −0.30, +0.30, all
   p ≈ 0.68–0.75) and never confirmed the direction
   (`docs/plans/2026-08-27-kitchen-parity-next-steps.md`). Independently, the
   K-series targets *single-view reconstruction*, which item 1 has now settled;
   it does not act on held-out-camera behaviour at all.
3. **The binding R1 negative is held-out-camera generalization, and it is real
   at the row level.** Re-read at 96 held-out lights from the committed
   checkpoints, no arm passes G1/G3/G4 and failing rows rise from 83 to 94 of
   180 (`docs/performance.md#encoding-redesign-campaign-re-read-at-96-held-out-lights-2026-08-29`).
   The negative is **not** an artifact of the 8-light estimator.

## What this plan does, and the one thing it must not become

The R1 generalization gate is conjunctive: G1 requires every one of 60 rows to
beat the baseline by ≥ 1.0 dB, G3 requires every seed to pass, and G4 requires
the worst row across three world rotations to clear the same margin. That is
structurally the rule the track already retired once. The per-seed parity rule
rejected an at-parity arm 76–91% of the time, its false-rejection rate *rose*
with seeds, and `world_sparse` demonstrated the failure concretely — 23/32 seeds
under the old rule, `pass` under the interval rule
(`docs/performance.md#the-equivalence-gate-from-2026-08-28`).

Against that, `world_sparse` at 96 lights already beats the baseline **on
average** by +1.412 dB against a +1.0 dB margin, on all five seeds
(+1.02, +2.12, +1.33, +1.51, +1.07). It fails only because 24 of 60 individual
rows dip below.

**This plan proposes exactly one re-specification of the R1 generalization gate,
from a conjunctive per-row rule to the interval gate already built, tested, and
in use on the parity question, and exactly one confirmatory run against it.**

**The thing it must not become:** a search over gate structures until R1 passes.
The re-specification below is chosen *after* seeing that the conjunctive rule
fails, which is precisely the move that requires pre-registration to stay
honest. If the confirmatory run does not pass, **R1 stays closed and no further
gate change may be proposed under this plan** — the next move becomes
re-diagnosis or a new representation, not a third rule.

## Why a gate change alone is not enough (and why this plan needs new seeds)

Re-scoring the committed 96-light checkpoints under an interval gate — no
retraining, evaluation only — does **not** flip R1:

| arm | mean Δ | interval over 60 rows | interval over 5 seeds | vs +1.0 margin |
|---|---:|---:|---:|---|
| `world_sparse` | +1.412 | [0.907, 1.917] | [0.448, 2.376] | not pass |
| `world_normal_triplane` | +1.079 | [0.511, 1.648] | [−0.658, 2.817] | not pass |
| `world3d` | +0.992 | [0.472, 1.512] | [−0.516, 2.500] | not pass |

(99.17% Student-t, the gate's six-look confidence.) `world_sparse` misses by
0.093 dB on the row reading and by 0.552 dB on the conservative seed reading.
**A re-specification that is only a re-scoring is therefore already known to
fail, and this plan does not propose one.** The deficit is power: 5 seeds.

## Pre-registered design

Every item below is fixed before any training starts. Deviations invalidate the
run and must be recorded as such rather than absorbed.

**Arm.** `world_sparse` **only**, as the single confirmatory arm. It is chosen
on grounds that pre-date this analysis: collision-free by construction (0.0%,
asserted not merely reported), essentially flat across world rotation
(+2.00 / +1.83 / +1.80 dB at 0°/90°/180°, where both hashed arms degrade
markedly), and the only arm to pass the parity gate. The other two arms are
trained and reported **descriptively** and are not gated — this is what removes
the across-arm multiplicity that the n=32 and n=48 parity looks had to carry as
a caveat.

**Estimand.** The mean, over seeds, of each seed's mean per-row delta in dB
against the campaign's existing baseline (`pixel2d` trained at the nearest
trained camera and reused at the held-out camera), at held-out cameras, scored
at 96 held-out lights as twelve independent draws of the campaign's own 8-light
configuration. Unchanged from the committed campaign in every respect except the
seed count.

**Unit of analysis: the seed, not the row.** The 12 rows within a seed
(4 held-out cameras × 3 rotations) share one trained model and one traced cache,
so they are not independent, and the row-level interval above understates its own
width. Using the seed is the conservative choice, and it is why the power
calculation lands at 12 rather than 7.

**Gate G1′.** `nrp/experiment_gate.py`'s `EquivalenceGate`, unmodified code,
with `threshold_db = +1.0` (**the margin is inherited unchanged from the R2
ladder and is not being weakened**), applied to the per-seed mean deltas. Pass
iff the interval's lower bound is at or above +1.0 dB; fail iff it lies entirely
below; `underpowered` at the cap.

**Look schedule.** Two looks, `(12, 20)`, α = 0.05 split two ways → 97.5%
per-look confidence. A single look at n = 12 would suffice at 95%
(7 seeds needed), but the two-look schedule buys one honest extension if the
sample sd comes in above the 0.444 dB this is powered on.

**Seeds: 12 fresh seeds (5–16) as the confirmatory sample.** Seeds 0–4 are
**excluded from the primary analysis**. They generated the hypothesis, the
mean, and the sd this run is powered on; reusing them would let the same data
both motivate and confirm the change. The pooled 17-seed result is reported as
secondary and is explicitly not the verdict.

**Retained guards, unchanged.** The 15 dB absolute floor stays as a conjunctive
safety check (non-binding to date — the dimmest row at 96 lights is 19.60 dB —
and kept precisely because it is a broken-arm detector, not a discriminator).
G2 capacity and G5 fallback decomposition continue to be reported.

**G4′ (rotation), asymmetric by design.** Per-rotation intervals over the same
12 seeds are reported, and a **fail** is declared if any single rotation's
interval lies entirely below +1.0 dB. Passing is *not* required per rotation:
three strata of 12 seeds are not powered for that, and demanding it would
reintroduce a conjunctive rule through the back door. This is a frame-dependence
detector, not a promotion criterion, and it is stated as weaker than the
original G4 rather than presented as equivalent.

## Prediction, falsifier, stop condition

**Prediction.** If `world_sparse`'s mean (+1.412 dB) and between-seed sd
(0.444 dB) hold on 12 unseen seeds, the interval is [1.080, 1.744] and G1′
**passes**. At n = 20 it would be [1.170, 1.654].

**Falsifier.** If the n = 12 interval straddles +1.0 dB, take the single
scheduled n = 20 look. If that also straddles, the verdict is `underpowered` at
the cap and **R1 stays closed**. If either interval lies entirely below +1.0 dB,
the verdict is `fail` and R1 is closed harder than before.

**Stop condition.** The verdict is read and recorded before anything else is
started, and **no rung R2–R6 is unblocked by this plan except on an explicit
`pass`**. A `pass` unblocks the R1 prerequisite only; R2's own pilot was
negative on three-view deltas and must be re-run on its own terms.

## Threats to validity, stated in advance

- **The gate was chosen after seeing the conjunctive rule fail.** Mitigated by:
  fresh seeds for the confirmatory sample, a single pre-declared arm, a fixed
  seed count and schedule, an unchanged margin, and a one-attempt limit. Not
  fully mitigated — a reader should weigh this as a re-specified gate, not an
  independent replication, and the write-up must say so.
- **The baseline is weak.** `pixel2d` reused from the nearest trained camera is
  a screen-space proxy applied at a camera it never saw; beating it by 1.0 dB is
  a low bar for a world-anchored encoder. A `pass` certifies "beats a reused
  screen-space proxy at interpolated held-out cameras on toy 64²" and nothing
  more. It is **not** evidence of production-grade novel-view quality, and R3's
  extrapolation question remains untouched.
- **The unit-of-analysis change (rows → seeds) moves the gate.** It is the
  conservative direction here (wider intervals, harder to pass), and it is
  declared before the run, but it is a change and must be reported as one.
- **The mean may not hold.** Two of five current seeds sit near the margin
  (+1.02, +1.07). If the true mean is nearer 1.2 than 1.4, 12 seeds will not
  settle it and the n = 20 look will be needed.

## Scope and cost

The campaign trains 11 models per (seed, rotation) — 8 per-camera `pixel2d`
baselines plus 3 world arms — over 3 rotations, i.e. 33 per seed. Twelve fresh
seeds is **396 trainings** plus 12 cameras' path-cache exports per seed, on toy
64² at 3,000 iterations.

The committed campaign report records no per-training time, so **the first step
is a one-seed timing run to price the full 12 before it is funded** — not an
estimate carried in prose. `--skip-export` already allows reusing exported
caches across re-runs.

Evidence lands in `out/r1-generalization-eq/report.json`, following the existing
report schema, citing the committed 5-seed campaign
(`out/r1-encoding-redesign/report.json`) and its 96-light re-read
(`out/r1-encoding-redesign/rescore-96.json`) by path rather than re-measuring
them. The runner is `examples/r1_encoding_redesign.py` with a seed list and the
new gate wiring; the gate itself is `nrp/experiment_gate.py` used as-is, with no
new statistics code.

## Approval

This plan requires sign-off on three points specifically, since they are the
ones that cannot be undone after the fact: (1) that an interval gate is an
acceptable re-specification of G1/G3 at all, (2) the 12-fresh-seed confirmatory
design with seeds 0–4 excluded, and (3) the one-attempt limit — that a
non-passing result closes R1 rather than prompting a third rule.
