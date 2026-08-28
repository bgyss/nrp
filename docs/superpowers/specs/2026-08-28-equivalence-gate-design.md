# Equivalence Gate — Replacing the Underpowered Per-Seed Parity Gate

## Why

The representation track's promotion gate is: *every* seed's paired PSNR delta
versus the same-run `pixel2d` control must be ≥ −0.5 dB. Five seeds, all must
clear.

Measured against the per-seed spreads from the deterministic Kitchen re-run
(`out/r1-parity-kitchen-det/report.json`), that rule rejects an arm that is
*exactly at parity* most of the time:

| arm (measured std) | P(one seed clears) | P(all 5 clear) | P(all 10 clear) |
|---|---:|---:|---:|
| `world_normal_triplane` (0.73) | 0.753 | **0.243** | 0.059 |
| `world_sparse` (1.00) | 0.691 | **0.158** | 0.025 |
| `world3d` (1.67) | 0.618 | **0.090** | 0.008 |

A true-parity arm fails 76–91% of the time. Worse, because the rule requires
every seed to clear, its false-rejection rate *rises* with the seed count: at ten
seeds a perfect arm passes under 6% of the time. **The rule cannot be repaired by
adding seeds** — more data makes it stricter in the wrong direction, punishing
sample size rather than rewarding it.

The spread is not measurement noise. The OIDN nondeterminism found on 2026-08-27
(`docs/performance.md`, "Run-to-run nondeterminism exceeds the gate") is fixed;
these runs are bit-reproducible. What remains is genuine sensitivity of these arms
to initialization and light sampling, and it is larger than the threshold being
tested.

This spec replaces the rule's *structure*, not its −0.5 dB threshold, which is
unchanged and not up for negotiation here.

## The rule

The per-seed statistic is unchanged: `d_i` is the mean over the held-out
validation lights of (candidate − control) PSNR, paired by light index, for seed
`i` — exactly what `pair_validation_metrics` already produces. What changes is how
the collection `{d_i}` yields a verdict.

Let `CI(n)` be a two-sided confidence interval for the mean of `d` over `n` seeds,
at the per-look confidence defined below, and let `T = -0.5` dB.

| verdict | condition |
|---|---|
| `pass` | `CI_lower >= T` — confidently not worse than the threshold |
| `fail` | `CI_upper < T` — confidently worse than the threshold |
| `continue` | interval straddles `T`, cap not reached |
| `underpowered` | interval straddles `T` at the cap |

`underpowered` is **never** a pass and never a fail; it reports that the
experiment did not answer its question. A one-sided question is deliberately
stated with a two-sided interval: the extra conservatism is accepted rather than
argued away.

This rule gets **stronger** with more seeds — a wider sample tightens `CI` and
makes both `pass` and `fail` harder to reach spuriously — which is the property
the current rule inverts.

## Interval estimator: Student-t, binding

The gate's interval is a **Student-t interval** on `{d_i}`:
`mean ± t(1-α'/2, n-1) · s / √n`, with `s` the sample standard deviation.

The existing reports use a percentile bootstrap over the per-seed deltas. At n=8
and 99.17% confidence the bootstrap's endpoints are effectively the min and max of
eight numbers, and percentile-bootstrap coverage that far into the tails is
unreliable — it would undermine the guarantee this gate exists to provide. Each
`d_i` is itself a mean over ~12 validation lights, so approximate normality of
`d_i` is a reasonable assumption and the t-interval's coverage is the better bet
at these sample sizes.

The bootstrap interval is still computed and recorded, as a descriptive statistic
and for continuity with existing reports. It is **not** binding. Every report
states which estimator was binding so no reader has to infer it.

**Implementing the quantile.** The repo installs numpy and torch only; scipy is
not a dependency and one quantile does not justify making it one — this codebase
hand-rolls its numerics elsewhere (the numpy backend's autodiff). The module
computes `t_ppf` itself: the regularized incomplete beta by Lentz's
continued-fraction method, the t CDF from it, and the quantile by bisection,
memoized over the six distinct degrees of freedom the look schedule uses. Unit
tests pin it against published table values (t(0.975, df=10) = 2.228,
t(0.975, df=47) = 2.012), which the prototype reproduces exactly.

## Looks, stopping, and the multiplicity correction

Seeds accumulate and the gate is evaluated at pre-registered milestones only:

- **Looks:** n = 8, 16, 24, 32, 40, 48 (six looks)
- **Cap:** 48 seeds
- **α:** 0.05 overall, Bonferroni-split across the six looks → α' = 0.008333 per
  look → **99.1667% two-sided CI at every look**

Evaluating at a look that is not in the schedule is an error, not a convenience:
unscheduled peeking is exactly what the correction exists to prevent.

Seeds are consumed in fixed ascending order (0, 1, 2, …), so an n=16 run is a
strict prefix of an n=48 run. Two consequences, both wanted: a longer run can
reuse a shorter one's trainings, and two runs at different caps remain directly
comparable.

Every verdict records `n`, `look_index`, `looks_taken`, `confidence`, and the
threshold, so the correction applied is auditable from the report alone.

**Not corrected:** multiplicity across the three world arms. Each arm is treated
as its own pre-registered question, tested once against a fixed control. Testing
three arms and promoting whichever passes would inflate the error rate; this spec
does not do that, and any future "best of N arms" selection needs its own
correction. Stated here so the omission is deliberate and visible rather than
silent.

## Module

`nrp/experiment_gate.py` — the gate rule, its arithmetic, and nothing else. It
knows about sequences of per-seed deltas; it knows nothing about caches, models,
or training.

```python
class EquivalenceGate:
    def __init__(self, threshold_db=-0.5, looks=(8,16,24,32,40,48), alpha=0.05): ...

    @property
    def confidence_per_look(self) -> float:      # 1 - alpha/len(looks)

    def next_look(self, n: int) -> int | None:   # next milestone, or None past the cap

    def evaluate(self, per_seed_deltas) -> dict:
        """Verdict at n = len(deltas). Raises unless n is one of `looks`."""


def per_seed_verdict(deltas, threshold_db=-0.5) -> dict:
    """The legacy every-seed rule, kept so both verdicts appear in every report."""


def seeds_needed(std_db, half_width_db, confidence) -> int:
    """Seeds required for a t-interval of the given half-width.

    Reported with every `underpowered` verdict, so a run that cannot decide says
    what it would take to decide. Measured values at the gate's confidence and a
    0.5 dB half-width: 19 seeds at std 0.73, 32 at std 1.00, 82 at std 1.67.
    """
```

`evaluate` returns: `rule`, `verdict`, `threshold_db`, `n`, `look_index`,
`looks_taken`, `confidence`, `estimator` (`"student_t"`), `ci_lower_db`,
`ci_upper_db`, `mean_db`, `std_db`, and a `definition` string stating the rule in
words. Raising rather than guessing is the house style here — a gate that reports
a verdict it cannot support is the defect this whole exercise is correcting, so
zero seeds, an off-schedule `n`, and n < 2 (no variance estimate) all raise.

## Runner integration

`examples/r1_parity.py` gains:

- `--gate equivalence|per-seed` (default: `equivalence`)
- `--max-seeds` (default: the cap, 48)
- training in look-sized chunks, evaluating at each look and stopping as soon as a
  verdict is reached

Both verdicts are always written to `report.json` — the equivalence verdict and
the legacy per-seed verdict — with a top-level `gate_rule` naming which one is
binding. Computing both is nearly free and lets any report be read against either
convention.

Exit status: unchanged in spirit — non-zero when no world arm passes.
`underpowered` is not a pass.

The other three runners (`r1a_variance.py`, `r1_failure_analysis.py`,
`r1_kitchen_k1.py`) keep their current gate. Their experiments are closed, and
re-labelling a finished result with a gate it was never measured under would
misrepresent it. Historical reports are untouched.

## Testing

Unit tests for the arithmetic and the boundaries:

- a CI lower bound exactly at −0.5 passes (boundary is inclusive)
- `fail` when the whole interval sits below the threshold
- `continue` before the cap, `underpowered` at the cap, for a straddling interval
- `confidence_per_look` equals 1 − α/len(looks); `next_look` arithmetic; `None`
  past the cap
- off-schedule `n`, empty input, and n < 2 all raise
- more seeds at the same mean and spread narrow the interval (monotonicity)
- `seeds_needed` matches the closed form and rounds up

Simulation tests for the property that motivated the change — these are the point
of the spec, so they are enforced by test rather than by argument. Thresholds come
from a 4,000-experiment prototype simulation at cap 48, not from aspiration:

| true mean | std | pass | fail | underpowered | median n |
|---|---:|---:|---:|---:|---:|
| 0.0 (at parity) | 0.73 | 0.977 | 0.000 | 0.023 | 24 |
| 0.0 | 1.00 | 0.798 | 0.000 | 0.202 | 32 |
| 0.0 | 1.67 | 0.341 | 0.000 | 0.659 | 48 |
| −0.25 | 1.00 | 0.241 | 0.000 | 0.759 | 48 |
| −1.5 (clearly worse) | 0.73 | 0.000 | 1.000 | 0.000 | 8 |
| −1.5 | 1.00 | 0.000 | 1.000 | 0.000 | 16 |
| −1.5 | 1.67 | 0.000 | 0.927 | 0.073 | 24 |

Tests assert, with margin below the measured values (600 trials each, fixed
seeds):

- at parity, std 0.73: pass rate **≥ 0.95**
- at parity, std 1.00: pass rate **≥ 0.70**
- 1.5 dB worse, any std: pass rate **≤ 0.01**, fail rate **≥ 0.85**
- at parity, std 1.67: **≥ 0.50 underpowered** — the gate must refuse to certify
  what 48 seeds cannot resolve, and that refusal is a tested behavior, not an
  accident
- the legacy per-seed rule through the same simulation: pass rate **≤ 0.30** at
  parity, a regression test on the diagnosis that motivated this spec

Note what the middle rows say: at cap 48 an at-parity arm with `world3d`'s spread
(1.67) resolves only a third of the time, because certifying ±0.5 dB at that
spread needs ~82 seeds. The cap stays at 48 by decision; the gate reports
`seeds_needed` alongside every `underpowered` verdict so the cost of a definitive
answer is stated rather than guessed. Failure detection is unaffected — a clearly
worse arm is caught at every spread, usually by seed 8–24.

All simulations use fixed seeds so the suite stays deterministic.

## Documentation

`docs/performance.md` gains a section defining the gate, its power properties, and
the measured false-rejection rates of the rule it replaces.
`docs/representation-track.md` notes which gate applies from this date and that
earlier verdicts were measured under the per-seed rule.

## Out of scope

- **Re-running any experiment.** This spec defines the gate. Re-measuring Kitchen
  under it costs up to ~8 h (48 seeds × 4 arms) and is a separate decision.
- **Changing the −0.5 dB threshold.** Unchanged.
- **Migrating the other three runners.**
- **Cross-arm multiplicity correction**, as discussed above.
