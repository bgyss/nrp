# Equivalence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the representation track's underpowered "every seed must clear −0.5 dB" promotion rule with a three-outcome equivalence gate whose false-rejection rate is controlled and which gets stronger, not weaker, as seeds are added.

**Architecture:** One dependency-free module, `nrp/experiment_gate.py`, owns the whole rule: a hand-rolled Student-t quantile, the interval, the four verdicts, the Bonferroni-corrected look schedule, and a planning helper. It knows only about sequences of per-seed deltas — no caches, models, or training. `examples/r1_parity.py` then consumes it, training seeds in look-sized chunks and stopping as soon as a verdict is reached, writing both the new and the legacy verdict into every report.

**Tech Stack:** Python 3.12, numpy (no scipy — see Global Constraints), `unittest` (NOT pytest), ruff, uv. Spec: `docs/superpowers/specs/2026-08-28-equivalence-gate-design.md`.

## Global Constraints

- **No new dependencies.** `uv sync` installs numpy + torch only. scipy is NOT available and must NOT be added. The t-quantile is hand-rolled in Task 1.
- **Test framework is `unittest`**, not pytest. Run tests with `uv run python -m unittest tests.test_experiment_gate` — a bare `pytest` invocation will not work.
- **Line length 100** (`[tool.ruff] line-length = 100`). Lint select: `E`, `F`, `I`, `UP`, `B`. `B905` means every `zip()` needs an explicit `strict=`.
- **Gate threshold is −0.5 dB and does not change.** This work changes the rule's structure only.
- **Looks are `(8, 16, 24, 32, 40, 48)`, α = 0.05**, Bonferroni-split six ways → α' = 0.0083333…, per-look two-sided confidence 0.9916667.
- **`underpowered` is never a pass and never a fail.** No code path may collapse it into either.
- **Every simulation test uses a fixed seed** so the suite stays deterministic.
- Repo is not an installed package (`[tool.uv] package = false`); tests import via the `sys.path` shim already present in `tests/`.
- Commit messages end with the two trailer lines used throughout this repo:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01144Jx91dfeyZW5DDgVqfWZ`

---

## File Structure

| File | Responsibility |
|---|---|
| `nrp/experiment_gate.py` (create) | The entire gate rule: t-quantile numerics, `EquivalenceGate`, `per_seed_verdict`, `seeds_needed`. Pure functions over per-seed deltas. |
| `tests/test_experiment_gate.py` (create) | Numerics against published table values, verdict boundaries, look arithmetic, and the power simulations that enforce the spec's whole point. |
| `examples/r1_parity.py` (modify) | Consumes the gate: `--gate`, `--max-seeds`, look-chunked training with early stop, both verdicts in the report. |
| `tests/test_r1_parity.py` (modify) | Covers the runner's new pure helpers (seed planning, report assembly with both verdicts). |
| `docs/performance.md` (modify) | Defines the gate, its measured power, and the false-rejection rates of the rule it replaces. |
| `docs/representation-track.md` (modify) | Notes which gate applies from 2026-08-28 onward. |

Task 1 delivers the numerics, Task 2 the verdict logic, Task 3 the power guarantees, Task 4 the runner integration, Task 5 the docs. Tasks 1–3 are all inside one module and one test file but are separated because a reviewer can meaningfully reject the statistics while approving the numerics.

---

### Task 1: Student-t quantile without scipy

**Files:**
- Create: `nrp/experiment_gate.py`
- Test: `tests/test_experiment_gate.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `t_ppf(p: float, df: int) -> float`, `t_cdf(t: float, df: int) -> float`, `betainc(a: float, b: float, x: float) -> float`. Task 2 calls `t_ppf` only.

- [ ] **Step 1: Write the failing test**

Create `tests/test_experiment_gate.py`:

```python
"""Tests for the equivalence gate (nrp/experiment_gate.py).

The gate this module implements replaces a rule that rejected an arm sitting
exactly at parity 76-91% of the time and got worse as seeds were added (spec:
docs/superpowers/specs/2026-08-28-equivalence-gate-design.md). Its power
properties are therefore not incidental -- they are the reason it exists, so
they are asserted by simulation here rather than argued for in a docstring.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nrp.experiment_gate import t_cdf, t_ppf  # noqa: E402


class StudentTNumericsTests(unittest.TestCase):
    """Pinned against published table values: scipy is not available here."""

    def test_quantiles_match_published_table_values(self):
        # Two-sided 95% critical values, standard t table.
        self.assertAlmostEqual(t_ppf(0.975, 10), 2.228, places=3)
        self.assertAlmostEqual(t_ppf(0.975, 47), 2.012, places=3)
        self.assertAlmostEqual(t_ppf(0.995, 7), 3.499, places=3)

    def test_large_df_approaches_the_normal_quantile(self):
        self.assertAlmostEqual(t_ppf(0.975, 100000), 1.960, places=3)

    def test_cdf_is_symmetric_about_zero(self):
        for df in (2, 7, 47):
            self.assertAlmostEqual(t_cdf(-1.3, df), 1.0 - t_cdf(1.3, df), places=12)

    def test_cdf_at_zero_is_one_half(self):
        for df in (1, 5, 47):
            self.assertAlmostEqual(t_cdf(0.0, df), 0.5, places=12)

    def test_ppf_inverts_cdf(self):
        for df in (3, 12, 47):
            for p in (0.6, 0.9, 0.99, 0.9958333):
                self.assertAlmostEqual(t_cdf(t_ppf(p, df), df), p, places=9)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            t_ppf(0.0, 10)
        with self.assertRaises(ValueError):
            t_ppf(1.0, 10)
        with self.assertRaises(ValueError):
            t_ppf(0.975, 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_experiment_gate -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'nrp.experiment_gate'`

- [ ] **Step 3: Write the implementation**

Create `nrp/experiment_gate.py`:

```python
"""Promotion gates over per-seed experiment deltas.

The representation track's original rule -- every seed's paired PSNR delta versus
the control must clear -0.5 dB -- rejects an arm sitting exactly at parity 76-91%
of the time at the per-seed spreads actually measured on Country Kitchen, and its
false-rejection rate RISES with the seed count (a true-parity arm passes under 6%
of the time at ten seeds). It cannot be fixed by adding seeds; only by changing
its structure. See docs/performance.md and
docs/superpowers/specs/2026-08-28-equivalence-gate-design.md.

`EquivalenceGate` replaces it: a confidence interval for the mean per-seed delta
decides between four outcomes, and adding seeds tightens the interval, making both
promotion and rejection harder to reach spuriously. `per_seed_verdict` keeps the
old rule available so every report can carry both.

No scipy: this repo installs numpy and torch only, so the Student-t quantile is
computed here from the regularized incomplete beta (Lentz's continued fraction)
by bisection, pinned against published table values in the tests.
"""

from __future__ import annotations

import functools
import math

import numpy as np

#: Convergence controls for the incomplete-beta continued fraction.
_CF_TINY = 1e-30
_CF_MAX_ITERATIONS = 300
_CF_EPSILON = 3e-16


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _CF_TINY:
        d = _CF_TINY
    d = 1.0 / d
    h = d
    for m in range(1, _CF_MAX_ITERATIONS):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + numerator / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        h *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < _CF_TINY:
            d = _CF_TINY
        c = 1.0 + numerator / c
        if abs(c) < _CF_TINY:
            c = _CF_TINY
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _CF_EPSILON:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
    # Use the reflected series where the direct one converges slowly.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """CDF of Student's t with `df` degrees of freedom."""
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


@functools.cache
def t_ppf(p: float, df: int) -> float:
    """Quantile of Student's t: the value whose CDF equals `p`.

    Bisection rather than a closed form -- there isn't one -- and memoized because
    a look schedule uses only a handful of distinct (p, df) pairs while a power
    simulation calls this thousands of times.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must lie in (0, 1), got {p}")
    if df <= 0:
        raise ValueError(f"df must be positive, got {df}")
    low, high = -1e3, 1e3
    for _ in range(300):
        mid = 0.5 * (low + high)
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m unittest tests.test_experiment_gate -v`
Expected: PASS, 6 tests.

Then lint: `uv run ruff format nrp/experiment_gate.py tests/test_experiment_gate.py && uv run ruff check nrp tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add nrp/experiment_gate.py tests/test_experiment_gate.py
git commit -m "feat: dependency-free Student-t quantile for the equivalence gate

The gate needs a t quantile and this repo installs numpy and torch only, so
the regularized incomplete beta (Lentz continued fraction) and a bisection
quantile live here rather than pulling in scipy for one function. Pinned
against published table values.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01144Jx91dfeyZW5DDgVqfWZ"
```

---

### Task 2: The gate rule — interval, verdicts, look schedule

**Files:**
- Modify: `nrp/experiment_gate.py` (append; do not touch Task 1's numerics)
- Test: `tests/test_experiment_gate.py` (append)

**Interfaces:**
- Consumes: `t_ppf(p, df)` from Task 1.
- Produces:
  - `DEFAULT_THRESHOLD_DB = -0.5`, `DEFAULT_LOOKS = (8, 16, 24, 32, 40, 48)`, `DEFAULT_ALPHA = 0.05`
  - `EquivalenceGate(threshold_db=-0.5, looks=DEFAULT_LOOKS, alpha=0.05)` with `.confidence_per_look -> float`, `.cap -> int`, `.next_look(n: int) -> int | None`, `.evaluate(per_seed_deltas) -> dict`
  - `per_seed_verdict(deltas, threshold_db=-0.5) -> dict`
  - `seeds_needed(std_db, half_width_db, confidence) -> int`
  - `evaluate` returns keys: `rule`, `verdict`, `threshold_db`, `n`, `look_index`, `looks_taken`, `looks`, `confidence`, `estimator`, `ci_lower_db`, `ci_upper_db`, `mean_db`, `std_db`, `seeds_needed`, `definition`. Task 4 reads `verdict`, `n`, `ci_lower_db`, `ci_upper_db`, `mean_db`, `seeds_needed`.
  - `verdict` is exactly one of `"pass"`, `"fail"`, `"continue"`, `"underpowered"`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_gate.py` (and extend the import line at the top to
`from nrp.experiment_gate import EquivalenceGate, per_seed_verdict, seeds_needed, t_cdf, t_ppf  # noqa: E402`):

```python
class GateScheduleTests(unittest.TestCase):
    def test_confidence_is_bonferroni_split_across_looks(self):
        gate = EquivalenceGate()
        self.assertAlmostEqual(gate.confidence_per_look, 1.0 - 0.05 / 6, places=12)
        self.assertEqual(gate.cap, 48)

    def test_next_look_walks_the_schedule_and_ends_at_the_cap(self):
        gate = EquivalenceGate()
        self.assertEqual(gate.next_look(0), 8)
        self.assertEqual(gate.next_look(8), 16)
        self.assertEqual(gate.next_look(47), 48)
        self.assertIsNone(gate.next_look(48))

    def test_evaluating_off_schedule_raises_rather_than_peeking(self):
        gate = EquivalenceGate()
        with self.assertRaises(ValueError):
            gate.evaluate([0.0] * 9)

    def test_empty_and_single_seed_inputs_raise(self):
        gate = EquivalenceGate()
        with self.assertRaises(ValueError):
            gate.evaluate([])
        with self.assertRaises(ValueError):
            gate.evaluate([0.1])

    def test_a_first_look_below_two_seeds_raises(self):
        """Two seeds is the floor for a variance estimate, so the schedule must respect it."""
        with self.assertRaises(ValueError):
            EquivalenceGate(looks=(1, 8))

    def test_non_monotonic_or_duplicate_looks_raise(self):
        with self.assertRaises(ValueError):
            EquivalenceGate(looks=(16, 8))
        with self.assertRaises(ValueError):
            EquivalenceGate(looks=(8, 8))
        with self.assertRaises(ValueError):
            EquivalenceGate(looks=())


class GateVerdictTests(unittest.TestCase):
    def test_tight_interval_above_threshold_passes(self):
        gate = EquivalenceGate()
        deltas = [0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.02, 0.01]
        result = gate.evaluate(deltas)
        self.assertEqual(result["verdict"], "pass")
        self.assertGreaterEqual(result["ci_lower_db"], -0.5)
        self.assertEqual(result["estimator"], "student_t")
        self.assertEqual(result["n"], 8)

    def test_interval_entirely_below_threshold_fails(self):
        gate = EquivalenceGate()
        deltas = [-3.0, -3.1, -2.9, -3.0, -3.2, -2.8, -3.1, -3.0]
        result = gate.evaluate(deltas)
        self.assertEqual(result["verdict"], "fail")
        self.assertLess(result["ci_upper_db"], -0.5)

    def test_straddling_interval_continues_before_the_cap(self):
        gate = EquivalenceGate()
        deltas = [1.5, -2.0, 0.5, -1.5, 2.0, -1.0, 0.0, -0.5]
        result = gate.evaluate(deltas)
        self.assertEqual(result["verdict"], "continue")

    def test_straddling_interval_at_the_cap_is_underpowered(self):
        gate = EquivalenceGate()
        rng = np.random.default_rng(11)
        deltas = list(rng.normal(0.0, 2.0, 48))
        result = gate.evaluate(deltas)
        self.assertEqual(result["verdict"], "underpowered")
        self.assertGreater(result["seeds_needed"], 48)

    def test_boundary_lower_bound_exactly_at_threshold_passes(self):
        """A gate that rejects its own boundary silently moves the threshold."""
        gate = EquivalenceGate(threshold_db=-0.5, looks=(8,))
        deltas = [0.0] * 8
        result = gate.evaluate(deltas)
        # Zero spread puts the interval at exactly the mean, above -0.5.
        self.assertEqual(result["verdict"], "pass")
        self.assertEqual(result["ci_lower_db"], 0.0)

    def test_more_seeds_narrow_the_interval_at_equal_mean_and_spread(self):
        gate = EquivalenceGate()
        base = [0.5, -0.5] * 4
        narrow = gate.evaluate(base * 6)
        wide = gate.evaluate(base)
        self.assertAlmostEqual(narrow["mean_db"], wide["mean_db"], places=12)
        span_narrow = narrow["ci_upper_db"] - narrow["ci_lower_db"]
        span_wide = wide["ci_upper_db"] - wide["ci_lower_db"]
        self.assertLess(span_narrow, span_wide)

    def test_verdict_records_the_multiplicity_correction_for_auditing(self):
        gate = EquivalenceGate()
        result = gate.evaluate([0.0, 0.1] * 4)
        self.assertEqual(result["looks_taken"], 1)
        self.assertEqual(result["look_index"], 0)
        self.assertEqual(result["looks"], [8, 16, 24, 32, 40, 48])
        self.assertAlmostEqual(result["confidence"], 1.0 - 0.05 / 6, places=12)

    def test_looks_taken_counts_every_look_up_to_this_one(self):
        gate = EquivalenceGate()
        result = gate.evaluate([0.0, 0.1] * 12)
        self.assertEqual(result["n"], 24)
        self.assertEqual(result["looks_taken"], 3)


class LegacyPerSeedVerdictTests(unittest.TestCase):
    def test_all_seeds_clearing_passes(self):
        result = per_seed_verdict([0.1, -0.2, 0.0])
        self.assertTrue(result["pass"])
        self.assertEqual(result["passing_seed_count"], 3)

    def test_one_failing_seed_fails_the_whole_arm(self):
        result = per_seed_verdict([0.1, -0.6, 0.0])
        self.assertFalse(result["pass"])

    def test_delta_exactly_at_threshold_passes(self):
        self.assertTrue(per_seed_verdict([-0.5, -0.5])["pass"])

    def test_empty_input_raises_rather_than_reporting_a_vacuous_pass(self):
        with self.assertRaises(ValueError):
            per_seed_verdict([])


class SeedsNeededTests(unittest.TestCase):
    def test_matches_the_measured_planning_numbers(self):
        confidence = 1.0 - 0.05 / 6
        self.assertEqual(seeds_needed(0.73, 0.5, confidence), 19)
        self.assertEqual(seeds_needed(1.00, 0.5, confidence), 32)
        self.assertEqual(seeds_needed(1.67, 0.5, confidence), 82)

    def test_tighter_half_width_needs_more_seeds(self):
        confidence = 1.0 - 0.05 / 6
        self.assertGreater(seeds_needed(1.0, 0.25, confidence), seeds_needed(1.0, 0.5, confidence))

    def test_zero_spread_needs_the_minimum_sample(self):
        self.assertEqual(seeds_needed(0.0, 0.5, 0.99), 2)

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            seeds_needed(-1.0, 0.5, 0.99)
        with self.assertRaises(ValueError):
            seeds_needed(1.0, 0.0, 0.99)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_experiment_gate -v`
Expected: FAIL — `ImportError: cannot import name 'EquivalenceGate' from 'nrp.experiment_gate'`

- [ ] **Step 3: Write the implementation**

Append to `nrp/experiment_gate.py`:

```python
#: The unchanged promotion threshold: a candidate arm may not be more than this
#: many dB worse than its paired control.
DEFAULT_THRESHOLD_DB = -0.5

#: Pre-registered look schedule. Evaluating anywhere else is an error, because
#: unscheduled peeking is exactly what the alpha correction below exists to price.
DEFAULT_LOOKS = (8, 16, 24, 32, 40, 48)

#: Overall false-pass budget, split evenly across the looks (Bonferroni).
DEFAULT_ALPHA = 0.05

VERDICT_PASS = "pass"
VERDICT_FAIL = "fail"
VERDICT_CONTINUE = "continue"
VERDICT_UNDERPOWERED = "underpowered"


def seeds_needed(std_db: float, half_width_db: float, confidence: float) -> int:
    """Seeds required for a t-interval of at most `half_width_db`.

    Reported with every `underpowered` verdict so a run that cannot decide states
    what deciding would cost, instead of leaving the reader to guess. At the gate's
    own confidence and a 0.5 dB half-width this returns 19 seeds at std 0.73, 32 at
    std 1.00, and 82 at std 1.67 -- the three spreads measured on Kitchen.
    """
    if std_db < 0.0:
        raise ValueError(f"std_db must be non-negative, got {std_db}")
    if half_width_db <= 0.0:
        raise ValueError(f"half_width_db must be positive, got {half_width_db}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1), got {confidence}")
    if std_db == 0.0:
        return 2
    quantile_p = 1.0 - (1.0 - confidence) / 2.0
    n = 2
    while t_ppf(quantile_p, n - 1) * std_db / math.sqrt(n) > half_width_db:
        n += 1
        if n > 100000:  # pragma: no cover - unreachable for any realistic spread
            raise ValueError("seeds_needed did not converge; check std_db/half_width_db")
    return n


def per_seed_verdict(deltas, threshold_db: float = DEFAULT_THRESHOLD_DB) -> dict:
    """The legacy rule: every seed's delta must clear the threshold.

    Kept so reports can carry both verdicts during and after the migration, NOT
    because it is a defensible gate -- see this module's docstring for the
    false-rejection rates that motivated replacing it.
    """
    values = [float(d) for d in deltas]
    if not values:
        raise ValueError("cannot compute a verdict with zero seeds")
    per_seed_pass = [bool(value >= threshold_db) for value in values]
    return {
        "rule": "per_seed",
        "threshold_db": threshold_db,
        "seed_count": len(values),
        "passing_seed_count": int(sum(per_seed_pass)),
        "per_seed_pass": per_seed_pass,
        "per_seed_delta_db": values,
        "pass": bool(all(per_seed_pass)),
        "definition": (
            "every seed's paired mean PSNR delta versus the same-run control must be "
            "at least the threshold (the original R1 gate; underpowered, retained for "
            "comparison only)"
        ),
    }


class EquivalenceGate:
    """Three-outcome equivalence gate over per-seed paired deltas.

    Passes when a confidence interval for the mean delta sits entirely above the
    threshold, fails when it sits entirely below, and otherwise reports that the
    experiment has not answered its question -- `continue` if seeds remain,
    `underpowered` at the cap. Unlike the per-seed rule it replaces, adding seeds
    tightens the interval and so makes both promotion and rejection harder to reach
    by luck.
    """

    def __init__(
        self,
        threshold_db: float = DEFAULT_THRESHOLD_DB,
        looks: tuple[int, ...] = DEFAULT_LOOKS,
        alpha: float = DEFAULT_ALPHA,
    ):
        looks = tuple(int(look) for look in looks)
        if not looks:
            raise ValueError("looks must contain at least one milestone")
        if any(b <= a for a, b in zip(looks[:-1], looks[1:], strict=True)):
            raise ValueError(f"looks must be strictly increasing, got {looks}")
        if looks[0] < 2:
            raise ValueError("the first look needs at least 2 seeds for a variance estimate")
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha}")
        self.threshold_db = float(threshold_db)
        self.looks = looks
        self.alpha = float(alpha)

    @property
    def confidence_per_look(self) -> float:
        """Bonferroni-corrected per-look confidence: the whole point of fixed looks."""
        return 1.0 - self.alpha / len(self.looks)

    @property
    def cap(self) -> int:
        return self.looks[-1]

    def next_look(self, n: int) -> int | None:
        """The next scheduled milestone strictly above `n`, or None past the cap."""
        for look in self.looks:
            if look > n:
                return look
        return None

    def evaluate(self, per_seed_deltas) -> dict:
        """Verdict at n = len(per_seed_deltas); raises unless n is a scheduled look."""
        values = np.asarray([float(d) for d in per_seed_deltas], dtype=np.float64)
        n = int(values.size)
        if n == 0:
            raise ValueError("cannot evaluate a gate with zero seeds")
        if n not in self.looks:
            raise ValueError(
                f"n={n} is not a scheduled look {self.looks}; evaluating off schedule "
                "is unscheduled peeking, which the alpha correction does not cover"
            )
        if n < 2:
            raise ValueError(f"need at least 2 seeds for a variance estimate, got {n}")

        look_index = self.looks.index(n)
        confidence = self.confidence_per_look
        mean = float(values.mean())
        std = float(values.std(ddof=1))
        half_width = t_ppf(1.0 - (1.0 - confidence) / 2.0, n - 1) * std / math.sqrt(n)
        lower, upper = mean - half_width, mean + half_width

        if lower >= self.threshold_db:
            verdict = VERDICT_PASS
        elif upper < self.threshold_db:
            verdict = VERDICT_FAIL
        elif n >= self.cap:
            verdict = VERDICT_UNDERPOWERED
        else:
            verdict = VERDICT_CONTINUE

        return {
            "rule": "equivalence",
            "verdict": verdict,
            "threshold_db": self.threshold_db,
            "n": n,
            "look_index": look_index,
            "looks_taken": look_index + 1,
            "looks": list(self.looks),
            "confidence": confidence,
            "alpha_overall": self.alpha,
            "estimator": "student_t",
            "ci_lower_db": float(lower),
            "ci_upper_db": float(upper),
            "mean_db": mean,
            "std_db": std,
            "seeds_needed": seeds_needed(std, abs(self.threshold_db), confidence),
            "definition": (
                "pass when the confidence interval for the mean paired delta lies "
                "entirely at or above the threshold, fail when it lies entirely below, "
                "underpowered when it straddles the threshold at the seed cap; "
                "underpowered is never a pass"
            ),
        }
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run python -m unittest tests.test_experiment_gate -v`
Expected: PASS, 24 tests.

Lint: `uv run ruff format nrp/experiment_gate.py tests/test_experiment_gate.py && uv run ruff check nrp tests`
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add nrp/experiment_gate.py tests/test_experiment_gate.py
git commit -m "feat: three-outcome equivalence gate over per-seed deltas

Passes when the CI for the mean paired delta clears -0.5 dB, fails when it
sits entirely below, and reports continue/underpowered rather than forcing a
verdict it cannot support. Fixed Bonferroni-corrected looks make adaptive
stopping honest; off-schedule evaluation raises rather than peeking.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01144Jx91dfeyZW5DDgVqfWZ"
```

---

### Task 3: Power simulations — enforce the property that motivated the gate

**Files:**
- Test: `tests/test_experiment_gate.py` (append)

**Interfaces:**
- Consumes: `EquivalenceGate`, `per_seed_verdict` from Task 2.
- Produces: no new production API. A test-local helper
  `simulate_experiment(gate, rng, true_mean, std) -> str` returning the final verdict.

These thresholds are set below measured prototype values (4,000 experiments per
cell) so they assert real behavior with margin, not aspiration:
at parity std 0.73 → 0.977 pass; std 1.00 → 0.798; std 1.67 → 0.341 pass /
0.659 underpowered; 1.5 dB worse → 0.000 pass at every spread.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_experiment_gate.py`:

```python
def simulate_experiment(gate, rng, true_mean, std):
    """Run one adaptive experiment to a verdict, mimicking a runner's look loop."""
    deltas = []
    for look in gate.looks:
        deltas.extend(rng.normal(true_mean, std, look - len(deltas)))
        result = gate.evaluate(deltas)
        if result["verdict"] in ("pass", "fail"):
            return result["verdict"]
    return "underpowered"


def verdict_rates(gate, true_mean, std, trials, seed):
    rng = np.random.default_rng(seed)
    verdicts = [simulate_experiment(gate, rng, true_mean, std) for _ in range(trials)]
    return {
        "pass": verdicts.count("pass") / trials,
        "fail": verdicts.count("fail") / trials,
        "underpowered": verdicts.count("underpowered") / trials,
    }


class GatePowerTests(unittest.TestCase):
    """The gate exists for these numbers; they are asserted, not argued.

    Thresholds sit below the measured prototype rates (4,000 experiments per cell)
    so ordinary Monte Carlo wobble at 600 trials cannot flip them.
    """

    TRIALS = 600

    def test_at_parity_arm_with_tight_spread_is_promoted(self):
        rates = verdict_rates(EquivalenceGate(), 0.0, 0.73, self.TRIALS, seed=101)
        self.assertGreaterEqual(rates["pass"], 0.95)

    def test_at_parity_arm_with_medium_spread_is_usually_promoted(self):
        rates = verdict_rates(EquivalenceGate(), 0.0, 1.0, self.TRIALS, seed=102)
        # Measured 0.798 over 4,000 experiments; 0.70 clears 600-trial Monte Carlo
        # wobble (se ~0.016) with room to spare.
        self.assertGreaterEqual(rates["pass"], 0.70)

    def test_clearly_worse_arm_is_essentially_never_promoted(self):
        for index, std in enumerate((0.73, 1.0, 1.67)):
            with self.subTest(std=std):
                rates = verdict_rates(EquivalenceGate(), -1.5, std, self.TRIALS, seed=200 + index)
                self.assertLessEqual(rates["pass"], 0.01)
                # Measured 1.000 / 1.000 / 0.927 at std 0.73 / 1.00 / 1.67.
                self.assertGreaterEqual(rates["fail"], 0.85)

    def test_gate_refuses_to_certify_a_spread_the_cap_cannot_resolve(self):
        """world3d's spread needs ~82 seeds; at cap 48 the honest answer is 'unknown'."""
        rates = verdict_rates(EquivalenceGate(), 0.0, 1.67, self.TRIALS, seed=103)
        self.assertGreaterEqual(rates["underpowered"], 0.50)
        self.assertEqual(rates["fail"], 0.0)

    def test_legacy_per_seed_rule_rejects_at_parity_arms(self):
        """Regression test on the diagnosis: the old rule fails a perfect arm."""
        rng = np.random.default_rng(104)
        passes = 0
        for _ in range(self.TRIALS):
            deltas = rng.normal(0.0, 1.0, 5)
            if per_seed_verdict(deltas)["pass"]:
                passes += 1
        self.assertLessEqual(passes / self.TRIALS, 0.30)

    def test_legacy_rule_gets_worse_with_more_seeds_and_the_gate_does_not(self):
        """The structural defect: the old rule punishes sample size."""
        rng = np.random.default_rng(105)
        rates = {}
        for n in (5, 10):
            passes = sum(
                1 for _ in range(self.TRIALS) if per_seed_verdict(rng.normal(0.0, 1.0, n))["pass"]
            )
            rates[n] = passes / self.TRIALS
        self.assertLess(rates[10], rates[5])

        gate_small = verdict_rates(EquivalenceGate(looks=(8,)), 0.0, 0.73, self.TRIALS, seed=106)
        gate_large = verdict_rates(
            EquivalenceGate(looks=(8, 16, 24, 32, 40, 48)), 0.0, 0.73, self.TRIALS, seed=106
        )
        self.assertGreaterEqual(gate_large["pass"], gate_small["pass"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_experiment_gate.GatePowerTests -v`
Expected: FAIL — `NameError: name 'simulate_experiment' is not defined` if the helper
was not appended, otherwise these pass immediately because Task 2 already implements
the behavior. Both outcomes are acceptable here: this task adds guarantees over
existing code rather than driving new code, so confirm the assertions actually
exercise the gate by temporarily changing `VERDICT_UNDERPOWERED` handling to return
`"pass"` in `nrp/experiment_gate.py`, re-running (expect
`test_gate_refuses_to_certify_a_spread_the_cap_cannot_resolve` to FAIL), then
reverting.

- [ ] **Step 3: Verify the simulation runs in reasonable time**

Run: `time uv run python -m unittest tests.test_experiment_gate.GatePowerTests`
Expected: PASS in under 60 seconds. If slower, reduce `TRIALS` to 400 and re-check
that every assertion still holds with margin.

- [ ] **Step 4: Run the whole suite**

Run: `uv run python -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK` with the pre-existing skip count (4).

- [ ] **Step 5: Commit**

```bash
git add tests/test_experiment_gate.py
git commit -m "test: assert the equivalence gate's power properties by simulation

Thresholds come from a 4,000-experiment prototype: an at-parity arm is
promoted 0.95+ of the time at std 0.73 and 0.75+ at std 1.00, a 1.5 dB-worse
arm essentially never, and a std-1.67 arm comes back underpowered rather than
certified because 48 seeds cannot resolve it. Also pins the diagnosis that
motivated the gate: the legacy per-seed rule rejects a perfect arm and gets
worse as seeds are added.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01144Jx91dfeyZW5DDgVqfWZ"
```

---

### Task 4: Wire the gate into `examples/r1_parity.py`

**Files:**
- Modify: `examples/r1_parity.py`
- Test: `tests/test_r1_parity.py` (append)

**Interfaces:**
- Consumes: `EquivalenceGate`, `per_seed_verdict` from Task 2.
- Produces:
  - `plan_seed_batches(gate, max_seeds) -> list[tuple[int, ...]]` — seed batches per look
  - `arm_gate_verdict(per_seed_deltas, seeds, gate=None) -> dict` — existing name, now
    returning `{"equivalence": {...}, "per_seed": {...}, "binding": "equivalence"|"per_seed", "pass": bool}`
  - report gains top-level `gate_rule` (`"equivalence"` or `"per_seed"`)

Existing behavior that must not change: `build_arm_models`, `make_arm_config`,
`evaluate_model`, the cache/validation plumbing, and the report's existing keys.
`any_world_arm_passes` keeps its signature and reads the new `pass` field.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_r1_parity.py`:

```python
class SeedPlanningTests(unittest.TestCase):
    def test_batches_follow_the_look_schedule_and_respect_max_seeds(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        batches = runner.plan_seed_batches(EquivalenceGate(), max_seeds=24)
        self.assertEqual([len(b) for b in batches], [8, 8, 8])
        self.assertEqual(batches[0], tuple(range(8)))
        self.assertEqual(batches[2], tuple(range(16, 24)))

    def test_max_seeds_below_the_first_look_raises(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        with self.assertRaises(ValueError):
            runner.plan_seed_batches(EquivalenceGate(), max_seeds=4)


class DualVerdictTests(unittest.TestCase):
    def test_verdict_carries_both_rules_with_equivalence_binding(self):
        runner = load_runner()
        deltas = [0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.02, 0.01]
        verdict = runner.arm_gate_verdict(deltas, tuple(range(8)))
        self.assertEqual(verdict["binding"], "equivalence")
        self.assertEqual(verdict["equivalence"]["verdict"], "pass")
        self.assertIn("per_seed", verdict)
        self.assertTrue(verdict["pass"])

    def test_underpowered_is_not_a_pass(self):
        runner = load_runner()
        rng = np.random.default_rng(11)
        deltas = list(rng.normal(0.0, 2.0, 48))
        verdict = runner.arm_gate_verdict(deltas, tuple(range(48)))
        self.assertEqual(verdict["equivalence"]["verdict"], "underpowered")
        self.assertFalse(verdict["pass"])

    def test_per_seed_rule_can_be_selected_as_binding(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        deltas = [0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.02, 0.01]
        verdict = runner.arm_gate_verdict(
            deltas, tuple(range(8)), gate=EquivalenceGate(), binding="per_seed"
        )
        self.assertEqual(verdict["binding"], "per_seed")
        self.assertTrue(verdict["pass"])

    def test_a_report_records_which_rule_was_binding(self):
        runner = load_runner()
        deltas = [0.02, -0.01, 0.03, 0.00, 0.01, -0.02, 0.02, 0.01]
        gates = {"world_sparse": runner.arm_gate_verdict(deltas, tuple(range(8)))}
        report = runner.build_report(
            seeds=tuple(range(8)),
            training_arms=[],
            world_gates=gates,
            hardware={"device": "cpu"},
            extra={"gate_rule": "equivalence"},
        )
        self.assertEqual(report["gate_rule"], "equivalence")
        self.assertTrue(report["any_world_arm_pass"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run python -m unittest tests.test_r1_parity -v`
Expected: FAIL — `AttributeError: module 'r1_parity' has no attribute 'plan_seed_batches'`

- [ ] **Step 3: Write the implementation**

In `examples/r1_parity.py`, add to the imports (after the existing
`from nrp.path_cache import PathCache` line):

```python
from nrp.experiment_gate import EquivalenceGate, per_seed_verdict  # noqa: E402
```

Add `plan_seed_batches` next to the other module-level helpers:

```python
def plan_seed_batches(gate: EquivalenceGate, max_seeds: int) -> list[tuple[int, ...]]:
    """Seed batches, one per scheduled look, in fixed ascending order.

    Ascending order matters: it makes an n=16 run a strict prefix of an n=48 run, so
    a longer run reuses a shorter one's trainings and two runs at different caps stay
    comparable.
    """
    if max_seeds < gate.looks[0]:
        raise ValueError(
            f"max_seeds={max_seeds} is below the first look ({gate.looks[0]}); the gate "
            "cannot reach a verdict"
        )
    batches = []
    previous = 0
    for look in gate.looks:
        if look > max_seeds:
            break
        batches.append(tuple(range(previous, look)))
        previous = look
    return batches
```

Replace the existing `arm_gate_verdict` function body with the dual-verdict version
(keep the function name — `run_experiment` already calls it):

```python
def arm_gate_verdict(
    per_seed_deltas: list[float] | np.ndarray,
    seeds: tuple[int, ...],
    gate: EquivalenceGate | None = None,
    binding: str = "equivalence",
) -> dict:
    """Both gate verdicts for one arm, with `binding` naming the decisive one.

    Reporting both is nearly free and lets any report be read against either
    convention; naming the binding rule explicitly keeps a reader from having to
    infer which number decided the promotion.
    """
    if len(seeds) == 0:
        raise ValueError("cannot compute a gate verdict with zero seeds")
    deltas = [float(d) for d in per_seed_deltas]
    if len(deltas) != len(seeds):
        raise ValueError(
            f"per_seed_deltas has {len(deltas)} entries, expected one per seed ({len(seeds)})"
        )
    if binding not in ("equivalence", "per_seed"):
        raise ValueError(f"unknown binding rule {binding!r}")
    gate = gate or EquivalenceGate()
    equivalence = gate.evaluate(deltas)
    legacy = per_seed_verdict(deltas, threshold_db=gate.threshold_db)
    decisive = equivalence["verdict"] == "pass" if binding == "equivalence" else legacy["pass"]
    return {
        "binding": binding,
        "equivalence": equivalence,
        "per_seed": legacy,
        "pass": bool(decisive),
    }
```

Update `any_world_arm_passes` to read the new field — it currently reads
`gate["pass"]`, which the dict above still provides, so **no change is needed**;
confirm by reading the function.

Replace `run_experiment` entirely with the look-chunked version below. The paired
delta computation is lifted into `seed_mean_delta` so the look loop and the final
summary share one definition instead of two that can drift:

```python
def seed_mean_delta(control_metrics: list[dict], candidate_metrics: list[dict]) -> float:
    """Mean paired PSNR delta over the validation lights, for one arm and one seed."""
    per_light = pair_validation_metrics(control_metrics, candidate_metrics)
    return float(np.mean([row["delta_db"] for row in per_light]))


def run_experiment(
    base_cfg: dict,
    cache: PathCache,
    *,
    root: Path,
    out_root: Path,
    seeds: tuple[int, ...] | None = None,
    resamples: int,
    bootstrap_seed: int,
    arm_models: dict[str, dict] = ARM_MODELS,
    gate: EquivalenceGate | None = None,
    binding: str = "equivalence",
    max_seeds: int | None = None,
) -> dict:
    """Train every arm over the gate's look schedule, stopping at the first verdict.

    `seeds` forces an explicit seed list (the per-seed rule's mode, and how a caller
    reproduces a historical run); otherwise seeds come from the look schedule. Early
    stopping only ever happens AT a look, which is what keeps the alpha correction
    honest.
    """
    gate = gate or EquivalenceGate()
    if seeds is not None:
        seed_batches = [tuple(seeds)]
    else:
        seed_batches = plan_seed_batches(gate, max_seeds or gate.cap)

    validation_sets: dict[int, list[dict]] = {}
    validation_specs: dict[str, list[dict]] = {}
    metrics_by_arm: dict[tuple[str, int], list[dict]] = {}
    training_arms: list[dict] = []
    trained_seeds: list[int] = []

    for batch in seed_batches:
        batch_sets, batch_specs = build_frozen_validation_sets(cache, base_cfg, batch)
        validation_sets.update(batch_sets)
        validation_specs.update(batch_specs)

        for seed in batch:
            for arm in ARMS:
                arm_dir = out_root / "train" / arm / f"seed{seed}"
                arm_cfg = make_arm_config(base_cfg, arm, seed, arm_dir, arm_models=arm_models)
                train_report = train(arm_cfg)
                model_path = arm_dir / "model.pt"
                model = load_trained_model(str(model_path), cache)
                metrics = evaluate_model(model, cache, validation_sets[seed])
                metrics_by_arm[(arm, seed)] = metrics
                capacity_report = model.encoding.capacity_report() if model.encoding else None
                psnrs = np.asarray([row["psnr_db_vs_raw"] for row in metrics], dtype=np.float64)
                training_arms.append(
                    {
                        "arm": arm,
                        "seed": seed,
                        "parameter_count": int(train_report["parameter_count"]),
                        "capacity_report": capacity_report,
                        "iters_per_second": train_report["iters_per_second"],
                        "train_seconds": train_report["train_seconds"],
                        "report": _relative_path(arm_dir / "torch_train_report.json", root),
                        "validation": metrics,
                        "validation_psnr_db_mean": float(psnrs.mean()),
                        "validation_psnr_db_std": float(psnrs.std()),
                    }
                )
                print(
                    f"{arm} seed {seed}: {psnrs.mean():.2f} dB, "
                    f"{train_report['parameter_count']} params"
                )
                del model
            trained_seeds.append(seed)

        # Look boundary: stop as soon as every world arm has a terminal verdict.
        if binding == "equivalence" and len(trained_seeds) in gate.looks:
            verdicts = []
            for arm in WORLD_ARMS:
                deltas = [
                    seed_mean_delta(metrics_by_arm[(CONTROL_ARM, seed)], metrics_by_arm[(arm, seed)])
                    for seed in trained_seeds
                ]
                verdicts.append(gate.evaluate(deltas)["verdict"])
            if all(verdict in ("pass", "fail") for verdict in verdicts):
                break

    seeds_run = tuple(trained_seeds)
    world_gates = {}
    per_arm_comparisons = {}
    for arm_index, arm in enumerate(WORLD_ARMS):
        per_seed = []
        seed_mean_deltas = []
        for seed_index, seed in enumerate(seeds_run):
            control = metrics_by_arm[(CONTROL_ARM, seed)]
            candidate = metrics_by_arm[(arm, seed)]
            per_light = pair_validation_metrics(control, candidate)
            deltas = np.asarray([row["delta_db"] for row in per_light], dtype=np.float64)
            per_seed.append(
                {
                    "seed": seed,
                    "per_light_deltas": per_light,
                    # Bootstrap stays in the report as a DESCRIPTIVE statistic; the
                    # gate's Student-t interval is what binds (see the module docstring).
                    "summary": summarize_values(
                        deltas,
                        resamples=resamples,
                        bootstrap_seed=bootstrap_seed + arm_index * 100 + seed_index,
                    ),
                }
            )
            seed_mean_deltas.append(float(deltas.mean()))
        verdict = arm_gate_verdict(seed_mean_deltas, seeds_run, gate=gate, binding=binding)
        verdict["across_seed_summary"] = summarize_values(
            np.asarray(seed_mean_deltas, dtype=np.float64),
            resamples=resamples,
            bootstrap_seed=bootstrap_seed + 1000 + arm_index,
        )
        world_gates[arm] = verdict
        per_arm_comparisons[arm] = per_seed

    return {
        "seeds_run": list(seeds_run),
        "validation_specs": validation_specs,
        "validation_fingerprints": {
            str(seed): validation_fingerprint(validation_specs[str(seed)]) for seed in seeds_run
        },
        "training_arms": training_arms,
        "world_gates": world_gates,
        "comparisons": per_arm_comparisons,
    }
```

Note what this preserves: the bootstrap `summarize_values` calls stay exactly where
they were, so every report keeps its existing descriptive statistics. Only the
binding verdict changes.

Add the CLI flags in `main`:

```python
    parser.add_argument(
        "--gate",
        default="equivalence",
        choices=["equivalence", "per-seed"],
        help="Which rule is binding for promotion (default: equivalence). Both verdicts "
        "are always recorded in the report.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=EquivalenceGate().cap,
        help="Seed cap for the adaptive look schedule (default: 48).",
    )
```

`--seeds` keeps its meaning but changes its default: it is now `None`, and passing
it explicitly forces that exact seed list (how a caller reproduces a historical
five-seed run, and the only mode `--gate per-seed` uses). When it is omitted under
`--gate equivalence`, seeds come from the look schedule. In `main`, pass
`seeds=tuple(args.seeds) if args.seeds else None`, `max_seeds=args.max_seeds`,
`gate=EquivalenceGate()`, and `binding=args.gate.replace("-", "_")` into
`run_experiment`, and take the seed list for the report from the returned
`result["seeds_run"]` rather than from `args`. Record `gate_rule` in the report's
`extra` dict:

```python
            "gate_rule": args.gate.replace("-", "_"),
            "gate_schedule": {
                "looks": list(gate.looks),
                "cap": gate.cap,
                "alpha_overall": gate.alpha,
                "confidence_per_look": gate.confidence_per_look,
            },
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_r1_parity -v`
Expected: PASS, including the pre-existing tests (the old `arm_gate_verdict` tests
that assert `result["pass"]` and `result["per_seed_delta_db"]` need updating to read
`result["per_seed"]["per_seed_delta_db"]` — do that as part of this step and note it
in the commit).

Then the whole suite: `uv run python -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK`.

Lint: `uv run ruff format examples/r1_parity.py tests/test_r1_parity.py && uv run ruff check .`
Expected: `All checks passed!`

- [ ] **Step 5: Smoke-test the runner end to end**

Run a tiny real run to prove the wiring works (this trains, so it takes a few minutes):

```bash
UV_CACHE_DIR=.uv-cache uv run python examples/r1_parity.py \
  --cache out/r1-encoding-redesign/seed0/train0.npz \
  --out-dir out/gate-smoke --gate equivalence --max-seeds 8 --iters 50
```

Expected: completes and writes `out/gate-smoke/report.json` containing
`gate_rule: "equivalence"`, a `gate_schedule` block, and per-arm entries with both
`equivalence` and `per_seed` verdicts. Verify with:

```bash
python3 -c "
import json; d=json.load(open('out/gate-smoke/report.json'))
print(d['gate_rule'], d['gate_schedule']['looks'])
for arm, g in d['gate'].items():
    print(arm, g['binding'], g['equivalence']['verdict'], g['equivalence']['n'], g['pass'])
"
```

- [ ] **Step 6: Commit**

```bash
git add examples/r1_parity.py tests/test_r1_parity.py
git commit -m "feat: r1_parity adopts the equivalence gate with adaptive looks

Trains seeds in look-sized batches and stops as soon as every world arm
reaches a terminal verdict, so a clearly-worse arm costs 8-24 seeds instead of
the full cap. Both verdicts land in every report with gate_rule naming the
binding one; --gate per-seed reproduces the old behavior for comparison.
underpowered never counts as a pass.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01144Jx91dfeyZW5DDgVqfWZ"
```

---

### Task 5: Document the gate

**Files:**
- Modify: `docs/performance.md` (append a section)
- Modify: `docs/representation-track.md` (R1 status row)

**Interfaces:**
- Consumes: the measured power table from Task 3's simulation.
- Produces: no code.

- [ ] **Step 1: Append the section to `docs/performance.md`**

```markdown
## The equivalence gate (from 2026-08-28)

Promotion decisions on this track previously required every seed's paired PSNR
delta to clear −0.5 dB. At the per-seed spreads measured on Country Kitchen under
the deterministic denoiser, that rule rejects an arm sitting *exactly at parity*
76–91% of the time, and its false-rejection rate rises with the seed count — a
true-parity arm passes under 6% of the time at ten seeds. It punished sample size,
so it could not be repaired by running more seeds.

`nrp/experiment_gate.py` replaces it. The threshold is unchanged at −0.5 dB; the
rule's structure is what changed:

| verdict | condition |
|---|---|
| `pass` | CI lower bound ≥ −0.5 dB |
| `fail` | CI upper bound < −0.5 dB |
| `underpowered` | interval straddles −0.5 dB at the 48-seed cap |

`underpowered` is never a pass. The interval is a Student-t interval (binding);
the percentile bootstrap is still reported but is descriptive only, because its
coverage at n=8 and 99.17% confidence is unreliable. Seeds accumulate to six
pre-registered looks (n = 8, 16, 24, 32, 40, 48) with α = 0.05 split six ways, so
adaptive stopping cannot inflate the false-pass rate; evaluating off schedule
raises.

Measured behavior (4,000 simulated experiments per cell, cap 48):

| true mean | std | pass | fail | underpowered | median n |
|---|---:|---:|---:|---:|---:|
| 0.0 (at parity) | 0.73 | 0.977 | 0.000 | 0.023 | 24 |
| 0.0 | 1.00 | 0.798 | 0.000 | 0.202 | 32 |
| 0.0 | 1.67 | 0.341 | 0.000 | 0.659 | 48 |
| −1.5 (clearly worse) | 0.73 / 1.00 / 1.67 | 0.000 | 1.00 / 1.00 / 0.93 | ≤0.07 | 8–24 |

Certifying parity at `world3d`'s spread (1.67) needs ~82 seeds; the cap stays at
48, and every `underpowered` verdict reports the `seeds_needed` figure so the cost
of a definitive answer is stated rather than guessed. Failure detection is
unaffected by the cap — a clearly worse arm is caught at every spread, usually by
seed 24.

Not corrected: multiplicity across arms. Each arm is one pre-registered question
against a fixed control; a future "promote whichever of N arms passes" selection
would need its own correction.

Verdicts recorded before 2026-08-28 used the per-seed rule and are labelled as
such in their reports.
```

- [ ] **Step 2: Update the R1 row in `docs/representation-track.md`**

Append to the R1 status cell, before the closing `**`:

```
. Promotion on this track uses the equivalence gate from 2026-08-28 (`nrp/experiment_gate.py`, `docs/performance.md#the-equivalence-gate-from-2026-08-28`); every verdict recorded before that date used the per-seed rule, which rejects an at-parity arm 76-91% of the time
```

- [ ] **Step 3: Verify the docs render and links resolve**

Run:

```bash
grep -n "equivalence-gate-from-2026-08-28" docs/performance.md docs/representation-track.md
```

Expected: the anchor appears in `performance.md` as a heading and in
`representation-track.md` as a link target.

- [ ] **Step 4: Run the full suite one final time**

Run: `uv run python -m unittest discover -s tests 2>&1 | tail -3`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add docs/performance.md docs/representation-track.md
git commit -m "docs: define the equivalence gate and the rule it replaces

Records the measured false-rejection rates of the per-seed rule (76-91% for an
arm at parity, worse as seeds are added), the new three-outcome rule, its
Bonferroni-corrected look schedule, and its simulated power, including the
regime the 48-seed cap cannot certify.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01144Jx91dfeyZW5DDgVqfWZ"
```

---

## Out of Scope

Stated so an implementer does not drift into them:

- **Re-running any experiment under the new gate.** A Kitchen sweep at cap 48 costs
  up to ~8 h. Separate decision.
- **Changing the −0.5 dB threshold.**
- **Migrating `r1a_variance.py`, `r1_failure_analysis.py`, or `r1_kitchen_k1.py`.**
  Their experiments are closed; re-labelling finished results with a gate they were
  never measured under would misrepresent them.
- **Cross-arm multiplicity correction.**
- **Editing historical reports** in `out/`.
