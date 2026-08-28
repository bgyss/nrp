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
    low, high = -1.0, 1.0
    expansions = 0
    max_expansions = 200
    while t_cdf(low, df) >= p:
        low *= 2.0
        expansions += 1
        if expansions > max_expansions:
            raise ValueError(f"could not bracket root for t_ppf(p={p}, df={df}): low diverged")
    expansions = 0
    while t_cdf(high, df) < p:
        high *= 2.0
        expansions += 1
        if expansions > max_expansions:
            raise ValueError(f"could not bracket root for t_ppf(p={p}, df={df}): high diverged")
    for _ in range(300):
        mid = 0.5 * (low + high)
        if t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


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
        import numpy as np

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
