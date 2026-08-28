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
