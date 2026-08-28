"""Tests for the equivalence gate (nrp/experiment_gate.py).

The gate this module implements replaces a rule that rejected an arm sitting
exactly at parity 76-91% of the time and got worse as seeds were added (spec:
docs/superpowers/specs/2026-08-28-equivalence-gate-design.md). Its power
properties are therefore not incidental -- they are the reason it exists, so
they are asserted by simulation here rather than argued for in a docstring.
"""

import math
import sys
import unittest
from pathlib import Path

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

    def test_extreme_quantile_matches_cauchy_closed_form(self):
        # df=1 is the standard Cauchy distribution, whose quantile function has
        # a closed form independent of this module's bisection: tan(pi * (p - 0.5)).
        # The old hard-coded [-1e3, 1e3] bracket silently clamps p=0.999999 to
        # 1000.0 instead of the true ~318310, so this pins against an external
        # oracle rather than the implementation under test.
        for p in (0.9, 0.999999):
            expected = math.tan(math.pi * (p - 0.5))
            self.assertAlmostEqual(t_ppf(p, 1), expected, places=3)

    def test_ppf_symmetric_at_extreme_quantile(self):
        q = 1e-6
        for df in (1, 5, 47):
            high, low = t_ppf(1.0 - q, df), t_ppf(q, df)
            self.assertAlmostEqual(high, -low, delta=1e-6 * abs(high))

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            t_ppf(0.0, 10)
        with self.assertRaises(ValueError):
            t_ppf(1.0, 10)
        with self.assertRaises(ValueError):
            t_ppf(0.975, 0)


if __name__ == "__main__":
    unittest.main()
