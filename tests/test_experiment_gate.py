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

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nrp.experiment_gate import (  # noqa: E402
    EquivalenceGate,
    per_seed_verdict,
    seeds_needed,
    t_cdf,
    t_ppf,
)


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

    def test_matches_brute_force_linear_scan(self):
        """Pin the bracket/binary search to the old linear scan's semantics.

        The old implementation walked n = 2, 3, 4, ... and returned the first n
        whose t-interval half-width cleared half_width_db. Reimplement that scan
        independently here (not by calling seeds_needed) and check the new
        search returns exactly the same integer for a range of spreads.
        """
        cases = [
            (0.3, 0.5),
            (0.73, 0.5),
            (1.0, 0.5),
            (1.67, 0.5),
            (4.0, 0.5),
            (1.0, 0.25),
        ]
        confidence = 1.0 - 0.05 / 6
        quantile_p = 1.0 - (1.0 - confidence) / 2.0
        for std_db, half_width_db in cases:
            with self.subTest(std_db=std_db, half_width_db=half_width_db):
                n = 2
                while t_ppf(quantile_p, n - 1) * std_db / math.sqrt(n) > half_width_db:
                    n += 1
                self.assertEqual(seeds_needed(std_db, half_width_db, confidence), n)

    def test_outlier_spread_returns_promptly_instead_of_hanging(self):
        """A single-seed outlier (measured on Kitchen: one seed at -3.8 dB and
        worse) used to make evaluate() walk a linear scan of millions of steps,
        each recomputing a bisection root, and then raise past the n > 100000
        guard instead of returning the verdict it had already decided. Both the
        hang and the spurious raise are regressions this test guards against.
        """
        import time

        deltas = [0.0] * 7 + [1000.0]
        start = time.perf_counter()
        result = EquivalenceGate().evaluate(deltas)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 5.0, "evaluate() regressed to a linear seeds_needed scan")
        self.assertEqual(result["verdict"], "continue")
        self.assertIsInstance(result["seeds_needed"], int)


if __name__ == "__main__":
    unittest.main()
