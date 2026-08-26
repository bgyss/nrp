"""Gate logic for the encoding redesign, separated from the expensive runner."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding_gates import (  # noqa: E402
    g1_generalization,
    g3_stability,
    g4_frame_robustness,
    g5_fallback_decomposition,
    stop_reason,
)


def _row(**kw):
    base = {
        "arm": "world_sparse",
        "seed": 0,
        "camera": "held0",
        "rotation_degrees": 0.0,
        "delta_db": 2.0,
        "psnr_db": 22.0,
        "baseline_psnr_db": 20.0,
        "out_of_occupancy_fraction": 0.0,
        "in_occupancy_psnr_db": 22.0,
        "out_occupancy_psnr_db": None,
    }
    base.update(kw)
    return base


class TestG1(unittest.TestCase):
    def test_passes_when_every_row_clears_the_threshold(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        gate = g1_generalization(rows)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failures"], [])

    def test_one_failing_camera_fails_the_whole_gate(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        rows[7]["delta_db"] = 0.4
        gate = g1_generalization(rows)
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 1)

    def test_a_good_mean_does_not_rescue_a_failing_seed(self):
        rows = [_row(seed=0, delta_db=-5.0), _row(seed=1, delta_db=9.0)]
        gate = g1_generalization(rows)
        self.assertFalse(gate["passed"])
        self.assertGreater(gate["mean_delta_db"], 1.0)

    def test_threshold_is_inclusive(self):
        gate = g1_generalization([_row(delta_db=1.0)], threshold_db=1.0)
        self.assertTrue(gate["passed"])

    def test_coverage_omitted_preserves_behaviour_and_reports_none(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        gate = g1_generalization(rows)
        self.assertTrue(gate["passed"])
        self.assertIsNone(gate["coverage_complete"])

    def test_coverage_supplied_and_complete_passes(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(2) for c in range(2)]
        gate = g1_generalization(rows, expected_seeds={0, 1}, expected_cameras={"held0", "held1"})
        self.assertTrue(gate["coverage_complete"])
        self.assertTrue(gate["passed"])

    def test_coverage_supplied_and_missing_a_camera_fails(self):
        # seed 1 is missing camera held1 entirely -- the deltas that do exist all
        # clear the threshold, but coverage is incomplete and must fail the gate.
        rows = [
            _row(seed=0, camera="held0"),
            _row(seed=0, camera="held1"),
            _row(seed=1, camera="held0"),
        ]
        gate = g1_generalization(rows, expected_seeds={0, 1}, expected_cameras={"held0", "held1"})
        self.assertFalse(gate["coverage_complete"])
        self.assertFalse(gate["passed"])

    def test_only_expected_cameras_supplied_is_malformed_not_a_pass(self):
        # expected_seeds omitted -- `all()` over an empty want_seeds would be
        # vacuously True. Coverage that was never actually specified for seeds
        # must never read as complete.
        rows = [_row(seed=s, camera=f"held{c}") for s in range(2) for c in range(1)]
        gate = g1_generalization(rows, expected_cameras={"held0", "held1"})
        self.assertFalse(gate["coverage_complete"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["coverage_status"], "malformed_request")

    def test_only_expected_seeds_supplied_is_malformed_not_a_pass(self):
        rows = [_row(seed=s, camera="held0") for s in range(2)]
        gate = g1_generalization(rows, expected_seeds={0, 1, 2})
        self.assertFalse(gate["coverage_complete"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["coverage_status"], "malformed_request")

    def test_expected_seeds_empty_set_with_real_cameras_is_malformed_not_a_pass(self):
        # An explicit empty set is the second vacuous-True trap: `set() or set()`
        # collapses to the same empty want_seeds as omission.
        rows = [_row(seed=0, camera=f"held{c}") for c in range(2)]
        gate = g1_generalization(rows, expected_seeds=set(), expected_cameras={"held0", "held1"})
        self.assertFalse(gate["coverage_complete"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["coverage_status"], "malformed_request")

    def test_both_supplied_nonempty_and_complete_still_passes(self):
        # Guard against over-correcting into always-False: a genuinely complete,
        # well-formed coverage request must still report True.
        rows = [_row(seed=s, camera=f"held{c}") for s in range(2) for c in range(2)]
        gate = g1_generalization(rows, expected_seeds={0, 1}, expected_cameras={"held0", "held1"})
        self.assertIs(gate["coverage_complete"], True)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["coverage_status"], "complete")

    def test_both_omitted_reports_none_and_matches_pre_fix_behaviour(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        gate = g1_generalization(rows)
        self.assertIsNone(gate["coverage_complete"])
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["coverage_status"], "not_requested")

    def test_row_above_floor_and_above_delta_passes(self):
        rows = [_row(delta_db=2.0, psnr_db=25.0)]
        gate = g1_generalization(rows, absolute_floor_db=20.0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failures"], [])
        self.assertEqual(gate["absolute_floor_db"], 20.0)

    def test_row_above_delta_but_below_floor_fails_naming_the_floor(self):
        rows = [_row(delta_db=2.0, psnr_db=15.0)]
        gate = g1_generalization(rows, threshold_db=1.0, absolute_floor_db=20.0)
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 1)
        self.assertIn("below_absolute_floor", gate["failures"][0]["reasons"])
        self.assertNotIn("below_delta_threshold", gate["failures"][0]["reasons"])

    def test_row_above_floor_but_below_delta_fails_naming_the_delta(self):
        rows = [_row(delta_db=0.2, psnr_db=25.0)]
        gate = g1_generalization(rows, threshold_db=1.0, absolute_floor_db=20.0)
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 1)
        self.assertIn("below_delta_threshold", gate["failures"][0]["reasons"])
        self.assertNotIn("below_absolute_floor", gate["failures"][0]["reasons"])

    def test_absolute_floor_omitted_matches_pre_change_result(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        rows[7]["delta_db"] = 0.4
        gate = g1_generalization(rows)
        # Pre-change shape: failures entries have no "reasons" key, and passed/
        # failures/coverage fields match exactly what the prior implementation
        # produced for this input.
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 1)
        self.assertNotIn("reasons", gate["failures"][0])
        self.assertEqual(
            gate["failures"][0],
            {
                "arm": rows[7]["arm"],
                "seed": rows[7]["seed"],
                "camera": rows[7]["camera"],
                "delta_db": 0.4,
            },
        )
        self.assertIsNone(gate["absolute_floor_db"])

    def test_row_exactly_at_floor_passes_inclusive(self):
        rows = [_row(delta_db=2.0, psnr_db=20.0)]
        gate = g1_generalization(rows, absolute_floor_db=20.0)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failures"], [])

    def test_empty_rows_with_floor_supplied_is_not_a_vacuous_pass(self):
        gate = g1_generalization([], absolute_floor_db=20.0)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["absolute_floor_db"], 20.0)


class TestG3(unittest.TestCase):
    def test_reports_per_seed_pass_and_spread(self):
        rows = [_row(seed=s, delta_db=float(s)) for s in range(5)]
        gate = g3_stability(rows)
        self.assertEqual(gate["seeds_passing"], 4)  # seeds 1..4 clear 1.0 dB
        self.assertEqual(gate["seeds_total"], 5)
        self.assertIn("std_delta_db", gate)

    def test_world_sparse_measured_and_zero_collision_passes(self):
        rows = [_row(seed=0, arm="world_sparse")]
        gate = g3_stability(rows, collision_fractions={"world_sparse": 0.0})
        self.assertTrue(gate["collision_assertions_checked"])
        self.assertTrue(gate["collision_assertions_passed"])

    def test_world_sparse_measured_but_missing_from_collision_fractions_is_a_failure(self):
        # This is the case the brief's `all()`-over-filtered-dict draft got wrong:
        # world_sparse rows were produced, but no collision fraction was ever
        # recorded for it. A missing entry must NOT read as a pass.
        rows = [_row(seed=0, arm="world_sparse")]
        gate = g3_stability(rows, collision_fractions={})
        self.assertTrue(gate["collision_assertions_checked"])
        self.assertFalse(gate["collision_assertions_passed"])

    def test_world_sparse_measured_and_nonzero_collision_is_a_failure(self):
        rows = [_row(seed=0, arm="world_sparse")]
        gate = g3_stability(rows, collision_fractions={"world_sparse": 0.01})
        self.assertTrue(gate["collision_assertions_checked"])
        self.assertFalse(gate["collision_assertions_passed"])

    def test_world_sparse_not_measured_is_reported_not_applicable(self):
        rows = [_row(seed=0, arm="world_normal_triplane")]
        gate = g3_stability(rows, collision_fractions={})
        self.assertFalse(gate["collision_assertions_checked"])
        # Not-applicable must never read as a pass either.
        self.assertFalse(gate["collision_assertions_passed"])

    def test_good_deltas_but_failing_collision_assertion_is_not_a_pass(self):
        # Every seed clears the delta threshold, but the sparse arm has a genuine
        # key-collision defect. `passed` must not ignore that.
        rows = [_row(seed=s, arm="world_sparse", delta_db=5.0) for s in range(3)]
        gate = g3_stability(rows, collision_fractions={"world_sparse": 0.02})
        self.assertEqual(gate["seeds_passing"], gate["seeds_total"])
        self.assertFalse(gate["collision_assertions_passed"])
        self.assertFalse(gate["passed"])


class TestG4(unittest.TestCase):
    def test_worst_orientation_governs(self):
        rows = [
            _row(rotation_degrees=0.0, delta_db=3.0),
            _row(rotation_degrees=90.0, delta_db=3.0),
            _row(rotation_degrees=180.0, delta_db=0.2),
        ]
        gate = g4_frame_robustness(rows)
        self.assertFalse(gate["passed"])
        self.assertAlmostEqual(gate["worst_delta_db"], 0.2)

    def test_incomplete_rotation_matrix_is_not_a_pass(self):
        rows = [_row(rotation_degrees=0.0, delta_db=3.0)]
        gate = g4_frame_robustness(rows)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["coverage_complete"])

    def test_full_matrix_across_seeds_and_cameras_passes(self):
        rows = [
            _row(seed=s, camera=c, rotation_degrees=r, delta_db=3.0)
            for s in range(2)
            for c in ("held0", "held1")
            for r in (0.0, 90.0, 180.0)
        ]
        gate = g4_frame_robustness(rows)
        self.assertTrue(gate["coverage_complete"])
        self.assertTrue(gate["passed"])

    def test_pooled_rotation_set_complete_but_one_pair_missing_90_fails_coverage(self):
        # 20 rows at 0 degrees plus one row each at 90 and 180 -- the pooled set of
        # rotation values is {0, 90, 180}, but almost no seed/camera pair was ever
        # tested off-axis. Coverage must require every pair to hit every rotation.
        rows = [_row(seed=s, camera="held0", rotation_degrees=0.0, delta_db=3.0) for s in range(20)]
        rows.append(_row(seed=0, camera="held0", rotation_degrees=90.0, delta_db=3.0))
        rows.append(_row(seed=0, camera="held0", rotation_degrees=180.0, delta_db=3.0))
        gate = g4_frame_robustness(rows)
        self.assertFalse(gate["coverage_complete"])
        self.assertFalse(gate["passed"])


class TestG5(unittest.TestCase):
    def test_decomposes_error_by_occupancy(self):
        rows = [
            _row(
                out_of_occupancy_fraction=0.1, in_occupancy_psnr_db=24.0, out_occupancy_psnr_db=15.0
            )
        ]
        gate = g5_fallback_decomposition(rows)
        self.assertAlmostEqual(gate["mean_out_of_occupancy_fraction"], 0.1)
        self.assertAlmostEqual(gate["mean_in_occupancy_psnr_db"], 24.0)
        self.assertAlmostEqual(gate["mean_out_occupancy_psnr_db"], 15.0)

    def test_missing_decomposition_is_flagged(self):
        rows = [_row(out_of_occupancy_fraction=0.3, out_occupancy_psnr_db=None)]
        gate = g5_fallback_decomposition(rows)
        self.assertFalse(gate["complete"])

    def test_no_rows_is_not_vacuously_complete(self):
        gate = g5_fallback_decomposition([])
        self.assertFalse(gate["complete"])


class TestStopReason(unittest.TestCase):
    def test_no_arm_passing_g1_stops_the_track(self):
        gates = {"arms": {"world_sparse": {"g1": {"passed": False}, "g4": {"passed": False}}}}
        self.assertIsNotNone(stop_reason(gates))

    def test_a_passing_arm_clears_the_stop(self):
        gates = {
            "arms": {
                "world_sparse": {
                    "g1": {"passed": True},
                    "g3": {"passed": True},
                    "g4": {"passed": True},
                }
            }
        }
        self.assertIsNone(stop_reason(gates))

    def test_g1_and_g4_passing_but_g3_failing_does_not_clear_the_stop(self):
        gates = {
            "arms": {
                "world_sparse": {
                    "g1": {"passed": True},
                    "g3": {"passed": False},
                    "g4": {"passed": True},
                }
            }
        }
        self.assertIsNotNone(stop_reason(gates))


if __name__ == "__main__":
    unittest.main()
