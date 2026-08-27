"""Tests for the K1 sweep runner's pure helpers (examples/r1_kitchen_k1.py).

K1 tests the vertex-support hypothesis by sweeping `world_sparse`'s finest
resolution against a FIXED, pre-committed `pixel2d` control. These tests cover the
logic that protects that fixture -- control extraction, scene/ladder compatibility,
validation-light fingerprint identity -- and the directional verdict, which must
never turn a falsified prediction into a pass. Training itself is exercised by
nrp/torch_backend/train.py's tests.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "r1_kitchen_k1", ROOT / "examples" / "r1_kitchen_k1.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def control_report(seeds=(0, 1), cache="out/kitchen/path_cache.npz", base_resolution=4):
    return {
        "control_arm": "pixel2d",
        "cache": cache,
        "resolution_ladder": {"base_resolution": base_resolution, "finest_resolution": 128},
        "command": "uv run python examples/r1_parity.py ...",
        "hardware": {"cpu_brand": "Apple M1 Max"},
        "training_config": {"iters": 3000, "denoise": {"enabled": True, "method": "oidn"}},
        "training_arms": [
            {
                "arm": arm,
                "seed": seed,
                "validation": [{"light": {"radius": 0.1}, "psnr_db_vs_raw": 20.0 + seed}],
            }
            for seed in seeds
            for arm in ("pixel2d", "world_sparse")
        ],
        "validation": {"fingerprints": {str(seed): f"fp{seed}" for seed in seeds}},
    }


class ControlExtractionTests(unittest.TestCase):
    def test_extracts_only_the_control_arms_metrics(self):
        runner = load_runner()
        by_seed = runner.control_metrics_by_seed(control_report((0, 1)), (0, 1))
        self.assertEqual(sorted(by_seed), [0, 1])
        self.assertEqual(by_seed[1][0]["psnr_db_vs_raw"], 21.0)

    def test_missing_seed_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.control_metrics_by_seed(control_report((0,)), (0, 1))

    def test_duplicate_control_seed_raises(self):
        runner = load_runner()
        report = control_report((0,))
        report["training_arms"].append(dict(report["training_arms"][0]))
        with self.assertRaises(ValueError):
            runner.control_metrics_by_seed(report, (0,))

    def test_zero_seeds_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.control_metrics_by_seed(control_report((0,)), ())

    def test_wrong_control_arm_raises(self):
        runner = load_runner()
        report = control_report((0,))
        report["control_arm"] = "world3d"
        with self.assertRaises(ValueError):
            runner.control_metrics_by_seed(report, (0,))


class CompatibilityTests(unittest.TestCase):
    def test_matching_cache_and_ladder_returns_provenance(self):
        runner = load_runner()
        control = runner.check_control_compatibility(
            control_report(), cache="out/kitchen/path_cache.npz", base_resolution=4
        )
        self.assertEqual(control["finest_resolution"], 128)
        self.assertEqual(control["arm"], "pixel2d")

    def test_different_cache_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.check_control_compatibility(
                control_report(), cache="out/toy/path_cache.npz", base_resolution=4
            )

    def test_different_base_resolution_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.check_control_compatibility(
                control_report(), cache="out/kitchen/path_cache.npz", base_resolution=8
            )


class FingerprintTests(unittest.TestCase):
    def test_identical_fingerprints_pass(self):
        runner = load_runner()
        runner.check_validation_fingerprints(
            control_report((0, 1)), {"0": "fp0", "1": "fp1"}, (0, 1)
        )

    def test_mismatched_fingerprint_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.check_validation_fingerprints(
                control_report((0, 1)), {"0": "fp0", "1": "different"}, (0, 1)
            )

    def test_absent_fingerprint_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.check_validation_fingerprints(control_report((0,)), {"0": "fp0"}, (0, 1))


def resolution_row(finest, mean_db, passes=False):
    return {
        "finest_resolution": finest,
        "gate": {"pass": passes, "across_seed_summary": {"mean_db": mean_db}},
    }


class VerdictTests(unittest.TestCase):
    def test_improving_as_resolution_falls_supports_the_prediction(self):
        runner = load_runner()
        rows = [
            resolution_row(32, -0.1),
            resolution_row(64, -0.4),
            resolution_row(128, -0.9),
        ]
        verdict = runner.prediction_verdict(rows)
        self.assertLess(verdict["spearman_resolution_vs_mean_delta"], 0.0)
        self.assertTrue(verdict["direction_supports_prediction"])
        self.assertTrue(verdict["monotonic_in_predicted_direction"])
        self.assertEqual(verdict["best_resolution"], 32)

    def test_worse_as_resolution_falls_falsifies(self):
        runner = load_runner()
        rows = [
            resolution_row(32, -1.2),
            resolution_row(64, -0.7),
            resolution_row(128, -0.3),
        ]
        verdict = runner.prediction_verdict(rows)
        self.assertGreater(verdict["spearman_resolution_vs_mean_delta"], 0.0)
        self.assertFalse(verdict["direction_supports_prediction"])
        self.assertEqual(verdict["best_resolution"], 128)

    def test_flat_sweep_reports_zero_correlation_and_no_support(self):
        runner = load_runner()
        rows = [resolution_row(res, -0.8) for res in (32, 64, 128)]
        verdict = runner.prediction_verdict(rows)
        self.assertEqual(verdict["spearman_resolution_vs_mean_delta"], 0.0)
        self.assertFalse(verdict["direction_supports_prediction"])

    def test_non_monotonic_direction_is_reported_separately(self):
        runner = load_runner()
        rows = [
            resolution_row(32, -0.2),
            resolution_row(64, -0.6),
            resolution_row(96, -0.3),
            resolution_row(128, -1.0),
        ]
        verdict = runner.prediction_verdict(rows)
        self.assertLess(verdict["spearman_resolution_vs_mean_delta"], 0.0)
        self.assertFalse(verdict["monotonic_in_predicted_direction"])

    def test_single_resolution_raises_rather_than_reporting_a_direction(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.prediction_verdict([resolution_row(64, -0.1)])

    def test_gate_pass_is_reported_but_does_not_imply_support(self):
        runner = load_runner()
        rows = [resolution_row(32, -1.0), resolution_row(128, -0.1, passes=True)]
        verdict = runner.prediction_verdict(rows)
        self.assertTrue(verdict["any_resolution_passes_gate"])
        self.assertFalse(verdict["direction_supports_prediction"])


class ReportTests(unittest.TestCase):
    def test_report_records_per_resolution_gate_and_verdict(self):
        runner = load_runner()
        rows = [resolution_row(32, -0.1, passes=True), resolution_row(128, -0.9)]
        report = runner.build_report(
            seeds=(0, 1),
            resolutions=(32, 128),
            per_resolution=rows,
            control={"report": "out/r1-parity-kitchen/report.json"},
            hardware={"device": "cpu"},
        )
        self.assertEqual(report["swept_arm"], "world_sparse")
        self.assertEqual(report["control_arm"], "pixel2d")
        self.assertEqual(report["gate"], {"32": True, "128": False})
        self.assertEqual(report["verdict"]["best_resolution"], 32)
        self.assertEqual(report["control"]["report"], "out/r1-parity-kitchen/report.json")

    def test_single_resolution_report_records_that_the_prediction_is_not_evaluable(self):
        runner = load_runner()
        report = runner.build_report(
            seeds=(0,),
            resolutions=(32,),
            per_resolution=[resolution_row(32, -0.9)],
            control={"report": "out/r1-parity-kitchen/report.json"},
            hardware={"device": "cpu"},
        )
        self.assertIn("prediction_not_evaluable", report["verdict"])
        self.assertNotIn("direction_supports_prediction", report["verdict"])
        self.assertEqual(report["verdict"]["resolutions_measured"], [32])


class SpearmanTests(unittest.TestCase):
    def test_perfect_negative_rank_correlation(self):
        runner = load_runner()
        self.assertAlmostEqual(runner.spearman([1, 2, 3], [9, 5, 1]), -1.0)

    def test_perfect_positive_rank_correlation(self):
        runner = load_runner()
        self.assertAlmostEqual(runner.spearman([1, 2, 3], [1, 5, 9]), 1.0)

    def test_constant_series_is_zero_not_nan(self):
        runner = load_runner()
        self.assertEqual(runner.spearman([1, 2, 3], [4, 4, 4]), 0.0)

    def test_length_mismatch_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.spearman([1, 2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
