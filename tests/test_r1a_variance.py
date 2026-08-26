"""Tests for the R1A variance-decomposition runner helpers."""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "r1a_variance", ROOT / "examples" / "r1a_variance.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class R1AStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_is_deterministic_and_reports_a_95_percent_interval(self):
        runner = load_runner()

        first = runner.paired_bootstrap_ci(
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), resamples=2000, seed=7
        )
        second = runner.paired_bootstrap_ci(
            np.array([-1.0, -0.5, 0.0, 0.5, 1.0]), resamples=2000, seed=7
        )

        self.assertEqual(first, second)
        self.assertEqual(first["n"], 5)
        self.assertEqual(first["resamples"], 2000)
        self.assertLessEqual(first["lower_db"], first["upper_db"])

    def test_pair_validation_metrics_keeps_light_order_and_computes_deltas(self):
        runner = load_runner()
        control = [
            {"light": {"center": [1.0, 0.0, 0.0]}, "psnr_db_vs_raw": 20.0},
            {"light": {"center": [2.0, 0.0, 0.0]}, "psnr_db_vs_raw": 22.0},
        ]
        candidate = [
            {"light": {"center": [1.0, 0.0, 0.0]}, "psnr_db_vs_raw": 19.5},
            {"light": {"center": [2.0, 0.0, 0.0]}, "psnr_db_vs_raw": 22.5},
        ]

        paired = runner.pair_validation_metrics(control, candidate)

        self.assertEqual([row["light_index"] for row in paired], [0, 1])
        self.assertEqual([row["delta_db"] for row in paired], [-0.5, 0.5])
        self.assertEqual([row["light"] for row in paired], [row["light"] for row in control])

    def test_make_arm_config_makes_output_bias_policy_explicit(self):
        runner = load_runner()
        base = {"model": {"encoding": {"levels": 1}}, "seed": 99, "device": "mps"}

        target = runner.make_arm_config(base, "world3d", "target_scale", 3, Path("out"))
        default = runner.make_arm_config(base, "world3d", "framework_default", 3, Path("out"))

        self.assertEqual(target["seed"], 3)
        self.assertEqual(target["device"], "cpu")
        self.assertEqual(target["model"]["spatial_encoding"], "world3d")
        self.assertTrue(target["model"]["init_output_scale"])
        self.assertFalse(default["model"]["init_output_scale"])

    def test_per_seed_gate_does_not_average_away_one_failing_seed(self):
        runner = load_runner()

        metrics = {}
        seed_deltas = [-1.0, 1.0, 1.0, 1.0, 1.0]
        for seed, delta in enumerate(seed_deltas):
            control = [
                {"light": {"id": 0}, "psnr_db_vs_raw": 20.0},
                {"light": {"id": 1}, "psnr_db_vs_raw": 20.0},
            ]
            candidate = [
                {"light": {"id": 0}, "psnr_db_vs_raw": 20.0 + delta},
                {"light": {"id": 1}, "psnr_db_vs_raw": 20.0 + delta},
            ]
            metrics[("target_scale", "pixel2d", seed)] = control
            metrics[("target_scale", "world3d", seed)] = candidate
            metrics[("target_scale", "world_triplane", seed)] = candidate
            metrics[("framework_default", "pixel2d", seed)] = control
            metrics[("framework_default", "world3d", seed)] = candidate
            metrics[("framework_default", "world_triplane", seed)] = candidate

        comparisons = runner.build_comparisons(
            metrics, tuple(range(5)), resamples=200, bootstrap_seed=7
        )

        world3d = comparisons["world3d"]["target_scale"]
        self.assertGreater(world3d["across_seed_summary"]["mean_db"], -0.5)
        self.assertFalse(world3d["gate"]["pass"])
        self.assertEqual(world3d["gate"]["passing_seed_count"], 4)


if __name__ == "__main__":
    unittest.main()
