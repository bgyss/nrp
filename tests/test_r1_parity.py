"""Tests for the R1 parity runner's pure helpers (examples/r1_parity.py).

This rung re-asks R1's original question -- does a fairly-allocated world-anchored
encoding match pixel2d at a single trained view -- under a resolution ladder common
to every arm, capacity reported rather than matched. These tests cover only the
runner's pure logic (arm-config construction, the per-seed gate, report assembly);
the training path itself is exercised by nrp/torch_backend/train.py's own tests.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location("r1_parity", ROOT / "examples" / "r1_parity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ArmConfigTests(unittest.TestCase):
    def test_every_arm_shares_the_common_resolution_ladder(self):
        runner = load_runner()
        base = {"model": {"hidden_width": 128}, "seed": 99, "device": "mps"}
        for arm in runner.ARMS:
            cfg = runner.make_arm_config(base, arm, seed=2, out_dir=Path("out/x"))
            encoding = cfg["model"]["encoding"]
            self.assertEqual(encoding["base_resolution"], 4, arm)
            self.assertEqual(encoding["finest_resolution"], 64, arm)
            self.assertEqual(encoding["features_per_level"], 2, arm)

    def test_pixel2d_control_uses_eight_levels(self):
        runner = load_runner()
        cfg = runner.make_arm_config({"model": {}}, "pixel2d", seed=0, out_dir=Path("out/x"))
        self.assertEqual(cfg["model"]["spatial_encoding"], "pixel2d")
        self.assertEqual(cfg["model"]["encoding"]["levels"], 8)
        self.assertEqual(cfg["model"]["encoding"]["table_size_log2"], 14)

    def test_world_normal_triplane_uses_four_levels(self):
        runner = load_runner()
        cfg = runner.make_arm_config(
            {"model": {}}, "world_normal_triplane", seed=0, out_dir=Path("out/x")
        )
        self.assertEqual(cfg["model"]["spatial_encoding"], "world_normal_triplane")
        self.assertEqual(cfg["model"]["encoding"]["levels"], 4)

    def test_world3d_opts_into_occupancy_allocation(self):
        runner = load_runner()
        cfg = runner.make_arm_config({"model": {}}, "world3d", seed=0, out_dir=Path("out/x"))
        self.assertEqual(cfg["model"]["spatial_encoding"], "world3d")
        self.assertEqual(cfg["model"]["encoding"]["allocation"], "occupancy")

    def test_world_sparse_has_no_table_size_log2(self):
        # SparseVoxelEncoding has no dense/hashed table policy to size -- passing
        # table_size_log2 would be silently ignored (**_ignored), which would hide
        # a config-authoring mistake rather than catch it.
        runner = load_runner()
        cfg = runner.make_arm_config({"model": {}}, "world_sparse", seed=0, out_dir=Path("out/x"))
        self.assertNotIn("table_size_log2", cfg["model"]["encoding"])

    def test_seed_and_out_dir_are_applied(self):
        runner = load_runner()
        cfg = runner.make_arm_config({"model": {}}, "pixel2d", seed=7, out_dir=Path("out/foo"))
        self.assertEqual(cfg["seed"], 7)
        self.assertEqual(cfg["device"], "cpu")
        self.assertEqual(cfg["out_dir"], str(Path("out/foo")))

    def test_base_config_is_not_mutated(self):
        runner = load_runner()
        base = {"model": {"hidden_width": 128}, "seed": 99}
        runner.make_arm_config(base, "world3d", seed=3, out_dir=Path("out/x"))
        self.assertEqual(base["seed"], 99)
        self.assertNotIn("spatial_encoding", base["model"])

    def test_unknown_arm_raises(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.make_arm_config({"model": {}}, "not_an_arm", seed=0, out_dir=Path("out/x"))


class SeedGateTests(unittest.TestCase):
    def test_all_seeds_passing_gate_passes(self):
        runner = load_runner()
        verdict = runner.arm_gate_verdict([0.1, -0.2, 0.0, -0.5, 1.0], seeds=(0, 1, 2, 3, 4))
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["passing_seed_count"], 5)
        self.assertEqual(verdict["seed_count"], 5)

    def test_one_failing_seed_fails_the_whole_arm(self):
        # A favourable mean must never rescue a single failing seed: mean of
        # [-1.0, 1.0, 1.0, 1.0, 1.0] is 0.6, well above threshold, but seed 0 fails.
        runner = load_runner()
        verdict = runner.arm_gate_verdict([-1.0, 1.0, 1.0, 1.0, 1.0], seeds=(0, 1, 2, 3, 4))
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["passing_seed_count"], 4)
        self.assertGreater(float(np.mean([-1.0, 1.0, 1.0, 1.0, 1.0])), runner.GATE_DELTA_DB)

    def test_delta_exactly_at_threshold_passes(self):
        runner = load_runner()
        verdict = runner.arm_gate_verdict([runner.GATE_DELTA_DB] * 3, seeds=(0, 1, 2))
        self.assertTrue(verdict["pass"])

    def test_delta_just_below_threshold_fails(self):
        runner = load_runner()
        verdict = runner.arm_gate_verdict(
            [runner.GATE_DELTA_DB - 1e-9] + [10.0, 10.0], seeds=(0, 1, 2)
        )
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["passing_seed_count"], 2)

    def test_empty_seeds_raises_rather_than_reporting_a_pass(self):
        # The recurring defect this branch has hit fifteen times: a gate that
        # reports a pass when zero seeds were evaluated. Must raise, not return
        # {"pass": True}.
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.arm_gate_verdict([], seeds=())

    def test_mismatched_lengths_raise(self):
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.arm_gate_verdict([0.1, 0.2], seeds=(0, 1, 2))


class AnyWorldArmPassesTests(unittest.TestCase):
    def test_true_when_one_arm_passes(self):
        runner = load_runner()
        gates = {
            "world_sparse": {"pass": False},
            "world3d": {"pass": True},
        }
        self.assertTrue(runner.any_world_arm_passes(gates))

    def test_false_when_no_arm_passes(self):
        runner = load_runner()
        gates = {
            "world_sparse": {"pass": False},
            "world3d": {"pass": False},
        }
        self.assertFalse(runner.any_world_arm_passes(gates))

    def test_empty_gate_mapping_raises_rather_than_reporting_a_pass(self):
        # Same recurring defect at the whole-experiment level: no world arms
        # evaluated must never resolve to a vacuous pass.
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.any_world_arm_passes({})


class ReportAssemblyTests(unittest.TestCase):
    def _fake_arm_summary(self, arm, seed, param_count, delta):
        return {
            "arm": arm,
            "seed": seed,
            "parameter_count": param_count,
            "capacity_report": {"encoding": arm, "total_slots": param_count},
        }

    def test_build_report_records_gate_per_world_arm_and_overall_verdict(self):
        runner = load_runner()
        seeds = (0, 1)
        world_arms = ("world_sparse", "world3d")
        per_seed_deltas = {
            "world_sparse": [0.2, -0.1],
            "world3d": [-1.0, 1.0],
        }
        world_gates = {
            arm: runner.arm_gate_verdict(per_seed_deltas[arm], seeds=seeds) for arm in world_arms
        }
        training_arms = [
            self._fake_arm_summary("pixel2d", 0, 100, 0.0),
            self._fake_arm_summary("pixel2d", 1, 100, 0.0),
            self._fake_arm_summary("world_sparse", 0, 50, 0.2),
            self._fake_arm_summary("world_sparse", 1, 50, -0.1),
            self._fake_arm_summary("world3d", 0, 8, -1.0),
            self._fake_arm_summary("world3d", 1, 8, 1.0),
        ]
        report = runner.build_report(
            seeds=seeds,
            training_arms=training_arms,
            world_gates=world_gates,
            hardware={"platform": "test"},
        )
        self.assertEqual(report["gate"]["world_sparse"]["pass"], True)
        self.assertEqual(report["gate"]["world3d"]["pass"], False)
        self.assertTrue(report["any_world_arm_pass"])
        self.assertEqual(report["training_arms"], training_arms)
        self.assertEqual(report["hardware"], {"platform": "test"})

    def test_build_report_all_world_arms_failing_reports_overall_failure(self):
        runner = load_runner()
        seeds = (0, 1)
        world_gates = {
            "world_sparse": runner.arm_gate_verdict([-1.0, -1.0], seeds=seeds),
            "world_normal_triplane": runner.arm_gate_verdict([-1.0, -1.0], seeds=seeds),
            "world3d": runner.arm_gate_verdict([-1.0, -1.0], seeds=seeds),
        }
        report = runner.build_report(
            seeds=seeds, training_arms=[], world_gates=world_gates, hardware={}
        )
        self.assertFalse(report["any_world_arm_pass"])


if __name__ == "__main__":
    unittest.main()
