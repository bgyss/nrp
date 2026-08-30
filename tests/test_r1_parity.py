"""Tests for the R1 parity runner's pure helpers (examples/r1_parity.py).

This rung re-asks R1's original question -- does a fairly-allocated world-anchored
encoding match pixel2d at a single trained view -- under a resolution ladder common
to every arm, capacity reported rather than matched. These tests cover only the
runner's pure logic (arm-config construction, the per-seed gate, report assembly);
the training path itself is exercised by nrp/torch_backend/train.py's own tests.
"""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from nrp.toy_tracer import trace_path_cache

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location("r1_parity", ROOT / "examples" / "r1_parity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_r1a_variance():
    spec = importlib.util.spec_from_file_location(
        "r1a_variance", ROOT / "examples" / "r1a_variance.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _small_cache():
    return trace_path_cache(5, 4, spp=2, max_bounces=1, seed=9)


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


class ResolutionLadderTests(unittest.TestCase):
    """--finest-resolution/--base-resolution build a ladder shared by every arm.

    `build_arm_models` must never let one arm drift to a different base/finest
    resolution than the others -- that would silently reintroduce the kind of
    capacity mismatch this experiment exists to rule out.
    """

    def test_default_ladder_matches_todays_64(self):
        runner = load_runner()
        arm_models = runner.build_arm_models()
        for arm in runner.ARMS:
            encoding = arm_models[arm]["encoding"]
            self.assertEqual(encoding["base_resolution"], 4, arm)
            self.assertEqual(encoding["finest_resolution"], 64, arm)
        # The module-level default must also match (nothing bypasses build_arm_models).
        for arm in runner.ARMS:
            encoding = runner.ARM_MODELS[arm]["encoding"]
            self.assertEqual(encoding["base_resolution"], 4, arm)
            self.assertEqual(encoding["finest_resolution"], 64, arm)

    def test_custom_ladder_is_shared_by_every_arm(self):
        runner = load_runner()
        arm_models = runner.build_arm_models(base_resolution=8, finest_resolution=128)
        for arm in runner.ARMS:
            encoding = arm_models[arm]["encoding"]
            self.assertEqual(encoding["base_resolution"], 8, arm)
            self.assertEqual(encoding["finest_resolution"], 128, arm)

    def test_arm_models_flows_through_make_arm_config(self):
        # This is the assertion that would actually catch one arm drifting from the
        # rest: build a ladder, then verify make_arm_config's output -- what the
        # trainer actually receives -- carries it through for every arm.
        runner = load_runner()
        arm_models = runner.build_arm_models(base_resolution=8, finest_resolution=128)
        for arm in runner.ARMS:
            cfg = runner.make_arm_config(
                {"model": {}}, arm, seed=0, out_dir=Path("out/x"), arm_models=arm_models
            )
            encoding = cfg["model"]["encoding"]
            self.assertEqual(encoding["base_resolution"], 8, arm)
            self.assertEqual(encoding["finest_resolution"], 128, arm)


class SeedGateTests(unittest.TestCase):
    """These exercise the legacy per-seed rule, now nested under `result["per_seed"]`
    (dual-verdict `arm_gate_verdict` still reports both rules; `binding="per_seed"`
    makes the legacy rule decisive so `result["pass"]` mirrors it directly)."""

    def test_all_seeds_passing_gate_passes(self):
        runner = load_runner()
        verdict = runner.arm_gate_verdict(
            [0.1, -0.2, 0.0, -0.5, 1.0], seeds=(0, 1, 2, 3, 4), binding="per_seed"
        )
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["per_seed"]["passing_seed_count"], 5)
        self.assertEqual(verdict["per_seed"]["seed_count"], 5)

    def test_one_failing_seed_fails_the_whole_arm(self):
        # A favourable mean must never rescue a single failing seed: mean of
        # [-1.0, 1.0, 1.0, 1.0, 1.0] is 0.6, well above threshold, but seed 0 fails.
        runner = load_runner()
        verdict = runner.arm_gate_verdict(
            [-1.0, 1.0, 1.0, 1.0, 1.0], seeds=(0, 1, 2, 3, 4), binding="per_seed"
        )
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["per_seed"]["passing_seed_count"], 4)
        self.assertGreater(
            float(np.mean([-1.0, 1.0, 1.0, 1.0, 1.0])), verdict["per_seed"]["threshold_db"]
        )

    def test_delta_exactly_at_threshold_passes(self):
        runner = load_runner()
        verdict = runner.arm_gate_verdict(
            [runner.GATE_DELTA_DB] * 3, seeds=(0, 1, 2), binding="per_seed"
        )
        self.assertTrue(verdict["pass"])

    def test_delta_just_below_threshold_fails(self):
        runner = load_runner()
        verdict = runner.arm_gate_verdict(
            [runner.GATE_DELTA_DB - 1e-9] + [10.0, 10.0], seeds=(0, 1, 2), binding="per_seed"
        )
        self.assertFalse(verdict["pass"])
        self.assertEqual(verdict["per_seed"]["passing_seed_count"], 2)

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
            arm: runner.arm_gate_verdict(per_seed_deltas[arm], seeds=seeds, binding="per_seed")
            for arm in world_arms
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
            "world_sparse": runner.arm_gate_verdict([-1.0, -1.0], seeds=seeds, binding="per_seed"),
            "world_normal_triplane": runner.arm_gate_verdict(
                [-1.0, -1.0], seeds=seeds, binding="per_seed"
            ),
            "world3d": runner.arm_gate_verdict([-1.0, -1.0], seeds=seeds, binding="per_seed"),
        }
        report = runner.build_report(
            seeds=seeds, training_arms=[], world_gates=world_gates, hardware={}
        )
        self.assertFalse(report["any_world_arm_pass"])


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


def make_args(**overrides):
    import argparse

    defaults = dict(
        cache="out/cache.npz",
        out_dir="out/r1-parity",
        iters=3000,
        finest_resolution=64,
        base_resolution=4,
        denoise_method="bilateral",
        gate="equivalence",
        max_seeds=48,
        bootstrap_seed=1234,
        bootstrap_resamples=2000,
        gate_lights=96,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class ReproduceCommandTests(unittest.TestCase):
    """Regression test: the recorded command must include every argument that
    changes the run's numbers, not just the ones that change its output location.
    Replaying a command missing --denoise-method silently falls back to the
    bilateral default and reproduces a DIFFERENT measurement.
    """

    def test_command_includes_denoise_method(self):
        runner = load_runner()
        args = make_args(denoise_method="oidn")
        command = runner.reproduce_command(args, (0, 1, 2, 3, 4))
        self.assertIn("--denoise-method oidn", command)

    def test_command_includes_bootstrap_seed_and_resamples(self):
        runner = load_runner()
        args = make_args(bootstrap_seed=99, bootstrap_resamples=500)
        command = runner.reproduce_command(args, (0,))
        self.assertIn("--bootstrap-seed 99", command)
        self.assertIn("--bootstrap-resamples 500", command)

    def test_command_includes_every_result_affecting_argument(self):
        runner = load_runner()
        args = make_args()
        command = runner.reproduce_command(args, (0, 1))
        for flag in (
            "--cache",
            "--out-dir",
            "--seeds",
            "--iters",
            "--finest-resolution",
            "--base-resolution",
            "--denoise-method",
            "--gate",
            "--max-seeds",
            "--bootstrap-seed",
            "--bootstrap-resamples",
            "--gate-lights",
        ):
            self.assertIn(flag, command)

    def test_command_includes_gate_lights_value(self):
        runner = load_runner()
        args = make_args(gate_lights=96)
        command = runner.reproduce_command(args, (0, 1))
        self.assertIn("--gate-lights 96", command)


class SeedBindingCompatibilityTests(unittest.TestCase):
    def test_none_forced_seeds_is_always_fine(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        runner.check_seed_binding_compatibility(None, "equivalence", EquivalenceGate())

    def test_per_seed_binding_never_raises(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        runner.check_seed_binding_compatibility((0, 1, 2, 3, 4), "per_seed", EquivalenceGate())

    def test_scheduled_look_under_equivalence_is_fine(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        runner.check_seed_binding_compatibility(tuple(range(8)), "equivalence", EquivalenceGate())

    def test_off_schedule_seed_count_under_equivalence_raises_before_training(self):
        runner = load_runner()
        from nrp.experiment_gate import EquivalenceGate

        with self.assertRaises(ValueError) as ctx:
            runner.check_seed_binding_compatibility(
                (0, 1, 2, 3, 4), "equivalence", EquivalenceGate()
            )
        message = str(ctx.exception)
        self.assertIn("per-seed", message)
        self.assertIn("5", message)


class GateExitCodeTests(unittest.TestCase):
    def test_any_pass_is_zero(self):
        runner = load_runner()
        gates = {"world_sparse": {"pass": True, "equivalence": {"verdict": "pass"}}}
        self.assertEqual(runner.gate_exit_code(gates), 0)

    def test_all_underpowered_is_three(self):
        runner = load_runner()
        gates = {
            "world_sparse": {"pass": False, "equivalence": {"verdict": "underpowered"}},
            "world3d": {"pass": False, "equivalence": {"verdict": "underpowered"}},
        }
        self.assertEqual(runner.gate_exit_code(gates), 3)

    def test_any_real_fail_is_two_even_alongside_underpowered(self):
        runner = load_runner()
        gates = {
            "world_sparse": {"pass": False, "equivalence": {"verdict": "fail"}},
            "world3d": {"pass": False, "equivalence": {"verdict": "underpowered"}},
        }
        self.assertEqual(runner.gate_exit_code(gates), 2)

    def test_per_seed_binding_with_no_equivalence_verdict_is_two(self):
        runner = load_runner()
        gates = {"world_sparse": {"pass": False, "equivalence": None}}
        self.assertEqual(runner.gate_exit_code(gates), 2)


class GateLightsPreRegistrationTests(unittest.TestCase):
    def test_gate_lights_default_is_pre_registered_at_96(self):
        runner = load_runner()
        self.assertEqual(runner.BASE_TRAIN_CONFIG["n_gate_lights"], 96)

    def test_reproduce_command_records_gate_lights(self):
        import argparse

        runner = load_runner()
        args = argparse.Namespace(
            cache="c.npz",
            out_dir="o",
            iters=3000,
            base_resolution=4,
            finest_resolution=128,
            denoise_method="oidn",
            gate="equivalence",
            max_seeds=8,
            bootstrap_seed=0,
            bootstrap_resamples=2000,
            gate_lights=96,
        )
        cmd = runner.reproduce_command(args, (0, 1))
        self.assertIn("--gate-lights 96", cmd)


class FrozenValidationSetGateSizeTests(unittest.TestCase):
    def test_frozen_sets_honor_an_explicit_gate_light_count(self):
        """The gate's held-out set is sized by n_gate_lights, not by the
        training-time n_val_lights -- raising it must not slow training."""
        runner = load_runner()
        r1a_variance = load_r1a_variance()
        cache = _small_cache()
        base = dict(runner.BASE_TRAIN_CONFIG)
        base["n_val_lights"] = 2
        base["denoise"] = {"enabled": False}
        sets, specs = r1a_variance.build_frozen_validation_sets(cache, base, (0,), n_gate_lights=6)
        self.assertEqual(len(sets[0]), 6)
        self.assertEqual(len(specs["0"]), 6)
        small, _ = r1a_variance.build_frozen_validation_sets(cache, base, (0,))
        self.assertEqual(len(small[0]), 2)
        for a, b in zip(small[0], sets[0], strict=False):
            self.assertEqual(a["light"].to_dict(), b["light"].to_dict())


class RunExperimentGateLightsTest(unittest.TestCase):
    """Regression test for the Critical defect `FrozenValidationSetGateSizeTests`
    above does not catch: that test calls `build_frozen_validation_sets` directly, a
    hand-copied mirror of the call `run_experiment` makes at
    examples/r1_parity.py:463-465. Reverting that call site to drop
    `n_gate_lights=base_cfg.get("n_gate_lights")` (silently falling back to the
    training-time `n_val_lights`) leaves every test above green, because none of
    them ever go through `run_experiment` itself.

    This test drives the real `run_experiment`, with training, model loading, and
    evaluation monkeypatched to synthetic, near-instant stand-ins -- training four
    arms for real would take minutes and is exercised independently by
    nrp/torch_backend/train.py's own tests -- and `build_frozen_validation_sets`
    replaced with a spy that records the `n_gate_lights` it was actually called
    with, while still returning a usable fake validation set so `run_experiment`
    completes. `n_gate_lights` is set to a value that differs from both
    `n_val_lights` and the 96 default, so a fallback to either would be caught.
    """

    def test_run_experiment_passes_n_gate_lights_through_to_the_frozen_sets(self):
        runner = load_runner()
        n_gate_lights = 20  # != n_val_lights (12) and != the 96 default
        base_cfg = copy.deepcopy(runner.BASE_TRAIN_CONFIG)
        base_cfg["n_val_lights"] = 12
        base_cfg["n_gate_lights"] = n_gate_lights
        base_cfg["iters"] = 1
        self.assertNotEqual(n_gate_lights, base_cfg["n_val_lights"])

        recorded_n_gate_lights = []

        def fake_build_frozen_validation_sets(cache, cfg, seeds, n_gate_lights=None):
            recorded_n_gate_lights.append(n_gate_lights)
            # Mirrors build_frozen_validation_sets' real fallback (n_val_lights when
            # n_gate_lights is None) so a reverted call site produces a
            # DIFFERENT-sized set here too, rather than accidentally still 20.
            count = n_gate_lights if n_gate_lights is not None else cfg.get("n_val_lights", 12)
            sets = {}
            specs = {}
            for seed in seeds:
                lights = [{"radius": 0.05 + 0.001 * i} for i in range(count)]
                sets[seed] = [{"light": light} for light in lights]
                specs[str(seed)] = lights
            return sets, specs

        def fake_train(cfg):
            return {"parameter_count": 1, "iters_per_second": 1.0, "train_seconds": 0.001}

        class _FakeEncoding:
            def capacity_report(self):
                return None

        class _FakeModel:
            encoding = _FakeEncoding()

        def fake_load_trained_model(path, cache):
            return _FakeModel()

        def fake_evaluate_model(model, cache, val_set):
            return [{"light": row["light"], "psnr_db_vs_raw": 20.0} for row in val_set]

        cache = _small_cache()
        with (
            mock.patch.object(
                runner, "build_frozen_validation_sets", fake_build_frozen_validation_sets
            ),
            mock.patch.object(runner, "train", fake_train),
            mock.patch.object(runner, "load_trained_model", fake_load_trained_model),
            mock.patch.object(runner, "evaluate_model", fake_evaluate_model),
        ):
            with tempfile.TemporaryDirectory() as out_dir:
                report = runner.run_experiment(
                    base_cfg,
                    cache,
                    root=ROOT,
                    out_root=Path(out_dir),
                    seeds=(0,),
                    resamples=10,
                    bootstrap_seed=0,
                    binding="per_seed",
                )

        # The spy proves the real call site handed n_gate_lights through (not None,
        # not n_val_lights) -- this is what a reverted call site would break.
        self.assertEqual(recorded_n_gate_lights, [n_gate_lights])
        # The report's own held-out set size confirms the gate actually used the
        # requested count end to end, not just that the spy received it.
        self.assertEqual(len(report["validation_specs"]["0"]), n_gate_lights)


if __name__ == "__main__":
    unittest.main()
