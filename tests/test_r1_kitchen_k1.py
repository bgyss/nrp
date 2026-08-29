"""Tests for the K1 sweep runner's pure helpers (examples/r1_kitchen_k1.py).

K1 tests the vertex-support hypothesis by sweeping `world_sparse`'s finest
resolution against a FIXED, pre-committed `pixel2d` control. These tests cover the
logic that protects that fixture -- control extraction, scene/ladder compatibility,
validation-light fingerprint identity -- and the directional verdict, which must
never turn a falsified prediction into a pass. Training itself is exercised by
nrp/torch_backend/train.py's tests.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.path_cache import PathCache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location(
        "r1_kitchen_k1", ROOT / "examples" / "r1_kitchen_k1.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def control_report(
    seeds=(0, 1), cache="out/kitchen/path_cache.npz", base_resolution=4, n_gate_lights=None
):
    training_config = {"iters": 3000, "denoise": {"enabled": True, "method": "oidn"}}
    if n_gate_lights is not None:
        training_config["n_gate_lights"] = n_gate_lights
    return {
        "control_arm": "pixel2d",
        "cache": cache,
        "resolution_ladder": {"base_resolution": base_resolution, "finest_resolution": 128},
        "command": "uv run python examples/r1_parity.py ...",
        "hardware": {"cpu_brand": "Apple M1 Max"},
        "training_config": training_config,
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

    def test_no_run_training_config_skips_the_strict_check(self):
        """Backward-compatible: omitting run_training_config compares nothing new."""
        runner = load_runner()
        control = runner.check_control_compatibility(
            control_report(), cache="out/kitchen/path_cache.npz", base_resolution=4
        )
        self.assertIn("training_config", control)

    def test_matching_training_config_passes(self):
        runner = load_runner()
        report = control_report()
        report["training_config"]["pool"] = {"size": 64}
        runner.check_control_compatibility(
            report,
            cache="out/kitchen/path_cache.npz",
            base_resolution=4,
            run_training_config={"pool": {"size": 64}},
        )

    def test_mismatched_pool_raises_naming_key_and_both_values(self):
        runner = load_runner()
        report = control_report()
        report["training_config"]["pool"] = {"size": 64}
        with self.assertRaises(ValueError) as ctx:
            runner.check_control_compatibility(
                report,
                cache="out/kitchen/path_cache.npz",
                base_resolution=4,
                run_training_config={"pool": {"size": 128}},
            )
        message = str(ctx.exception)
        self.assertIn("pool", message)
        self.assertIn("64", message)
        self.assertIn("128", message)

    def test_mismatched_lr_raises(self):
        runner = load_runner()
        report = control_report()
        report["training_config"]["lr"] = 0.005
        with self.assertRaises(ValueError):
            runner.check_control_compatibility(
                report,
                cache="out/kitchen/path_cache.npz",
                base_resolution=4,
                run_training_config={"lr": 0.01},
            )

    def test_mismatched_model_raises(self):
        runner = load_runner()
        report = control_report()
        report["training_config"]["model"] = {"hidden_width": 128, "hidden_layers": 4}
        with self.assertRaises(ValueError):
            runner.check_control_compatibility(
                report,
                cache="out/kitchen/path_cache.npz",
                base_resolution=4,
                run_training_config={"model": {"hidden_width": 64, "hidden_layers": 4}},
            )

    def test_mismatched_light_bounds_raises(self):
        runner = load_runner()
        report = control_report()
        report["training_config"]["light_bounds"] = {"radius_min": 0.05, "radius_max": 0.25}
        with self.assertRaises(ValueError):
            runner.check_control_compatibility(
                report,
                cache="out/kitchen/path_cache.npz",
                base_resolution=4,
                run_training_config={"light_bounds": {"radius_min": 0.1, "radius_max": 0.25}},
            )

    def test_iters_mismatch_is_not_checked_here_only_via_the_caller_warning(self):
        """iters stays a warning in main(), never a raise from this function."""
        runner = load_runner()
        report = control_report()
        report["training_config"]["iters"] = 3000
        runner.check_control_compatibility(
            report,
            cache="out/kitchen/path_cache.npz",
            base_resolution=4,
            run_training_config={"iters": 50},
        )


class GateLightsCompatibilityTests(unittest.TestCase):
    """A sweep read against a control scored on a different number of held-out
    lights would attribute an estimator-precision difference to the swept
    resolution -- exactly the class of confound `_STRICT_TRAINING_CONFIG_KEYS`
    already exists to catch (2026-08-29).
    """

    def test_control_with_a_different_gate_light_count_is_rejected(self):
        runner = load_runner()
        control = control_report(n_gate_lights=12)
        run_cfg = dict(control["training_config"])
        run_cfg["n_gate_lights"] = 96
        with self.assertRaisesRegex(ValueError, "n_gate_lights"):
            runner.check_control_compatibility(
                control,
                cache="out/kitchen/path_cache.npz",
                base_resolution=4,
                run_training_config=run_cfg,
            )

    def test_a_control_predating_the_key_reads_as_twelve_lights(self):
        """Committed controls have no n_gate_lights. Absence is 12, not None --
        otherwise every historical control is rejected by a 12-light run."""
        runner = load_runner()
        control = control_report()
        control["training_config"].pop("n_gate_lights", None)
        run_cfg = dict(control["training_config"])
        run_cfg["n_gate_lights"] = 12
        runner.check_control_compatibility(
            control,
            cache="out/kitchen/path_cache.npz",
            base_resolution=4,
            run_training_config=run_cfg,
        )

    def test_a_control_predating_the_key_is_rejected_by_a_96_light_run(self):
        runner = load_runner()
        control = control_report()
        control["training_config"].pop("n_gate_lights", None)
        run_cfg = dict(control["training_config"])
        run_cfg["n_gate_lights"] = 96
        with self.assertRaisesRegex(ValueError, "n_gate_lights"):
            runner.check_control_compatibility(
                control,
                cache="out/kitchen/path_cache.npz",
                base_resolution=4,
                run_training_config=run_cfg,
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
        # A per-seed-bound row carries no equivalence verdict; the word is derived
        # from the binding rule's pass/fail rather than dropped from the report.
        self.assertEqual(report["gate_verdict"], {"32": "pass", "128": "fail"})
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


class GateBindingTests(unittest.TestCase):
    """Regression test: `run_sweep` must not use the equivalence gate's default
    binding, because K1's 5-seed default is not a scheduled look (8/16/24/32/40/48)
    and `arm_gate_verdict`'s default `binding="equivalence"` raises off schedule.
    `run_sweep` itself needs real training, so this exercises the exact call it
    makes (`arm_gate_verdict(deltas, seeds, binding="per_seed")`) against the real
    gate machinery -- no mocking -- with K1's documented 5-seed default.
    """

    def test_five_seed_gate_call_returns_a_verdict_rather_than_raising(self):
        runner = load_runner()
        seeds = (0, 1, 2, 3, 4)
        deltas = [0.1, -0.2, 0.05, -0.4, 0.0]
        result = runner.arm_gate_verdict(deltas, seeds, binding="per_seed")
        self.assertIn("pass", result)
        self.assertIsInstance(result["pass"], bool)
        self.assertEqual(result["binding"], "per_seed")

    def test_default_equivalence_binding_raises_at_five_seeds(self):
        """Sanity check on the bug this regression test guards against."""
        runner = load_runner()
        with self.assertRaises(ValueError):
            runner.arm_gate_verdict([0.1] * 5, (0, 1, 2, 3, 4))


def _tiny_cache() -> PathCache:
    """A minimal one-pixel, one-segment cache. `run_sweep`'s real callers that would
    read this cache's contents (training, evaluation, vertex-support measurement) are
    monkeypatched away by `RunSweepGateBindingTest` below, so only `PathCache.validate`
    (called implicitly by construction patterns elsewhere) needs a shape-correct
    object -- the numeric content is never inspected.
    """
    return PathCache(
        width=1,
        height=1,
        n_paths=np.array([1], dtype=np.int64),
        seg_pixel=np.array([0], dtype=np.int64),
        seg_origin=np.array([[0.0, 0.0, 0.0]]),
        seg_dir=np.array([[0.0, 0.0, 1.0]]),
        seg_tmax=np.array([1.0]),
        seg_throughput=np.full((1, 3), 1.0),
        albedo=np.full((1, 1, 3), 0.5),
        depth=np.ones((1, 1)),
        normal=np.tile(np.array([0.0, 1.0, 0.0]), (1, 1, 1)),
        position=np.array([[[0.0, 0.0, 1.0]]]),
    )


def run_sweep_with_fakes(runner, seeds=(0, 1, 2, 3, 4), binding=None):
    """Drive `run_sweep` with training/evaluation/vertex-support faked out."""
    # Deterministic per-seed deltas the real per-light pairing math on the way to
    # the gate call must reproduce exactly -- if `run_sweep`'s aggregation logic
    # regressed this would also fail even though that is not what this test is for.
    cycle = (0.1, -0.2, 0.05, -0.4, 0.0)
    per_seed_delta = {seed: cycle[index % len(cycle)] for index, seed in enumerate(seeds)}
    control_light = {"radius": 0.1}
    control_by_seed = {seed: [{"light": control_light, "psnr_db_vs_raw": 20.0}] for seed in seeds}
    validation_sets = {seed: seed for seed in seeds}  # opaque marker, read back below

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
        seed = val_set
        return [{"light": control_light, "psnr_db_vs_raw": 20.0 + per_seed_delta[seed]}]

    def fake_cache_vertex_support(cache, levels, base_resolution, finest):
        return {
            "finest": {"median_support": 1.0, "fraction_touched_by_le1_pixel": 0.5},
        }

    cache = _tiny_cache()
    with (
        mock.patch.object(runner, "train", fake_train),
        mock.patch.object(runner, "load_trained_model", fake_load_trained_model),
        mock.patch.object(runner, "evaluate_model", fake_evaluate_model),
        mock.patch.object(runner, "cache_vertex_support", fake_cache_vertex_support),
    ):
        with tempfile.TemporaryDirectory() as out_dir:
            per_resolution = runner.run_sweep(
                runner.BASE_TRAIN_CONFIG,
                cache,
                root=ROOT,
                out_root=Path(out_dir),
                seeds=seeds,
                resolutions=(32,),
                base_resolution=4,
                control_by_seed=control_by_seed,
                validation_sets=validation_sets,
                resamples=10,
                bootstrap_seed=0,
                **({} if binding is None else {"binding": binding}),
            )
    return per_resolution, per_seed_delta


class RunSweepGateBindingTest(unittest.TestCase):
    """Regression test for the Critical defect `GateBindingTests` above does not
    actually catch: those tests call `arm_gate_verdict` directly, a hand-copied
    mirror of the call `run_sweep` makes at examples/r1_kitchen_k1.py:406. Reverting
    that call to the buggy `arm_gate_verdict(seed_mean_deltas, seeds)` (dropping
    `binding="per_seed"`) leaves the direct tests above passing, because they never
    go through `run_sweep` at all.

    This test drives `run_sweep` itself with the historical 5-seed reproduction mode
    (`--gate per-seed`; 5 is not a scheduled equivalence look), so it fails if the real
    call site regresses to ignoring `binding`. Real
    training/evaluation/vertex-support measurement are monkeypatched to synthetic,
    near-instant stand-ins -- constructing a real cache large enough to train on
    would take minutes per seed, and the gate-binding bug lives entirely in
    `run_sweep`'s post-training bookkeeping, not in training or evaluation
    themselves. The monkeypatched pieces are exactly the ones `nrp/torch_backend`'s
    own tests already cover independently (training, model loading, evaluation).
    """

    def test_run_sweep_binds_the_five_seed_gate_on_per_seed(self):
        runner = load_runner()
        per_resolution, per_seed_delta = run_sweep_with_fakes(runner, binding="per_seed")
        self.assertEqual(len(per_resolution), 1)
        gate = per_resolution[0]["gate"]
        self.assertEqual(gate["binding"], "per_seed")
        expected_pass = all(delta >= runner.GATE_DELTA_DB for delta in per_seed_delta.values())
        self.assertEqual(gate["pass"], expected_pass)

    def test_run_sweep_raises_if_the_default_equivalence_binding_is_used(self):
        """Demonstrates the failure mode this test class exists to catch: with the
        buggy call (`arm_gate_verdict(seed_mean_deltas, seeds)`, no `binding=`),
        `run_sweep` would raise `ValueError` at 5 seeds instead of returning a report,
        since 5 is not one of the equivalence gate's scheduled looks.
        """
        runner = load_runner()
        real_arm_gate_verdict = runner.arm_gate_verdict

        def buggy_arm_gate_verdict(seed_mean_deltas, seeds, *args, **kwargs):
            kwargs.pop("binding", None)
            return real_arm_gate_verdict(seed_mean_deltas, seeds, *args, **kwargs)

        with mock.patch.object(runner, "arm_gate_verdict", buggy_arm_gate_verdict):
            with self.assertRaises(ValueError):
                run_sweep_with_fakes(runner, binding="per_seed")


class EquivalenceBindingTests(unittest.TestCase):
    """K1 re-run under the equivalence gate (2026-08-28).

    The per-seed rule K1 originally bound on rejects a true-parity arm 76-91% of the
    time at Kitchen's measured spreads, so K1's "no support" verdict was read under a
    rule that could not have supported the hypothesis reliably in the first place.
    `run_sweep` must therefore be able to bind on the equivalence rule -- but only at a
    scheduled look, and it must refuse BEFORE training rather than after hours of it.
    """

    def test_run_sweep_binds_the_eight_seed_gate_on_equivalence(self):
        runner = load_runner()
        seeds = tuple(range(8))
        per_resolution, per_seed_delta = run_sweep_with_fakes(
            runner, seeds=seeds, binding="equivalence"
        )
        gate = per_resolution[0]["gate"]
        self.assertEqual(gate["binding"], "equivalence")
        equivalence = gate["equivalence"]
        self.assertEqual(equivalence["n"], len(seeds))
        self.assertIn(equivalence["verdict"], {"pass", "fail", "continue", "underpowered"})
        # `pass` in the report is the DECISIVE rule's outcome, not the legacy one's.
        self.assertEqual(gate["pass"], equivalence["verdict"] == "pass")
        # Both verdicts stay recorded so a report can be read against either rule.
        self.assertEqual(gate["per_seed"]["seed_count"], len(seeds))

    def test_equivalence_binding_refuses_an_off_schedule_seed_count_before_training(self):
        """Five seeds is not a scheduled look, so the run must raise without training."""
        runner = load_runner()
        trained = []

        def counting_train(cfg):
            trained.append(cfg)
            raise AssertionError("run_sweep trained before validating the gate binding")

        with mock.patch.object(runner, "train", counting_train):
            with self.assertRaises(ValueError):
                runner.run_sweep(
                    runner.BASE_TRAIN_CONFIG,
                    _tiny_cache(),
                    root=ROOT,
                    out_root=ROOT,
                    seeds=(0, 1, 2, 3, 4),
                    resolutions=(32,),
                    base_resolution=4,
                    control_by_seed={},
                    validation_sets={},
                    resamples=10,
                    bootstrap_seed=0,
                    binding="equivalence",
                )
        self.assertEqual(trained, [])

    def test_default_seed_count_is_a_scheduled_look(self):
        """K1's default invocation must be runnable under its default gate."""
        runner = load_runner()
        gate = runner.EquivalenceGate()
        self.assertEqual(runner.DEFAULT_GATE, "equivalence")
        self.assertIn(len(runner.DEFAULT_SEEDS), gate.looks)


class MainGateLightsCallSiteTest(unittest.TestCase):
    """Regression test for the call site `main()` makes to
    `build_frozen_validation_sets` at examples/r1_kitchen_k1.py (around line 605).

    Unlike `r1_parity.py`, where that call lives inside `run_experiment` (the
    function `RunExperimentGateLightsTest` in tests/test_r1_parity.py drives
    directly), in this module the call lives in `main()` itself -- `run_sweep`
    receives an already-built `validation_sets` dict and never calls
    `build_frozen_validation_sets`. `run_sweep_with_fakes` therefore cannot see this
    call site at all; this test drives the real `main()` instead, with `sys.argv`
    pointed at a synthetic control report and cache, and with training, model
    loading, evaluation, vertex-support measurement, cache loading, and
    `build_frozen_validation_sets` itself replaced with fast, deterministic
    stand-ins -- the same style `run_sweep_with_fakes` and
    `RunExperimentGateLightsTest` use, just wrapping one more layer (`main`'s
    argument parsing and control-compatibility/fingerprint checks) to reach the
    call site that actually exists here.

    `build_frozen_validation_sets` is replaced with a spy that records the
    `n_gate_lights` it was called with, and returns a small deterministic
    validation set/spec so the rest of `main()` (fingerprint check, `run_sweep`,
    report-writing) completes normally. `--gate-lights 20` is chosen because it is
    neither the 12-light legacy default nor the 96-light BASE_TRAIN_CONFIG default,
    so a call-site regression to either fallback would produce a different
    (wrong) recorded value.
    """

    def test_main_passes_gate_lights_through_to_build_frozen_validation_sets(self):
        runner = load_runner()
        self._run_main(runner, gate_lights=20)
        self.assertEqual(runner._recorded_n_gate_lights, [20])

    def _run_main(self, runner, *, gate_lights: int) -> None:
        seeds = (0, 1, 2, 3, 4)
        cycle = (0.1, -0.2, 0.05, -0.4, 0.0)
        per_seed_delta = {seed: cycle[index % len(cycle)] for index, seed in enumerate(seeds)}
        control_light = {"radius": 0.1}

        runner._recorded_n_gate_lights = []

        def spy_build_frozen_validation_sets(cache, base_cfg, seeds, n_gate_lights=None):
            runner._recorded_n_gate_lights.append(n_gate_lights)
            sets = {seed: seed for seed in seeds}
            specs = {str(seed): [{"seed": seed}] for seed in seeds}
            return sets, specs

        fingerprints = {
            str(seed): runner.validation_fingerprint([{"seed": seed}]) for seed in seeds
        }

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
            seed = val_set
            return [{"light": control_light, "psnr_db_vs_raw": 20.0 + per_seed_delta[seed]}]

        def fake_cache_vertex_support(cache, levels, base_resolution, finest):
            return {"finest": {"median_support": 1.0, "fraction_touched_by_le1_pixel": 0.5}}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cache_path = tmp_path / "path_cache.npz"
            cache_path.write_bytes(b"")  # only existence is checked; load is faked below
            out_dir = tmp_path / "out"

            strict_cfg = {
                key: runner.BASE_TRAIN_CONFIG[key]
                for key in runner._STRICT_TRAINING_CONFIG_KEYS
                if key != "n_gate_lights"
            }
            strict_cfg["n_gate_lights"] = gate_lights
            strict_cfg["iters"] = 1
            strict_cfg["denoise"] = {"method": "oidn"}
            control = {
                "control_arm": "pixel2d",
                "cache": str(cache_path),
                "resolution_ladder": {"base_resolution": 4, "finest_resolution": 128},
                "command": "uv run python examples/r1_parity.py ...",
                "hardware": {"cpu_brand": "test"},
                "training_config": strict_cfg,
                "training_arms": [
                    {
                        "arm": "pixel2d",
                        "seed": seed,
                        "validation": [{"light": control_light, "psnr_db_vs_raw": 20.0}],
                    }
                    for seed in seeds
                ],
                "validation": {"fingerprints": fingerprints},
            }
            control_path = tmp_path / "control_report.json"
            control_path.write_text(json.dumps(control))

            argv = [
                "r1_kitchen_k1.py",
                "--cache",
                str(cache_path),
                "--control-report",
                str(control_path),
                "--out-dir",
                str(out_dir),
                "--seeds",
                *[str(seed) for seed in seeds],
                "--resolutions",
                "32",
                "--iters",
                "1",
                "--base-resolution",
                "4",
                "--gate",
                "per-seed",
                "--denoise-method",
                "oidn",
                "--gate-lights",
                str(gate_lights),
            ]

            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(runner.PathCache, "load", lambda path: _tiny_cache()),
                mock.patch.object(
                    runner, "build_frozen_validation_sets", spy_build_frozen_validation_sets
                ),
                mock.patch.object(runner, "train", fake_train),
                mock.patch.object(runner, "load_trained_model", fake_load_trained_model),
                mock.patch.object(runner, "evaluate_model", fake_evaluate_model),
                mock.patch.object(runner, "cache_vertex_support", fake_cache_vertex_support),
            ):
                runner.main()


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
