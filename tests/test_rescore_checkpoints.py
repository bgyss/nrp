"""Tests for the evaluation-only re-scorer (examples/rescore_checkpoints.py)."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    """Same file-location import the other examples/ runners' tests use."""
    spec = importlib.util.spec_from_file_location(
        "rescore_checkpoints", ROOT / "examples" / "rescore_checkpoints.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRescore(unittest.TestCase):
    def test_rescore_at_the_original_count_reproduces_the_committed_deltas(self):
        """The correctness test for the whole re-read: at n_gate_lights=12 the
        re-scorer must return the committed report's numbers exactly, because it is
        reading the same checkpoints against the same lights."""
        run_dir = ROOT / "out/r1-parity-kitchen-eq"
        if not (run_dir / "report.json").exists():
            self.skipTest("committed 8-seed Kitchen parity run not present")

        from nrp.path_cache import PathCache

        runner = load_runner()
        report = json.loads((run_dir / "report.json").read_text())
        cache = PathCache.load(str(ROOT / "out/kitchen/path_cache.npz"))
        got = runner.rescore(
            run_dir,
            cache,
            seeds=report["seeds"],
            arms=["pixel2d", *report["world_arms"]],
            control_arm=report["control_arm"],
            base_cfg=report["training_config"],
            n_gate_lights=12,
        )
        for arm in report["world_arms"]:
            expected = [
                float(
                    sum(r["delta_db"] for r in row["per_light_deltas"])
                    / len(row["per_light_deltas"])
                )
                for row in report["comparisons"][arm]
            ]
            got_deltas = got["arms"][arm]["per_seed_mean_delta_db"]
            self.assertEqual(len(got_deltas), len(expected))
            for a, b in zip(got_deltas, expected, strict=True):
                self.assertAlmostEqual(a, b, places=6)

    def test_redesign_rescore_at_the_original_count_reproduces_the_committed_rows(self):
        """The encoding-redesign campaign's own correctness guard.

        `frozen_lights` extends rather than replaces its draw, so one group of
        `N_EVAL_LIGHTS` lights is exactly the light set the committed run used: at
        `n_gate_lights=N_EVAL_LIGHTS` every re-scored row must equal the committed
        row to 6 decimal places. One seed at one rotation is enough to catch a wrong
        cache, a wrong camera, an unrotated evaluation light, or a mis-rebuilt
        occupancy table; the full five-seed three-rotation check (all 180 rows) is
        run out of band and recorded in docs/performance.md, because it costs ~95 s.
        """
        run_dir = ROOT / "out/r1-encoding-redesign"
        if not (run_dir / "report.json").exists():
            self.skipTest("committed encoding-redesign run not present")

        runner = load_runner()
        committed = json.loads((run_dir / "report.json").read_text())
        got = runner.rescore_encoding_redesign(
            run_dir,
            seeds=[0],
            arms=["world_sparse"],
            rotations=[90.0],
            n_gate_lights=runner.N_EVAL_LIGHTS,
        )
        expected = {
            (row["seed"], row["rotation_degrees"], row["camera"]): row
            for row in committed["arms"]["world_sparse"]["rows"]
        }
        rows = got["arms"]["world_sparse"]["rows"]
        self.assertEqual(len(rows), 4)
        for row in rows:
            want = expected[(row["seed"], row["rotation_degrees"], row["camera"])]
            self.assertEqual(row["baseline_camera"], want["baseline_camera"])
            for field in ("psnr_db", "baseline_psnr_db", "delta_db"):
                self.assertAlmostEqual(row[field], want[field], places=6)

    def test_redesign_light_groups_extend_the_committed_draw(self):
        """A larger held-out draw must start with the committed one, or the
        reproduction check above would be comparing different lights."""
        cache_path = ROOT / "out/r1-encoding-redesign/seed0/train0.npz"
        if not cache_path.exists():
            self.skipTest("committed encoding-redesign caches not present")

        from nrp.path_cache import PathCache

        runner = load_runner()
        cache = PathCache.load(str(cache_path))
        one = runner.redesign_light_groups(cache, 0, runner.N_EVAL_LIGHTS)
        many = runner.redesign_light_groups(cache, 0, 3 * runner.N_EVAL_LIGHTS)
        self.assertEqual(len(one), 1)
        self.assertEqual(len(many), 3)
        for first, later in zip(one[0], many[0], strict=True):
            self.assertEqual(list(first.center), list(later.center))
            self.assertEqual(first.radius, later.radius)
        with self.assertRaises(ValueError):
            runner.redesign_light_groups(cache, 0, runner.N_EVAL_LIGHTS + 1)

    def test_redesign_light_groups_rejects_nonpositive_n_gate_lights(self):
        """n_gate_lights=0 is a multiple of N_EVAL_LIGHTS, so the multiple-of-8 check
        alone lets it through; it must still be rejected before it reaches
        `_aggregate_groups` with zero groups (which raised IndexError)."""
        cache_path = ROOT / "out/r1-encoding-redesign/seed0/train0.npz"
        if not cache_path.exists():
            self.skipTest("committed encoding-redesign caches not present")

        from nrp.path_cache import PathCache

        runner = load_runner()
        cache = PathCache.load(str(cache_path))
        with self.assertRaises(ValueError):
            runner.redesign_light_groups(cache, 0, 0)
        with self.assertRaises(ValueError):
            runner.redesign_light_groups(cache, 0, -runner.N_EVAL_LIGHTS)

    def test_rescore_sweep_at_the_original_count_reproduces_the_committed_deltas(self):
        """rescore_sweep at n_gate_lights=12 must reproduce the committed K1 sweep's
        per-resolution per-seed deltas exactly, because it reloads the same
        checkpoints against the same held-out lights."""
        sweep_dir = ROOT / "out/r1-kitchen-parity-k1-eq"
        control_dir = ROOT / "out/r1-parity-kitchen-eq"
        if not (sweep_dir / "report.json").exists() or not (control_dir / "report.json").exists():
            self.skipTest("committed K1 sweep or control run not present")

        from nrp.path_cache import PathCache

        runner = load_runner()
        sweep_report = json.loads((sweep_dir / "report.json").read_text())
        control_report = json.loads((control_dir / "report.json").read_text())
        cache = PathCache.load(str(ROOT / "out/kitchen/path_cache.npz"))
        got = runner.rescore_sweep(
            sweep_dir,
            control_dir,
            cache,
            seeds=sweep_report["seeds"],
            resolutions=sweep_report["resolutions"],
            control_arm=sweep_report["control_arm"],
            base_cfg=control_report["training_config"],
            n_gate_lights=12,
        )
        for res_row in sweep_report["per_resolution"]:
            res = str(res_row["finest_resolution"])
            expected = res_row["gate"]["per_seed"]["per_seed_delta_db"]
            got_deltas = got["resolutions"][res]["per_seed"]
            self.assertEqual(len(got_deltas), len(expected))
            for a, b in zip(got_deltas, expected, strict=True):
                self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
