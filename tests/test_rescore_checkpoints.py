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
            for a, b in zip(got["arms"][arm]["per_seed_mean_delta_db"], expected, strict=False):
                self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
