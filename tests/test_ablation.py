"""Ablation runner (roadmap item 10): one command produces a deterministic report
with every cell's config embedded and all four metrics per cell."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location("ablation", ROOT / "examples" / "ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AblationRunnerTests(unittest.TestCase):
    def test_tiny_toy_run_produces_complete_report(self):
        ablation = load_runner()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            argv = [
                "ablation.py",
                "--out",
                str(out),
                "--producer",
                "toy",
                "--width",
                "12",
                "--height",
                "12",
                "--spp",
                "4",
                "--variants",
                "none",
                "aux_enc_den",
                "--iters",
                "40",
                "--pool-size",
                "4",
                "--batch-pixels",
                "256",
                "--n-val",
                "2",
                "--ref-spp",
                "8",
                "--denoise-method",
                "bilateral",
            ]
            with mock.patch.object(sys, "argv", argv):
                ablation.main()
            report = json.loads(out.read_text())

        self.assertEqual(report["spp"], [4])
        self.assertEqual(report["variants"], ["none", "aux_enc_den"])
        cells = report["cells"]["4"]
        for name in ("none", "aux_enc_den"):
            cell = cells[name]
            # Every cell embeds its full training config for reproducibility.
            self.assertEqual(cell["config"]["iters"], 40)
            self.assertEqual(cell["config"]["seed"], report["seed"])
            self.assertIn("use_aux", cell["config"]["model"])
            # All four paper metrics, per light and aggregated.
            for metric in ("psnr_db", "smape", "ssim", "flip"):
                self.assertEqual(len(cell[f"{metric}_per_light"]), 2)
                self.assertIn(f"{metric}_mean", cell)
            self.assertTrue(0.0 <= cell["flip_mean"] <= 1.0)
            self.assertTrue(-1.0 <= cell["ssim_mean"] <= 1.0)
        # The two variants really trained different architectures.
        self.assertFalse(cells["none"]["config"]["model"]["use_aux"])
        self.assertTrue(cells["aux_enc_den"]["config"]["model"]["use_aux"])
        self.assertNotEqual(
            cells["none"]["parameter_count"], cells["aux_enc_den"]["parameter_count"]
        )


if __name__ == "__main__":
    unittest.main()
