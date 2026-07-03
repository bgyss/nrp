import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.generative_loop import fixture_masks, masked_psnr, write_provenance  # noqa: E402


class GenerativeLoopTests(unittest.TestCase):
    def test_fixture_masks_have_objective_and_protect_regions(self):
        objective, protect = fixture_masks(12, 10)
        self.assertEqual(objective.shape, (12, 10))
        self.assertEqual(protect.shape, (12, 10))
        self.assertGreater(float(objective.max()), 1.0)
        self.assertEqual(float(protect[:8].sum()), 0.0)
        self.assertGreater(float(protect[9:].sum()), 0.0)

    def test_masked_psnr_uses_only_selected_pixels(self):
        a = np.zeros((2, 2, 3))
        b = np.zeros((2, 2, 3))
        b[0, 0] = 10.0
        mask = np.array([[False, True], [False, False]])
        self.assertGreater(masked_psnr(a, b, mask), 300.0)

    def test_cli_report_includes_required_latency_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    "examples/generative_loop.py",
                    "--out",
                    str(out),
                    "--width",
                    "8",
                    "--height",
                    "8",
                    "--steps",
                    "4",
                ],
                check=True,
            )
            report = json.loads(out.read_text())
        self.assertEqual(
            [row["pixel_fraction"] for row in report["latency_sweep"]],
            [1.0, 0.25, 0.05],
        )
        self.assertTrue(all(row["wall_ms"] > 0.0 for row in report["latency_sweep"]))
        self.assertEqual(report["outputs"]["provenance"], "provenance.json")
        self.assertEqual(report["provenance"]["external_generator"], None)

    def test_write_provenance_hashes_files_and_records_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "target.npy").write_bytes(b"target")
            write_provenance(
                base,
                {"target": "target.npy"},
                {"width": 2, "height": 2},
            )
            written = json.loads((base / "provenance.json").read_text())
        expected_hash = hashlib.sha256(b"target").hexdigest()
        self.assertEqual(written["files"]["target"]["sha256"], expected_hash)
        self.assertEqual(
            written["generation"]["method"],
            "deterministic repo-local numpy fixture generation",
        )


if __name__ == "__main__":
    unittest.main()
