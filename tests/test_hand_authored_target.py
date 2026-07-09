import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.hand_authored_target import (  # noqa: E402
    ACCENT_RGB,
    BACKGROUND_RGB,
    hand_authored_strokes,
    render_hand_authored_target,
)


class HandAuthoredTargetTests(unittest.TestCase):
    def test_strokes_are_within_bounds_and_well_formed(self):
        strokes = hand_authored_strokes()
        self.assertGreater(len(strokes), 0)
        for r, c, rgb in strokes:
            self.assertGreaterEqual(r, 0)
            self.assertGreaterEqual(c, 0)
            self.assertLess(r, 14)
            self.assertLess(c, 14)
            self.assertEqual(len(rgb), 3)
            self.assertTrue(all(0.0 <= v <= 1.0 for v in rgb))

    def test_render_places_background_and_strokes_correctly(self):
        image = render_hand_authored_target()
        self.assertEqual(image.shape, (14, 14, 3))
        np.testing.assert_allclose(image[0, 0], BACKGROUND_RGB)
        for r, c, rgb in hand_authored_strokes():
            np.testing.assert_allclose(image[r, c], rgb)

    def test_accent_dot_present_and_distinct_from_plus(self):
        image = render_hand_authored_target()
        np.testing.assert_allclose(image[1, 11], ACCENT_RGB)
        self.assertFalse(np.allclose(image[1, 11], image[6, 6]))

    def test_cli_report_includes_provenance_and_realization_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    "examples/hand_authored_target.py",
                    "--out",
                    str(out),
                    "--steps",
                    "4",
                    "--restarts",
                    "1",
                ],
                check=True,
            )
            report = json.loads(out.read_text())
            provenance = json.loads(
                (Path(tmp) / "hand_authored_provenance.json").read_text()
            )
        self.assertTrue(provenance["generation"]["hand_authored"])
        self.assertFalse(provenance["generation"]["derived_from_render"])
        self.assertIn("target_vs_realized_psnr_db", report)
        self.assertIn("best", report)


if __name__ == "__main__":
    unittest.main()
