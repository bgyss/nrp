"""Camera-arc construction and report shape for the encoding redesign runner."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.r1_encoding_redesign import (  # noqa: E402
    camera_arc,
    nearest_trained_camera,
)
from nrp.metrics import psnr  # noqa: E402


class TestCameraArc(unittest.TestCase):
    def test_returns_the_requested_counts(self):
        trained, held_out = camera_arc(8, 4)
        self.assertEqual(len(trained), 8)
        self.assertEqual(len(held_out), 4)

    def test_held_out_cameras_are_never_trained_cameras(self):
        trained, held_out = camera_arc(8, 4)
        trained_origins = {tuple(np.round(c["origin"], 9)) for c in trained}
        for camera in held_out:
            self.assertNotIn(tuple(np.round(camera["origin"], 9)), trained_origins)

    def test_held_out_cameras_are_interpolated_not_extrapolated(self):
        trained, held_out = camera_arc(8, 4)
        # Every held-out camera must lie inside the convex hull of the trained arc
        # along each axis; extrapolation is R3's question, not this gate's.
        lo = np.min([c["origin"] for c in trained], axis=0)
        hi = np.max([c["origin"] for c in trained], axis=0)
        for camera in held_out:
            origin = np.asarray(camera["origin"])
            self.assertTrue(np.all(origin >= lo - 1e-9))
            self.assertTrue(np.all(origin <= hi + 1e-9))

    def test_every_camera_has_a_distinct_name(self):
        trained, held_out = camera_arc(8, 4)
        names = [c["name"] for c in trained + held_out]
        self.assertEqual(len(names), len(set(names)))

    def test_targets_are_inside_the_toy_box(self):
        trained, held_out = camera_arc(8, 4)
        for camera in trained + held_out:
            target = np.asarray(camera["target"])
            self.assertTrue(np.all(target > 0.0) and np.all(target < 1.0))


class TestNearestTrainedCamera(unittest.TestCase):
    def test_picks_the_closest_origin(self):
        trained = [
            {"name": "a", "origin": [0.0, 0.0, 0.0], "target": [0.5, 0.5, 0.5]},
            {"name": "b", "origin": [1.0, 0.0, 0.0], "target": [0.5, 0.5, 0.5]},
        ]
        held = {"name": "h", "origin": [0.9, 0.0, 0.0], "target": [0.5, 0.5, 0.5]}
        self.assertEqual(nearest_trained_camera(held, trained)["name"], "b")


class TestDeltaIsPeakInvariant(unittest.TestCase):
    """Pins the property this runner's fixed-peak change relies on: G1's comparative
    delta is peak-independent (the peak cancels in `10*log10(MSE_baseline/MSE_pred)`)
    even though the absolute PSNR values it's built from are not."""

    def test_delta_matches_across_two_peaks_while_absolute_psnr_differs(self):
        rng = np.random.default_rng(0)
        reference = rng.random((8, 8, 3)) + 0.1
        prediction = reference + rng.normal(scale=0.05, size=reference.shape)
        baseline = reference + rng.normal(scale=0.15, size=reference.shape)

        peak_a, peak_b = 1.0, 5.0
        self.assertNotEqual(peak_a, peak_b)

        pred_psnr_a = psnr(prediction, reference, peak=peak_a)
        base_psnr_a = psnr(baseline, reference, peak=peak_a)
        delta_a = pred_psnr_a - base_psnr_a

        pred_psnr_b = psnr(prediction, reference, peak=peak_b)
        base_psnr_b = psnr(baseline, reference, peak=peak_b)
        delta_b = pred_psnr_b - base_psnr_b

        # The absolute numbers really do move with the peak...
        self.assertNotAlmostEqual(pred_psnr_a, pred_psnr_b, places=6)
        self.assertNotAlmostEqual(base_psnr_a, base_psnr_b, places=6)
        # ...but the delta between them does not, which is what makes it safe for a
        # fixed campaign peak to change the absolute floor without touching G1's win.
        self.assertAlmostEqual(delta_a, delta_b, places=9)


if __name__ == "__main__":
    unittest.main()
