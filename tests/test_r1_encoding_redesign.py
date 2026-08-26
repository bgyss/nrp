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


if __name__ == "__main__":
    unittest.main()
