import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import SphereLight, segment_hits_sphere  # noqa: E402


class SegmentSphereTests(unittest.TestCase):
    def hit(self, origin, direction, t_max, center=(0.0, 0.0, 0.0), radius=1.0) -> bool:
        d = np.asarray(direction, dtype=np.float64)
        d = d / np.linalg.norm(d)
        return bool(
            segment_hits_sphere(
                np.array([origin], dtype=np.float64),
                np.array([d]),
                np.array([t_max], dtype=np.float64),
                np.asarray(center, dtype=np.float64),
                radius,
            )[0]
        )

    def test_straight_through_hit(self):
        self.assertTrue(self.hit((-3, 0, 0), (1, 0, 0), 10.0))

    def test_miss_offset_ray(self):
        self.assertFalse(self.hit((-3, 2, 0), (1, 0, 0), 10.0))

    def test_segment_too_short(self):
        # Sphere entry is at t=2; segment ends before it.
        self.assertFalse(self.hit((-3, 0, 0), (1, 0, 0), 1.5))

    def test_sphere_behind_origin(self):
        self.assertFalse(self.hit((3, 0, 0), (1, 0, 0), 10.0))

    def test_origin_inside_sphere_counts_as_hit(self):
        self.assertTrue(self.hit((0.2, 0, 0), (1, 0, 0), 10.0))

    def test_escape_segment_infinite_tmax(self):
        self.assertTrue(self.hit((-3, 0, 0), (1, 0, 0), np.inf))

    def test_radius_change_flips_result(self):
        self.assertFalse(self.hit((-3, 0.5, 0), (1, 0, 0), 10.0, radius=0.3))
        self.assertTrue(self.hit((-3, 0.5, 0), (1, 0, 0), 10.0, radius=0.8))

    def test_vectorized_matches_scalar(self):
        rng = np.random.default_rng(7)
        origins = rng.normal(size=(64, 3)) * 3.0
        dirs = rng.normal(size=(64, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        t_max = rng.random(64) * 8.0
        batched = segment_hits_sphere(origins, dirs, t_max, np.zeros(3), 1.0)
        for i in range(64):
            single = segment_hits_sphere(
                origins[i : i + 1], dirs[i : i + 1], t_max[i : i + 1], np.zeros(3), 1.0
            )[0]
            self.assertEqual(bool(batched[i]), bool(single))


class SphereLightTests(unittest.TestCase):
    def test_round_trip(self):
        light = SphereLight(center=[0.5, 0.4, 0.3], radius=0.1, rgb=[2.0, 1.0, 0.5])
        again = SphereLight.from_dict(light.to_dict())
        np.testing.assert_allclose(again.center, light.center)
        np.testing.assert_allclose(again.rgb, light.rgb)
        self.assertEqual(again.radius, light.radius)

    def test_rejects_nonpositive_radius(self):
        with self.assertRaises(ValueError):
            SphereLight(center=[0, 0, 0], radius=0.0)


if __name__ == "__main__":
    unittest.main()
