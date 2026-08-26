"""Characterization tests pinning inherited R2 camera/rotation behaviour.

These functions arrive from codex/goal-implement-r2 and are reused by the
encoding redesign. They were not covered by dedicated unit tests before.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.r1_promotion import (  # noqa: E402
    out_of_bounds_fraction,
    rotation_matrix_y,
    transform_cache,
)
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.conditioned_multiview import (  # noqa: E402
    camera_direction,
    global_world_bounds,
)


def _tiny_cache() -> PathCache:
    return PathCache(
        width=2,
        height=1,
        n_paths=np.array([1, 1], dtype=np.int64),
        seg_pixel=np.array([0, 1], dtype=np.int64),
        seg_origin=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        seg_dir=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        seg_tmax=np.array([1.0, 1.0]),
        seg_throughput=np.ones((2, 3)),
        albedo=np.full((1, 2, 3), 0.5),
        depth=np.ones((1, 2)),
        normal=np.tile(np.array([0.0, 1.0, 0.0]), (1, 2, 1)),
        position=np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
    )


class TestRotationMatrix(unittest.TestCase):
    def test_is_orthonormal_and_right_handed(self):
        for degrees in (0.0, 90.0, 180.0, 37.5):
            r = rotation_matrix_y(degrees)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=12)

    def test_ninety_degrees_maps_x_axis_into_z(self):
        r = rotation_matrix_y(90.0)
        rotated = np.array([1.0, 0.0, 0.0]) @ r.T
        self.assertAlmostEqual(abs(float(rotated[2])), 1.0, places=12)
        self.assertAlmostEqual(float(rotated[1]), 0.0, places=12)


class TestTransformCache(unittest.TestCase):
    def test_identity_rotation_preserves_geometry(self):
        cache = _tiny_cache()
        out = transform_cache(cache, np.eye(3))
        np.testing.assert_allclose(out.position, cache.position)
        np.testing.assert_allclose(out.normal, cache.normal)
        np.testing.assert_allclose(out.seg_dir, cache.seg_dir)

    def test_rotation_preserves_segment_lengths_and_normal_unit_norm(self):
        cache = _tiny_cache()
        out = transform_cache(cache, rotation_matrix_y(90.0))
        np.testing.assert_allclose(out.seg_tmax, cache.seg_tmax)
        norms = np.linalg.norm(out.normal.reshape(-1, 3), axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)
        out.validate()

    def test_rejects_non_orthonormal_rotation(self):
        with self.assertRaises(ValueError):
            transform_cache(_tiny_cache(), np.full((3, 3), 2.0))


class TestOutOfBoundsFraction(unittest.TestCase):
    def test_all_inside_is_zero(self):
        positions = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        self.assertEqual(out_of_bounds_fraction(positions, bounds), 0.0)

    def test_half_outside_is_one_half(self):
        positions = np.array([[0.5, 0.5, 0.5], [9.0, 0.1, 0.1]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        self.assertAlmostEqual(out_of_bounds_fraction(positions, bounds), 0.5)


class TestCameraDirection(unittest.TestCase):
    def test_returns_unit_vector(self):
        d = camera_direction({"origin": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 4.0]})
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=12)

    def test_rejects_degenerate_direction(self):
        with self.assertRaises(ValueError):
            camera_direction({"origin": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0]})


class TestGlobalWorldBounds(unittest.TestCase):
    def test_bounds_enclose_every_cache(self):
        a = _tiny_cache()
        b = _tiny_cache()
        b.position = b.position + 3.0
        bounds = global_world_bounds([a, b])
        lo = np.asarray(bounds["min"])
        hi = np.asarray(bounds["max"])
        for cache in (a, b):
            p = cache.position.reshape(-1, 3)
            self.assertTrue(np.all(p >= lo - 1e-9))
            self.assertTrue(np.all(p <= hi + 1e-9))


if __name__ == "__main__":
    unittest.main()
