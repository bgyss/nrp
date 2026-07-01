import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light, gather_lights, gather_throughput_quad  # noqa: E402
from nrp.lights import (  # noqa: E402
    QuadLight,
    light_from_dict,
    quad_tangent_frame,
    segment_hits_quad,
)
from nrp.path_cache import PathCache  # noqa: E402


def hit(origin, direction, t_max, center, normal, width, height) -> bool:
    d = np.asarray(direction, dtype=np.float64)
    d = d / np.linalg.norm(d)
    return bool(
        segment_hits_quad(
            np.array([origin], dtype=np.float64),
            np.array([d]),
            np.array([t_max], dtype=np.float64),
            np.asarray(center, dtype=np.float64),
            np.asarray(normal, dtype=np.float64),
            width,
            height,
        )[0]
    )


class QuadTangentFrameTests(unittest.TestCase):
    def test_orthonormal_for_arbitrary_normals(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            n = rng.normal(size=3)
            n /= np.linalg.norm(n)
            u, v = quad_tangent_frame(n)
            for a, b in [(u, v), (u, n), (v, n)]:
                self.assertAlmostEqual(float(a @ b), 0.0, places=12)
            self.assertAlmostEqual(float(u @ u), 1.0, places=12)
            self.assertAlmostEqual(float(v @ v), 1.0, places=12)

    def test_deterministic(self):
        n = np.array([0.0, 1.0, 0.0])
        u1, v1 = quad_tangent_frame(n)
        u2, v2 = quad_tangent_frame(n)
        np.testing.assert_array_equal(u1, u2)
        np.testing.assert_array_equal(v1, v2)


class SegmentQuadTests(unittest.TestCase):
    # A unit quad at z=1 facing -z; its plane frame spans x/y.
    C = (0.0, 0.0, 1.0)
    N = (0.0, 0.0, -1.0)

    def test_center_crossing_hits(self):
        self.assertTrue(hit((0, 0, 0), (0, 0, 1), 2.0, self.C, self.N, 1.0, 1.0))

    def test_crossing_beyond_tmax_misses(self):
        self.assertFalse(hit((0, 0, 0), (0, 0, 1), 0.5, self.C, self.N, 1.0, 1.0))

    def test_escape_segment_hits(self):
        self.assertTrue(hit((0, 0, 0), (0, 0, 1), np.inf, self.C, self.N, 1.0, 1.0))

    def test_behind_origin_misses(self):
        self.assertFalse(hit((0, 0, 2), (0, 0, 1), np.inf, self.C, self.N, 1.0, 1.0))

    def test_outside_rectangle_misses(self):
        self.assertFalse(hit((0.6, 0, 0), (0, 0, 1), 2.0, self.C, self.N, 1.0, 1.0))
        self.assertTrue(hit((0.4, 0, 0), (0, 0, 1), 2.0, self.C, self.N, 1.0, 1.0))

    def test_rectangular_extents_are_respected(self):
        # width spans one in-plane axis, height the other; a point inside one but
        # outside the other must miss.
        wide = hit((0.4, 0.0, 0.0), (0, 0, 1), 2.0, self.C, self.N, 2.0, 0.1)
        tall = hit((0.4, 0.0, 0.0), (0, 0, 1), 2.0, self.C, self.N, 0.1, 2.0)
        self.assertNotEqual(wide, tall)

    def test_parallel_segment_misses(self):
        self.assertFalse(hit((0, 0, 1.0), (1, 0, 0), np.inf, self.C, self.N, 4.0, 4.0))

    def test_normal_orientation_is_irrelevant_for_hits(self):
        # A pure emitter has no facing side: flipping the normal flips the tangent
        # frame but the hit set is identical for a symmetric rectangle.
        self.assertTrue(hit((0, 0, 0), (0, 0, 1), 2.0, self.C, (0, 0, 1), 1.0, 1.0))


class GatherQuadTests(unittest.TestCase):
    def micro_cache(self) -> PathCache:
        # 1x2 image; pixel 0 has one segment crossing the quad, pixel 1 misses it.
        return PathCache(
            width=2,
            height=1,
            n_paths=np.array([1, 1]),
            seg_pixel=np.array([0, 1]),
            seg_origin=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            seg_dir=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            seg_tmax=np.array([2.0, 2.0]),
            seg_throughput=np.array([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]]),
            albedo=np.zeros((1, 2, 3)),
            position=np.zeros((1, 2, 3)),
            depth=np.zeros((1, 2)),
            normal=np.zeros((1, 2, 3)),
        )

    def test_gather_throughput_quad(self):
        cache = self.micro_cache()
        img = gather_throughput_quad(cache, (0, 0, 1), (0, 0, -1), 1.0, 1.0)
        np.testing.assert_allclose(img[0, 0], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(img[0, 1], [0.0, 0.0, 0.0])

    def test_gather_light_scales_by_rgb(self):
        cache = self.micro_cache()
        light = QuadLight(center=(0, 0, 1), normal=(0, 0, -1), width=1.0, height=1.0, rgb=(2, 4, 8))
        img = gather_light(cache, light)
        np.testing.assert_allclose(img[0, 0], [1.0, 2.0, 4.0])

    def test_gather_lights_sums_linearly(self):
        cache = self.micro_cache()
        a = QuadLight(center=(0, 0, 1), normal=(0, 0, -1), width=1.0, height=1.0, rgb=(1, 1, 1))
        b = QuadLight(center=(0, 0, 1), normal=(0, 0, -1), width=1.0, height=1.0, rgb=(2, 2, 2))
        img = gather_lights(cache, [a, b])
        np.testing.assert_allclose(img, gather_light(cache, a) + gather_light(cache, b))


class LightFromDictTests(unittest.TestCase):
    def test_dispatch(self):
        sphere = light_from_dict({"center": [0, 0, 0], "radius": 0.5})
        quad = light_from_dict(
            {"type": "quad", "center": [0, 0, 0], "normal": [0, 0, 1], "width": 1, "height": 2}
        )
        self.assertEqual(sphere.__class__.__name__, "SphereLight")
        self.assertEqual(quad.__class__.__name__, "QuadLight")

    def test_quad_roundtrip(self):
        q = QuadLight(center=(1, 2, 3), normal=(0, 1, 0), width=0.5, height=0.25, rgb=(1, 2, 3))
        q2 = light_from_dict(q.to_dict())
        np.testing.assert_allclose(q2.center, q.center)
        np.testing.assert_allclose(q2.normal, q.normal)
        self.assertEqual((q2.width, q2.height), (q.width, q.height))


if __name__ == "__main__":
    unittest.main()
