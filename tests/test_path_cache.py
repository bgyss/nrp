"""NRP M1 exit criteria: serialization round-trips, a tiny hand-authored cache loads
without renderer dependencies, and the schema supports sphere-light queries."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402


def tiny_cache() -> PathCache:
    """Hand-authored 2x1 cache: pixel 0 has one path of two segments (the second an
    escape ray), pixel 1 has one single-segment path."""
    return PathCache(
        width=2,
        height=1,
        n_paths=np.array([1, 1], dtype=np.int64),
        seg_pixel=np.array([0, 0, 1], dtype=np.int64),
        seg_origin=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        seg_dir=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64),
        seg_tmax=np.array([1.0, np.inf, 2.0]),
        seg_throughput=np.array([[1, 1, 1], [0.5, 0.5, 0.5], [0.8, 0.2, 0.1]]),
        albedo=np.full((1, 2, 3), 0.5),
        position=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]]]),
        depth=np.array([[1.0, 2.0]]),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, 2, 1)),
    )


class PathCacheTests(unittest.TestCase):
    def test_hand_authored_cache_validates(self):
        tiny_cache().validate()

    def test_npz_round_trip(self):
        cache = tiny_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cache.npz")
            cache.save(path)
            again = PathCache.load(path)
        np.testing.assert_allclose(again.seg_origin, cache.seg_origin)
        np.testing.assert_allclose(again.seg_throughput, cache.seg_throughput)
        np.testing.assert_allclose(again.seg_tmax, cache.seg_tmax)
        np.testing.assert_array_equal(again.seg_pixel, cache.seg_pixel)
        np.testing.assert_allclose(again.albedo, cache.albedo)

    def test_sharded_round_trip_restores_monolithic_cache(self):
        cache = tiny_cache()
        light = SphereLight(center=[1.0, 3.0, 0.0], radius=0.5, rgb=[1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as tmp:
            cache.save_sharded(str(Path(tmp) / "shards"), tile_size=1)
            again = PathCache.load_sharded(str(Path(tmp) / "shards"))
        np.testing.assert_array_equal(again.seg_pixel, cache.seg_pixel)
        np.testing.assert_allclose(again.seg_origin, cache.seg_origin)
        np.testing.assert_allclose(again.seg_dir, cache.seg_dir)
        np.testing.assert_allclose(again.seg_tmax, cache.seg_tmax)
        np.testing.assert_allclose(again.seg_throughput, cache.seg_throughput)
        np.testing.assert_allclose(again.albedo, cache.albedo)
        np.testing.assert_allclose(gather_light(again, light), gather_light(cache, light))

    def test_json_round_trip_preserves_escape_segments(self):
        cache = tiny_cache()
        again = PathCache.from_dict(cache.to_dict())
        self.assertTrue(np.isinf(again.seg_tmax[1]))
        np.testing.assert_allclose(again.seg_tmax[[0, 2]], cache.seg_tmax[[0, 2]])
        np.testing.assert_allclose(again.seg_dir, cache.seg_dir)

    def test_validate_rejects_bad_shapes_and_values(self):
        cache = tiny_cache()
        cache.seg_pixel = np.array([0, 0, 5], dtype=np.int64)  # out of range
        with self.assertRaises(ValueError):
            cache.validate()
        cache = tiny_cache()
        cache.seg_dir[0] = [2.0, 0.0, 0.0]  # not unit length
        with self.assertRaises(ValueError):
            cache.validate()
        cache = tiny_cache()
        cache.seg_tmax[0] = -1.0
        with self.assertRaises(ValueError):
            cache.validate()

    def test_supports_sphere_light_query(self):
        # The escape segment of pixel 0 (origin (1,0,0), dir +y) passes through a
        # sphere at (1,3,0) r=0.5 -> pixel 0 accumulates its throughput.
        cache = tiny_cache()
        light = SphereLight(center=[1.0, 3.0, 0.0], radius=0.5, rgb=[1.0, 1.0, 1.0])
        image = gather_light(cache, light)
        np.testing.assert_allclose(image[0, 0], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(image[0, 1], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
