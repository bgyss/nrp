"""Packed cache layout (§4.2): rgb9e5 encode/decode properties and fp16+rgb9e5
`.npz` round-trips whose GATHERLIGHT images stay within 0.5 dB of the float64 cache."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.rgb9e5 import MAX_RGB9E5, rgb9e5_decode, rgb9e5_encode  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

MITSUBA_CACHE = str(Path(__file__).resolve().parent.parent / "out" / "mitsuba" / "path_cache.npz")


class Rgb9e5Tests(unittest.TestCase):
    def test_round_trip_relative_error_bound(self):
        # Log-uniform sweep across the whole normal range. The dominant channel
        # carries a full 9-bit mantissa, so its relative error is <= 2^-9 after
        # round-to-nearest; the other channels share its exponent, so their
        # *absolute* error is bounded by half an ulp at the dominant scale.
        rng = np.random.default_rng(0)
        mag = 2.0 ** rng.uniform(-14.0, np.log2(MAX_RGB9E5), size=(20000, 1))
        rgb = mag * rng.uniform(0.0, 1.0, size=(20000, 3))
        dec = rgb9e5_decode(rgb9e5_encode(rgb))
        max_c = rgb.max(axis=-1)
        self.assertLessEqual(np.max(np.abs(dec - rgb) / max_c[:, None]), 2.0**-9)
        dom = np.take_along_axis(rgb, rgb.argmax(axis=-1)[:, None], axis=-1)[:, 0]
        dom_dec = np.take_along_axis(dec, rgb.argmax(axis=-1)[:, None], axis=-1)[:, 0]
        self.assertLessEqual(np.max(np.abs(dom_dec - dom) / dom), 2.0**-9)

    def test_zeros_negatives_nan_and_overflow(self):
        rgb = np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, -0.5, 0.0],
                [np.nan, 1.0, 2.0],
                [1e6, MAX_RGB9E5 * 2, 3.0],
            ]
        )
        dec = rgb9e5_decode(rgb9e5_encode(rgb))
        np.testing.assert_array_equal(dec[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(dec[1], [0.0, 0.0, 0.0])
        self.assertEqual(dec[2, 0], 0.0)
        np.testing.assert_allclose(dec[2, 1:], [1.0, 2.0], rtol=2.0**-9)
        np.testing.assert_allclose(dec[3, :2], MAX_RGB9E5, rtol=2.0**-9)

    def test_denormal_range(self):
        # Below 2^-15 the exponent clamps and the format becomes fixed point with
        # step 2^-24: everything representable to within half a step, and values
        # under half the smallest step flush to zero.
        step = 2.0**-24
        rgb = np.array(
            [
                [step, 3.4 * step, 200.0 * step],
                [0.2 * step, 0.6 * step, 0.0],
            ]
        )
        dec = rgb9e5_decode(rgb9e5_encode(rgb))
        self.assertLessEqual(np.max(np.abs(dec - rgb)), step / 2)
        self.assertEqual(dec[1, 0], 0.0)  # rounds down to zero
        self.assertEqual(dec[1, 1], step)  # rounds up to one step

    def test_exponent_bump_on_mantissa_overflow(self):
        # A value just below a power of two rounds its mantissa up to 512, which
        # must bump the shared exponent instead of wrapping.
        rgb = np.array([[1.0 - 2**-11, 0.5, 0.25]])
        dec = rgb9e5_decode(rgb9e5_encode(rgb))
        np.testing.assert_allclose(dec[0], rgb[0], atol=2.0**-10)


class PackedCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(16, 16, 8, 3, seed=3)

    def round_trip(self, cache: PathCache) -> PathCache:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "packed.npz")
            cache.save(path, compressed=True)
            self.packed_bytes = os.path.getsize(path)
            return PathCache.load(path)

    def test_packed_round_trip_validates_and_preserves_structure(self):
        again = self.round_trip(self.cache)
        again.validate()
        self.assertEqual(again.segment_count, self.cache.segment_count)
        np.testing.assert_array_equal(again.seg_pixel, self.cache.seg_pixel)
        np.testing.assert_array_equal(np.isinf(again.seg_tmax), np.isinf(self.cache.seg_tmax))
        self.assertTrue(np.all(again.seg_tmax > 0))
        np.testing.assert_allclose(np.linalg.norm(again.seg_dir, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(again.seg_origin, self.cache.seg_origin, atol=2e-2)

    def test_packed_gather_within_half_db_of_float64(self):
        again = self.round_trip(self.cache)
        light = SphereLight(center=[0.3, 0.9, -0.4], radius=0.3, rgb=[4.0, 3.0, 2.0])
        ref = gather_light(self.cache, light)
        packed = gather_light(again, light)
        # "within 0.5 dB" means the packed image is a >=~40 dB-faithful copy of the
        # float64 image, so any downstream PSNR against a common reference moves
        # by far less than 0.5 dB.
        self.assertGreaterEqual(psnr(packed, ref), 40.0)

    def test_packed_cache_is_smaller(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = str(Path(tmp) / "full.npz")
            packed = str(Path(tmp) / "packed.npz")
            self.cache.save(full)
            self.cache.save(packed, compressed=True)
            self.assertLess(os.path.getsize(packed) * 2, os.path.getsize(full))

    def test_packed_sharded_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            full_dir = Path(tmp) / "full"
            packed_dir = Path(tmp) / "packed"
            self.cache.save_sharded(str(full_dir), tile_size=4)
            self.cache.save_sharded(str(packed_dir), tile_size=4, packed=True)
            again = PathCache.load_sharded(str(packed_dir))
            light = SphereLight(center=[0.3, 0.9, -0.4], radius=0.3, rgb=[4.0, 3.0, 2.0])
            self.assertGreaterEqual(
                psnr(gather_light(again, light), gather_light(self.cache, light)), 40.0
            )
            full_bytes = sum(p.stat().st_size for p in full_dir.glob("*.npz"))
            packed_bytes = sum(p.stat().st_size for p in packed_dir.glob("*.npz"))
            self.assertLess(packed_bytes, full_bytes)

    def test_medium_metadata_survives_packing(self):
        cache = trace_path_cache(8, 8, 4, 3, seed=5, medium={"sigma_t": 0.4, "albedo": 0.8})
        again = self.round_trip(cache)
        self.assertIsNotNone(again.medium)
        self.assertAlmostEqual(again.medium["sigma_t"], 0.4)
        self.assertAlmostEqual(again.medium["albedo"], 0.8)

    @unittest.skipUnless(os.path.exists(MITSUBA_CACHE), "no exported Mitsuba cache in out/")
    def test_mitsuba_cache_packed_gather_within_half_db(self):
        cache = PathCache.load(MITSUBA_CACHE)
        again = self.round_trip(cache)
        rng = np.random.default_rng(11)
        for _ in range(5):
            center = rng.uniform([-0.6, 0.2, -0.6], [0.6, 1.6, 0.6])
            light = SphereLight(center=center, radius=0.2, rgb=[5.0, 5.0, 5.0])
            ref = gather_light(cache, light)
            packed = gather_light(again, light)
            self.assertGreaterEqual(psnr(packed, ref), 40.0)


if __name__ == "__main__":
    unittest.main()
