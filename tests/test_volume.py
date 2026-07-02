"""Volumetric path export and volume GATHERLIGHT (roadmap item 2, paper §3.1).

The decoupling insight under test: with free-flight sampling, transmittance is
implicit in the recorded segment lengths — P(segment reaches distance d) =
exp(-sigma_t * d) — so GATHERLIGHT needs no volume-specific code at all.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.toy_tracer import CAM_POS, sample_free_flight, trace_path_cache  # noqa: E402


class FreeFlightSamplingTests(unittest.TestCase):
    def test_lengths_follow_transmittance_ks(self):
        # Kolmogorov-Smirnov sanity check: sampled flight distances against the
        # exponential CDF 1 - exp(-sigma_t * t). 1.36/sqrt(n) is the 5% critical
        # value; the fixed seed makes the assertion deterministic.
        rng = np.random.default_rng(0)
        sigma_t = 3.0
        n = 20000
        d = np.sort(sample_free_flight(rng, sigma_t, n))
        cdf = 1.0 - np.exp(-sigma_t * d)
        empirical_hi = np.arange(1, n + 1) / n
        empirical_lo = np.arange(0, n) / n
        ks = max(np.abs(empirical_hi - cdf).max(), np.abs(empirical_lo - cdf).max())
        self.assertLess(ks, 1.36 / np.sqrt(n), f"KS statistic {ks:.5f}")

    def test_mean_flight_is_inverse_sigma(self):
        rng = np.random.default_rng(1)
        d = sample_free_flight(rng, 4.0, 50000)
        self.assertAlmostEqual(float(d.mean()), 0.25, delta=0.005)


class VolumeGatherTests(unittest.TestCase):
    def test_single_scatter_falloff_matches_transmittance(self):
        # Slab-style fixture: pure absorption (albedo 0), primary segments only.
        # The gathered value at the center pixel is Le * P(segment reaches the
        # sphere) which must follow exp(-sigma_t * entry_distance) within 5%.
        sigma_t = 2.0
        radius = 0.12
        for z_center in [0.35, 0.55, 0.75]:
            cache = trace_path_cache(
                9, 9, spp=4000, max_bounces=1, seed=3, medium={"sigma_t": sigma_t, "albedo": 0.0}
            )
            light = SphereLight(center=[0.5, 0.5, z_center], radius=radius, rgb=[1.0, 1.0, 1.0])
            got = float(gather_light(cache, light)[4, 4].mean())
            entry = (z_center - CAM_POS[2]) - radius
            want = float(np.exp(-sigma_t * entry))
            self.assertLess(
                abs(got - want) / want, 0.05, f"z={z_center}: got {got:.4f}, want {want:.4f}"
            )

    def test_scattering_illuminates_medium_beyond_absorption(self):
        # With single-scattering albedo > 0, continuation segments from scatter
        # vertices also cross the light: a light inside the medium illuminates it.
        light = SphereLight(center=[0.75, 0.6, 0.5], radius=0.1, rgb=[1.0, 1.0, 1.0])
        means = {}
        for albedo in [0.0, 0.9]:
            cache = trace_path_cache(
                24, 24, spp=300, max_bounces=3, seed=5, medium={"sigma_t": 2.0, "albedo": albedo}
            )
            means[albedo] = float(gather_light(cache, light).mean())
        self.assertGreater(means[0.9], means[0.0] * 1.5)

    def test_medium_shortens_segments_not_counts(self):
        # The toy tracer uses a fixed event budget, so the medium replaces surface
        # events with scatter events: same segment count, shorter mean segment.
        surf = trace_path_cache(12, 12, spp=20, max_bounces=3, seed=7)
        vol = trace_path_cache(
            12, 12, spp=20, max_bounces=3, seed=7, medium={"sigma_t": 2.0, "albedo": 0.8}
        )
        self.assertEqual(surf.segment_count, vol.segment_count)
        self.assertLess(
            float(vol.seg_tmax[np.isfinite(vol.seg_tmax)].mean()),
            float(surf.seg_tmax[np.isfinite(surf.seg_tmax)].mean()),
        )


class SchemaVersionTests(unittest.TestCase):
    def test_v1_npz_without_version_loads_with_no_medium(self):
        # A v1 cache written before the schema_version field existed must load
        # unchanged (backward compatibility requirement).
        cache = trace_path_cache(6, 6, spp=2, max_bounces=2, seed=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "v1.npz")
            np.savez_compressed(
                path,
                width=cache.width,
                height=cache.height,
                n_paths=cache.n_paths,
                seg_pixel=cache.seg_pixel,
                seg_origin=cache.seg_origin,
                seg_dir=cache.seg_dir,
                seg_tmax=cache.seg_tmax,
                seg_throughput=cache.seg_throughput,
                albedo=cache.albedo,
                position=cache.position,
                depth=cache.depth,
                normal=cache.normal,
            )
            loaded = PathCache.load(path)
        self.assertIsNone(loaded.medium)
        np.testing.assert_array_equal(loaded.seg_tmax, cache.seg_tmax)

    def test_v2_roundtrip_preserves_medium(self):
        medium = {"sigma_t": 2.5, "albedo": 0.7}
        cache = trace_path_cache(6, 6, spp=2, max_bounces=2, seed=0, medium=medium)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "v2.npz")
            cache.save(path)
            loaded = PathCache.load(path)
        self.assertEqual(loaded.medium, medium)

    def test_json_roundtrip_preserves_medium_and_defaults_to_none(self):
        medium = {"sigma_t": 1.5, "albedo": 0.4}
        cache = trace_path_cache(6, 6, spp=2, max_bounces=2, seed=0, medium=medium)
        again = PathCache.from_dict(cache.to_dict())
        self.assertEqual(again.medium, medium)
        d = trace_path_cache(6, 6, spp=2, max_bounces=2, seed=0).to_dict()
        del d["medium"], d["schema_version"]  # a hand-authored v1 dict
        self.assertIsNone(PathCache.from_dict(d).medium)

    def test_validate_rejects_bad_medium(self):
        cache = trace_path_cache(6, 6, spp=2, max_bounces=2, seed=0)
        cache.medium = {"sigma_t": 0.0, "albedo": 0.5}
        with self.assertRaises(ValueError):
            cache.validate()
        cache.medium = {"sigma_t": 1.0, "albedo": 1.5}
        with self.assertRaises(ValueError):
            cache.validate()


if __name__ == "__main__":
    unittest.main()
