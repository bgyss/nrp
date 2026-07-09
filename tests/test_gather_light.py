"""NRP M2 exit criteria: reference GATHERLIGHT returns expected per-pixel
contributions on hand-authored caches, covering no-hit, one-hit, multiple-hit, radius
change, intensity change, and occluded/undersampled cases."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import (  # noqa: E402
    GatherControls,
    gather_light,
    gather_light_controlled,
    gather_throughput,
    undersampled_mask,
)
from nrp.lights import EnvironmentLight, QuadLight, SphereLight, TexturedQuadLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402

TOL = 1e-12


def make_cache(segments: list[dict], n_paths: list[int], width: int = 2, height: int = 1):
    """segments: list of {pixel, origin, dir, tmax, throughput}; dirs normalized here."""
    if segments:
        dirs = np.array([s["dir"] for s in segments], dtype=np.float64)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        origin = np.array([s["origin"] for s in segments], dtype=np.float64)
        tmax = np.array([s["tmax"] for s in segments], dtype=np.float64)
        thr = np.array([s["throughput"] for s in segments], dtype=np.float64)
        pix = np.array([s["pixel"] for s in segments], dtype=np.int64)
    else:
        dirs = np.zeros((0, 3))
        origin = np.zeros((0, 3))
        tmax = np.zeros(0)
        thr = np.zeros((0, 3))
        pix = np.zeros(0, dtype=np.int64)
    return PathCache(
        width=width,
        height=height,
        n_paths=np.asarray(n_paths, dtype=np.int64),
        seg_pixel=pix,
        seg_origin=origin,
        seg_dir=dirs,
        seg_tmax=tmax,
        seg_throughput=thr,
        albedo=np.full((height, width, 3), 0.5),
        position=np.zeros((height, width, 3)),
        depth=np.ones((height, width)),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (height, width, 1)),
    )


# A segment from the origin along +x, long enough to reach x=5.
SEG = {
    "pixel": 0,
    "origin": [0, 0, 0],
    "dir": [1, 0, 0],
    "tmax": 10.0,
    "throughput": [0.5, 0.4, 0.3],
}
LIGHT_ON_X = SphereLight(center=[5.0, 0.0, 0.0], radius=1.0, rgb=[1.0, 1.0, 1.0])


class GatherLightTests(unittest.TestCase):
    def test_no_hit_returns_zero(self):
        cache = make_cache([SEG], n_paths=[1, 1])
        light = SphereLight(center=[5.0, 5.0, 0.0], radius=1.0)  # off-axis, missed
        np.testing.assert_allclose(gather_light(cache, light), 0.0, atol=TOL)

    def test_one_hit_returns_throughput_over_n_paths(self):
        cache = make_cache([SEG], n_paths=[2, 1])  # pixel 0 traced with 2 paths
        image = gather_light(cache, LIGHT_ON_X)
        np.testing.assert_allclose(image[0, 0], np.array([0.5, 0.4, 0.3]) / 2.0, atol=TOL)
        np.testing.assert_allclose(image[0, 1], 0.0, atol=TOL)

    def test_multiple_hits_accumulate(self):
        seg2 = dict(SEG, throughput=[0.1, 0.1, 0.1])
        cache = make_cache([SEG, seg2], n_paths=[1, 1])
        image = gather_light(cache, LIGHT_ON_X)
        np.testing.assert_allclose(image[0, 0], [0.6, 0.5, 0.4], atol=TOL)

    def test_radius_change_flips_contribution(self):
        # Segment passes 0.8 units from the light center: r=0.5 misses, r=1.0 hits.
        seg = dict(SEG, origin=[0, 0.8, 0])
        cache = make_cache([seg], n_paths=[1, 1])
        small = SphereLight(center=[5.0, 0.0, 0.0], radius=0.5)
        big = SphereLight(center=[5.0, 0.0, 0.0], radius=1.0)
        np.testing.assert_allclose(gather_light(cache, small), 0.0, atol=TOL)
        self.assertGreater(gather_light(cache, big)[0, 0].sum(), 0.0)

    def test_intensity_scales_linearly(self):
        cache = make_cache([SEG], n_paths=[1, 1])
        base = gather_light(cache, LIGHT_ON_X)
        doubled = gather_light(
            cache, SphereLight(center=[5.0, 0.0, 0.0], radius=1.0, rgb=[2.0, 2.0, 2.0])
        )
        np.testing.assert_allclose(doubled, 2.0 * base, atol=TOL)

    def test_occluded_segment_does_not_reach_light(self):
        # Same geometry as SEG but the segment terminates at t=2 (a blocker) before
        # the light region at x in [4, 6] -> no contribution.
        occluded = dict(SEG, tmax=2.0)
        cache = make_cache([occluded], n_paths=[1, 1])
        np.testing.assert_allclose(gather_light(cache, LIGHT_ON_X), 0.0, atol=TOL)

    def test_undersampled_pixel_is_zero_and_flagged(self):
        cache = make_cache([SEG], n_paths=[1, 0])  # pixel 1 has zero paths
        image = gather_light(cache, LIGHT_ON_X)
        np.testing.assert_allclose(image[0, 1], 0.0, atol=TOL)
        mask = undersampled_mask(cache)
        self.assertFalse(mask[0, 0])
        self.assertTrue(mask[0, 1])

    def test_gather_throughput_is_pre_emission_scaling(self):
        cache = make_cache([SEG], n_paths=[1, 1])
        thr = gather_throughput(cache, LIGHT_ON_X.center, LIGHT_ON_X.radius)
        rgb = np.array([3.0, 2.0, 1.0])
        image = gather_light(cache, SphereLight(center=[5.0, 0.0, 0.0], radius=1.0, rgb=rgb))
        np.testing.assert_allclose(image, thr * rgb, atol=TOL)

    def test_constant_textured_quad_matches_quad_light(self):
        seg = {
            "pixel": 0,
            "origin": [0, 0, 0],
            "dir": [0, 0, 1],
            "tmax": 2.0,
            "throughput": [0.5, 0.4, 0.3],
        }
        cache = make_cache([seg], n_paths=[2, 1])
        rgb = np.array([2.0, 1.0, 0.5])
        quad = QuadLight(
            center=[0, 0, 1],
            normal=[0, 0, -1],
            width=2.0,
            height=2.0,
            rgb=rgb,
        )
        textured = TexturedQuadLight(
            center=[0, 0, 1],
            normal=[0, 0, -1],
            width=2.0,
            height=2.0,
            texture=np.tile(rgb, (4, 4, 1)),
        )
        np.testing.assert_allclose(
            gather_light(cache, textured),
            gather_light(cache, quad),
            atol=TOL,
        )

    def test_constant_environment_gathers_escaped_segments(self):
        escaped = dict(SEG, tmax=np.inf, throughput=[0.2, 0.4, 0.6])
        blocked = dict(SEG, pixel=1, tmax=3.0, throughput=[9.0, 9.0, 9.0])
        cache = make_cache([escaped, blocked], n_paths=[2, 1])
        coeffs = np.zeros((9, 3))
        coeffs[0] = [3.0, 2.0, 1.0]
        image = gather_light(cache, EnvironmentLight(coeffs))
        np.testing.assert_allclose(image[0, 0], [0.3, 0.4, 0.3], atol=TOL)
        np.testing.assert_allclose(image[0, 1], 0.0, atol=TOL)

    def test_gather_time_linking_excludes_owned_pixels(self):
        pixel1 = dict(SEG, pixel=1, throughput=[0.1, 0.2, 0.3])
        cache = make_cache([SEG, pixel1], n_paths=[1, 1])
        mask = np.array([[True, False]])
        linked = gather_light_controlled(
            cache,
            LIGHT_ON_X,
            GatherControls(exclude_pixel_mask=mask),
        )
        np.testing.assert_allclose(linked[0, 0], 0.0, atol=TOL)
        np.testing.assert_allclose(linked[0, 1], [0.1, 0.2, 0.3], atol=TOL)

    def test_linear_distance_attenuation_fixture(self):
        cache = make_cache([SEG], n_paths=[1, 1])
        # Segment origin is distance 5 from LIGHT_ON_X. intercept 2 + slope -0.2
        # gives a 1.0 multiplier, while intercept 1 + slope -0.1 gives 0.5.
        half = gather_light_controlled(
            cache,
            LIGHT_ON_X,
            GatherControls(
                attenuation={"type": "linear_distance", "intercept": 1.0, "slope": -0.1}
            ),
        )
        np.testing.assert_allclose(half[0, 0], np.array([0.5, 0.4, 0.3]) * 0.5, atol=TOL)


if __name__ == "__main__":
    unittest.main()
