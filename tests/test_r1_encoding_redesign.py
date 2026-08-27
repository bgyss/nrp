"""Camera-arc construction and report shape for the encoding redesign runner."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.r1_encoding_redesign import (  # noqa: E402
    ARM_ENCODING_CONFIG,
    ARM_NAMES,
    camera_arc,
    campaign_peak,
    nearest_trained_camera,
    rotated_light,
    rotated_lights,
)
from examples.r1_promotion import rotation_matrix_y, transform_cache  # noqa: E402
from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import QuadLight, SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.encoder_registry import encoder_schedule_params  # noqa: E402


def _cache_with_throughput(value: float) -> PathCache:
    """A one-path-per-pixel cache whose sole segment carries a known, constant
    throughput. With a sphere light of rgb=(1,1,1) covering both segments, GATHERLIGHT
    reduces to exactly `throughput` per pixel (n_paths=1, full hit, no attenuation) --
    so the expected peak below is derived independently of `gather_lights`/`campaign_peak`.
    """
    return PathCache(
        width=2,
        height=1,
        n_paths=np.array([1, 1], dtype=np.int64),
        seg_pixel=np.array([0, 1], dtype=np.int64),
        seg_origin=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        seg_dir=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        seg_tmax=np.array([1.0, 1.0]),
        seg_throughput=np.full((2, 3), value),
        albedo=np.full((1, 2, 3), 0.5),
        depth=np.ones((1, 2)),
        normal=np.tile(np.array([0.0, 1.0, 0.0]), (1, 2, 1)),
        position=np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
    )


_COVERING_LIGHT = [SphereLight(center=[0.5, 0.0, 0.5], radius=10.0, rgb=[1.0, 1.0, 1.0])]


class TestCampaignPeak(unittest.TestCase):
    """`campaign_peak` must reduce over TRAINED-view references only: held-out
    references never influence the scale held-out cameras are judged against."""

    def test_peak_comes_from_trained_cache_not_held_out(self):
        trained_cache = _cache_with_throughput(1.0)
        held_out_cache = _cache_with_throughput(5.0)  # deliberately the larger max

        peak = campaign_peak([trained_cache], _COVERING_LIGHT, seed=0)

        # Derived independently: throughput=1.0, rgb=1.0, one segment per pixel, full
        # hit -> every pixel's GATHERLIGHT value is exactly 1.0, so max == 1.0.
        self.assertAlmostEqual(peak, 1.0, places=12)
        # Not the held-out cache's max (5.0), not a mean of the two (3.0), not the
        # first cache in some other ordering -- pinning the exact reduction.
        self.assertNotAlmostEqual(peak, 5.0, places=6)
        self.assertNotAlmostEqual(peak, 3.0, places=6)
        del held_out_cache  # only used to prove it must NOT be passed in below

    def test_widening_the_reduction_to_include_held_out_would_change_the_answer(self):
        # Sanity check on the fixture itself: if a future refactor fed
        # `trained_caches + held_out_caches` into the same max-reduction, the peak
        # would change (to 5.0), which is exactly the regression FIX 1 guards against.
        trained_cache = _cache_with_throughput(1.0)
        held_out_cache = _cache_with_throughput(5.0)
        widened_peak = campaign_peak([trained_cache, held_out_cache], _COVERING_LIGHT, seed=0)
        self.assertAlmostEqual(widened_peak, 5.0, places=12)


class TestCampaignPeakGuards(unittest.TestCase):
    def test_empty_trained_caches_raises_with_useful_message(self):
        with self.assertRaisesRegex(ValueError, "no trained caches"):
            campaign_peak([], _COVERING_LIGHT, seed=7)

    def test_non_finite_or_nonpositive_peak_raises(self):
        zero_cache = _cache_with_throughput(0.0)
        with self.assertRaisesRegex(ValueError, "seed=3"):
            campaign_peak([zero_cache], _COVERING_LIGHT, seed=3)

        nan_cache = _cache_with_throughput(float("nan"))
        with self.assertRaisesRegex(ValueError, "not finite"):
            campaign_peak([nan_cache], _COVERING_LIGHT, seed=9)


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


class TestArmEncodingConfigSharesResolutionSchedule(unittest.TestCase):
    """The three arms must be compared on a COMMON resolution ladder (same
    base_resolution/finest_resolution), even though their level counts and
    parameter budgets are deliberately left unequal. An empty per-arm config
    (falling through to that encoder class's own constructor default) is exactly
    the bug this guards against -- each class's default schedule differs, which
    would make any measured difference between arms inseparable from schedule and
    capacity."""

    def test_all_arms_resolve_to_the_same_base_and_finest_resolution(self):
        schedules = {
            name: encoder_schedule_params(name, ARM_ENCODING_CONFIG.get(name, {}))
            for name in ARM_NAMES
        }
        bases = {base for _levels, base, _finest in schedules.values()}
        finests = {finest for _levels, _base, finest in schedules.values()}
        self.assertEqual(
            bases,
            {4},
            f"arms disagree on base_resolution: {schedules}",
        )
        self.assertEqual(
            finests,
            {64},
            f"arms disagree on finest_resolution: {schedules}",
        )


class TestRotatedLight(unittest.TestCase):
    """Pins the G4 fix: eval lights must rotate with the cache and camera, or a frame
    change silently becomes a different physical setup (see `rotated_light`'s
    docstring for the run this invalidated)."""

    def test_sphere_center_rotates_radius_and_rgb_untouched(self):
        light = SphereLight(center=[1.0, 2.0, 3.0], radius=0.25, rgb=[0.1, 0.2, 0.3])
        rotation = rotation_matrix_y(90.0)

        rotated = rotated_light(light, rotation)

        # Independently derived expected center: a +90 degree right-handed rotation
        # about Y sends (x, y, z) -> (z, y, -x) -- worked out from first principles,
        # not by calling `rotation_matrix_y` output through the production helper's
        # own matmul convention.
        expected_center = np.array([3.0, 2.0, -1.0])
        np.testing.assert_allclose(rotated.center, expected_center, atol=1e-10)
        self.assertEqual(rotated.radius, light.radius)
        np.testing.assert_allclose(rotated.rgb, light.rgb)

    def test_rotated_lights_maps_over_a_list(self):
        lights = [
            SphereLight(center=[1.0, 0.0, 0.0], radius=0.1),
            SphereLight(center=[0.0, 0.0, 1.0], radius=0.2),
        ]
        rotation = rotation_matrix_y(180.0)

        out = rotated_lights(lights, rotation)

        np.testing.assert_allclose(out[0].center, [-1.0, 0.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(out[1].center, [0.0, 0.0, -1.0], atol=1e-10)

    def test_unhandled_light_type_raises_instead_of_passing_through_unrotated(self):
        quad = QuadLight(center=[0.0, 0.0, 0.0], normal=[0.0, 1.0, 0.0], width=1.0, height=1.0)
        with self.assertRaisesRegex(TypeError, "QuadLight"):
            rotated_light(quad, rotation_matrix_y(90.0))


def _non_axis_aligned_cache() -> PathCache:
    """A tiny two-pixel, two-segment-per-pixel cache with off-axis geometry, so a 90
    degree rotation about Y actually moves things instead of trivially fixing them.
    """
    d0 = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    d1 = np.array([1.0, -1.0, 0.5]) / np.linalg.norm([1.0, -1.0, 0.5])
    return PathCache(
        width=2,
        height=1,
        n_paths=np.array([1, 1], dtype=np.int64),
        seg_pixel=np.array([0, 1], dtype=np.int64),
        seg_origin=np.array([[0.2, 0.1, -0.3], [-0.4, 0.2, 0.1]]),
        seg_dir=np.array([d0, d1]),
        seg_tmax=np.array([0.8, 0.6]),
        seg_throughput=np.array([[0.7, 0.6, 0.5], [0.3, 0.4, 0.2]]),
        albedo=np.full((1, 2, 3), 0.5),
        depth=np.ones((1, 2)),
        normal=np.tile(np.array([0.0, 1.0, 0.0]), (1, 2, 1)),
        position=np.array([[[0.5, 0.3, 0.1], [-0.2, 0.4, 0.3]]]),
    )


def _rendered_stats(cache: PathCache, light: SphereLight) -> tuple[float, float]:
    image = gather_lights(cache, [light])
    return float(image.mean()), float(image.max())


class TestRotationPreservesPhysics(unittest.TestCase):
    """The actual bug class: rotating the cache and the light together must render
    the identical physics -- a frame change, not a different scene. Rotating the
    cache while leaving the light behind is the mistake that produced 4.4 hours of
    invalid G4 results (delta_db as bad as -18 dB at 180 degrees, scaling with the
    rotation angle, versus a healthy 0 degree control)."""

    def test_rotated_cache_and_rotated_light_match_unrotated_reference(self):
        cache = _non_axis_aligned_cache()
        light = SphereLight(center=[0.3, 0.2, 0.4], radius=0.6, rgb=[1.0, 0.8, 0.6])
        rotation = rotation_matrix_y(90.0)

        mean_before, max_before = _rendered_stats(cache, light)
        self.assertGreater(max_before, 0.0, "fixture must actually produce a hit")

        rotated_cache = transform_cache(cache, rotation)
        light_correctly_rotated = rotated_light(light, rotation)
        mean_after, max_after = _rendered_stats(rotated_cache, light_correctly_rotated)

        np.testing.assert_allclose(mean_after, mean_before, rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(max_after, max_before, rtol=1e-9, atol=1e-12)

    def test_break_restore_demonstration_bug_leaves_light_unrotated(self):
        """Demonstrates the actual bug: reusing the UNROTATED light against a rotated
        cache changes the rendered result. This is the exact defect `rotated_light`
        fixes; it must fail here to prove the covering test above is not vacuous."""
        cache = _non_axis_aligned_cache()
        light = SphereLight(center=[0.3, 0.2, 0.4], radius=0.6, rgb=[1.0, 0.8, 0.6])
        rotation = rotation_matrix_y(90.0)

        mean_before, max_before = _rendered_stats(cache, light)

        rotated_cache = transform_cache(cache, rotation)
        # Bug reproduction: pass the ORIGINAL unrotated light against the rotated
        # cache, exactly what `main`'s rotation loop did before the fix.
        mean_buggy, max_buggy = _rendered_stats(rotated_cache, light)

        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(mean_buggy, mean_before, rtol=1e-9, atol=1e-12)
        with self.assertRaises(AssertionError):
            np.testing.assert_allclose(max_buggy, max_before, rtol=1e-9, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
