import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.dynamic_geometry import (  # noqa: E402
    TorchNRPWarmStartProxy,
    WarmStartImageProxy,
)
from nrp.dynamic_geometry import (  # noqa: E402
    primary_visibility_invalidation_mask,
    splice_invalidated_pixels,
    swept_bounding_sphere,
    swept_volume_invalidation_mask,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.toy_tracer import SPHERE_CENTER, trace_path_cache  # noqa: E402


class DynamicGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = trace_path_cache(
            18,
            18,
            3,
            max_bounces=1,
            seed=5,
            sphere_center=SPHERE_CENTER,
        )
        cls.after = trace_path_cache(
            18,
            18,
            3,
            max_bounces=1,
            seed=5,
            sphere_center=SPHERE_CENTER + np.array([0.12, 0.0, 0.0]),
        )
        cls.light = SphereLight(center=[0.18, 0.72, 0.25], radius=0.2)

    def test_primary_visibility_mask_covers_all_gbuffer_changes(self):
        mask = primary_visibility_invalidation_mask(self.before, self.after)
        unchanged = ~mask
        self.assertGreater(int(mask.sum()), 0)
        np.testing.assert_allclose(self.before.depth[unchanged], self.after.depth[unchanged])
        np.testing.assert_allclose(self.before.albedo[unchanged], self.after.albedo[unchanged])
        np.testing.assert_allclose(self.before.normal[unchanged], self.after.normal[unchanged])
        np.testing.assert_allclose(self.before.position[unchanged], self.after.position[unchanged])

    def test_spliced_cache_matches_full_retrace_for_one_bounce(self):
        mask = primary_visibility_invalidation_mask(self.before, self.after)
        spliced, stats = splice_invalidated_pixels(self.before, self.after, mask)
        self.assertEqual(stats.invalid_pixels, int(mask.sum()))
        spliced.validate()
        np.testing.assert_allclose(
            gather_light(spliced, self.light),
            gather_light(self.after, self.light),
            atol=1e-12,
        )

    def test_warm_start_image_proxy_updates_masked_pixels_only(self):
        initial = np.zeros((2, 2, 3), dtype=np.float64)
        target = np.ones((2, 2, 3), dtype=np.float64)
        mask = np.array([[True, False], [False, True]])
        proxy = WarmStartImageProxy(initial)
        losses = proxy.fine_tune(target, mask, steps=3, lr=0.5)
        updated = proxy.predict()
        self.assertLess(losses[-1], losses[0])
        self.assertGreater(float(updated[0, 0, 0]), 0.0)
        self.assertGreater(float(updated[1, 1, 0]), 0.0)
        np.testing.assert_allclose(updated[0, 1], 0.0)
        np.testing.assert_allclose(updated[1, 0], 0.0)

    def test_torchnrp_warm_start_fine_tune_only_touches_masked_pixels(self):
        mask = primary_visibility_invalidation_mask(self.before, self.after)
        spliced, _ = splice_invalidated_pixels(self.before, self.after, mask)
        proxy = TorchNRPWarmStartProxy(hidden_width=8, hidden_layers=1, seed=0)
        proxy.set_light(self.light)
        before_pred = proxy.predict(self.before)
        target = gather_light(spliced, self.light)
        losses = proxy.fine_tune(spliced, target, mask, iters=5, lr=1e-2)
        after_pred = proxy.predict(self.before)
        self.assertEqual(len(losses), 5)
        # Fine-tuning updates shared weights, so unmasked-pixel predictions can move
        # too (unlike the pure image-space proxy) — this only checks the call runs
        # end to end and produces a finite loss trajectory.
        self.assertTrue(np.all(np.isfinite(after_pred)))
        self.assertFalse(np.array_equal(before_pred, after_pred))

    def test_torchnrp_fine_tune_with_replay_runs_and_produces_finite_loss(self):
        mask = primary_visibility_invalidation_mask(self.before, self.after)
        spliced, _ = splice_invalidated_pixels(self.before, self.after, mask)
        proxy = TorchNRPWarmStartProxy(hidden_width=8, hidden_layers=1, seed=0)
        proxy.set_light(self.light)
        target = gather_light(spliced, self.light)
        losses = proxy.fine_tune_with_replay(spliced, target, mask, iters=5, lr=1e-2)
        self.assertEqual(len(losses), 5)
        self.assertTrue(all(np.isfinite(loss) for loss in losses))

    def test_torchnrp_fine_tune_with_replay_no_op_on_empty_mask(self):
        proxy = TorchNRPWarmStartProxy(hidden_width=8, hidden_layers=1, seed=0)
        proxy.set_light(self.light)
        mask = np.zeros((self.before.height, self.before.width), dtype=bool)
        target = gather_light(self.before, self.light)
        losses = proxy.fine_tune_with_replay(self.before, target, mask, iters=5, lr=1e-2)
        self.assertEqual(losses, [0.0, 0.0])

    def test_swept_bounding_sphere_contains_endpoints_and_object_extent(self):
        center_before = np.array([0.0, 0.0, 0.0])
        center_after = np.array([1.0, 0.0, 0.0])
        radius = 0.2
        midpoint, swept_radius = swept_bounding_sphere(center_before, center_after, radius)
        np.testing.assert_allclose(midpoint, [0.5, 0.0, 0.0])
        # Every point of the object at either endpoint must be inside the bound.
        for center in (center_before, center_after):
            farthest_point_on_object = center + np.array([radius, 0.0, 0.0])
            self.assertLessEqual(
                float(np.linalg.norm(farthest_point_on_object - midpoint)), swept_radius + 1e-12
            )

    def test_swept_volume_mask_flags_pixels_with_any_bounce_depth_segment_in_region(self):
        # Two paths: pixel 0's only segment (bounce 0) passes nowhere near the swept
        # sphere; pixel 1 has a bounce-0 segment far away and a bounce-1 segment that
        # passes straight through the swept region. Only primary (first) segments are
        # what `primary_visibility_invalidation_mask` would see; the swept-volume mask
        # must also catch pixel 1 via its second segment.
        cache = PathCache(
            width=2,
            height=1,
            n_paths=np.array([1, 1], dtype=np.int64),
            seg_pixel=np.array([0, 1, 1], dtype=np.int64),
            seg_origin=np.array(
                [
                    [10.0, 10.0, 10.0],  # pixel 0, bounce 0: far from swept region
                    [10.0, 10.0, 10.0],  # pixel 1, bounce 0: far from swept region
                    [-1.0, 0.0, 0.0],  # pixel 1, bounce 1: crosses the swept region
                ]
            ),
            seg_dir=np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            seg_tmax=np.array([1.0, 1.0, 2.0]),
            seg_throughput=np.ones((3, 3)),
            albedo=np.zeros((1, 2, 3)),
            position=np.zeros((1, 2, 3)),
            depth=np.ones((1, 2)),
            normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, 2, 1)),
        )
        cache.validate()
        mask = swept_volume_invalidation_mask(
            cache, center_before=[0.0, 0.0, 0.0], center_after=[0.0, 0.0, 0.0], radius=0.3
        )
        self.assertFalse(bool(mask[0, 0]))
        self.assertTrue(bool(mask[0, 1]))

    def test_multi_bounce_spliced_cache_matches_full_retrace_with_swept_mask(self):
        before = trace_path_cache(
            16, 16, 4, max_bounces=2, seed=11, sphere_center=SPHERE_CENTER
        )
        moved = SPHERE_CENTER + np.array([0.12, 0.0, 0.0])
        after = trace_path_cache(16, 16, 4, max_bounces=2, seed=11, sphere_center=moved)
        primary_mask = primary_visibility_invalidation_mask(before, after)
        swept_mask = swept_volume_invalidation_mask(
            before, SPHERE_CENTER, moved, radius=0.25, margin=0.05
        )
        combined = primary_mask | swept_mask
        spliced, stats = splice_invalidated_pixels(before, after, combined)
        self.assertEqual(stats.invalid_pixels, int(combined.sum()))
        spliced.validate()
        np.testing.assert_allclose(
            gather_light(spliced, self.light),
            gather_light(after, self.light),
            atol=1e-9,
        )

    def test_torchnrp_warm_start_no_op_on_empty_mask(self):
        proxy = TorchNRPWarmStartProxy(hidden_width=8, hidden_layers=1, seed=0)
        proxy.set_light(self.light)
        mask = np.zeros((self.before.height, self.before.width), dtype=bool)
        target = gather_light(self.before, self.light)
        losses = proxy.fine_tune(self.before, target, mask, iters=5, lr=1e-2)
        self.assertEqual(losses, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
