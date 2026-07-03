import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.dynamic_geometry import (  # noqa: E402
    primary_visibility_invalidation_mask,
    splice_invalidated_pixels,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
