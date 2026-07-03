import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.light_aware_sampling import region_density  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.toy_tracer import render_reference, trace_path_cache  # noqa: E402


class LightAwareSamplingTests(unittest.TestCase):
    def test_guided_sampling_increases_region_segment_density(self):
        region = {"type": "sphere", "center": [0.45, 0.75, 0.45], "radius": 0.12}
        standard = trace_path_cache(18, 18, 6, 3, seed=13)
        guided = trace_path_cache(
            18,
            18,
            6,
            3,
            seed=13,
            light_region=region,
            guide_probability=0.5,
        )
        self.assertGreater(
            region_density(guided, region)["region_hit_fraction"],
            2.0 * region_density(standard, region)["region_hit_fraction"],
        )
        self.assertEqual(guided.segment_count, standard.segment_count)
        guided.validate()

    def test_guided_gather_is_consistent_with_independent_guided_reference(self):
        region = {"type": "sphere", "center": [0.45, 0.75, 0.45], "radius": 0.12}
        light = SphereLight(center=region["center"], radius=region["radius"])
        cache = trace_path_cache(
            16,
            16,
            24,
            3,
            seed=41,
            light_region=region,
            guide_probability=0.5,
        )
        reference = render_reference(
            16,
            16,
            96,
            3,
            seed=99,
            light=light,
            light_region=region,
            guide_probability=0.5,
        )
        self.assertGreater(psnr(gather_light(cache, light), reference), 18.0)


if __name__ == "__main__":
    unittest.main()
