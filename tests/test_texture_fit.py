import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.textured_quad_fit import (  # noqa: E402
    CENTER,
    HEIGHT,
    NORMAL,
    WIDTH,
    make_full_rank_cache,
    make_random_uv_cache,
    make_reference_texture,
    run_proxy_scaling_case,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import TexturedQuadLight  # noqa: E402
from nrp.texture_fit import fit_textured_quad_light, textured_quad_design_matrices  # noqa: E402


class TextureFitTests(unittest.TestCase):
    def test_design_matrices_are_full_rank_for_synthetic_cache(self):
        cache = make_full_rank_cache(4)
        designs = textured_quad_design_matrices(cache, CENTER, NORMAL, WIDTH, HEIGHT, (4, 4))
        for design in designs:
            self.assertEqual(np.linalg.matrix_rank(design), 16)

    def test_fit_textured_quad_recovers_reference_texture(self):
        cache = make_full_rank_cache(4)
        reference = TexturedQuadLight(
            center=CENTER,
            normal=NORMAL,
            width=WIDTH,
            height=HEIGHT,
            texture=make_reference_texture(4),
        )
        target = gather_light(cache, reference)
        fit = fit_textured_quad_light(
            cache,
            target,
            CENTER,
            NORMAL,
            WIDTH,
            HEIGHT,
            (4, 4),
            reference=reference,
        )
        self.assertTrue(all(rank == 16 for rank in fit.ranks))
        self.assertIsNotNone(fit.relative_texture_error)
        self.assertLess(fit.relative_texture_error, 1e-10)
        np.testing.assert_allclose(fit.light.texture, reference.texture, atol=1e-10)
        np.testing.assert_allclose(gather_light(cache, fit.light), target, atol=1e-10)

    def test_random_uv_cache_is_valid(self):
        cache = make_random_uv_cache(4, 12, seed=9)
        self.assertEqual(cache.segment_count, 12)
        cache.validate()

    def test_proxy_scaling_case_reports_underdetermined_high_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = run_proxy_scaling_case(8, Path(tmp), train_samples=16)
        self.assertEqual(case["texture_parameter_count"], 192)
        self.assertTrue(case["underdetermined"])
        self.assertLess(max(case["rank_per_channel"]), case["unknowns_per_channel"])


if __name__ == "__main__":
    unittest.main()
