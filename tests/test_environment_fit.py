import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.environment_fit import make_escaped_cache, make_reference_light  # noqa: E402
from nrp.environment_fit import environment_design_matrix, fit_environment_light  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402


class EnvironmentFitTests(unittest.TestCase):
    def test_design_matrix_reproduces_gatherlight(self):
        cache = make_escaped_cache(width=32)
        light = make_reference_light()
        design = environment_design_matrix(cache)
        coeff_vector = light.coeffs.T.reshape(-1)
        np.testing.assert_allclose(
            design @ coeff_vector,
            gather_light(cache, light).reshape(-1),
            atol=1e-12,
        )

    def test_inverse_recovery_meets_e4_threshold(self):
        cache = make_escaped_cache(width=48)
        reference = make_reference_light()
        target = gather_light(cache, reference)
        fit = fit_environment_light(cache, target, reference=reference)
        self.assertEqual(fit.rank, 27)
        self.assertIsNotNone(fit.relative_coeff_error)
        self.assertLess(fit.relative_coeff_error, 0.10)
        np.testing.assert_allclose(gather_light(cache, fit.light), target, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
