import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.production_controls import BinaryLinkProxy, LinearAttenuationProxy  # noqa: E402


class ProductionControlsTests(unittest.TestCase):
    def test_binary_link_proxy_predicts_both_states(self):
        inactive = np.ones((2, 2, 3))
        active = np.zeros((2, 2, 3))
        proxy = BinaryLinkProxy(inactive, active)
        np.testing.assert_allclose(proxy.predict(0.0), inactive)
        np.testing.assert_allclose(proxy.predict(1.0), active)
        np.testing.assert_allclose(proxy.predict(0.25), 0.75 * inactive + 0.25 * active)
        self.assertEqual(proxy.parameter_count, inactive.size + active.size)

    def test_binary_link_proxy_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            BinaryLinkProxy(np.zeros((1, 1, 3)), np.zeros((2, 1, 3)))

    def test_linear_attenuation_proxy_predicts_heldout_control(self):
        base = np.arange(12, dtype=np.float64).reshape(2, 2, 3) * 0.1
        distance_weighted = np.flip(base, axis=1) + 0.25
        controls = np.array([[0.8, -0.05], [1.0, -0.1], [1.2, -0.15]])
        images = [intercept * base + slope * distance_weighted for intercept, slope in controls]
        proxy = LinearAttenuationProxy.fit(controls, images)
        expected = 1.1 * base - 0.12 * distance_weighted
        np.testing.assert_allclose(proxy.predict(1.1, -0.12), expected, atol=1e-12)
        self.assertEqual(proxy.parameter_count, 2 * base.size)

    def test_linear_attenuation_proxy_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            LinearAttenuationProxy.fit(np.ones((2, 3)), [np.zeros((1, 1, 3))] * 2)
        with self.assertRaises(ValueError):
            LinearAttenuationProxy.fit(np.ones((2, 2)), [np.zeros((1, 1, 3))])


if __name__ == "__main__":
    unittest.main()
