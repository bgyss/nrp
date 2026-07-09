import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.production_controls import (  # noqa: E402
    BasisControlProxy,
    BinaryLinkProxy,
    LinearAttenuationProxy,
    apply_soft_link_mask,
    mask_from_weights,
)


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

    def test_basis_control_proxy_predicts_heldout_linear_features(self):
        base = np.arange(12, dtype=np.float64).reshape(2, 2, 3) * 0.1
        term_a = base + 0.25
        term_b = np.flip(base, axis=0) + 0.5
        features = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.2, 0.0],
                [1.0, 0.0, 0.3],
                [1.0, 0.4, 0.5],
            ],
            dtype=np.float64,
        )
        images = [f[0] * base + f[1] * term_a + f[2] * term_b for f in features]
        proxy = BasisControlProxy.fit(features, images)
        heldout = np.array([1.0, 0.15, 0.25], dtype=np.float64)
        np.testing.assert_allclose(
            proxy.predict(heldout),
            heldout[0] * base + heldout[1] * term_a + heldout[2] * term_b,
            atol=1e-12,
        )

    def test_basis_control_proxy_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            BasisControlProxy.fit(np.ones(3), [np.zeros((1, 1, 3))] * 3)
        with self.assertRaises(ValueError):
            BasisControlProxy.fit(np.ones((2, 2)), [np.zeros((1, 1, 3))])
        with self.assertRaises(ValueError):
            BasisControlProxy.fit(
                np.ones((2, 2)),
                [np.zeros((1, 1, 3)), np.zeros((2, 1, 3))],
            )

    def test_soft_mask_helpers_clip_and_apply_masks(self):
        basis = [
            np.ones((2, 2), dtype=np.float64),
            np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64),
        ]
        mask = mask_from_weights(basis, np.array([0.25, 0.9]))
        np.testing.assert_allclose(mask, np.array([[0.25, 1.0], [0.25, 1.0]]))
        image = np.ones((2, 2, 3), dtype=np.float64)
        np.testing.assert_allclose(
            apply_soft_link_mask(image, mask),
            np.broadcast_to(1.0 - mask[..., None], image.shape),
        )
        with self.assertRaises(ValueError):
            mask_from_weights(basis, np.array([0.5]))


if __name__ == "__main__":
    unittest.main()
