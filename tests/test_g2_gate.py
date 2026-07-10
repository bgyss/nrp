import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.g2_gate import apply_controls, effective_light  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.toy_tracer import SPHERE_CENTER, layer_ownership_mask, trace_path_cache  # noqa: E402


def make_state(link=False, k=0.0, tint=(1.0, 1.0, 1.0)):
    return {
        "light": {
            "type": "sphere",
            "center": [0.35, 0.28, 0.62],
            "radius": 0.2,
            "rgb": [1.3, 1.0, 0.8],
        },
        "controls": {"rgb": list(tint), "attenuation_k": k, "link": link},
    }


class EffectiveLightTests(unittest.TestCase):
    def test_tint_folds_into_emission(self):
        light = effective_light(make_state(tint=(0.5, 1.0, 2.0)))
        np.testing.assert_allclose(light.rgb, [0.65, 1.0, 1.6])
        np.testing.assert_allclose(light.center, [0.35, 0.28, 0.62])


class ApplyControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(12, 12, 2, max_bounces=1, seed=5, sphere_center=SPHERE_CENTER)
        cls.image = gather_light(cls.cache, effective_light(make_state()))
        cls.positions = cls.cache.position.reshape(-1, 3).astype(np.float32)

    def test_linking_zeroes_exactly_the_masked_pixels(self):
        """Pixel-level linking matches E8's gather-time algebra: with the layer
        ownership mask as the link mask, masked pixels go to zero and the rest are
        untouched (the layer caches partition segments by first hit, so zeroing a
        pixel equals excluding the light from that layer)."""
        mask = layer_ownership_mask(12, 12, "sphere").astype(np.float32)
        state = make_state(link=True)
        out = apply_controls(self.image, state, mask, self.positions, np.array([0.35, 0.28, 0.62]))
        self.assertTrue((out[mask > 0.5] == 0.0).all())
        np.testing.assert_array_equal(out[mask <= 0.5], self.image[mask <= 0.5])

    def test_attenuation_matches_closed_form(self):
        state = make_state(k=0.1)
        center = np.array([0.35, 0.28, 0.62])
        mask = np.zeros((12, 12), dtype=np.float32)
        out = apply_controls(self.image, state, mask, self.positions, center)
        dist = np.linalg.norm(self.positions - center[None, :], axis=1).reshape(12, 12)
        expected = self.image * np.maximum(0.0, 1.0 - 0.1 * dist)[..., None]
        np.testing.assert_allclose(out, expected)

    def test_no_controls_is_identity(self):
        state = make_state()
        mask = np.ones((12, 12), dtype=np.float32)
        out = apply_controls(self.image, state, mask, self.positions, np.array([0.35, 0.28, 0.62]))
        np.testing.assert_array_equal(out, self.image)

    def test_gate_wiring_identical_pair_passes(self):
        gate = evaluate_gate(self.image, self.image.copy(), "preview")
        self.assertTrue(gate["passed"])


if __name__ == "__main__":
    unittest.main()
