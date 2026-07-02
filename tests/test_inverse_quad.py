"""Quad-light inverse optimization (roadmap item 4, paper §5.3 + Fig. 13).

Fast tests: reparameterization round-trip and gradient flow (including through the
normal normalization). The expensive 1-light quad recovery check (center error
< 0.05 on a trained toy quad model) lives in `examples/inverse_grid.py --quad-check`,
which trains a full-quality toy quad proxy — too slow for the unit suite.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.optimize_lights import (  # noqa: E402
    DEFAULT_BOUNDS,
    ReparamQuadLights,
    make_reparam,
    predicted_image,
    random_init,
    reinhard,
)

INIT = [
    {
        "center": [0.4, 0.5, 0.6],
        "normal": [0.2, -1.0, 0.3],
        "width": 0.25,
        "height": 0.18,
        "rgb": [2.0, 1.5, 1.0],
    },
    {
        "center": [0.7, 0.3, 0.4],
        "normal": [0.0, 0.0, 1.0],
        "width": 0.1,
        "height": 0.4,
        "rgb": [0.5, 0.5, 3.0],
    },
]


class QuadReparamTests(unittest.TestCase):
    def test_roundtrip_recovers_init(self):
        rep = ReparamQuadLights(INIT, DEFAULT_BOUNDS, torch.device("cpu"))
        lights = rep.to_lights()
        for light, spec in zip(lights, INIT, strict=True):
            np.testing.assert_allclose(light.center, spec["center"], atol=1e-4)
            self.assertAlmostEqual(light.width, spec["width"], places=4)
            self.assertAlmostEqual(light.height, spec["height"], places=4)
            np.testing.assert_allclose(light.rgb, spec["rgb"], atol=1e-4)
            # QuadLight normalizes; compare directions.
            want = np.asarray(spec["normal"]) / np.linalg.norm(spec["normal"])
            np.testing.assert_allclose(light.normal, want, atol=1e-4)

    def test_constrained_respects_bounds_under_extreme_params(self):
        rep = ReparamQuadLights(INIT, DEFAULT_BOUNDS, torch.device("cpu"))
        with torch.no_grad():
            rep.u_geo += 100.0  # push far past the logit range
        centers, _, widths, heights, _ = rep.constrained()
        self.assertTrue((centers <= torch.tensor(DEFAULT_BOUNDS["center_max"]) + 1e-5).all())
        self.assertTrue((widths <= DEFAULT_BOUNDS["size_max"] + 1e-5).all())
        self.assertTrue((heights <= DEFAULT_BOUNDS["size_max"] + 1e-5).all())

    def test_gradients_flow_to_all_parameters_including_normal(self):
        torch.manual_seed(0)
        model = TorchNRP(
            light_type="quad",
            hidden_width=16,
            hidden_layers=2,
            encoding={"levels": 2, "table_size_log2": 8, "finest_resolution": 8},
        )
        rep = ReparamQuadLights(INIT, DEFAULT_BOUNDS, torch.device("cpu"))
        xy = torch.rand((32, 2))
        aux = torch.rand((32, 7))
        pred = predicted_image(model, rep, xy, aux)
        loss = (reinhard(pred) ** 2).mean()
        loss.backward()
        for name, u in [("geo", rep.u_geo), ("normal", rep.u_normal), ("rgb", rep.u_rgb)]:
            self.assertIsNotNone(u.grad, name)
            self.assertGreater(float(u.grad.abs().sum()), 0.0, f"no gradient into {name}")

    def test_normal_gradient_is_tangential(self):
        # quad_params normalizes the raw normal, so the gradient must be orthogonal
        # to it (pure scaling of the raw vector cannot change the light).
        torch.manual_seed(1)
        model = TorchNRP(
            light_type="quad",
            hidden_width=16,
            hidden_layers=2,
            encoding={"levels": 2, "table_size_log2": 8, "finest_resolution": 8},
        )
        rep = ReparamQuadLights(INIT[:1], DEFAULT_BOUNDS, torch.device("cpu"))
        pred = predicted_image(model, rep, torch.rand((64, 2)), torch.rand((64, 7)))
        (reinhard(pred) ** 2).mean().backward()
        n = rep.u_normal.detach()[0]
        g = rep.u_normal.grad[0]
        cos = float(torch.dot(n, g) / (n.norm() * g.norm()).clamp(min=1e-12))
        self.assertLess(abs(cos), 1e-5)

    def test_make_reparam_and_random_init_dispatch(self):
        rng = np.random.default_rng(0)
        for light_type, keys in [("sphere", {"radius"}), ("quad", {"normal", "width", "height"})]:
            init = random_init(rng, light_type, DEFAULT_BOUNDS, 3)
            self.assertEqual(len(init), 3)
            self.assertTrue(keys <= set(init[0]))
            rep = make_reparam(light_type, init, DEFAULT_BOUNDS, torch.device("cpu"))
            self.assertEqual(len(rep.to_lights()), 3)
        with self.assertRaises(ValueError):
            make_reparam("disc", [], DEFAULT_BOUNDS, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
