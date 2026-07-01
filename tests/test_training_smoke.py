"""NRP M4 exit criterion (smoke scale): a training run over a tiny traced scene
reduces validation loss against GATHERLIGHT. Also verifies the hand-rolled autodiff
against finite differences (both weight and input gradients), since the whole M5
inverse-optimization path depends on those input gradients being right."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.dataset import build_inputs  # noqa: E402
from nrp.model import (  # noqa: E402
    INPUT_DIM,
    ProxyMLP,
    light_param_gradient,
    relative_mse_loss,
)
from nrp.toy_tracer import trace_path_cache  # noqa: E402
from nrp.train import load_config, train  # noqa: E402


class AutodiffTests(unittest.TestCase):
    def test_weight_and_input_gradients_match_finite_differences(self):
        rng = np.random.default_rng(3)
        model = ProxyMLP(hidden=(8, 8), seed=3)
        x = rng.normal(size=(5, INPUT_DIM))
        y = np.abs(rng.normal(size=(5, 3)))

        pred = model.forward(x)
        _, dpred = relative_mse_loss(pred, y)
        dx, d_w, d_b = model.backward(dpred)

        def loss_at(xv):
            p = model.forward(xv)
            return relative_mse_loss(p, y)[0]

        eps = 1e-6
        # Input gradients (spot-check a handful of coordinates).
        for i, j in [(0, 0), (2, 5), (4, INPUT_DIM - 1), (1, INPUT_DIM - 4)]:
            xp = x.copy()
            xp[i, j] += eps
            xm = x.copy()
            xm[i, j] -= eps
            fd = (loss_at(xp) - loss_at(xm)) / (2 * eps)
            self.assertAlmostEqual(dx[i, j], fd, places=5)
        # Weight gradient spot-check (first layer).
        w = model.weights[0]
        for i, j in [(0, 0), (3, 7)]:
            orig = w[i, j]
            w[i, j] = orig + eps
            lp = loss_at(x)
            w[i, j] = orig - eps
            lm = loss_at(x)
            w[i, j] = orig
            fd = (lp - lm) / (2 * eps)
            self.assertAlmostEqual(d_w[0][i, j], fd, places=5)

    def test_light_param_gradient_matches_finite_differences(self):
        """The chain rule through the derived (diff, dist) columns — the gradient
        inverse optimization (M5) actually uses — checked against finite differences
        of the full encode+forward pipeline."""
        rng = np.random.default_rng(11)
        model = ProxyMLP(hidden=(8, 8), seed=11)
        n = 6
        px = {
            "pixel_xy": rng.random((n, 2)),
            "albedo": rng.random((n, 3)),
            "depth": rng.random(n) + 0.1,
            "normal": np.tile([0.0, 0.0, 1.0], (n, 1)),
            "position": rng.random((n, 3)),
        }
        center = np.array([0.6, 0.5, 0.4])
        radius = 0.15
        target = np.abs(rng.normal(size=(n, 3)))

        def loss_at(c, r):
            pred = model.forward(build_inputs(px, c, r))
            return relative_mse_loss(pred, target)[0]

        pred = model.forward(build_inputs(px, center, radius))
        _, dpred = relative_mse_loss(pred, target)
        dx, _, _ = model.backward(dpred)
        d_center, d_radius = light_param_gradient(dx, px["position"], center)

        eps = 1e-6
        for axis in range(3):
            cp = center.copy()
            cp[axis] += eps
            cm = center.copy()
            cm[axis] -= eps
            fd = (loss_at(cp, radius) - loss_at(cm, radius)) / (2 * eps)
            self.assertAlmostEqual(d_center[axis], fd, places=5)
        fd_r = (loss_at(center, radius + eps) - loss_at(center, radius - eps)) / (2 * eps)
        self.assertAlmostEqual(d_radius, fd_r, places=5)


class TracerSmokeTests(unittest.TestCase):
    def test_traced_cache_validates_and_has_expected_shape(self):
        cache = trace_path_cache(width=8, height=8, spp=2, max_bounces=2, seed=5)
        cache.validate()
        self.assertEqual(cache.segment_count, 8 * 8 * 2 * 2)
        # Closed box: every segment terminates (no escape rays).
        self.assertTrue(np.all(np.isfinite(cache.seg_tmax)))
        # Throughput never grows along a path.
        self.assertTrue(np.all(cache.seg_throughput <= 1.0 + 1e-12))
        self.assertTrue(np.all(cache.depth > 0.0))


class TrainingSmokeTests(unittest.TestCase):
    def test_training_reduces_validation_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "cache": str(Path(tmp) / "cache.npz"),
                "out_dir": str(Path(tmp) / "out"),
                "trace": {"width": 12, "height": 12, "spp": 6, "bounces": 2, "seed": 2},
                "light_bounds": {
                    "center_min": [0.2, 0.2, 0.25],
                    "center_max": [0.8, 0.8, 0.8],
                    "radius_min": 0.08,
                    "radius_max": 0.25,
                },
                "n_train_lights": 16,
                "n_val_lights": 4,
                "epochs": 8,
                "batch_size": 1024,
                "lr": 0.002,
                "hidden": [32, 32],
                "seed": 0,
            }
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps(cfg))
            report = train(load_config(str(cfg_path)))
            val = report["history"]["val_loss"]
            self.assertLess(val[-1], val[0] * 0.7, f"val loss did not drop: {val}")
            self.assertTrue((Path(tmp) / "out" / "model.npz").exists())
            # Round-trip: loaded model reproduces param count.
            model = ProxyMLP.load(str(Path(tmp) / "out" / "model.npz"))
            self.assertEqual(model.parameter_count, report["parameter_count"])


if __name__ == "__main__":
    unittest.main()
