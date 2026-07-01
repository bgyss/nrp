import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.torch_backend.denoise import joint_bilateral_denoise  # noqa: E402
from nrp.torch_backend.encoding import HashEncoding2D  # noqa: E402
from nrp.torch_backend.model import (  # noqa: E402
    TorchNRP,
    quad_params,
    relative_mse_loss,
    sphere_params,
)
from nrp.torch_backend.optimize_lights import (  # noqa: E402
    ReparamSphereLights,
    optimize,
    reinhard,
)
from nrp.torch_backend.sampling import sample_light, sample_positions  # noqa: E402
from nrp.torch_backend.train import light_param_vector, load_config, train  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


class HashEncodingTests(unittest.TestCase):
    def test_output_shape_and_dim(self):
        enc = HashEncoding2D(levels=4, features_per_level=2, finest_resolution=32)
        xy = torch.rand(17, 2)
        out = enc(xy)
        self.assertEqual(out.shape, (17, 8))
        self.assertEqual(enc.output_dim, 8)

    def test_deterministic_and_continuous(self):
        enc = HashEncoding2D(levels=4, finest_resolution=32)
        xy = torch.tensor([[0.5, 0.5]])
        out1, out2 = enc(xy), enc(xy)
        torch.testing.assert_close(out1, out2)
        # Bilinear interpolation: a tiny move produces a tiny change.
        near = enc(xy + 1e-5)
        self.assertLess(float((near - out1).abs().max()), 1e-3)

    def test_gradients_reach_tables(self):
        enc = HashEncoding2D(levels=2, finest_resolution=8)
        out = enc(torch.rand(5, 2)).sum()
        out.backward()
        grads = [t.grad for t in enc.tables]
        self.assertTrue(any(g is not None and float(g.abs().sum()) > 0 for g in grads))

    def test_resolutions_grow_geometrically(self):
        enc = HashEncoding2D(levels=5, base_resolution=4, finest_resolution=64)
        self.assertEqual(enc.resolutions[0], 4)
        self.assertEqual(enc.resolutions[-1], 64)
        self.assertEqual(enc.resolutions, sorted(enc.resolutions))


class RelativeMSELossTests(unittest.TestCase):
    def test_matches_eq4_value(self):
        pred = torch.tensor([[2.0, 0.0, 1.0]])
        target = torch.tensor([[1.0, 1.0, 1.0]])
        expected = np.mean([1.0 / (4.0 + 0.01), 1.0 / 0.01, 0.0])
        self.assertAlmostEqual(float(relative_mse_loss(pred, target)), expected, places=5)

    def test_denominator_is_stop_gradient(self):
        # With sg(pred) in the denominator, d/dpred [(pred-t)^2 / (sg(pred)^2+eps)]
        # = 2(pred-t) / (pred^2+eps): the denominator contributes no gradient term.
        pred = torch.tensor([3.0], requires_grad=True)
        relative_mse_loss(pred, torch.tensor([1.0])).backward()
        expected = 2.0 * (3.0 - 1.0) / (9.0 + 0.01)
        self.assertAlmostEqual(float(pred.grad[0]), expected, places=6)


class ModelTests(unittest.TestCase):
    def test_forward_shapes_sphere_and_quad(self):
        for light_type, dim in [("sphere", 4), ("quad", 8)]:
            model = TorchNRP(light_type=light_type, hidden_width=16, hidden_layers=2)
            out = model(torch.rand(9, 2), torch.rand(9, 7), torch.rand(9, dim))
            self.assertEqual(out.shape, (9, 3))
            self.assertTrue(bool((out >= 0).all()), "softplus head must be non-negative")

    def test_save_load_roundtrip(self):
        model = TorchNRP(hidden_width=16, hidden_layers=2)
        xy, aux, lp = torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pt")
            model.save(path)
            loaded = TorchNRP.load(path)
        torch.testing.assert_close(model(xy, aux, lp), loaded(xy, aux, lp))

    def test_param_broadcast_helpers(self):
        sp = sphere_params(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(0.5), 4)
        self.assertEqual(sp.shape, (4, 4))
        qp = quad_params(
            torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]), torch.tensor(1.0), torch.tensor(2.0), 4
        )
        self.assertEqual(qp.shape, (4, 8))
        torch.testing.assert_close(qp[0, 3:6], torch.tensor([0.0, 0.0, 1.0]))


class DenoiseTests(unittest.TestCase):
    def test_reduces_noise_on_flat_region(self):
        rng = np.random.default_rng(0)
        clean = np.full((16, 16, 3), 2.0)
        noisy = clean + rng.normal(0, 0.5, clean.shape)
        flat_aux = np.zeros((16, 16, 3))
        out = joint_bilateral_denoise(noisy, flat_aux, flat_aux, np.zeros((16, 16)))
        self.assertLess(np.mean((out - clean) ** 2), np.mean((noisy - clean) ** 2) / 2)

    def test_preserves_aux_guided_edge(self):
        # Two flat halves with different albedo: the filter must not blur across.
        image = np.zeros((8, 8, 3))
        image[:, 4:] = 10.0
        albedo = np.zeros((8, 8, 3))
        albedo[:, 4:] = 1.0
        out = joint_bilateral_denoise(
            image, albedo, np.zeros((8, 8, 3)), np.zeros((8, 8)), sigma_albedo=0.05
        )
        self.assertLess(abs(out[0, 3].mean() - 0.0), 0.5)
        self.assertLess(abs(out[0, 4].mean() - 10.0), 0.5)


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(8, 8, 4, 2, seed=7)

    def test_segment_positions_lie_on_segments_range(self):
        rng = np.random.default_rng(1)
        pos = sample_positions(self.cache, rng, 64, "segments")
        self.assertEqual(pos.shape, (64, 3))
        self.assertTrue(np.isfinite(pos).all())

    def test_bbox_positions_within_bbox(self):
        rng = np.random.default_rng(1)
        pos = sample_positions(self.cache, rng, 64, "bbox")
        p = self.cache.position.reshape(-1, 3)
        self.assertTrue((pos >= p.min(axis=0) - 1e-9).all())
        self.assertTrue((pos <= p.max(axis=0) + 1e-9).all())

    def test_sample_light_types(self):
        rng = np.random.default_rng(2)
        s = sample_light(self.cache, rng, "sphere", {"radius_min": 0.1, "radius_max": 0.2})
        q = sample_light(self.cache, rng, "quad", {"size_min": 0.1, "size_max": 0.3})
        self.assertTrue(0.1 <= s.radius <= 0.2)
        self.assertTrue(0.1 <= q.width <= 0.3 and 0.1 <= q.height <= 0.3)
        self.assertEqual(len(light_param_vector(s)), 4)
        self.assertEqual(len(light_param_vector(q)), 8)


class TrainingSmokeTests(unittest.TestCase):
    def test_tiny_training_run_improves_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "cache": str(Path(tmp) / "cache.npz"),
                "out_dir": str(Path(tmp) / "out"),
                "trace": {"width": 12, "height": 12, "spp": 6, "bounces": 2, "seed": 1},
                "light_type": "sphere",
                "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
                "sampling": "segments",
                "pool": {"size": 8, "replace_every": 5, "replace_count": 1},
                "denoise": {"enabled": True, "radius": 1},
                "iters": 300,
                "batch_pixels": 512,
                "lr": 0.01,
                "model": {
                    "hidden_width": 32,
                    "hidden_layers": 2,
                    "encoding": {"levels": 4, "table_size_log2": 10, "finest_resolution": 12},
                },
                "n_val_lights": 3,
                "seed": 0,
            }
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps(cfg))
            report = train(load_config(str(cfg_path)))
            # Single-minibatch losses are noisy; compare windowed means instead.
            curve = report["loss_curve"]
            head = float(np.mean(curve[: len(curve) // 5]))
            tail = float(np.mean(curve[-len(curve) // 5 :]))
            self.assertLess(tail, head)
            self.assertTrue((Path(tmp) / "out" / "model.pt").exists())

    def test_quad_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "cache": str(Path(tmp) / "cache.npz"),
                "out_dir": str(Path(tmp) / "out"),
                "trace": {"width": 10, "height": 10, "spp": 4, "bounces": 2, "seed": 2},
                "light_type": "quad",
                "light_bounds": {"size_min": 0.1, "size_max": 0.4},
                "pool": {"size": 6, "replace_every": 10, "replace_count": 1},
                "denoise": {"enabled": False},
                "iters": 30,
                "batch_pixels": 256,
                "lr": 0.01,
                "model": {
                    "hidden_width": 16,
                    "hidden_layers": 2,
                    "encoding": {"levels": 2, "table_size_log2": 8, "finest_resolution": 10},
                },
                "n_val_lights": 2,
                "seed": 0,
            }
            cfg_path = Path(tmp) / "cfg.json"
            cfg_path.write_text(json.dumps(cfg))
            report = train(load_config(str(cfg_path)))
            self.assertIn("val_psnr_db_vs_raw_mean", report)


class InverseOptimizationTests(unittest.TestCase):
    def test_reparam_roundtrip_and_bounds(self):
        bounds = {
            "center_min": [0.0, 0.0, 0.0],
            "center_max": [1.0, 1.0, 1.0],
            "radius_min": 0.05,
            "radius_max": 0.5,
        }
        init = [{"center": [0.3, 0.6, 0.4], "radius": 0.2, "rgb": [1.0, 2.0, 3.0]}]
        rp = ReparamSphereLights(init, bounds, torch.device("cpu"))
        centers, radii, rgbs = rp.constrained()
        torch.testing.assert_close(centers[0], torch.tensor([0.3, 0.6, 0.4]), atol=1e-3, rtol=1e-3)
        self.assertAlmostEqual(float(radii[0]), 0.2, places=3)
        torch.testing.assert_close(rgbs[0], torch.tensor([1.0, 2.0, 3.0]), atol=1e-3, rtol=1e-3)
        # Unconstrained steps can never leave the bounded domain.
        with torch.no_grad():
            rp.u_geo += 100.0
        centers, radii, _ = rp.constrained()
        self.assertTrue(bool((centers <= 1.0).all() and (radii <= 0.5).all()))

    def test_reinhard(self):
        x = torch.tensor([0.0, 1.0, 9.0])
        torch.testing.assert_close(reinhard(x), torch.tensor([0.0, 0.5, 0.9]))

    def test_single_light_recovery_improves_loss(self):
        cache = trace_path_cache(10, 10, 6, 2, seed=3)
        model = TorchNRP(
            hidden_width=32,
            hidden_layers=2,
            encoding={"levels": 4, "table_size_log2": 10, "finest_resolution": 10},
        )
        true = SphereLight(center=[0.5, 0.7, 0.5], radius=0.15, rgb=[5.0, 5.0, 5.0])
        target = gather_light(cache, true)
        bounds = {
            "center_min": [0.1, 0.1, 0.15],
            "center_max": [0.9, 0.9, 0.9],
            "radius_min": 0.05,
            "radius_max": 0.3,
        }
        init = [{"center": [0.3, 0.3, 0.3], "radius": 0.1, "rgb": [1.0, 1.0, 1.0]}]
        report = optimize(model, cache, target, init, bounds, steps=25, lr=0.05)
        self.assertLess(report["proxy_loss_last"], report["proxy_loss_first"])
        self.assertEqual(len(report["optimized_lights"]), 1)

    def test_pixel_fraction_subset_runs(self):
        cache = trace_path_cache(10, 10, 4, 2, seed=4)
        model = TorchNRP(
            hidden_width=16,
            hidden_layers=2,
            encoding={"levels": 2, "table_size_log2": 8, "finest_resolution": 10},
        )
        target = gather_light(cache, SphereLight(center=[0.5, 0.5, 0.5], radius=0.2))
        bounds = {
            "center_min": [0.1, 0.1, 0.15],
            "center_max": [0.9, 0.9, 0.9],
            "radius_min": 0.05,
            "radius_max": 0.3,
        }
        init = [
            {"center": [0.4, 0.4, 0.4], "radius": 0.1, "rgb": [1.0, 1.0, 1.0]},
            {"center": [0.6, 0.6, 0.6], "radius": 0.1, "rgb": [1.0, 1.0, 1.0]},
        ]
        report = optimize(
            model, cache, target, init, bounds, steps=10, lr=0.05, pixel_fraction=0.25
        )
        self.assertEqual(len(report["optimized_lights"]), 2)
        self.assertEqual(len(report["proxy_loss_curve"]), 10)


if __name__ == "__main__":
    unittest.main()
