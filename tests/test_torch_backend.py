import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light, gather_lights  # noqa: E402
from nrp.lights import SphereLight, TexturedQuadLight  # noqa: E402
from nrp.torch_backend.denoise import joint_bilateral_denoise  # noqa: E402
from nrp.torch_backend.encoding import HashEncoding2D  # noqa: E402
from nrp.torch_backend.model import (  # noqa: E402
    TorchNRP,
    inverse_softplus,
    quad_params,
    relative_mse_loss,
    sphere_params,
)
from nrp.torch_backend.optimize_lights import (  # noqa: E402
    ReparamSphereLights,
    optimize,
    reinhard,
)
from nrp.torch_backend.relight import (  # noqa: E402
    relight,
    relight_tiled,
    render_quality_tier,
    write_image_with_metadata,
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

    def test_forward_shape_textured_quad_with_configured_param_dim(self):
        model = TorchNRP(
            light_type="textured_quad",
            light_param_dim=20,
            hidden_width=16,
            hidden_layers=2,
        )
        out = model(torch.rand(9, 2), torch.rand(9, 7), torch.rand(9, 20))
        self.assertEqual(out.shape, (9, 3))
        self.assertEqual(model.light_param_dim, 20)

    def test_save_load_roundtrip(self):
        model = TorchNRP(hidden_width=16, hidden_layers=2)
        xy, aux, lp = torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pt")
            model.save(path)
            loaded = TorchNRP.load(path)
        torch.testing.assert_close(model(xy, aux, lp), loaded(xy, aux, lp))

    def test_textured_quad_save_load_roundtrip(self):
        model = TorchNRP(
            light_type="textured_quad",
            light_param_dim=20,
            hidden_width=16,
            hidden_layers=2,
        )
        xy, aux, lp = torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pt")
            model.save(path)
            loaded = TorchNRP.load(path)
        self.assertEqual(loaded.light_type, "textured_quad")
        self.assertEqual(loaded.light_param_dim, 20)
        torch.testing.assert_close(model(xy, aux, lp), loaded(xy, aux, lp))

    def test_ablation_switches_shapes_and_input_dims(self):
        # Roadmap item 10: {None, Aux, Aux+Enc, ...} variants toggle the aux features
        # and the hashgrid encoding; forward keeps its signature in every variant.
        enc = {"levels": 2, "table_size_log2": 8, "finest_resolution": 8}
        cases = [
            (False, False, 2 + 4),  # "None": raw px + light params
            (True, False, 2 + 7 + 4),  # "Aux"
            (False, True, 4 + 4),  # encoding only
            (True, True, 4 + 7 + 4),  # "Aux+Enc"
        ]
        xy, aux, lp = torch.rand(9, 2), torch.rand(9, 7), torch.rand(9, 4)
        for use_aux, use_encoding, in_dim in cases:
            model = TorchNRP(
                hidden_width=16,
                hidden_layers=2,
                encoding=enc,
                use_aux=use_aux,
                use_encoding=use_encoding,
            )
            self.assertEqual(model.mlp[0].in_features, in_dim)
            self.assertEqual(model(xy, aux, lp).shape, (9, 3))

    def test_disabled_inputs_are_ignored(self):
        model = TorchNRP(hidden_width=16, hidden_layers=2, use_aux=False, use_encoding=False)
        xy, lp = torch.rand(5, 2), torch.rand(5, 4)
        torch.testing.assert_close(model(xy, torch.rand(5, 7), lp), model(xy, torch.rand(5, 7), lp))

    def test_ablation_save_load_roundtrip(self):
        model = TorchNRP(hidden_width=16, hidden_layers=2, use_aux=False, use_encoding=False)
        xy, aux, lp = torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 4)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model.pt")
            model.save(path)
            loaded = TorchNRP.load(path)
        self.assertFalse(loaded.use_aux)
        self.assertFalse(loaded.use_encoding)
        torch.testing.assert_close(model(xy, aux, lp), loaded(xy, aux, lp))

    def test_param_broadcast_helpers(self):
        sp = sphere_params(torch.tensor([1.0, 2.0, 3.0]), torch.tensor(0.5), 4)
        self.assertEqual(sp.shape, (4, 4))
        qp = quad_params(
            torch.zeros(3), torch.tensor([0.0, 0.0, 2.0]), torch.tensor(1.0), torch.tensor(2.0), 4
        )
        self.assertEqual(qp.shape, (4, 8))
        torch.testing.assert_close(qp[0, 3:6], torch.tensor([0.0, 0.0, 1.0]))

    def test_tiled_relight_matches_untiled(self):
        cache = trace_path_cache(5, 4, 2, 1, seed=9)
        model = TorchNRP(
            hidden_width=8,
            hidden_layers=1,
            encoding={"levels": 1, "finest_resolution": 5},
        )
        lights = [SphereLight(center=[0.0, 0.5, 0.0], radius=0.2, rgb=[1.0, 0.5, 2.0])]
        np.testing.assert_allclose(
            relight_tiled(model, cache, lights, tile_pixels=3),
            relight(model, cache, lights),
            rtol=0.0,
            atol=1e-6,
        )

    def test_residual_quality_tier_is_exact_at_approval_light(self):
        cache = trace_path_cache(5, 4, 2, 1, seed=10)
        model = TorchNRP(
            hidden_width=8,
            hidden_layers=1,
            encoding={"levels": 1, "finest_resolution": 5},
        )
        lights = [SphereLight(center=[0.0, 0.5, 0.0], radius=0.2, rgb=[1.0, 0.5, 2.0])]
        image, metadata = render_quality_tier(
            model,
            cache,
            lights,
            quality="preview",
            residual_lights=lights,
        )
        np.testing.assert_allclose(image, gather_lights(cache, lights), rtol=0.0, atol=1e-12)
        self.assertEqual(metadata["quality"], "preview")
        self.assertTrue(metadata["residual_applied"])
        self.assertEqual(metadata["source"], "proxy_plus_cached_residual")

    def test_init_output_scale_sets_output_to_target_scale(self):
        # H1 fix (docs/hardening-track.md): the output head should start near a
        # given scale instead of nn.Linear's default softplus(~0) ~= 0.69.
        model = TorchNRP(hidden_width=16, hidden_layers=2, use_encoding=False)
        model.init_output_scale(0.0123)
        out = model(torch.rand(11, 2), torch.rand(11, 7), torch.rand(11, 4))
        torch.testing.assert_close(out, torch.full_like(out, 0.0123), atol=1e-6, rtol=0)

    def test_inverse_softplus_roundtrips_softplus(self):
        for y in (1e-4, 0.005, 0.5, 3.0):
            z = inverse_softplus(y)
            self.assertAlmostEqual(float(F.softplus(torch.tensor(z))), y, places=5)

    def test_relight_metadata_sidecar_is_written(self):
        image = np.zeros((2, 2, 3))
        metadata = {"quality": "draft", "source": "gatherlight_cached"}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "image.npy")
            write_image_with_metadata(path, image, metadata)
            np.testing.assert_array_equal(np.load(path), image)
            with open(f"{path}.json") as f:
                written = json.load(f)
        self.assertEqual(written["quality"], "draft")
        self.assertEqual(written["source"], "gatherlight_cached")


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
        tq = sample_light(
            self.cache,
            rng,
            "textured_quad",
            {
                "center": [0.0, 0.0, 1.0],
                "normal": [0.0, 0.0, -1.0],
                "width": 2.0,
                "height": 2.0,
                "texture_size": [2, 2],
            },
        )
        self.assertTrue(0.1 <= s.radius <= 0.2)
        self.assertTrue(0.1 <= q.width <= 0.3 and 0.1 <= q.height <= 0.3)
        self.assertIsInstance(tq, TexturedQuadLight)
        self.assertEqual(len(light_param_vector(s)), 4)
        self.assertEqual(len(light_param_vector(q)), 8)
        self.assertEqual(len(light_param_vector(tq)), 20)


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

    def test_textured_quad_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = {
                "cache": str(Path(tmp) / "cache.npz"),
                "out_dir": str(Path(tmp) / "out"),
                "trace": {"width": 10, "height": 10, "spp": 4, "bounces": 2, "seed": 2},
                "light_type": "textured_quad",
                "light_bounds": {
                    "center": [0.0, 0.0, 1.0],
                    "normal": [0.0, 0.0, -1.0],
                    "width": 2.0,
                    "height": 2.0,
                    "texture_size": [2, 2],
                    "texture_min": 0.1,
                    "texture_max": 1.0,
                },
                "pool": {"size": 6, "replace_every": 10, "replace_count": 1},
                "denoise": {"enabled": False},
                "iters": 20,
                "batch_pixels": 128,
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
            self.assertEqual(report["config"]["light_type"], "textured_quad")
            self.assertIn("val_psnr_db_vs_raw_mean", report)


class QuadZeroCollapseTests(unittest.TestCase):
    """H1 (docs/hardening-track.md): pins the dying-softplus collapse diagnosed on
    the kitchen-512 QuadLight rig lights and the fix (TorchNRP.init_output_scale).
    Mirrors the failure's real conditions without a slow PathCache/gather: a pool
    of targets shaped like the kitchen cache's QuadLight pool (median ~0.005,
    3% of slots ~100x brighter) at the rig's real architecture/lr, which drives
    nn.Linear's default output-head init (softplus(~0) ~= 0.69, ~130x the target
    median) into a persistent negative-gradient walk that Adam (per-step
    displacement ~lr, confirmed independent of eps/grad-clipping) carries past
    float32's softplus-derivative-underflow point within a normal training budget."""

    def _make_pool(self, seed=0, n_px=48 * 48, n_pool=32):
        gen = torch.Generator().manual_seed(seed)
        bright = torch.rand(n_pool, n_px, 1, generator=gen) < 0.03
        dim = torch.rand(n_pool, n_px, 1, generator=gen) * 0.01
        bright_val = 0.3 + torch.rand(n_pool, n_px, 1, generator=gen) * 1.0
        targets = torch.where(bright, bright_val, dim).expand(-1, -1, 3).contiguous()
        xy = torch.rand(n_px, 2, generator=gen)
        aux = torch.rand(n_px, 7, generator=gen)
        light_params = torch.rand(n_pool, 8, generator=gen)
        return targets, xy, aux, light_params

    def _train(self, targets, xy, aux, light_params, init_fix, iters=100, batch=2048, lr=0.005):
        torch.manual_seed(1)
        enc = {
            "levels": 10,
            "features_per_level": 2,
            "table_size_log2": 16,
            "base_resolution": 4,
            "finest_resolution": 512,
        }
        model = TorchNRP(light_type="quad", hidden_width=128, hidden_layers=4, encoding=enc)
        if init_fix:
            model.init_output_scale(float(targets.mean(dim=-1).median().item()))
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        n_pool, n_px = targets.shape[0], targets.shape[1]
        for _ in range(iters):
            pool_ids = torch.randint(0, n_pool, (batch,))
            pixel_ids = torch.randint(0, n_px, (batch,))
            pred = model(xy[pixel_ids], aux[pixel_ids], light_params[pool_ids])
            loss = relative_mse_loss(pred, targets[pool_ids, pixel_ids])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            return model(xy, aux, light_params[0].expand(n_px, -1))

    def test_collapses_without_fix(self):
        # Documents the bug: pins that it reproduces (would fail loudly, not
        # silently, if the pool/model/optimizer setup above ever stops
        # reproducing it -- the assertion is on the un-fixed path, not the fix).
        targets, xy, aux, light_params = self._make_pool()
        pred = self._train(targets, xy, aux, light_params, init_fix=False)
        self.assertEqual(float(pred.mean()), 0.0)

    def test_init_output_scale_prevents_collapse(self):
        targets, xy, aux, light_params = self._make_pool()
        pred = self._train(targets, xy, aux, light_params, init_fix=True)
        self.assertGreater(float(pred.mean()), 1e-4)


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
