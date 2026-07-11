"""Mixed-light-type color recovery + headless slider loop (roadmap V2 art-direction
loop): `RigColorReparam` narrows optimize_lights.py's inverse machinery to per-light
RGB recovery only (geometry fixed) across a mixed sphere/quad/textured_quad rig, and
`slider_loop` measures render latency for a sequence of interactive rgb nudges."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import QuadLight, SphereLight  # noqa: E402
from nrp.torch_backend.art_loop import (  # noqa: E402
    RigColorReparam,
    optimize_colors,
    predicted_image,
    slider_loop,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.rig import LightRig, RigLight, light_type_of  # noqa: E402
from nrp.torch_backend.train import pixel_tensors  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

TINY_ENCODING = {
    "levels": 2,
    "features_per_level": 2,
    "table_size_log2": 6,
    "base_resolution": 4,
    "finest_resolution": 12,
}


def tiny_cache(seed: int = 2):
    return trace_path_cache(width=10, height=10, spp=4, max_bounces=2, seed=seed)


def tiny_model(light_type: str, light_param_dim=None) -> TorchNRP:
    return TorchNRP(
        light_type=light_type,
        light_param_dim=light_param_dim,
        hidden_width=8,
        hidden_layers=1,
        encoding=TINY_ENCODING,
    ).eval()


def make_sphere(seed=0, rgb=None) -> SphereLight:
    rng = np.random.default_rng(seed)
    return SphereLight(
        center=rng.uniform(-0.2, 0.2, size=3),
        radius=0.1,
        rgb=np.asarray(rgb) if rgb is not None else rng.uniform(1.0, 4.0, size=3),
    )


def make_quad(seed=1, rgb=None) -> QuadLight:
    rng = np.random.default_rng(seed)
    return QuadLight(
        center=rng.uniform(-0.2, 0.2, size=3),
        normal=[0.0, 1.0, 0.0],
        width=0.2,
        height=0.2,
        rgb=np.asarray(rgb) if rgb is not None else rng.uniform(1.0, 4.0, size=3),
    )


def make_two_light_rig(sphere_rgb=None, quad_rgb=None):
    """A 2-light mixed rig (sphere + quad), each with its own untrained tiny proxy."""
    sphere = make_sphere(0, rgb=sphere_rgb)
    quad = make_quad(1, rgb=quad_rgb)
    lights = [RigLight(name="key", light=sphere), RigLight(name="fill", light=quad)]
    models = {
        "key": tiny_model(light_type_of(sphere)),
        "fill": tiny_model(light_type_of(quad)),
    }
    return LightRig(lights, models)


class RigColorReparamTests(unittest.TestCase):
    def test_rig_color_reparam_roundtrip(self):
        rig = make_two_light_rig()
        init_rgbs = {rl.name: rl.light.rgb for rl in rig.lights}
        reparam = RigColorReparam(rig, init_rgbs, torch.device("cpu"))
        back = reparam.to_rig()
        back_by_name = {rl.name: rl.light.rgb for rl in back.lights}
        for name, rgb in init_rgbs.items():
            np.testing.assert_allclose(back_by_name[name], rgb, atol=1e-5)


class PredictedImageTests(unittest.TestCase):
    def test_predicted_image_matches_render_before_optimization(self):
        cache = tiny_cache(seed=3)
        rig = make_two_light_rig()
        init_rgbs = {rl.name: rl.light.rgb for rl in rig.lights}
        reparam = RigColorReparam(rig, init_rgbs, torch.device("cpu"))
        xy, aux = pixel_tensors(cache, torch.device("cpu"))
        pred = predicted_image(rig, reparam, xy, aux)
        pred_np = (
            pred.detach().cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)
        )
        rendered = rig.render(cache)
        np.testing.assert_allclose(pred_np, rendered, atol=1e-4)


class OptimizeColorsTests(unittest.TestCase):
    def test_optimize_colors_reduces_loss_and_recovers_target_rgb(self):
        cache = tiny_cache(seed=4)
        # Target rig: some "true" rgbs.
        target_rig = make_two_light_rig(sphere_rgb=[2.0, 0.5, 1.0], quad_rgb=[0.3, 1.5, 2.0])
        target = target_rig.render(cache)

        # Initial guess rig: same models/geometry, different starting rgbs.
        guess_rig = LightRig(
            [
                RigLight(name="key", light=make_sphere(0, rgb=[0.5, 0.5, 0.5])),
                RigLight(name="fill", light=make_quad(1, rgb=[0.5, 0.5, 0.5])),
            ],
            target_rig.models,
        )

        xy, aux = pixel_tensors(cache, torch.device("cpu"))
        init_rgbs = {rl.name: rl.light.rgb for rl in guess_rig.lights}
        pre_reparam = RigColorReparam(guess_rig, init_rgbs, torch.device("cpu"))
        with torch.no_grad():
            pre_pred = (
                predicted_image(guess_rig, pre_reparam, xy, aux)
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(cache.height, cache.width, 3)
            )
        from nrp.metrics import psnr

        pre_psnr = psnr(pre_pred, target)

        report = optimize_colors(guess_rig, cache, target, steps=150, lr=0.05, seed=0)
        self.assertLess(report["proxy_loss_last"], report["proxy_loss_first"])
        self.assertGreater(report["proxy_vs_target_psnr_db"], pre_psnr)
        self.assertIn("optimized_rig", report)
        self.assertIn("proxy_loss_curve", report)
        self.assertIn("proxy_vs_target_ssim", report)


class OptimizedRigReloadTests(unittest.TestCase):
    def test_optimized_rig_is_reloadable(self):
        cache = tiny_cache(seed=5)
        target_rig = make_two_light_rig(sphere_rgb=[1.5, 0.7, 1.2], quad_rgb=[0.6, 1.8, 1.0])
        target = target_rig.render(cache)
        guess_rig = LightRig(
            [
                RigLight(name="key", light=make_sphere(0, rgb=[0.4, 0.4, 0.4])),
                RigLight(name="fill", light=make_quad(1, rgb=[0.4, 0.4, 0.4])),
            ],
            target_rig.models,
        )
        report = optimize_colors(guess_rig, cache, target, steps=20, lr=0.05, seed=0)
        optimized_rig = report["optimized_rig"]
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "rig.json")
            optimized_rig.save(path)
            loaded = LightRig.load(path, target_rig.models)
        self.assertEqual(
            [rl.name for rl in loaded.lights], [rl.name for rl in optimized_rig.lights]
        )
        for a, b in zip(loaded.lights, optimized_rig.lights, strict=True):
            self.assertEqual(a.light.to_dict(), b.light.to_dict())


class SliderLoopTests(unittest.TestCase):
    def test_slider_loop_applies_adjustments_and_measures_latency(self):
        cache = tiny_cache(seed=6)
        rig = make_two_light_rig(sphere_rgb=[0.1, 0.1, 0.1], quad_rgb=[0.1, 0.1, 0.1])
        adjustments = [
            {"light": "key", "rgb": [0.5, 0.5, 0.5]},
            {"light": "key", "rgb": [1.5, 1.5, 1.5]},
            {"light": "key", "rgb": [3.0, 3.0, 3.0]},
        ]
        result = slider_loop(rig, cache, adjustments, device=torch.device("cpu"))
        self.assertEqual(result["n_adjustments"], 3)
        self.assertEqual(len(result["latency_ms"]), 3)
        self.assertGreater(result["latency_ms_mean"], 0.0)
        self.assertIn("latency_ms_p95", result)

        # Brightness should track the increasing rgb nudges (rig unmodified between
        # calls other than through slider_loop's own working copy).
        brightness = []
        working = copy.deepcopy(rig)
        for adj in adjustments:
            for rl in working.lights:
                if rl.name == adj["light"]:
                    rl.light.rgb = np.asarray(adj["rgb"], dtype=np.float64)
            brightness.append(float(working.render(cache).mean()))
        self.assertLess(brightness[0], brightness[1])
        self.assertLess(brightness[1], brightness[2])


if __name__ == "__main__":
    unittest.main()
