"""LightRig core (roadmap V1): per-light-name proxies, solo/mute, N-way composite,
and the non-relightable monolithic baseline used to evaluate it against."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import QuadLight, SphereLight, TexturedQuadLight  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.rig import (  # noqa: E402
    LightRig,
    RigLight,
    light_type_of,
    train_monolithic,
)
from nrp.torch_backend.train import light_param_vector  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

TINY_ENCODING = {
    "levels": 2,
    "features_per_level": 2,
    "table_size_log2": 6,
    "base_resolution": 4,
    "finest_resolution": 12,
}


def tiny_cache(seed: int = 2):
    """12x12/4spp toy cache, small enough for fast per-light proxy tests."""
    return trace_path_cache(width=12, height=12, spp=4, max_bounces=2, seed=seed)


def tiny_model(light_type: str, light_param_dim=None) -> TorchNRP:
    """An untrained, tiny-capacity model: the rig tests below only check structural
    identities (solo/mute selection, additivity, serialization), not render accuracy,
    so training would just slow the suite down for no assertion benefit."""
    return TorchNRP(
        light_type=light_type,
        light_param_dim=light_param_dim,
        hidden_width=8,
        hidden_layers=1,
        encoding=TINY_ENCODING,
    ).eval()


def make_sphere(seed=0) -> SphereLight:
    rng = np.random.default_rng(seed)
    return SphereLight(
        center=rng.uniform(-0.2, 0.2, size=3), radius=0.1, rgb=rng.uniform(1.0, 4.0, size=3)
    )


def make_quad(seed=1) -> QuadLight:
    rng = np.random.default_rng(seed)
    return QuadLight(
        center=rng.uniform(-0.2, 0.2, size=3),
        normal=[0.0, 1.0, 0.0],
        width=0.2,
        height=0.2,
        rgb=rng.uniform(1.0, 4.0, size=3),
    )


def make_textured_quad(seed=2) -> TexturedQuadLight:
    rng = np.random.default_rng(seed)
    return TexturedQuadLight(
        center=rng.uniform(-0.2, 0.2, size=3),
        normal=[0.0, 1.0, 0.0],
        width=0.2,
        height=0.2,
        texture=rng.uniform(0.5, 2.0, size=(2, 2, 3)),
    )


class RigLightRoundtripTests(unittest.TestCase):
    def test_rig_light_roundtrip(self):
        for light in (make_sphere(), make_quad(), make_textured_quad()):
            rl = RigLight(name="key", light=light, mute=True, solo=False)
            back = RigLight.from_dict(rl.to_dict())
            self.assertEqual(back.name, "key")
            self.assertTrue(back.mute)
            self.assertFalse(back.solo)
            self.assertEqual(back.light.to_dict(), light.to_dict())


class LightRigJsonRoundtripTests(unittest.TestCase):
    def test_light_rig_json_roundtrip(self):
        sphere, quad, tq = make_sphere(), make_quad(), make_textured_quad()
        lights = [
            RigLight(name="sphere_key", light=sphere),
            RigLight(name="quad_key", light=quad, mute=True),
            RigLight(name="tq_key", light=tq, solo=True),
        ]
        models = {
            "sphere_key": tiny_model(light_type_of(sphere)),
            "quad_key": tiny_model(light_type_of(quad)),
            "tq_key": tiny_model(light_type_of(tq), light_param_dim=len(light_param_vector(tq))),
        }
        rig = LightRig(lights, models)
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "rig.json")
            rig.save(path)
            loaded = LightRig.load(path, models)
        self.assertEqual(loaded.to_dict(), rig.to_dict())


class MuteSoloRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = tiny_cache()

    def test_mute_excludes_light_from_render(self):
        a, b = make_sphere(0), make_sphere(5)
        rig_lights = [
            RigLight(name="a", light=a, mute=True),
            RigLight(name="b", light=b),
        ]
        models = {"a": tiny_model("sphere"), "b": tiny_model("sphere")}
        rig = LightRig(rig_lights, models)
        rendered = rig.render(self.cache)
        per_light = rig.render_per_light(self.cache)
        self.assertEqual(list(per_light.keys()), ["b"])
        np.testing.assert_allclose(rendered, per_light["b"], atol=1e-5)

    def test_solo_overrides_mute_and_other_lights(self):
        a, b, c = make_sphere(0), make_sphere(5), make_sphere(9)
        rig_lights = [
            RigLight(name="a", light=a, mute=True),
            RigLight(name="b", light=b),
            RigLight(name="c", light=c, solo=True),
        ]
        models = {n: tiny_model("sphere") for n in ("a", "b", "c")}
        rig = LightRig(rig_lights, models)
        active = rig.active_lights()
        self.assertEqual([rl.name for rl in active], ["c"])

    def test_render_equals_sum_of_render_per_light(self):
        a, b, c = make_sphere(0), make_sphere(5), make_sphere(9)
        rig_lights = [
            RigLight(name="a", light=a),
            RigLight(name="b", light=b),
            RigLight(name="c", light=c),
        ]
        models = {n: tiny_model("sphere") for n in ("a", "b", "c")}
        rig = LightRig(rig_lights, models)
        rendered = rig.render(self.cache)
        per_light = rig.render_per_light(self.cache)
        summed = sum(per_light.values())
        np.testing.assert_allclose(rendered, summed, atol=1e-5)


class TrainMonolithicTests(unittest.TestCase):
    def test_train_monolithic_reduces_loss_and_is_not_light_conditioned(self):
        cache = tiny_cache(seed=3)
        rng = np.random.default_rng(11)
        target_image = rng.uniform(0.1, 2.0, size=(cache.height, cache.width, 3))
        model, loss_curve = train_monolithic(
            cache,
            target_image,
            hidden_width=16,
            hidden_layers=2,
            iters=150,
            lr=0.01,
            seed=0,
        )
        self.assertEqual(len(loss_curve), 150)
        first_mean = float(np.mean(loss_curve[:20]))
        last_mean = float(np.mean(loss_curve[-20:]))
        self.assertLess(last_mean, first_mean)
        self.assertEqual(model.light_param_dim, 0)

        n_px = cache.height * cache.width
        from nrp.torch_backend.train import pixel_tensors

        xy, aux = pixel_tensors(cache, torch.device("cpu"))
        params_a = torch.zeros((n_px, 0), dtype=torch.float32)
        params_b = torch.rand((n_px, 0), dtype=torch.float32)
        with torch.no_grad():
            out_a = model(xy, aux, params_a)
            out_b = model(xy, aux, params_b)
        np.testing.assert_allclose(out_a.numpy(), out_b.numpy())


class V1ReportSmokeTests(unittest.TestCase):
    """Fast smoke test for examples/v1_rig.py's core function at toy scale: a tiny
    cache, tiny iteration counts, and one light per type (not the full 8-light T1
    rig), just enough to exercise the per-light train -> LightRig -> additivity
    gate -> monolithic baseline -> compositing overhead pipeline end to end,
    including the textured_quad path (LightRig.render_per_light has no `.rgb` to
    multiply by for TexturedQuadLight, unlike sphere/quad)."""

    def test_build_and_evaluate_rig_smoke(self):
        import sys
        import tempfile
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
        from v1_rig import build_and_evaluate_rig  # noqa: E402

        cache = tiny_cache(seed=4)
        lights = [
            RigLight(name="a", light=make_sphere(0)),
            RigLight(name="b", light=make_quad(1)),
            RigLight(name="c", light=make_textured_quad(2)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = str(Path(tmp) / "path_cache.npz")
            cache.save(cache_path)
            base_cfg = {
                "cache": cache_path,
                "sampling": "segments",
                "gather_backend": "numpy",
                "pool": {"size": 4, "replace_every": 5, "replace_count": 1},
                "denoise": {"enabled": True, "method": "bilateral"},
                "batch_pixels": 32,
                "lr": 0.01,
                "model": {"hidden_width": 8, "hidden_layers": 1, "encoding": TINY_ENCODING},
                "n_val_lights": 2,
                "seed": 0,
                "device": "cpu",
            }
            report = build_and_evaluate_rig(
                cache,
                lights,
                out_dir=tmp,
                base_cfg=base_cfg,
                iters=3,
                monolithic_hidden_width=8,
                monolithic_hidden_layers=1,
                monolithic_iters=3,
                monolithic_lr=0.01,
                gate_tier="preview",
                overhead_frames=2,
                overhead_warmup=1,
            )
        for key in (
            "rig",
            "per_light_training",
            "additivity_gate",
            "monolithic_baseline",
            "sizes_bytes",
            "compositing_overhead_ms",
            "hardware",
        ):
            self.assertIn(key, report)
        self.assertTrue(np.isfinite(report["additivity_gate"]["metrics"]["psnr_db"]["value"]))


if __name__ == "__main__":
    unittest.main()
