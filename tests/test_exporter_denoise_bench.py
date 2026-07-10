import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.torch_backend.bench import available_devices, bench_model  # noqa: E402
from nrp.torch_backend.denoise import denoise_image, oidn_available  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402

HAVE_MITSUBA = importlib.util.find_spec("mitsuba") is not None


def _have_jit_variant() -> bool:
    if not HAVE_MITSUBA:
        return False
    import mitsuba as mi

    from nrp.mitsuba_exporter import pick_jit_variant

    return pick_jit_variant(mi) is not None


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba extra not installed")
class MitsubaExporterTests(unittest.TestCase):
    """Schema/semantics tests, run against the scalar loop (the reference)."""

    mode = "scalar"

    @classmethod
    def setUpClass(cls):
        from nrp.mitsuba_exporter import _load_mitsuba, _load_scene

        cls.mi = _load_mitsuba(cls.mode)
        scene = _load_scene(cls.mi, "builtin:cornell-box", 8, 8)
        cls.cache = cls._export(scene, cls.mi)

    @classmethod
    def _export(cls, scene, mi):
        from nrp.mitsuba_exporter import export_path_cache, export_path_cache_wavefront

        export = export_path_cache_wavefront if cls.mode == "wavefront" else export_path_cache
        return export(scene, mi, 8, 8, spp=2, max_bounces=3, seed=1, russian_roulette=False)

    def test_cache_validates_and_has_expected_counts(self):
        self.cache.validate()
        self.assertEqual((self.cache.width, self.cache.height), (8, 8))
        self.assertTrue((self.cache.n_paths == 2).all(), "spp paths per pixel without RR")
        # Closed box + 3 bounces: strictly more segments than paths, at most 3 per path.
        self.assertGreater(self.cache.segment_count, 128)
        self.assertLessEqual(self.cache.segment_count, 128 * 3)

    def test_aux_buffers_are_populated(self):
        self.assertGreater(float(self.cache.albedo.max()), 0.0)
        self.assertGreater(float(self.cache.depth.min()), 0.0, "camera outside geometry")
        norms = np.linalg.norm(self.cache.normal.reshape(-1, 3), axis=1)
        self.assertTrue(np.all(norms > 0.9), "first-hit normals must be unit length")

    def test_gather_light_in_scene_coordinates_is_nonzero(self):
        # The Mitsuba cornell box spans roughly [-1,1]^3; a big central emitter must
        # be crossed by many cached segments.
        light = SphereLight(center=[0.0, 0.0, 0.0], radius=0.6, rgb=[10.0, 10.0, 10.0])
        image = gather_light(self.cache, light)
        self.assertGreater(float(image.mean()), 0.0)

    def test_throughput_semantics_first_segment_is_unit(self):
        # Every path's first segment carries throughput 1 (before any bounce).
        firsts = {}
        for i in range(self.cache.segment_count):
            px = int(self.cache.seg_pixel[i])
            if px not in firsts:
                firsts[px] = self.cache.seg_throughput[i]
        for tp in firsts.values():
            np.testing.assert_allclose(tp, [1.0, 1.0, 1.0])

    def test_cli_optional_report_records_peak_rss_and_hardware(self):
        from nrp.mitsuba_exporter import main

        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "cache.npz"
            report_path = Path(tmp) / "report.json"
            argv = [
                "nrp.mitsuba_exporter",
                "--scene",
                "builtin:cornell-box",
                "--width",
                "4",
                "--height",
                "4",
                "--spp",
                "1",
                "--bounces",
                "2",
                "--mode",
                "scalar",
                "--out",
                str(cache_path),
                "--report",
                str(report_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                main()
            report = json.loads(report_path.read_text())
        self.assertEqual(report["resolution"], [4, 4])
        self.assertEqual(report["mode"], "scalar")
        self.assertGreater(report["peak_rss_bytes"], 0)
        self.assertIn("machine", report["hardware"])


@unittest.skipUnless(_have_jit_variant(), "no working Mitsuba JIT variant")
class MitsubaExporterWavefrontTests(MitsubaExporterTests):
    """The same schema/semantics suite against the drjit wavefront loop."""

    mode = "wavefront"


@unittest.skipUnless(_have_jit_variant(), "no working Mitsuba JIT variant")
class ScalarWavefrontEquivalenceTests(unittest.TestCase):
    def test_gather_light_means_statistically_compatible(self):
        # Fixed-seed exports of the 8x8 cornell box under both loops must produce
        # GATHERLIGHT images whose mean radiance agrees within 2% (independent MC
        # estimates of the same integral; 64 spp keeps the noise well below that).
        from nrp.mitsuba_exporter import (
            _load_mitsuba,
            _load_scene,
            export_path_cache,
            export_path_cache_wavefront,
        )

        light = SphereLight(center=[0.0, 0.0, 0.0], radius=0.6, rgb=[10.0, 10.0, 10.0])
        means = {}
        for mode, export in [
            ("scalar", export_path_cache),
            ("wavefront", export_path_cache_wavefront),
        ]:
            mi = _load_mitsuba(mode)
            scene = _load_scene(mi, "builtin:cornell-box", 8, 8)
            cache = export(scene, mi, 8, 8, spp=64, max_bounces=4, seed=0, russian_roulette=False)
            means[mode] = float(gather_light(cache, light).mean())
        rel = abs(means["scalar"] - means["wavefront"]) / means["scalar"]
        self.assertLess(rel, 0.02, f"means diverge: {means} ({rel * 100:.2f}%)")


@unittest.skipUnless(oidn_available(), "oidn extra not installed or lib unavailable")
class OIDNTests(unittest.TestCase):
    def test_denoises_hdr_and_preserves_mean(self):
        rng = np.random.default_rng(0)
        clean = np.full((32, 32, 3), 2.0)
        noisy = clean + rng.normal(0, 0.5, clean.shape)
        alb = np.full((32, 32, 3), 0.5)
        nrm = np.tile(np.array([0.0, 0.0, 1.0]), (32, 32, 1))
        out = denoise_image(noisy, alb, nrm, np.ones((32, 32)), method="oidn")
        self.assertLess(
            float(((out - clean) ** 2).mean()), float(((noisy - clean) ** 2).mean()) / 5
        )
        # HDR mode: values above 1.0 must survive (LDR mode would clamp toward 1).
        self.assertGreater(float(out.mean()), 1.5)


class DenoiseDispatchTests(unittest.TestCase):
    def test_unknown_method_raises(self):
        z3 = np.zeros((4, 4, 3))
        with self.assertRaises(ValueError):
            denoise_image(z3, z3, z3, np.zeros((4, 4)), method="nope")

    def test_bilateral_dispatch_matches_direct_call(self):
        from nrp.torch_backend.denoise import joint_bilateral_denoise

        rng = np.random.default_rng(1)
        img = rng.random((8, 8, 3))
        alb = rng.random((8, 8, 3))
        nrm = rng.random((8, 8, 3))
        dep = rng.random((8, 8))
        np.testing.assert_allclose(
            denoise_image(img, alb, nrm, dep, method="bilateral", radius=1),
            joint_bilateral_denoise(img, alb, nrm, dep, radius=1),
        )


class BenchTests(unittest.TestCase):
    def test_cpu_bench_reports_timing(self):
        model = TorchNRP(
            hidden_width=16,
            hidden_layers=2,
            encoding={"levels": 2, "table_size_log2": 8, "finest_resolution": 16},
        )
        row = bench_model(model, torch.device("cpu"), resolution=16, frames=3, warmup=1)
        self.assertEqual(row["pixels"], 256)
        self.assertGreater(row["ms_per_frame"], 0.0)
        self.assertGreater(row["hz"], 0.0)

    def test_available_devices_includes_cpu(self):
        self.assertIn("cpu", available_devices())


if __name__ == "__main__":
    unittest.main()
