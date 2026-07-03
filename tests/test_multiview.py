"""Multi-view NRPs (roadmap item 7, §6.1): per-view camera export, the view manifest,
cross-view consistency machinery, and the relight_multiview CLI."""

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend import relight_multiview  # noqa: E402
from nrp.torch_backend.relight_multiview import (  # noqa: E402
    cross_view_consistency,
    edit_latency_ms,
    load_views,
    relight_all,
)
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.train import train  # noqa: E402

HAVE_MITSUBA = importlib.util.find_spec("mitsuba") is not None


def tiny_view_cfg(tmp: str, name: str, seed: int) -> dict:
    """Tiny per-view training config (same shape as examples/multiview.py's, scaled
    down); both test views share one toy cache — the multi-view machinery only cares
    that each view has its own (model, cache) pair."""
    return {
        "cache": str(Path(tmp) / "cache.npz"),
        "out_dir": str(Path(tmp) / name),
        "trace": {"width": 12, "height": 12, "spp": 4, "bounces": 2, "seed": 2},
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
        "sampling": "segments",
        "pool": {"size": 8, "replace_every": 5, "replace_count": 1},
        "denoise": {"enabled": True, "radius": 1},
        "iters": 60,
        "batch_pixels": 256,
        "lr": 0.005,
        "model": {
            "hidden_width": 16,
            "hidden_layers": 2,
            "encoding": {
                "levels": 2,
                "features_per_level": 2,
                "table_size_log2": 6,
                "base_resolution": 4,
                "finest_resolution": 12,
            },
        },
        "n_val_lights": 2,
        "seed": seed,
        "device": "cpu",
    }


class MultiViewTests(unittest.TestCase):
    """Manifest loading, cross-view consistency, edit latency, and the CLI, on two
    tiny toy-scene views trained once for the whole class."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        for name, seed in (("front", 0), ("side", 1)):
            train(tiny_view_cfg(tmp, name, seed))
        cls.manifest = Path(tmp) / "views.json"
        cls.manifest.write_text(
            json.dumps(
                [
                    {"name": "front", "model": "front/model.pt", "cache": "cache.npz"},
                    {"name": "side", "model": "side/model.pt", "cache": "cache.npz"},
                ]
            )
        )
        cls.views = load_views(str(cls.manifest), device="cpu")
        rng = np.random.default_rng(7)
        cls.light = sample_light(
            cls.views[0].cache,
            rng,
            "sphere",
            {"radius_min": 0.1, "radius_max": 0.2},
            "segments",
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_load_views_resolves_paths_and_names(self):
        self.assertEqual([v.name for v in self.views], ["front", "side"])
        for view in self.views:
            self.assertEqual((view.cache.width, view.cache.height), (12, 12))
            self.assertGreater(view.model_bytes, 0)

    def test_one_edit_renders_every_view(self):
        images = relight_all(self.views, [self.light])
        self.assertEqual(sorted(images), ["front", "side"])
        for image in images.values():
            self.assertEqual(image.shape, (12, 12, 3))
            self.assertTrue(np.all(image >= 0.0), "softplus head: non-negative radiance")
        # Different seeds trained different weights: the views must not be clones.
        self.assertFalse(np.allclose(images["front"], images["side"]))

    def test_cross_view_consistency_reports_per_view_psnr_and_spread(self):
        result = cross_view_consistency(self.views, [self.light])
        self.assertEqual(len(result["per_view"]), 2)
        for row in result["per_view"]:
            self.assertTrue(np.isfinite(row["psnr_db"]))
        self.assertGreaterEqual(result["psnr_db_spread"], 0.0)
        self.assertAlmostEqual(
            result["psnr_db_spread"], result["psnr_db_max"] - result["psnr_db_min"]
        )

    def test_edit_latency_is_measured_synchronized(self):
        ms = edit_latency_ms(self.views, [self.light], frames=2, warmup=1)
        self.assertGreater(ms, 0.0)

    def test_cli_smoke_writes_one_image_per_view(self):
        out_dir = Path(self._tmp.name) / "edit"
        light = json.dumps(
            {
                "type": "sphere",
                "center": [float(c) for c in self.light.center],
                "radius": float(self.light.radius),
                "rgb": [1.0, 1.0, 1.0],
            }
        )
        argv = [
            "relight_multiview",
            "--views",
            str(self.manifest),
            "--light",
            light,
            "--out-dir",
            str(out_dir),
            "--bench",
            "2",
        ]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(buf):
            relight_multiview.main()
        for name in ("front", "side"):
            image = np.load(out_dir / f"{name}.npy")
            self.assertEqual(image.shape, (12, 12, 3))
        self.assertIn("ms/edit", buf.getvalue())


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba extra not installed")
class ExporterViewOverrideTests(unittest.TestCase):
    """--sensor-index / per-view camera override in the Mitsuba exporter (scalar loop)."""

    @classmethod
    def setUpClass(cls):
        from nrp.mitsuba_exporter import _load_mitsuba, _load_scene

        cls.mi = _load_mitsuba("scalar")
        cls.scene = _load_scene(cls.mi, "builtin:cornell-box", 8, 8)

    def _export(self, sensor=None, sensor_index=0):
        from nrp.mitsuba_exporter import export_path_cache

        return export_path_cache(
            self.scene,
            self.mi,
            8,
            8,
            spp=2,
            max_bounces=2,
            seed=1,
            russian_roulette=False,
            sensor_index=sensor_index,
            sensor=sensor,
        )

    def test_sensor_index_zero_matches_default(self):
        a = self._export()
        b = self._export(sensor_index=0)
        np.testing.assert_allclose(a.seg_origin, b.seg_origin)
        np.testing.assert_allclose(a.seg_tmax, b.seg_tmax)
        np.testing.assert_allclose(a.position, b.position)

    def test_camera_override_produces_a_different_valid_view(self):
        from nrp.mitsuba_exporter import build_sensor

        side = build_sensor(self.mi, 8, 8, origin=[1.2, 0.3, 3.5], target=[-0.2, 0.0, 0.0])
        default = self._export()
        moved = self._export(sensor=side)
        moved.validate()
        # Same scene, different camera: full path counts, but the first-hit G-buffer
        # (positions, depths) must differ substantially between the two views.
        self.assertTrue((moved.n_paths == 2).all())
        self.assertGreater(
            float(np.abs(moved.position - default.position).max()),
            0.1,
            "camera override did not change the first-hit positions",
        )
        self.assertFalse(np.allclose(moved.depth, default.depth))
        # Ray origins must sit at the overridden camera position (near-clip offset).
        first_origins = moved.seg_origin[np.unique(moved.seg_pixel, return_index=True)[1]]
        self.assertLess(
            float(np.abs(first_origins - np.array([1.2, 0.3, 3.5])).max()),
            0.05,
        )

    def test_sensor_index_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            self._export(sensor_index=5)


if __name__ == "__main__":
    unittest.main()
