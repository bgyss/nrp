"""Supervision accounting for the image-based baseline (roadmap item 9): a
replace_count-0 pool is a fixed dataset, and the recorded supervision lights are
exactly what training consumed."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.train import train  # noqa: E402
from nrp.train import load_config  # noqa: E402


def tiny_cfg(tmp: str, out: str, **overrides) -> dict:
    cfg = {
        "cache": str(Path(tmp) / "cache.npz"),
        "out_dir": str(Path(tmp) / out),
        "trace": {"width": 12, "height": 12, "spp": 4, "bounces": 2, "seed": 2},
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
        "sampling": "segments",
        "pool": {"size": 8, "replace_every": 5, "replace_count": 1},
        "denoise": {"enabled": True, "radius": 1},
        "iters": 40,
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
        "seed": 0,
        "device": "cpu",
        "record_supervision_lights": True,
    }
    cfg.update(overrides)
    cfg_path = Path(tmp) / f"{out}.json"
    cfg_path.write_text(json.dumps(cfg))
    return load_config(str(cfg_path))


class SupervisionAccountingTests(unittest.TestCase):
    def test_fixed_pool_consumes_exactly_pool_size_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp, "fixed", pool={"size": 8, "replace_every": 5, "replace_count": 0})
            report = train(cfg)
            self.assertEqual(report["supervision_images"], 8)
            self.assertEqual(len(report["supervision_light_params"]), 8)
            self.assertGreater(report["supervision_seconds"], 0.0)

    def test_replacing_pool_counts_build_plus_replacements(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp, "live")  # pool 8, 1 image every 5 iters, 40 iters
            report = train(cfg)
            self.assertEqual(report["supervision_images"], 8 + 40 // 5)
            # Recorded vectors are valid sphere params (center xyz + radius > 0).
            params = np.asarray(report["supervision_light_params"])
            self.assertEqual(params.shape, (16, 4))
            self.assertTrue(np.all(params[:, 3] > 0))
            # The two regimes' first 8 supervision lights coincide (same seed and
            # rng), which is what makes the image-based comparison seed-matched.
            fixed = train(
                tiny_cfg(tmp, "fixed2", pool={"size": 8, "replace_every": 5, "replace_count": 0})
            )
            np.testing.assert_allclose(np.asarray(fixed["supervision_light_params"]), params[:8])


if __name__ == "__main__":
    unittest.main()
