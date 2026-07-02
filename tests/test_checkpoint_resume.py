"""Checkpoint/resume and cosine LR decay for long torch runs (roadmap item 6):
resuming from a mid-run checkpoint must reproduce the uninterrupted loss curve."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import torch  # noqa: E402

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
        "iters": 200,
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
        "checkpoint": {"every": 100},
    }
    cfg.update(overrides)
    cfg_path = Path(tmp) / f"{out}.json"
    cfg_path.write_text(json.dumps(cfg))
    return load_config(str(cfg_path))


class CheckpointResumeTests(unittest.TestCase):
    def test_resume_reproduces_uninterrupted_loss_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Uninterrupted 200-iteration reference run.
            full = train(tiny_cfg(tmp, "full"))
            # Interrupted run: 100 iterations (checkpoint lands at exactly 100),
            # then resume the same out_dir with the full 200-iteration budget.
            train(tiny_cfg(tmp, "split", iters=100))
            resumed = train(tiny_cfg(tmp, "split"), resume=True)

            ck_full = torch.load(str(Path(tmp) / "full" / "checkpoint.pt"), weights_only=False)
            ck_split = torch.load(str(Path(tmp) / "split" / "checkpoint.pt"), weights_only=False)
            self.assertEqual(ck_full["iteration"], 200)
            self.assertEqual(ck_split["iteration"], 200)
            curve_full = np.array(ck_full["loss_curve"])
            curve_split = np.array(ck_split["loss_curve"])
            self.assertEqual(curve_full.shape, curve_split.shape)
            # Same seed, same trajectory: on CPU the resumed run replays the exact
            # RNG/pool/optimizer state, so the curves agree to float noise.
            np.testing.assert_allclose(curve_split, curve_full, rtol=1e-5, atol=1e-7)
            self.assertAlmostEqual(
                resumed["val_psnr_db_vs_raw_mean"],
                full["val_psnr_db_vs_raw_mean"],
                places=3,
            )
            # The checkpoint PSNR curve exists at both checkpoint iterations.
            self.assertEqual([c["iteration"] for c in resumed["checkpoint_metrics"]], [100, 200])

    def test_cosine_schedule_decays_to_lr_min(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = tiny_cfg(tmp, "cos", lr_schedule="cosine", lr_min=1e-5, iters=50)
            cfg.pop("checkpoint")

            from nrp.torch_backend import train as train_mod

            captured = {}
            orig = train_mod.torch.optim.lr_scheduler.CosineAnnealingLR

            class Spy(orig):
                def __init__(self, opt, **kw):
                    super().__init__(opt, **kw)
                    captured["sched"] = self

            train_mod.torch.optim.lr_scheduler.CosineAnnealingLR = Spy
            try:
                train(cfg)
            finally:
                train_mod.torch.optim.lr_scheduler.CosineAnnealingLR = orig
            final_lr = captured["sched"].get_last_lr()[0]
            self.assertAlmostEqual(final_lr, 1e-5, places=6)


if __name__ == "__main__":
    unittest.main()
