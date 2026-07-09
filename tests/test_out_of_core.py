import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.out_of_core import (  # noqa: E402
    cache_segment_bytes,
    stream_shard_targets,
    train_image_proxy_monolithic,
    train_image_proxy_streamed,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.torch_backend.model import TorchNRP, relative_mse_loss  # noqa: E402
from nrp.torch_backend.streamed_train import _pixel_tensors, train_streamed  # noqa: E402
from nrp.torch_backend.train import ImagePool  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def _train_monolithic_for_test(cache, cfg):
    import torch

    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device("cpu")
    xy, aux = _pixel_tensors(cache, device)
    pool = ImagePool(cache, cfg, rng, device)
    model = TorchNRP(
        hidden_width=cfg["model"]["hidden_width"],
        hidden_layers=cfg["model"]["hidden_layers"],
        encoding=cfg["model"]["encoding"],
        light_type="sphere",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0))
    batch = cfg.get("batch_pixels", 64)
    n_px = xy.shape[0]
    loss_curve = []
    for it in range(cfg["iters"]):
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen).to(device)
        pool_ids = torch.randint(0, pool.size, (batch,), generator=gen).to(device)
        pred = model(xy[pixel_ids], aux[pixel_ids], pool.params[pool_ids])
        loss = relative_mse_loss(pred, pool.targets[pool_ids, pixel_ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
        if (it + 1) % cfg["pool"]["replace_every"] == 0:
            pool.replace_round()
    return loss_curve


class OutOfCoreTests(unittest.TestCase):
    def test_streamed_torchnrp_training_matches_monolithic(self):
        cache = trace_path_cache(10, 10, 4, 2, seed=21)
        cfg = {
            "seed": 0,
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
            "sampling": "segments",
            "denoise": {"enabled": False},
            "pool": {"size": 6, "replace_count": 1, "replace_every": 4},
            "model": {
                "hidden_width": 16,
                "hidden_layers": 2,
                "encoding": {"levels": 2, "features_per_level": 2, "finest_resolution": 10},
            },
            "lr": 5e-3,
            "batch_pixels": 64,
            "iters": 24,
        }
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shards"
            cache.save_sharded(str(shard_dir), tile_size=4)
            mono_loss = _train_monolithic_for_test(cache, cfg)
            _, streamed_stats = train_streamed(shard_dir, cache, cfg)
        self.assertEqual(mono_loss, streamed_stats["loss_curve"])
        self.assertLess(streamed_stats["peak_segment_bytes_loaded"], cache_segment_bytes(cache))

    def test_streamed_targets_match_monolithic_gather(self):
        cache = trace_path_cache(12, 12, 4, 2, seed=17)
        lights = [
            SphereLight(center=[0.1, 0.6, 0.0], radius=0.2, rgb=[1.5, 1.0, 0.75]),
            SphereLight(center=[0.75, 0.75, 0.35], radius=0.12, rgb=[0.8, 1.2, 1.0]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shards"
            cache.save_sharded(str(shard_dir), tile_size=4)
            streamed, stats = stream_shard_targets(shard_dir, lights)
            mono_proxy, _ = train_image_proxy_monolithic(streamed, epochs=3, lr=0.5)
            streamed_proxy, opt_stats = train_image_proxy_streamed(
                shard_dir, lights, epochs=3, lr=0.5
            )
        mono = sum(gather_light(cache, light) for light in lights) / len(lights)
        np.testing.assert_allclose(streamed, mono, atol=1e-12)
        np.testing.assert_allclose(streamed_proxy, mono_proxy, atol=1e-12)
        self.assertLess(stats["stream_peak_segments_loaded"], cache.segment_count)
        self.assertLess(stats["stream_peak_segment_bytes_loaded"], cache_segment_bytes(cache))
        self.assertLess(opt_stats["streamed_optimizer_peak_segments_loaded"], cache.segment_count)
        self.assertLess(
            opt_stats["streamed_optimizer_peak_segment_bytes_loaded"], cache_segment_bytes(cache)
        )
        self.assertGreater(stats["stream_peak_shard_file_bytes"], 0)
        self.assertGreater(stats["stream_process_rss_after_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
