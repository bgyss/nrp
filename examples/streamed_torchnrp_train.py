"""E5 remaining criterion: streamed TorchNRP pool training vs an in-memory run.

Trains a real sphere-light TorchNRP proxy two ways from the *same* segments and the
*same* seed: (a) the standard in-memory `ImagePool` (nrp.torch_backend.train), and
(b) `StreamedImagePool` (nrp.torch_backend.streamed_train), which renders pool targets
by visiting on-disk shard tiles instead of holding the whole cache's segment arrays
resident. Both use an identical training loop (same rng draw order) so the loss
curves and held-out PSNR are directly comparable at equal iterations.

Verifies the E5 criterion: streamed training matches in-memory within 0.3 dB
held-out PSNR at equal iterations/seed. Reports peak resident segment bytes for
each path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP, relative_mse_loss  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.streamed_train import _pixel_tensors, train_streamed  # noqa: E402
from nrp.torch_backend.train import ImagePool, light_param_vector  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def cache_segment_bytes(cache) -> int:
    return int(
        cache.seg_pixel.nbytes
        + cache.seg_origin.nbytes
        + cache.seg_dir.nbytes
        + cache.seg_tmax.nbytes
        + cache.seg_throughput.nbytes
    )


def train_monolithic(cache, cfg: dict) -> tuple[TorchNRP, dict]:
    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))
    xy, aux = _pixel_tensors(cache, device)

    t_pool0 = time.perf_counter()
    pool = ImagePool(cache, cfg, rng, device)
    pool_seconds = time.perf_counter() - t_pool0

    model = TorchNRP(
        hidden_width=cfg["model"]["hidden_width"],
        hidden_layers=cfg["model"]["hidden_layers"],
        encoding=cfg["model"]["encoding"],
        light_type="sphere",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0))

    batch = cfg.get("batch_pixels", 512)
    replace_every = cfg["pool"]["replace_every"]
    n_px = xy.shape[0]
    loss_curve = []
    t0 = time.perf_counter()
    for it in range(cfg["iters"]):
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen).to(device)
        pool_ids = torch.randint(0, pool.size, (batch,), generator=gen).to(device)
        pred = model(xy[pixel_ids], aux[pixel_ids], pool.params[pool_ids])
        loss = relative_mse_loss(pred, pool.targets[pool_ids, pixel_ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
        if (it + 1) % replace_every == 0:
            pool.replace_round()
    train_seconds = time.perf_counter() - t0
    return model, {
        "pool_seconds": pool_seconds,
        "train_seconds": train_seconds,
        "loss_curve": loss_curve,
    }


def held_out_psnr(model, cache, cfg: dict, n_lights: int = 8) -> float:
    """PSNR of the trained proxy vs raw GATHERLIGHT on fresh sphere lights, dedicated RNG."""
    val_rng = np.random.default_rng([cfg.get("seed", 0), 0x5EED])
    device = torch.device(cfg.get("device", "cpu"))
    xy, aux = _pixel_tensors(cache, device)
    n_px = xy.shape[0]
    vals = []
    model.eval()
    with torch.no_grad():
        for _ in range(n_lights):
            light = sample_light(
                cache, val_rng, "sphere", cfg["light_bounds"], cfg.get("sampling", "segments")
            )
            raw = gather_light(cache, light).reshape(-1, 3)
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
            p = psnr(pred, raw)
            if np.isfinite(p):
                vals.append(p)
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/out-of-core/streamed_torchnrp_report.json")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=6)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(args.width, args.height, args.spp, args.bounces, seed=7)
    shard_dir = base / "streamed_cache_sharded"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    cache.save_sharded(str(shard_dir), tile_size=args.tile_size)

    cfg = {
        "seed": 0,
        "device": "cpu",
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
        "sampling": "segments",
        "denoise": {"enabled": False},
        "pool": {"size": 12, "replace_count": 1, "replace_every": 5},
        "model": {
            "hidden_width": 32,
            "hidden_layers": 2,
            "encoding": {
                "levels": 2,
                "features_per_level": 2,
                "finest_resolution": args.width,
            },
        },
        "lr": 5e-3,
        "batch_pixels": 256,
        "iters": args.iters,
    }

    mono_model, mono_stats = train_monolithic(cache, cfg)
    streamed_model, streamed_stats = train_streamed(shard_dir, cache, cfg)

    mono_psnr = held_out_psnr(mono_model, cache, cfg)
    streamed_psnr = held_out_psnr(streamed_model, cache, cfg)

    report = {
        "resolution": [args.width, args.height],
        "segments": cache.segment_count,
        "tile_size": args.tile_size,
        "iters": args.iters,
        "monolithic": {
            "pool_seconds": mono_stats["pool_seconds"],
            "train_seconds": mono_stats["train_seconds"],
            "loss_first": mono_stats["loss_curve"][0],
            "loss_last": mono_stats["loss_curve"][-1],
            "held_out_psnr_db": mono_psnr,
            "resident_segment_bytes": cache_segment_bytes(cache),
        },
        "streamed": {
            "pool_seconds": streamed_stats["pool_seconds"],
            "train_seconds": streamed_stats["train_seconds"],
            "loss_first": streamed_stats["loss_curve"][0],
            "loss_last": streamed_stats["loss_curve"][-1],
            "held_out_psnr_db": streamed_psnr,
            "peak_segment_bytes_loaded": streamed_stats["peak_segment_bytes_loaded"],
        },
        "psnr_gap_db": abs(mono_psnr - streamed_psnr),
        "matches_within_0p3db": bool(abs(mono_psnr - streamed_psnr) <= 0.3),
        "resident_segment_memory_ratio": cache_segment_bytes(cache)
        / max(streamed_stats["peak_segment_bytes_loaded"], 1),
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
