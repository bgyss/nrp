"""S2: 1024x1024 kitchen — packed sharding + S1-accelerated streamed training report.

Takes an exported 1024^2 Country Kitchen cache (spp chosen by
`examples/s2_spp_probe.py`), shards it packed with the S3 parallel writer, trains a
sphere-light proxy through the S1 streamed path (torch gather + threaded decode),
and measures held-out PSNR and tiled full-frame inference at 1024^2 — the first
run of the pipeline above 512^2.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.mitsuba_512_streamed import (  # noqa: E402
    cache_segment_bytes,
    current_rss_bytes,
    directory_bytes,
    held_out_psnr,
)
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.relight import relight_tiled  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.streamed_train import train_streamed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default="out/kitchen-1024/path_cache.npz")
    parser.add_argument("--out", default="out/s2-scale/kitchen_1024_report.json")
    parser.add_argument("--shard-dir", default="out/s2-scale/kitchen_1024_sharded")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--gather-device", default="mps")
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--tile-pixels", type=int, default=16384)
    parser.add_argument("--spp", type=int, required=True, help="spp of the export (recorded)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise SystemExit(
            f"{cache_path} missing; export first: uv run python -m nrp.mitsuba_exporter "
            f"--scene examples/scenes/kitchen/scene.xml --width 1024 --height 1024 "
            f"--spp {args.spp} --bounces 4 --out {cache_path} "
            f"--report {cache_path.parent / 'export_report.json'}"
        )
    cache = PathCache.load(str(cache_path))

    shard_dir = Path(args.shard_dir)
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    t0 = time.perf_counter()
    cache.save_sharded(str(shard_dir), tile_size=args.tile_size, packed=True, workers=8)
    shard_seconds = time.perf_counter() - t0

    cfg = {
        "seed": 0,
        "device": "cpu",
        "light_type": "sphere",
        # kitchen scene units (meters): same bounds family as T1/V1 kitchen runs
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.6},
        "sampling": "segments",
        "denoise": {"enabled": False},
        "pool": {"size": 16, "replace_count": 1, "replace_every": 6},
        "model": {
            "hidden_width": 64,
            "hidden_layers": 3,
            "encoding": {"levels": 4, "features_per_level": 2, "finest_resolution": 1024},
        },
        "lr": 5e-3,
        "batch_pixels": 4096,
        "iters": args.iters,
        "gather_backend": "torch",
        "gather_device": args.gather_device,
        "decode_workers": args.decode_workers,
    }

    rss_before = current_rss_bytes()
    t0 = time.perf_counter()
    model, stats = train_streamed(shard_dir, cache, cfg)
    total_train = time.perf_counter() - t0
    rss_after = current_rss_bytes()

    psnr_db = held_out_psnr(model, cache, cfg)

    light = sample_light(cache, np.random.default_rng(123), "sphere", cfg["light_bounds"])
    t0 = time.perf_counter()
    relight_tiled(model, cache, [light], tile_pixels=args.tile_pixels)
    tiled_seconds = time.perf_counter() - t0

    report = {
        "rung": "S2",
        "scope": "1024x1024 kitchen: packed shard + S1 streamed train + tiled inference",
        "resolution": [cache.width, cache.height],
        "spp": args.spp,
        "segments": cache.segment_count,
        "monolithic_cache_bytes": cache_path.stat().st_size,
        "monolithic_resident_segment_bytes": cache_segment_bytes(cache),
        "sharded_cache_bytes": directory_bytes(shard_dir),
        "save_sharded_seconds": shard_seconds,
        "tile_size": args.tile_size,
        "streamed_train": {
            "config": {k: v for k, v in cfg.items() if k != "model"},
            "pool_seconds": stats["pool_seconds"],
            "train_seconds": stats["train_seconds"],
            "total_seconds": total_train,
            "decode_seconds": stats["decode_seconds"],
            "gather_seconds": stats["gather_seconds"],
            "peak_segment_bytes_loaded": stats["peak_segment_bytes_loaded"],
            "peak_device_tensor_bytes": stats["peak_device_tensor_bytes"],
            "resident_segment_memory_ratio": cache_segment_bytes(cache)
            / max(stats["peak_segment_bytes_loaded"], 1),
            "loss_first": stats["loss_curve"][0],
            "loss_last": stats["loss_curve"][-1],
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
        },
        "held_out_psnr_db": psnr_db,
        "tiled_full_frame_inference_ms": tiled_seconds * 1000.0,
        "tile_pixels": args.tile_pixels,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
