"""E5's last open criterion: a 512x512 / 128 spp Mitsuba end-to-end streamed report.

Exports (or reuses) a 512x512/128spp Mitsuba cornell-box path cache, shards it,
trains a real TorchNRP sphere-light proxy via `nrp.torch_backend.streamed_train`
(bounded resident segment memory), and measures tiled full-frame inference. Reports
cache size, streamed vs monolithic peak segment bytes, training wall-clock, held-out
PSNR, and full-frame proxy inference latency — the "does it scale to film frames"
datapoint the extension asks for at production resolution (toy-scale scene content,
production-scale pixel/sample count).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.relight import relight_tiled  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.streamed_train import _pixel_tensors, train_streamed  # noqa: E402


def cache_segment_bytes(cache: PathCache) -> int:
    return int(
        cache.seg_pixel.nbytes
        + cache.seg_origin.nbytes
        + cache.seg_dir.nbytes
        + cache.seg_tmax.nbytes
        + cache.seg_throughput.nbytes
    )


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def current_rss_bytes() -> int:
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith("linux"):
        rss *= 1024
    return rss


def held_out_psnr(model, cache: PathCache, cfg: dict, n_lights: int = 6) -> float:
    val_rng = np.random.default_rng([cfg.get("seed", 0), 0x5EED])
    device = "cpu"
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
                np.concatenate([light.center, [light.radius]]),
                dtype=torch.float32,
            ).expand(n_px, -1)
            pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
            p = psnr(pred, raw)
            if np.isfinite(p):
                vals.append(p)
    model.train()
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/out-of-core/mitsuba_512_report.json")
    parser.add_argument("--cache", default="out/mitsuba-512/path_cache.npz")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-pixels", type=int, default=16384)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--gather-backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--gather-device", default="cpu")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache)

    if not cache_path.exists():
        raise SystemExit(
            f"{cache_path} does not exist; run "
            f"`uv run python -m nrp.mitsuba_exporter --scene builtin:cornell-box "
            f"--width {args.width} --height {args.height} --spp {args.spp} "
            f"--bounces {args.bounces} --out {cache_path}` first "
            f"(or `mise run export-mitsuba-512`)."
        )
    cache = PathCache.load(str(cache_path))

    shard_dir = out_path.parent / "mitsuba_512_sharded"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    t0 = time.perf_counter()
    cache.save_sharded(str(shard_dir), tile_size=args.tile_size)
    shard_s = time.perf_counter() - t0

    cfg = {
        "seed": 0,
        "device": "cpu",
        "light_type": "sphere",
        "light_bounds": {"radius_min": 5.0, "radius_max": 15.0},
        "sampling": "segments",
        "denoise": {"enabled": False},
        "pool": {"size": 16, "replace_count": 1, "replace_every": 6},
        "model": {
            "hidden_width": 64,
            "hidden_layers": 3,
            "encoding": {"levels": 4, "features_per_level": 2, "finest_resolution": args.width},
        },
        "lr": 5e-3,
        "batch_pixels": 4096,
        "iters": args.iters,
        "gather_backend": args.gather_backend,
        "gather_device": args.gather_device,
    }

    rss_before = current_rss_bytes()
    t0 = time.perf_counter()
    model, stats = train_streamed(shard_dir, cache, cfg)
    train_s = time.perf_counter() - t0
    rss_after = current_rss_bytes()

    held_psnr = held_out_psnr(model, cache, cfg)

    light = sample_light(cache, np.random.default_rng(123), "sphere", cfg["light_bounds"])
    t0 = time.perf_counter()
    relight_tiled(model, cache, [light], tile_pixels=args.tile_pixels)
    tiled_s = time.perf_counter() - t0

    report = {
        "extension": "E5",
        "scope": "512x512/128spp Mitsuba streamed TorchNRP end-to-end report",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "segments": cache.segment_count,
        "monolithic_cache_bytes": cache_path.stat().st_size,
        "monolithic_cache_gb": cache_path.stat().st_size / 1e9,
        "monolithic_resident_segment_bytes": cache_segment_bytes(cache),
        "sharded_cache_bytes": directory_bytes(shard_dir),
        "save_sharded_seconds": shard_s,
        "streamed_train": {
            "iters": args.iters,
            "pool_seconds": stats["pool_seconds"],
            "train_seconds": stats["train_seconds"],
            "total_seconds": train_s,
            "loss_first": stats["loss_curve"][0],
            "loss_last": stats["loss_curve"][-1],
            "peak_segment_bytes_loaded": stats["peak_segment_bytes_loaded"],
            "resident_segment_memory_ratio": cache_segment_bytes(cache)
            / max(stats["peak_segment_bytes_loaded"], 1),
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": rss_after,
        },
        "held_out_psnr_db": held_psnr,
        "tiled_full_frame_inference_ms": tiled_s * 1000.0,
        "tile_pixels": args.tile_pixels,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
