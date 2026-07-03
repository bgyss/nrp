"""E5 out-of-core foundation check: sharded cache + tiled proxy inference.

This is not the full E5 production-resolution training run. It verifies and measures
the primitives that run can build on:

- `PathCache.save_sharded` / `load_sharded` reconstruct the monolithic cache.
- GATHERLIGHT from the sharded round-trip matches the monolithic cache.
- `relight_tiled` matches standard relight within backend floating-point tolerance
  while bounding MLP activation chunks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight, segment_hits_sphere  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight, relight_tiled  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def directory_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += (Path(root) / name).stat().st_size
    return total


def cache_segment_bytes(cache: PathCache) -> int:
    """Bytes occupied by segment arrays in a resident monolithic cache."""
    return int(
        cache.seg_pixel.nbytes
        + cache.seg_origin.nbytes
        + cache.seg_dir.nbytes
        + cache.seg_tmax.nbytes
        + cache.seg_throughput.nbytes
    )


def current_rss_bytes() -> int:
    """Best-effort current process resident set size."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes, Linux reports KiB. Keep this deterministic enough for
    # coarse reporting without platform-specific dependencies.
    if sys.platform.startswith("linux"):
        rss *= 1024
    return rss


def stream_shard_targets(shard_dir: Path, lights: list[SphereLight]) -> tuple[np.ndarray, dict]:
    """Stream tile shards and build per-pixel mean GATHERLIGHT targets for fixed lights.

    This is a toy streamed-training primitive: for each shard, accumulate the same
    supervised target table an in-memory pass would build, without loading all segment
    arrays at once. The returned target is the per-pixel mean over `lights`.
    """
    with open(shard_dir / "manifest.json") as f:
        manifest = json.load(f)
    width = int(manifest["width"])
    height = int(manifest["height"])
    sums = np.zeros((height * width, 3), dtype=np.float64)
    n_paths = np.zeros(height * width, dtype=np.int64)
    peak_segments = 0
    peak_segment_bytes = 0
    peak_shard_file_bytes = 0
    visited_pixels = 0
    rss_before = current_rss_bytes()
    t0 = time.perf_counter()
    for shard in manifest["shards"]:
        shard_path = shard_dir / shard["path"]
        peak_shard_file_bytes = max(peak_shard_file_bytes, shard_path.stat().st_size)
        z = np.load(shard_path)
        y0, y1 = int(z["y0"]), int(z["y1"])
        x0, x1 = int(z["x0"]), int(z["x1"])
        tile_paths = z["n_paths"].reshape(-1)
        pixel_ids = (
            np.arange(y0, y1)[:, None] * width + np.arange(x0, x1)[None, :]
        ).reshape(-1)
        n_paths[pixel_ids] = tile_paths
        visited_pixels += int(pixel_ids.size)
        seg_pixel = z["seg_pixel"].astype(np.int64)
        seg_origin = z["seg_origin"].astype(np.float64)
        seg_dir = z["seg_dir"].astype(np.float64)
        seg_tmax = z["seg_tmax"].astype(np.float64)
        seg_throughput = z["seg_throughput"].astype(np.float64)
        peak_segments = max(peak_segments, int(seg_pixel.size))
        peak_segment_bytes = max(
            peak_segment_bytes,
            int(
                seg_pixel.nbytes
                + seg_origin.nbytes
                + seg_dir.nbytes
                + seg_tmax.nbytes
                + seg_throughput.nbytes
            ),
        )
        for light in lights:
            hits = segment_hits_sphere(seg_origin, seg_dir, seg_tmax, light.center, light.radius)
            if hits.any():
                np.add.at(sums, seg_pixel[hits], seg_throughput[hits] * light.rgb)
    denom = np.maximum(n_paths, 1).astype(np.float64)
    targets = sums / denom[:, None] / max(len(lights), 1)
    elapsed = time.perf_counter() - t0
    return targets.reshape(height, width, 3), {
        "stream_seconds": elapsed,
        "stream_pixels_visited": visited_pixels,
        "stream_peak_segments_loaded": peak_segments,
        "stream_peak_segment_bytes_loaded": peak_segment_bytes,
        "stream_peak_shard_file_bytes": peak_shard_file_bytes,
        "stream_process_rss_before_bytes": rss_before,
        "stream_process_rss_after_bytes": current_rss_bytes(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/out-of-core/report.json")
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=8)
    parser.add_argument("--tile-pixels", type=int, default=64)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(args.width, args.height, args.spp, args.bounces, seed=4)
    mono_path = base / "cache.npz"
    cache.save(str(mono_path))

    shard_dir = base / "cache_sharded"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    t0 = time.perf_counter()
    cache.save_sharded(str(shard_dir), tile_size=args.tile_size)
    save_sharded_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    sharded = PathCache.load_sharded(str(shard_dir))
    load_sharded_s = time.perf_counter() - t0

    light = SphereLight(center=[0.1, 0.6, 0.0], radius=0.2, rgb=[1.5, 1.0, 0.75])
    mono_gather = gather_light(cache, light)
    shard_gather = gather_light(sharded, light)
    train_lights = [
        light,
        SphereLight(center=[0.75, 0.75, 0.35], radius=0.12, rgb=[0.8, 1.2, 1.0]),
        SphereLight(center=[0.45, 0.25, 0.62], radius=0.16, rgb=[1.0, 0.8, 1.3]),
    ]
    t0 = time.perf_counter()
    mono_targets = sum(gather_light(cache, train_light) for train_light in train_lights) / len(
        train_lights
    )
    mono_target_s = time.perf_counter() - t0
    streamed_targets, stream_stats = stream_shard_targets(shard_dir, train_lights)

    model = TorchNRP(
        hidden_width=16,
        hidden_layers=2,
        encoding={"levels": 2, "features_per_level": 2, "finest_resolution": args.width},
    )
    full = relight(model, cache, [light])
    tiled = relight_tiled(model, cache, [light], tile_pixels=args.tile_pixels)
    gather_psnr = psnr(shard_gather, mono_gather)

    report = {
        "resolution": [args.width, args.height],
        "segments": cache.segment_count,
        "tile_size": args.tile_size,
        "tile_pixels": args.tile_pixels,
        "monolithic_cache_bytes": mono_path.stat().st_size,
        "monolithic_segment_bytes_resident": cache_segment_bytes(cache),
        "sharded_cache_bytes": directory_bytes(shard_dir),
        "save_sharded_seconds": save_sharded_s,
        "load_sharded_seconds": load_sharded_s,
        "sharded_roundtrip_exact_arrays": {
            "seg_pixel": bool(np.array_equal(sharded.seg_pixel, cache.seg_pixel)),
            "seg_origin": bool(np.array_equal(sharded.seg_origin, cache.seg_origin)),
            "seg_dir": bool(np.array_equal(sharded.seg_dir, cache.seg_dir)),
            "seg_tmax": bool(np.array_equal(sharded.seg_tmax, cache.seg_tmax)),
            "seg_throughput": bool(np.array_equal(sharded.seg_throughput, cache.seg_throughput)),
        },
        "gather_psnr_db_sharded_vs_monolithic": gather_psnr
        if math.isfinite(gather_psnr)
        else "inf",
        "gather_max_abs_diff_sharded_vs_monolithic": float(
            np.max(np.abs(shard_gather - mono_gather))
        ),
        "streamed_training_target": {
            "lights": len(train_lights),
            "monolithic_seconds": mono_target_s,
            **stream_stats,
            "stream_peak_segment_fraction": stream_stats["stream_peak_segments_loaded"]
            / max(cache.segment_count, 1),
            "stream_peak_segment_byte_fraction": stream_stats["stream_peak_segment_bytes_loaded"]
            / max(cache_segment_bytes(cache), 1),
            "estimated_resident_segment_memory_ratio": cache_segment_bytes(cache)
            / max(stream_stats["stream_peak_segment_bytes_loaded"], 1),
            "max_abs_diff_vs_monolithic": float(np.max(np.abs(streamed_targets - mono_targets))),
            "psnr_db_vs_monolithic": psnr(streamed_targets, mono_targets)
            if math.isfinite(psnr(streamed_targets, mono_targets))
            else "inf",
            "matches_monolithic_atol_1e_12": bool(
                np.allclose(streamed_targets, mono_targets, rtol=0.0, atol=1e-12)
            ),
        },
        "tiled_relight_max_abs_diff": float(np.max(np.abs(tiled - full))),
        "tiled_relight_allclose_atol_1e_6": bool(np.allclose(tiled, full, rtol=0.0, atol=1e-6)),
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
