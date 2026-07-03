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
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
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
        "tiled_relight_max_abs_diff": float(np.max(np.abs(tiled - full))),
        "tiled_relight_allclose_atol_1e_6": bool(np.allclose(tiled, full, rtol=0.0, atol=1e-6)),
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
