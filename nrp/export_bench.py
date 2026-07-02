"""Benchmark the Mitsuba exporter: scalar Python loop vs drjit wavefront loop.

Measures wall-clock and throughput (segments/s) for `export_path_cache` (scalar_rgb)
and `export_path_cache_wavefront` (first working JIT variant) on the same scene at a
list of resolutions, and writes a JSON report (roadmap item 1: target >= 20x wavefront
speedup at 48x48 and 128x128).

Usage:
  python -m nrp.export_bench --out out/export-bench.json
  python -m nrp.export_bench --sizes 48 128 --spp 16 --bounces 4 --out out/export-bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

from .mitsuba_exporter import (
    _load_mitsuba,
    _load_scene,
    export_path_cache,
    export_path_cache_wavefront,
)


def bench_mode(
    mode: str,
    scene_spec: str,
    size: int,
    spp: int,
    bounces: int,
    seed: int,
    repeats: int = 3,
) -> dict:
    mi = _load_mitsuba(mode)
    scene = _load_scene(mi, scene_spec, size, size)
    export = export_path_cache_wavefront if mode == "wavefront" else export_path_cache
    warmup_seconds = 0.0
    if mode == "wavefront":
        # First launch pays one-time JIT kernel compilation; keep it out of the
        # steady-state throughput number but report it.
        t0 = time.perf_counter()
        export(scene, mi, size, size, 1, bounces, seed=seed)
        warmup_seconds = time.perf_counter() - t0
    best = None
    for i in range(repeats):
        t0 = time.perf_counter()
        cache = export(scene, mi, size, size, spp, bounces, seed=seed + i)
        seconds = time.perf_counter() - t0
        if best is None or seconds < best[0]:
            best = (seconds, cache.segment_count)
    seconds, segments = best
    return {
        "mode": mode,
        "variant": mi.variant(),
        "resolution": size,
        "spp": spp,
        "bounces": bounces,
        "repeats": repeats,
        "warmup_seconds": warmup_seconds,
        "segments": segments,
        "seconds": seconds,
        "segments_per_s": segments / seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="builtin:cornell-box")
    parser.add_argument("--sizes", type=int, nargs="+", default=[48, 128])
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", required=True, help="output JSON report")
    args = parser.parse_args()

    rows = []
    for size in args.sizes:
        pair = {}
        for mode in ["scalar", "wavefront"]:
            row = bench_mode(
                mode, args.scene, size, args.spp, args.bounces, args.seed, args.repeats
            )
            pair[mode] = row
            print(
                f"{mode:10s} [{row['variant']}] {size}x{size} @ {args.spp} spp: "
                f"{row['segments']} segments in {row['seconds']:.2f}s "
                f"({row['segments_per_s']:.0f} seg/s)"
            )
        speedup = pair["scalar"]["seconds"] / pair["wavefront"]["seconds"]
        print(f"speedup @ {size}x{size}: {speedup:.1f}x")
        rows.append({"resolution": size, "speedup": speedup, **pair})

    report = {
        "scene": args.scene,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "results": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
