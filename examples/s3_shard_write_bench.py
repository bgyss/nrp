"""S3: sharded-cache write throughput — parallel vs serial save_sharded on a real cache.

Times `PathCache.save_sharded` (serial workers=1 vs threaded) on the 512x512/128spp
E5 cache against the committed 306.3 s baseline, for both the float64 and packed
layouts, and verifies the parallel output loads identically to the serial one
(spot-check here; the exhaustive bit-identical test lives in
tests/test_path_cache.py).
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

from nrp.path_cache import PathCache  # noqa: E402


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default="out/mitsuba-512/path_cache.npz")
    parser.add_argument("--out", default="out/s3-shard-write/report.json")
    parser.add_argument("--work-dir", default="out/s3-shard-write")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--layouts", nargs="+", default=["float64", "packed"], choices=["float64", "packed"]
    )
    parser.add_argument(
        "--keep",
        nargs="*",
        default=[],
        help="shard dirs to keep, as <layout>@<workers> (e.g. packed@8); others deleted",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    cache = PathCache.load(args.cache)
    report = {
        "rung": "S3",
        "scope": "save_sharded wall-clock, parallel vs serial, on the E5 512x512/128spp cache",
        "cache": args.cache,
        "segments": cache.segment_count,
        "tile_size": args.tile_size,
        "baseline_seconds": 306.3,
        "baseline_source": "docs/performance.md E5 512x512/128spp table (serial, pre-S3 code)",
        "runs": [],
    }

    serial_dirs: dict[str, Path] = {}
    for layout in args.layouts:
        packed = layout == "packed"
        for workers in args.workers:
            dest = work / f"shards_{layout}_w{workers}"
            if dest.exists():
                shutil.rmtree(dest)
            t0 = time.perf_counter()
            cache.save_sharded(str(dest), tile_size=args.tile_size, packed=packed, workers=workers)
            seconds = time.perf_counter() - t0
            row = {
                "layout": layout,
                "workers": workers,
                "seconds": seconds,
                "bytes": directory_bytes(dest),
                "speedup_vs_baseline": 306.3 / seconds,
            }
            report["runs"].append(row)
            print(json.dumps(row), flush=True)
            if workers == 1:
                serial_dirs[layout] = dest
            else:
                # spot parity: the largest shard's arrays match the serial write
                serial = serial_dirs.get(layout)
                if serial is not None:
                    name = max(
                        (p.name for p in dest.glob("*.npz")),
                        key=lambda n: (dest / n).stat().st_size,
                    )
                    with np.load(serial / name) as a, np.load(dest / name) as b:
                        for key in a.files:
                            np.testing.assert_array_equal(a[key], b[key])
                    row["spot_parity_vs_serial"] = f"{name} identical"
            keep_key = f"{layout}@{workers}"
            if keep_key not in args.keep:
                pass  # deletion deferred until parity checks are done
    # cleanup
    for layout in args.layouts:
        for workers in args.workers:
            keep_key = f"{layout}@{workers}"
            dest = work / f"shards_{layout}_w{workers}"
            if keep_key not in args.keep and dest.exists():
                shutil.rmtree(dest)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
