"""S3: profile the wavefront exporter's Python-side conversion cost.

Runs `export_path_cache_wavefront` on the builtin cornell box under cProfile and
reports where the wall-clock goes (drjit kernel evaluation is lazily triggered by
the first `np.array(...)` conversion of each launch, so conversion rows include the
kernel work they force — the caveat is recorded in the report). Writes the top-N
cumulative rows plus a coarse phase split to out/s3-shard-write/exporter_profile.json.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--out", default="out/s3-shard-write/exporter_profile.json")
    args = parser.parse_args()

    from nrp.mitsuba_exporter import (
        _load_mitsuba,
        _load_scene,
        export_path_cache_wavefront,
    )

    mi = _load_mitsuba("wavefront")
    scene = _load_scene(mi, "builtin:cornell-box", args.width, args.height)

    # warmup (JIT compile) outside the profile
    export_path_cache_wavefront(scene, mi, 32, 32, 4, 2)

    prof = cProfile.Profile()
    t0 = time.perf_counter()
    prof.enable()
    cache = export_path_cache_wavefront(scene, mi, args.width, args.height, args.spp, args.bounces)
    prof.disable()
    wall = time.perf_counter() - t0

    stream = io.StringIO()
    stats = pstats.Stats(prof, stream=stream)
    stats.sort_stats("cumulative").print_stats(args.top)
    text = stream.getvalue()

    rows = []
    for line in text.splitlines():
        parts = line.split(None, 5)
        if len(parts) == 6 and parts[0][0].isdigit():
            ncalls, tottime, _, cumtime, _, func = parts
            try:
                rows.append(
                    {
                        "ncalls": ncalls,
                        "tottime_s": float(tottime),
                        "cumtime_s": float(cumtime),
                        "function": func.strip(),
                    }
                )
            except ValueError:
                continue  # pstats summary/header lines

    report = {
        "rung": "S3",
        "scope": "wavefront exporter Python-side conversion profile",
        "scene": "builtin:cornell-box",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "segments": cache.segment_count,
        "wall_seconds": wall,
        "segments_per_second": cache.segment_count / wall,
        "caveat": (
            "drjit evaluates lazily: each bounce's kernel cost is attributed to the "
            "np.array()/drjit conversion that first forces it, so 'conversion' rows "
            "bound the Python-side cost from above"
        ),
        "top_cumulative": rows[: args.top],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(text[:4000])
    print(f"wall {wall:.2f}s, {cache.segment_count} segments; wrote {out_path}")


if __name__ == "__main__":
    main()
