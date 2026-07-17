"""S2: spp cost/quality probe for the 1024x1024 kitchen export.

The 1024^2 export's spp must be chosen deliberately (docs/scale-track.md). This
probe measures the two sides of the tradeoff on the real kitchen scene:

1. quality: at a fixed probe resolution (default 256^2, cheap), export caches at
   each candidate spp plus a high-spp reference, and compare GATHERLIGHT images for
   a fixed set of sphere lights (PSNR vs the reference gather). GATHERLIGHT noise
   is per-pixel MC noise from the path samples, so it depends on spp, not on how
   many pixels surround it — the 256^2 numbers transfer to 1024^2.
2. cost: the exporter's own --report JSONs (wall-clock, peak RSS, cache bytes) at
   the probe resolution, from which the 1024^2 cost scales ~4x per pixel count.

Writes out/s2-scale/spp_probe.json; the chosen spp is justified in
docs/performance.md from this report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402

PROBE_LIGHTS = [
    SphereLight(center=[-0.64, 2.19, -0.22], radius=0.3, rgb=[3.0, 3.0, 3.0]),
    SphereLight(center=[-1.81, 1.73, -0.21], radius=0.35, rgb=[1.2, 1.3, 1.6]),
    SphereLight(center=[-1.53, 2.8, -0.63], radius=0.2, rgb=[2.0, 1.0, 0.6]),
    SphereLight(center=[-1.2, 2.2, -0.4], radius=0.45, rgb=[1.0, 1.0, 1.0]),
]


def export(scene: str, res: int, spp: int, out: Path, report: Path) -> float:
    t0 = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "nrp.mitsuba_exporter",
            "--scene",
            scene,
            "--width",
            str(res),
            "--height",
            str(res),
            "--spp",
            str(spp),
            "--bounces",
            "4",
            "--out",
            str(out),
            "--report",
            str(report),
        ],
        check=True,
    )
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", default="examples/scenes/kitchen/scene.xml")
    parser.add_argument("--probe-res", type=int, default=256)
    parser.add_argument("--spp", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--reference-spp", type=int, default=256)
    parser.add_argument("--out", default="out/s2-scale/spp_probe.json")
    parser.add_argument("--work-dir", default="out/s2-scale/probe")
    args = parser.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ref_cache_path = work / f"probe_{args.probe_res}_{args.reference_spp}spp.npz"
    rows = []
    if not ref_cache_path.exists():
        export(
            args.scene,
            args.probe_res,
            args.reference_spp,
            ref_cache_path,
            work / f"export_{args.reference_spp}spp.json",
        )
    ref_cache = PathCache.load(str(ref_cache_path))
    ref_images = [gather_light(ref_cache, light) for light in PROBE_LIGHTS]
    del ref_cache

    for spp in args.spp:
        cache_path = work / f"probe_{args.probe_res}_{spp}spp.npz"
        report_path = work / f"export_{spp}spp.json"
        if not cache_path.exists():
            export(args.scene, args.probe_res, spp, cache_path, report_path)
        cache = PathCache.load(str(cache_path))
        psnrs = [
            float(psnr(gather_light(cache, light), ref))
            for light, ref in zip(PROBE_LIGHTS, ref_images, strict=True)
        ]
        exp_report = json.loads(report_path.read_text()) if report_path.exists() else {}
        rows.append(
            {
                "spp": spp,
                "segments": cache.segment_count,
                "cache_bytes": cache_path.stat().st_size,
                "gather_psnr_vs_refspp_db": {
                    "mean": float(np.mean(psnrs)),
                    "min": float(np.min(psnrs)),
                    "per_light": psnrs,
                },
                "export_report": {
                    k: exp_report[k]
                    for k in ("wall_seconds", "peak_rss_bytes", "segments_per_second")
                    if k in exp_report
                },
            }
        )
        del cache
        print(json.dumps(rows[-1]), flush=True)

    report = {
        "rung": "S2",
        "scope": "spp cost/quality probe for the 1024^2 kitchen export",
        "scene": args.scene,
        "probe_resolution": args.probe_res,
        "reference_spp": args.reference_spp,
        "probe_lights": [
            {"center": light.center.tolist(), "radius": light.radius} for light in PROBE_LIGHTS
        ],
        "rows": rows,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
