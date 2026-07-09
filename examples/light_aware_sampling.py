"""E3 light-aware sampling report for the toy tracer.

This is the first E3 slice: a surface-bounce cosine/cone mixture sampler aimed at a
declared spherical light-placement region. It measures segment density and Monte Carlo
consistency, but does not yet train the standard-vs-guided proxy A/B study.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight, segment_hits_sphere  # noqa: E402
from nrp.metrics import psnr, smape  # noqa: E402
from nrp.toy_tracer import render_reference, trace_path_cache  # noqa: E402


def json_psnr(a: np.ndarray, b: np.ndarray) -> float | str:
    value = psnr(a, b)
    return "inf" if math.isinf(value) else float(value)


def region_density(cache, region: dict) -> dict:
    hits = segment_hits_sphere(
        cache.seg_origin,
        cache.seg_dir,
        cache.seg_tmax,
        np.asarray(region["center"], dtype=np.float64),
        float(region["radius"]),
    )
    return {
        "segments": cache.segment_count,
        "region_hits": int(hits.sum()),
        "region_hit_fraction": float(hits.mean()) if cache.segment_count else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/light-aware-sampling/report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=12)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--guide-probability", type=float, default=0.5)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    region = {"type": "sphere", "center": [0.45, 0.75, 0.45], "radius": 0.12}
    light = SphereLight(center=region["center"], radius=region["radius"], rgb=[1.2, 1.0, 0.8])

    t0 = time.perf_counter()
    standard = trace_path_cache(args.width, args.height, args.spp, args.bounces, seed=31)
    standard_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    guided = trace_path_cache(
        args.width,
        args.height,
        args.spp,
        args.bounces,
        seed=31,
        light_region=region,
        guide_probability=args.guide_probability,
    )
    guided_s = time.perf_counter() - t0

    standard_density = region_density(standard, region)
    guided_density = region_density(guided, region)
    standard_image = gather_light(standard, light)
    guided_image = gather_light(guided, light)
    reference = render_reference(
        args.width,
        args.height,
        spp=max(args.spp * 8, 64),
        max_bounces=args.bounces,
        seed=991,
        light=light,
        light_region=region,
        guide_probability=args.guide_probability,
    )

    report = {
        "extension": "E3",
        "scope": "spherical light-placement region with cosine/cone mixture sampling",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "guide_probability": args.guide_probability,
        "light_region": region,
        "standard": {
            **standard_density,
            "trace_ms": standard_s * 1000.0,
            "psnr_db_vs_reference": json_psnr(standard_image, reference),
            "smape_vs_reference": smape(standard_image, reference),
        },
        "guided": {
            **guided_density,
            "trace_ms": guided_s * 1000.0,
            "psnr_db_vs_reference": json_psnr(guided_image, reference),
            "smape_vs_reference": smape(guided_image, reference),
        },
        "density_gain": guided_density["region_hit_fraction"]
        / max(standard_density["region_hit_fraction"], 1e-12),
        "cache_size_delta_segments": guided.segment_count - standard.segment_count,
        "artifacts": {},
        "limitations": [
            "No occluder/lamp-shade proxy A/B is trained in this slice.",
            "The guide distribution currently supports spherical placement regions only.",
        ],
    }
    np.save(out_path.parent / "standard.npy", standard_image)
    np.save(out_path.parent / "guided.npy", guided_image)
    np.save(out_path.parent / "reference.npy", reference)
    report["artifacts"] = {
        "standard": "standard.npy",
        "guided": "guided.npy",
        "reference": "reference.npy",
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
