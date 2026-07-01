"""NRP M3 check: decoupled GATHERLIGHT reconstruction vs an independent rendered
reference for the same light.

The reference is the toy tracer re-run with a different seed, evaluating the emissive
sphere inline over its own fresh paths. Both are Monte Carlo estimates of the same
integral, so agreement (PSNR rising with spp) is a consistency check between the
decoupled path and direct rendering, not a comparison against a fabricated constant.
"""

from __future__ import annotations

import argparse
import json

from .gather_light import gather_light
from .lights import SphereLight
from .metrics import psnr, smape
from .path_cache import PathCache
from .toy_tracer import render_reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True, help="path cache .npz (from toy_tracer)")
    parser.add_argument("--light", required=True, help="JSON file or inline JSON light spec")
    parser.add_argument("--ref-spp", type=int, default=64)
    parser.add_argument("--ref-bounces", type=int, default=3)
    parser.add_argument("--ref-seed", type=int, default=999)
    parser.add_argument("--out", help="optional JSON report path")
    args = parser.parse_args()

    cache = PathCache.load(args.cache)
    try:
        spec = json.loads(args.light)
    except json.JSONDecodeError:
        with open(args.light) as f:
            spec = json.load(f)
    light = SphereLight.from_dict(spec)

    recon = gather_light(cache, light)
    ref = render_reference(
        cache.width, cache.height, args.ref_spp, args.ref_bounces, args.ref_seed, light
    )
    report = {
        "light": light.to_dict(),
        "cache_segments": cache.segment_count,
        "ref_spp": args.ref_spp,
        "psnr_db": psnr(recon, ref),
        "smape": smape(recon, ref),
        "recon_mean": float(recon.mean()),
        "ref_mean": float(ref.mean()),
    }
    print(json.dumps(report, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
