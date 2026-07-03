"""E8 gather-time production controls: light linking and custom attenuation.

This report covers the cache/GATHERLIGHT workaround path only. It intentionally does
not claim live proxy-conditioned controls; the report records that as open work.
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

from nrp.gather_light import GatherControls, gather_light, gather_light_controlled  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.toy_tracer import layer_ownership_mask, trace_path_cache  # noqa: E402


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000.0


def finite_or_inf(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/production-controls/report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=12)
    parser.add_argument("--bounces", type=int, default=2)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = trace_path_cache(args.width, args.height, args.spp, args.bounces, seed=8)
    sphere_cache = trace_path_cache(
        args.width, args.height, args.spp, args.bounces, seed=8, layer="sphere"
    )
    box_cache = trace_path_cache(
        args.width, args.height, args.spp, args.bounces, seed=8, layer="box"
    )
    light = SphereLight(center=[0.0, 0.55, 0.0], radius=0.22, rgb=[1.0, 0.8, 0.6])

    full, full_ms = timed(lambda: gather_light(cache, light))
    sphere, _ = timed(lambda: gather_light(sphere_cache, light))
    box, _ = timed(lambda: gather_light(box_cache, light))
    linked, linked_ms = timed(
        lambda: gather_light_controlled(
            cache,
            light,
            GatherControls(
                exclude_pixel_mask=layer_ownership_mask(args.width, args.height, "sphere")
            ),
        )
    )
    attenuated, attenuation_ms = timed(
        lambda: gather_light_controlled(
            cache,
            light,
            GatherControls(
                attenuation={"type": "linear_distance", "intercept": 1.0, "slope": -0.1}
            ),
        )
    )

    report = {
        "resolution": [args.width, args.height],
        "segments": cache.segment_count,
        "linking": {
            "full_equals_sphere_plus_box_max_abs": float(np.max(np.abs(full - (sphere + box)))),
            "exclude_sphere_psnr_vs_box_layer_db": finite_or_inf(psnr(linked, box)),
            "exclude_sphere_max_abs_vs_box_layer": float(np.max(np.abs(linked - box))),
            "full_gather_ms": full_ms,
            "linked_gather_ms": linked_ms,
        },
        "attenuation": {
            "curve": {"type": "linear_distance", "intercept": 1.0, "slope": -0.1},
            "attenuated_gather_ms": attenuation_ms,
            "mean_radiance_ratio_vs_default": float(attenuated.mean() / max(full.mean(), 1e-12)),
        },
        "proxy_conditioned_controls": {
            "implemented": False,
            "finding": "gather-time controls work; proxy-conditioned live controls remain open",
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
