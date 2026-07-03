"""E2 dynamic-geometry report: primary-visibility invalidation and cache splicing."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.dynamic_geometry import (  # noqa: E402
    primary_visibility_invalidation_mask,
    splice_invalidated_pixels,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.toy_tracer import SPHERE_CENTER, trace_path_cache  # noqa: E402


def psnr_json(a: np.ndarray, b: np.ndarray) -> float | str:
    value = psnr(a, b)
    return "inf" if math.isinf(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/dynamic-geometry/report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--frames", type=int, default=10)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    light = SphereLight(center=[0.35, 0.28, 0.62], radius=0.2, rgb=[1.3, 1.0, 0.8])
    base_center = SPHERE_CENTER.copy()
    offsets = np.linspace(0.0, 0.16, args.frames)

    t0 = time.perf_counter()
    base = trace_path_cache(
        args.width,
        args.height,
        args.spp,
        max_bounces=1,
        seed=21,
        sphere_center=base_center,
    )
    base_trace_s = time.perf_counter() - t0
    stale_image = gather_light(base, light)

    frames = []
    for frame, dx in enumerate(offsets):
        center = base_center + np.array([dx, 0.0, 0.0])
        t0 = time.perf_counter()
        full = trace_path_cache(
            args.width,
            args.height,
            args.spp,
            max_bounces=1,
            seed=21,
            sphere_center=center,
        )
        full_trace_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask = primary_visibility_invalidation_mask(base, full)
        spliced, stats = splice_invalidated_pixels(base, full, mask)
        splice_s = time.perf_counter() - t0

        full_image = gather_light(full, light)
        spliced_image = gather_light(spliced, light)
        outside = ~mask
        outside_max_diff = 0.0
        if outside.any():
            outside_max_diff = float(np.max(np.abs(stale_image[outside] - full_image[outside])))
        frames.append(
            {
                "frame": frame,
                "sphere_center": center.tolist(),
                "full_trace_ms": full_trace_s * 1000.0,
                "incremental_splice_ms": splice_s * 1000.0,
                "invalid_pixels": stats.invalid_pixels,
                "invalid_fraction": stats.invalid_fraction,
                "segments_retraced_fraction": stats.new_segments_inserted
                / max(full.segment_count, 1),
                "outside_mask_full_vs_base_max_abs": outside_max_diff,
                "stale_psnr_vs_full": psnr_json(stale_image, full_image),
                "incremental_psnr_vs_full": psnr_json(spliced_image, full_image),
                "incremental_max_abs_vs_full": float(np.max(np.abs(spliced_image - full_image))),
                "spliced_cache_valid": True,
            }
        )

    mean_full_ms = float(np.mean([f["full_trace_ms"] for f in frames]))
    mean_splice_ms = float(np.mean([f["incremental_splice_ms"] for f in frames]))
    mean_invalid_fraction = float(np.mean([f["invalid_fraction"] for f in frames]))
    report = {
        "extension": "E2",
        "scope": "one-bounce primary-visibility cache invalidation and segment splicing",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "frames": args.frames,
        "base_trace_ms": base_trace_s * 1000.0,
        "mean_full_trace_ms": mean_full_ms,
        "mean_incremental_splice_ms": mean_splice_ms,
        "mean_invalid_fraction": mean_invalid_fraction,
        "mean_frame_budget_fraction_16ms_full_trace": mean_full_ms / 16.0,
        "mean_frame_budget_fraction_16ms_splice_only": mean_splice_ms / 16.0,
        "frames_detail": frames,
        "limitations": [
            "This is a one-bounce primary-visibility slice; secondary-bounce "
            "invalidation is not proven.",
            "No proxy fine-tuning is performed, so regime (b) quality is cache-splice "
            "quality only.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
