"""E1 animated-camera baseline: interpolate camera-keyframe GATHERLIGHT frames.

This is deliberately a baseline, not the full time-conditioned neural proxy requested
by E1. It traces K camera keyframe caches, gathers one fixed light per keyframe, and
linearly interpolates the resulting frames at held-out camera times. The report gives
cache-size-vs-K and the held-out gap that a real time-conditioned proxy must beat.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.toy_tracer import CAM_POS, trace_path_cache  # noqa: E402


def interpolate_frames(times: np.ndarray, frames: np.ndarray, query: float) -> np.ndarray:
    """Linearly interpolate an image sequence sampled at normalized ``times``."""
    times = np.asarray(times, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    if times.ndim != 1 or frames.shape[0] != times.size:
        raise ValueError("times must be 1D and match the first frame dimension")
    if times.size == 0:
        raise ValueError("at least one time sample is required")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    if query <= times[0]:
        return frames[0].copy()
    if query >= times[-1]:
        return frames[-1].copy()
    right = int(np.searchsorted(times, query, side="right"))
    left = right - 1
    alpha = float((query - times[left]) / (times[right] - times[left]))
    return ((1.0 - alpha) * frames[left] + alpha * frames[right]).copy()


def camera_at(time_value: float, offset_extent: float) -> np.ndarray:
    """Move the toy pinhole camera laterally inside the unit box."""
    offset = (float(time_value) * 2.0 - 1.0) * offset_extent
    return CAM_POS + np.array([offset, 0.0, 0.0], dtype=np.float64)


def cache_file_bytes(paths: list[Path]) -> int:
    return int(sum(os.path.getsize(path) for path in paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/time-camera/report.json")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--offset-extent", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    heldout_times = np.array([0.25, 0.75], dtype=np.float64)
    light = SphereLight(
        center=np.array([0.52, 0.72, 0.45]),
        radius=0.08,
        rgb=np.array([3.0, 2.2, 1.6]),
    )

    cache_paths: list[Path] = []
    keyframe_images: list[np.ndarray] = []
    keyframe_segments: list[int] = []
    t0 = time.perf_counter()
    for idx, time_value in enumerate(times):
        cache = trace_path_cache(
            args.width,
            args.height,
            args.spp,
            args.bounces,
            seed=args.seed + idx,
            camera_pos=camera_at(float(time_value), args.offset_extent),
        )
        cache_path = out_dir / f"camera_keyframe_{idx:02d}.npz"
        cache.save(cache_path)
        cache_paths.append(cache_path)
        keyframe_segments.append(cache.segment_count)
        image = gather_light(cache, light)
        np.save(out_dir / f"keyframe_{idx:02d}.npy", image)
        keyframe_images.append(image)
    keyframe_seconds = time.perf_counter() - t0

    frames = np.stack(keyframe_images, axis=0)
    heldout_reports: list[dict] = []
    for idx, time_value in enumerate(heldout_times):
        t1 = time.perf_counter()
        heldout_cache = trace_path_cache(
            args.width,
            args.height,
            args.spp,
            args.bounces,
            seed=args.seed + 100 + idx,
            camera_pos=camera_at(float(time_value), args.offset_extent),
        )
        direct = gather_light(heldout_cache, light)
        predicted = interpolate_frames(times, frames, float(time_value))
        np.save(out_dir / f"heldout_{idx:02d}_direct.npy", direct)
        np.save(out_dir / f"heldout_{idx:02d}_interpolated.npy", predicted)
        heldout_reports.append(
            {
                "time": float(time_value),
                "camera_pos": camera_at(float(time_value), args.offset_extent).tolist(),
                "segments": heldout_cache.segment_count,
                "interpolated_vs_direct_psnr_db": psnr(predicted, direct),
                "trace_and_gather_seconds": time.perf_counter() - t1,
            }
        )

    report = {
        "extension": "E1",
        "scope": "animated-camera image-space interpolation baseline",
        "status": "baseline_only_full_time_conditioned_proxy_still_open",
        "width": args.width,
        "height": args.height,
        "spp": args.spp,
        "bounces": args.bounces,
        "training_keyframes": [
            {
                "time": float(time_value),
                "camera_pos": camera_at(float(time_value), args.offset_extent).tolist(),
                "segments": int(segments),
                "cache_path": str(path),
            }
            for time_value, segments, path in zip(
                times, keyframe_segments, cache_paths, strict=True
            )
        ],
        "keyframe_cache_bytes_total": cache_file_bytes(cache_paths),
        "keyframe_cache_bytes_per_keyframe": cache_file_bytes(cache_paths) / len(cache_paths),
        "keyframe_trace_and_gather_seconds": keyframe_seconds,
        "heldout_intermediate": heldout_reports,
        "mean_heldout_psnr_db": float(
            np.mean([entry["interpolated_vs_direct_psnr_db"] for entry in heldout_reports])
        ),
        "completion_note": (
            "This establishes camera-cache K-scaling and a held-out interpolation baseline. "
            "It does not satisfy E1's neural time-conditioned proxy criterion."
        ),
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"wrote {out_path} with {len(times)} camera keyframes, "
        f"{report['mean_heldout_psnr_db']:.2f} dB mean held-out interpolation PSNR"
    )


if __name__ == "__main__":
    main()
