"""Animated-light image sequences from one resident torch NRP.

The cache is used only for per-pixel auxiliary features; no path segments are gathered
while rendering frames. A keyframe JSON contains either one moving light or a list of
moving lights:

{
  "frames": 24,
  "interpolation": "linear",
  "lights": [
    {
      "keyframes": [
        {"time": 0.0, "light": {"type": "sphere", "center": [...], "radius": 0.1}},
        {"time": 1.0, "light": {"type": "sphere", "center": [...], "radius": 0.1}}
      ]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..gather_light import gather_lights
from ..lights import light_from_dict
from ..path_cache import PathCache
from .model import TorchNRP
from .relight import relight

NUMERIC_LIGHT_FIELDS = ("center", "rgb", "normal", "radius", "width", "height")


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def interpolate_light_spec(keyframes: list[dict], t: float) -> dict:
    """Linearly interpolate one light spec at normalized time ``t``.

    Categorical fields such as ``type`` are copied from the bracketing keyframes and
    must agree. Numeric scalar/vector fields are interpolated componentwise.
    """
    if not keyframes:
        raise ValueError("at least one keyframe is required")
    ordered = sorted(keyframes, key=lambda k: float(k["time"]))
    if t <= float(ordered[0]["time"]):
        return dict(ordered[0]["light"])
    if t >= float(ordered[-1]["time"]):
        return dict(ordered[-1]["light"])

    before = ordered[0]
    after = ordered[-1]
    for left, right in zip(ordered, ordered[1:], strict=False):
        if float(left["time"]) <= t <= float(right["time"]):
            before, after = left, right
            break

    t0 = float(before["time"])
    t1 = float(after["time"])
    if t1 <= t0:
        raise ValueError("keyframe times must be strictly increasing")
    alpha = (t - t0) / (t1 - t0)
    a = before["light"]
    b = after["light"]
    if a.get("type", "sphere") != b.get("type", "sphere"):
        raise ValueError("cannot interpolate across different light types")

    out = dict(a)
    for field in NUMERIC_LIGHT_FIELDS:
        if field in a or field in b:
            if field not in a or field not in b:
                raise ValueError(f"field {field!r} must exist in both bracketing lights")
            value = (1.0 - alpha) * _as_float_array(a[field]) + alpha * _as_float_array(b[field])
            out[field] = float(value) if value.ndim == 0 else value.tolist()
    return out


def frame_times(n_frames: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("frames must be positive")
    if n_frames == 1:
        return np.array([0.0], dtype=np.float64)
    return np.linspace(0.0, 1.0, n_frames, dtype=np.float64)


def lights_at(spec: dict, t: float) -> list:
    if spec.get("interpolation", "linear") != "linear":
        raise ValueError("only linear interpolation is currently supported")
    tracks = spec.get("lights")
    if tracks is None:
        tracks = [{"keyframes": spec["keyframes"]}]
    return [light_from_dict(interpolate_light_spec(track["keyframes"], t)) for track in tracks]


def mean_frame_delta(images: list[np.ndarray]) -> float:
    """Mean absolute per-pixel delta between consecutive frames."""
    if len(images) < 2:
        return 0.0
    deltas = [np.mean(np.abs(b - a)) for a, b in zip(images, images[1:], strict=False)]
    return float(np.mean(deltas))


def sequence_latency_ms(model: TorchNRP, cache: PathCache, spec: dict, frame_count: int) -> float:
    """Render ``frame_count`` frames without writing them, returning ms/frame."""
    bench_spec = dict(spec)
    bench_spec["frames"] = frame_count
    times = frame_times(frame_count)
    t0 = time.perf_counter()
    for t in times:
        relight(model, cache, lights_at(bench_spec, float(t)))
    return (time.perf_counter() - t0) / frame_count * 1000.0


def render_sequence(
    model: TorchNRP,
    cache: PathCache,
    spec: dict,
    out_dir: Path,
    measure_reference: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    times = frame_times(int(spec["frames"]))
    frame_paths: list[str] = []
    images: list[np.ndarray] = []
    t0 = time.perf_counter()
    for idx, t in enumerate(times):
        image = relight(model, cache, lights_at(spec, float(t)))
        path = out_dir / f"frame_{idx:04d}.npy"
        np.save(path, image)
        frame_paths.append(str(path))
        images.append(image)
    elapsed = time.perf_counter() - t0
    per_frame_ms = elapsed / len(times) * 1000.0
    latency_counts = spec.get("latency_frame_counts", [1, len(times)])
    report = {
        "frames": len(times),
        "width": cache.width,
        "height": cache.height,
        "per_frame_ms": per_frame_ms,
        "latency_vs_frame_count": [
            {"frames": int(n), "per_frame_ms": sequence_latency_ms(model, cache, spec, int(n))}
            for n in latency_counts
        ],
        "total_seconds": elapsed,
        "frame_paths": frame_paths,
        "cache_access": "aux_features_only_no_gatherlight",
        "proxy_mean_frame_delta": mean_frame_delta(images),
    }
    if measure_reference:
        reference_images = [gather_lights(cache, lights_at(spec, float(t))) for t in times]
        report["reference_mean_frame_delta"] = mean_frame_delta(reference_images)
        report["proxy_vs_reference_delta_ratio"] = (
            report["proxy_mean_frame_delta"] / report["reference_mean_frame_delta"]
            if report["reference_mean_frame_delta"] > 0.0
            else None
        )
        report["cache_access_for_measurement"] = "gatherlight_reference_temporal_metrics_only"
    with open(out_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="trained model .pt")
    parser.add_argument("--cache", required=True, help="path cache .npz for aux features")
    parser.add_argument("--keyframes", required=True, help="keyframed light JSON")
    parser.add_argument(
        "--out-dir",
        required=True,
        help="directory for frame_*.npy and report.json",
    )
    parser.add_argument(
        "--measure-reference",
        action="store_true",
        help="also compute GATHERLIGHT frame-to-frame delta for temporal-stability reporting",
    )
    args = parser.parse_args()

    model = TorchNRP.load(args.model)
    cache = PathCache.load(args.cache)
    with open(args.keyframes) as f:
        spec = json.load(f)
    report = render_sequence(model, cache, spec, Path(args.out_dir), args.measure_reference)
    print(
        f"wrote {report['frames']} frames at {report['width']}x{report['height']} "
        f"({report['per_frame_ms']:.2f} ms/frame)"
    )


if __name__ == "__main__":
    main()
