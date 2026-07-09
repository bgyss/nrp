"""E6 exported-runtime report and minimal slider-loop frame dump.

The "viewer" here is intentionally a headless Python loop: it simulates live slider
positions for a sphere light, renders every frame through the exported TorchScript
artifact, and writes frame dumps as evidence. No training-module checkpoint is loaded
inside the loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.engine_runtime import (  # noqa: E402
    export_artifact,
    load_runtime,
    runtime_latency_ms,
    runtime_relight,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def synthetic_runtime_cache(width: int, height: int) -> PathCache:
    """Valid cache shell for exported-runtime full-frame inference benchmarks.

    Runtime inference needs pixel coordinates and G-buffer aux; it does not inspect
    path segments. This keeps the E6 resolution sweep focused on exported MLP cost.
    """
    n_px = width * height
    cache = PathCache(
        width=width,
        height=height,
        n_paths=np.ones(n_px, dtype=np.int64),
        seg_pixel=np.zeros(0, dtype=np.int64),
        seg_origin=np.zeros((0, 3), dtype=np.float64),
        seg_dir=np.zeros((0, 3), dtype=np.float64),
        seg_tmax=np.zeros(0, dtype=np.float64),
        seg_throughput=np.zeros((0, 3), dtype=np.float64),
        albedo=np.full((height, width, 3), 0.5, dtype=np.float64),
        position=np.zeros((height, width, 3), dtype=np.float64),
        depth=np.ones((height, width), dtype=np.float64),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (height, width, 1)),
    )
    cache.validate()
    return cache


def runtime_resolution_sweep(
    runtime,
    lights: list,
    frames: int,
    devices: tuple[str, ...] = ("cpu", "mps"),
) -> list[dict]:
    rows = []
    for device in devices:
        available = device == "cpu" or (device == "mps" and torch.backends.mps.is_available())
        for resolution in (128, 256, 512):
            row = {
                "resolution": [resolution, resolution],
                "device": device,
                "available": bool(available),
            }
            if available:
                cache = synthetic_runtime_cache(resolution, resolution)
                ms = runtime_latency_ms(runtime, cache, lights, frames, device=device)
                fps = 1000.0 / ms
                row.update(
                    {
                        "ms_per_frame": ms,
                        "fps": fps,
                        "meets_30_fps": bool(fps >= 30.0),
                        "meets_60_fps": bool(fps >= 60.0),
                    }
                )
            else:
                row.update(
                    {
                        "ms_per_frame": None,
                        "fps": None,
                        "meets_30_fps": False,
                        "meets_60_fps": False,
                        "unavailable_reason": f"{device} backend not available",
                    }
                )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/engine-runtime/report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--bench", type=int, default=20)
    parser.add_argument("--sweep-frames", type=int, default=5)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    frames_dir = base / "viewer_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(args.width, args.height, spp=4, max_bounces=1, seed=13)
    model = TorchNRP(
        light_type="sphere",
        hidden_width=32,
        hidden_layers=2,
        encoding={"levels": 3, "features_per_level": 2, "finest_resolution": args.width},
    )
    artifact = base / "sphere_runtime.pt"
    metadata = export_artifact(model, str(artifact))
    runtime = load_runtime(str(artifact))

    parity_light = [SphereLight(center=[0.0, 0.55, 0.0], radius=0.2, rgb=[1.0, 0.8, 0.6])]
    parity = runtime_relight(runtime, cache, parity_light)
    direct = relight(model, cache, parity_light)

    slider_ms = []
    for idx, t in enumerate(np.linspace(0.0, 1.0, args.frames)):
        x = -0.35 + 0.7 * float(t)
        lights = [SphereLight(center=[x, 0.55, 0.0], radius=0.2, rgb=[1.0, 0.8, 0.6])]
        t0 = time.perf_counter()
        image = runtime_relight(runtime, cache, lights)
        slider_ms.append((time.perf_counter() - t0) * 1000.0)
        np.save(frames_dir / f"frame_{idx:04d}.npy", image)

    latency_ms = runtime_latency_ms(runtime, cache, parity_light, args.bench)
    sweep = runtime_resolution_sweep(runtime, parity_light, args.sweep_frames)
    report = {
        "resolution": [args.width, args.height],
        "frames": args.frames,
        "artifact": str(artifact.relative_to(base)),
        "artifact_bytes": metadata["artifact_bytes"],
        "parity_max_abs_diff": float(np.max(np.abs(parity - direct))),
        "parity_allclose_rtol_1e_4": bool(np.allclose(parity, direct, rtol=1e-4, atol=1e-6)),
        "exported_runtime_ms": latency_ms,
        "exported_runtime_fps": 1000.0 / latency_ms,
        "slider_to_frame_ms_mean": float(np.mean(slider_ms)),
        "slider_to_frame_ms_max": float(np.max(slider_ms)),
        "resolution_sweep": sweep,
        "resolution_sweep_devices": sorted({row["device"] for row in sweep}),
        "viewer_frame_dir": str(frames_dir.relative_to(base)),
        "runtime_path": "torch.jit.load artifact; no TorchNRP checkpoint load in frame loop",
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
