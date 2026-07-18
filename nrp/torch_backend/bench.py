"""Proxy inference benchmark across resolutions and devices (paper Table 1 analogue).

Measures full-frame forward latency (pixel features + light params -> RGB) for a
trained model — or a freshly initialized one with a given architecture — at a range of
resolutions on every requested device (cpu, mps, cuda). Aux features are synthesized at
benchmark resolutions above the cache's own (the MLP cost is feature-independent), so
timings reflect pure network + encoding throughput, matching the paper's inference rows
rather than its GATHERLIGHT reconstruction rows.

Device timing uses proper synchronization (torch.cuda.synchronize /
torch.mps.synchronize) with warmup iterations before measurement.

Usage:
  python -m nrp.torch_backend.bench --model out/toy-torch/model.pt \
      --resolutions 48 128 256 512 --devices cpu mps --out out/bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from .device import resolve_device, synchronize
from .model import LIGHT_PARAM_DIMS, TorchNRP


def available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _synchronize(device: torch.device) -> None:
    synchronize(device)


def bench_model(
    model: TorchNRP,
    device: torch.device,
    resolution: int,
    frames: int = 30,
    warmup: int = 5,
) -> dict:
    model = model.to(device).eval()
    n = resolution * resolution
    gen = torch.Generator(device="cpu").manual_seed(0)
    xy = torch.rand((n, 2), generator=gen).to(device)
    aux = torch.rand((n, 7), generator=gen).to(device)
    params = torch.rand((1, LIGHT_PARAM_DIMS[model.light_type]), generator=gen).to(device)
    params = params.expand(n, -1)

    with torch.no_grad():
        for _ in range(warmup):
            model(xy, aux, params)
        _synchronize(device)
        t0 = time.perf_counter()
        for _ in range(frames):
            model(xy, aux, params)
        _synchronize(device)
        ms = (time.perf_counter() - t0) / frames * 1000.0
    return {
        "device": str(device),
        "resolution": resolution,
        "pixels": n,
        "ms_per_frame": ms,
        "hz": 1000.0 / ms if ms > 0 else None,
    }


def bench_gather(
    cache_path: str,
    devices: list[str],
    n_lights: int = 20,
    warmup: int = 3,
    seed: int = 0,
) -> list[dict]:
    """GATHERLIGHT ms/image: numpy-CPU reference vs batched torch per device.

    The same `n_lights` random sphere lights (sampled inside the cache's first-hit
    bbox) are gathered by every backend; reported time is the mean per image.
    """
    import numpy as np

    from ..gather_light import gather_light
    from ..lights import SphereLight
    from ..path_cache import PathCache
    from .gather import TorchPathCache

    cache = PathCache.load(cache_path)
    rng = np.random.default_rng(seed)
    lo = cache.position.reshape(-1, 3).min(axis=0)
    hi = cache.position.reshape(-1, 3).max(axis=0)
    scale = float(np.linalg.norm(hi - lo))
    lights = [
        SphereLight(
            center=rng.uniform(lo, hi),
            radius=float(rng.uniform(0.02, 0.1) * scale),
            rgb=[1.0, 1.0, 1.0],
        )
        for _ in range(n_lights)
    ]

    rows = []

    def timed(fn, sync=lambda: None) -> float:
        for light in lights[:warmup]:
            fn(light)
        sync()
        t0 = time.perf_counter()
        for light in lights:
            fn(light)
        sync()
        return (time.perf_counter() - t0) / n_lights * 1000.0

    ms = timed(lambda light: gather_light(cache, light))
    rows.append(
        {
            "backend": "numpy",
            "device": "cpu",
            "cache": cache_path,
            "resolution": cache.width,
            "segments": cache.segment_count,
            "ms_per_image": ms,
        }
    )
    for device_name in devices:
        device = resolve_device(device_name)
        tc = TorchPathCache(cache, device)
        ms = timed(
            lambda light, tc=tc: tc.gather_light(light),
            lambda device=device: _synchronize(device),
        )
        rows.append(
            {
                "backend": "torch",
                "device": device_name,
                "cache": cache_path,
                "resolution": cache.width,
                "segments": cache.segment_count,
                "ms_per_image": ms,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="trained model .pt (default: fresh paper-ish model)")
    parser.add_argument("--light-type", default="sphere", choices=sorted(LIGHT_PARAM_DIMS))
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[48, 128, 256, 512])
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="devices to benchmark (default: all available: cpu, mps, cuda)",
    )
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument(
        "--gather-caches",
        nargs="+",
        default=[],
        help="path caches to also benchmark GATHERLIGHT on (numpy vs torch per device)",
    )
    parser.add_argument("--out", help="write results JSON here")
    args = parser.parse_args()

    if args.model:
        model = TorchNRP.load(args.model)
    else:
        model = TorchNRP(
            light_type=args.light_type,
            hidden_width=args.hidden_width,
            hidden_layers=args.hidden_layers,
        )
    devices = args.devices or available_devices()

    results = []
    print(f"benchmarking {model.parameter_count} params ({model.light_type}) on {devices}")
    print(f"{'device':>8} {'resolution':>11} {'ms/frame':>10} {'Hz':>10}")
    for device_name in devices:
        device = resolve_device(device_name)
        for res in args.resolutions:
            row = bench_model(model, device, res, frames=args.frames)
            results.append(row)
            print(
                f"{row['device']:>8} {res:>7}x{res:<3} {row['ms_per_frame']:>10.2f} "
                f"{row['hz']:>10.1f}"
            )

    gather_results = []
    for cache_path in args.gather_caches:
        rows = bench_gather(cache_path, devices=[d for d in devices if d != "cpu"] + ["cpu"])
        gather_results.extend(rows)
        for row in rows:
            print(
                f"gather {row['backend']:>6}/{row['device']:<4} {cache_path} "
                f"({row['segments']} segments): {row['ms_per_image']:.2f} ms/image"
            )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(
                {
                    "parameter_count": model.parameter_count,
                    "results": results,
                    "gather_results": gather_results,
                },
                f,
                indent=2,
            )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
