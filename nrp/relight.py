"""Forward relighting through the trained proxy (NRP M4 CLI).

Loads a trained proxy + the scene's path cache (for auxiliary features), evaluates one
light configuration, optionally compares against reference GATHERLIGHT, and optionally
benchmarks steady-state relighting rate at the cache's resolution.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .dataset import build_inputs, pixel_feature_block
from .gather_light import gather_light
from .lights import SphereLight
from .metrics import psnr, smape
from .model import ProxyMLP
from .path_cache import PathCache


def relight(model: ProxyMLP, px: dict, light: SphereLight, height: int, width: int) -> np.ndarray:
    contribution = model.forward(build_inputs(px, light.center, light.radius))
    return (contribution * light.rgb).reshape(height, width, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="trained model .npz")
    parser.add_argument("--cache", required=True, help="path cache .npz")
    parser.add_argument("--light", required=True, help="JSON file or inline JSON light spec")
    parser.add_argument("--out", help="output image .npy")
    parser.add_argument(
        "--compare-gather", action="store_true", help="report PSNR/SMAPE vs GATHERLIGHT"
    )
    parser.add_argument("--bench", type=int, default=0, help="benchmark N relight frames")
    args = parser.parse_args()

    model = ProxyMLP.load(args.model)
    cache = PathCache.load(args.cache)
    px = pixel_feature_block(cache)
    try:
        spec = json.loads(args.light)
    except json.JSONDecodeError:
        with open(args.light) as f:
            spec = json.load(f)
    light = SphereLight.from_dict(spec)

    image = relight(model, px, light, cache.height, cache.width)
    if args.out:
        np.save(args.out, image)
        print(f"wrote {args.out}")

    if args.compare_gather:
        ref = gather_light(cache, light)
        print(f"vs GATHERLIGHT: PSNR {psnr(image, ref):.2f} dB, SMAPE {smape(image, ref):.4f}")

    if args.bench:
        t0 = time.perf_counter()
        for _ in range(args.bench):
            relight(model, px, light, cache.height, cache.width)
        ms = (time.perf_counter() - t0) / args.bench * 1000.0
        print(
            f"relight rate at {cache.width}x{cache.height}: {ms:.2f} ms/frame "
            f"({1000.0 / ms:.1f} Hz) over {args.bench} frames"
        )


if __name__ == "__main__":
    main()
