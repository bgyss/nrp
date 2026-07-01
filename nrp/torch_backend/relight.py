"""Interactive forward relighting through the torch NRP (Eq. 3), with benchmark mode.

The final image is the emission-weighted sum of per-light network outputs:
I(px) = sum_i E_i * N_type(px, F_px, v_i). Accepts one light or a JSON list; the light
type(s) must match the type the model was trained for.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..lights import light_from_dict
from ..path_cache import PathCache
from .model import TorchNRP
from .train import light_param_vector, pixel_tensors


def relight(model: TorchNRP, cache: PathCache, lights: list) -> np.ndarray:
    device = next(model.parameters()).device
    n_px = cache.height * cache.width
    xy, aux = pixel_tensors(cache, device)
    image = torch.zeros((n_px, 3), device=device)
    with torch.no_grad():
        for light in lights:
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            rgb = torch.as_tensor(light.rgb, dtype=torch.float32, device=device)
            image += model(xy, aux, params) * rgb
    return image.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="trained model .pt")
    parser.add_argument("--cache", required=True, help="path cache .npz (aux features)")
    parser.add_argument(
        "--light", required=True, help="JSON file or inline JSON: one light spec or a list"
    )
    parser.add_argument("--out", help="output image .npy")
    parser.add_argument("--bench", type=int, default=0, help="benchmark N relight frames")
    args = parser.parse_args()

    model = TorchNRP.load(args.model)
    cache = PathCache.load(args.cache)
    try:
        spec = json.loads(args.light)
    except json.JSONDecodeError:
        with open(args.light) as f:
            spec = json.load(f)
    specs = spec if isinstance(spec, list) else [spec]
    lights = [light_from_dict(s) for s in specs]

    image = relight(model, cache, lights)
    if args.out:
        np.save(args.out, image)
        print(f"wrote {args.out}")
    print(f"mean radiance {image.mean():.6f} over {len(lights)} light(s)")

    if args.bench:
        t0 = time.perf_counter()
        for _ in range(args.bench):
            relight(model, cache, lights)
        ms = (time.perf_counter() - t0) / args.bench * 1000.0
        print(
            f"relight rate at {cache.width}x{cache.height}: {ms:.2f} ms/frame "
            f"({1000.0 / ms:.1f} Hz) over {args.bench} frames"
        )


if __name__ == "__main__":
    main()
