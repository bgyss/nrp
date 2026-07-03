"""Per-layer compositing (§6.1, Fig. 11): relight one layer, hold the other fixed.

The scene is split into first-hit layers (e.g. foreground sphere vs background box,
`nrp.toy_tracer --layer`), each with its own path cache and trained proxy. Because
transport is linear (Eq. 1) and the layers partition the full cache's paths, the
full-scene image under any light is exactly the sum of the layer images — so an
artist can re-render *one* layer's proxy under a new light and add the other layer's
image back unchanged, at the cost of a single-layer inference.

The fixed layer image is any (H, W, 3) `.npy` — typically the other layer's
GATHERLIGHT render or proxy prediction under the lighting being kept.

Usage:
  python -m nrp.torch_backend.composite \
      --edit-model out/layers/sphere/model.pt --edit-cache out/layers/sphere/path_cache.npz \
      --light '{"type": "sphere", "center": [0.2, 0.8, 0.5], "radius": 0.1, "rgb": [4, 4, 4]}' \
      --fixed-image out/layers/box_fixed.npy --out out/layers/composite.npy --bench 20
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from ..lights import light_from_dict
from ..path_cache import PathCache
from .bench import _synchronize
from .model import TorchNRP
from .relight_multiview import ViewProxy


def composite(edit_layer: ViewProxy, lights: list, fixed_image: np.ndarray) -> np.ndarray:
    """Edited layer re-rendered by its proxy (Eq. 3) + the other layer held fixed."""
    image = edit_layer.render(lights)
    if fixed_image.shape != image.shape:
        raise ValueError(f"fixed image shape {fixed_image.shape} != edited layer's {image.shape}")
    return image + fixed_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--edit-model", required=True, help="edited layer's model .pt")
    parser.add_argument("--edit-cache", required=True, help="edited layer's path cache .npz")
    parser.add_argument(
        "--light", required=True, help="JSON file or inline JSON: new light(s) for the edited layer"
    )
    parser.add_argument(
        "--fixed-image", required=True, help="(H,W,3) .npy of the layer being held fixed"
    )
    parser.add_argument("--out", help="output composite image .npy")
    parser.add_argument("--device", default="cpu", help="cpu/mps/cuda")
    parser.add_argument("--bench", type=int, default=0, help="benchmark N composite edits")
    args = parser.parse_args()

    edit_layer = ViewProxy(
        "edit",
        TorchNRP.load(args.edit_model),
        PathCache.load(args.edit_cache),
        torch.device(args.device),
    )
    try:
        spec = json.loads(args.light)
    except json.JSONDecodeError:
        with open(args.light) as f:
            spec = json.load(f)
    specs = spec if isinstance(spec, list) else [spec]
    lights = [light_from_dict(s) for s in specs]
    fixed_image = np.load(args.fixed_image)

    image = composite(edit_layer, lights, fixed_image)
    if args.out:
        np.save(args.out, image)
        print(f"wrote {args.out}")
    print(
        f"composite mean radiance {image.mean():.6f} "
        f"({len(lights)} edited light(s) + fixed {args.fixed_image})"
    )

    if args.bench:
        for _ in range(2):
            composite(edit_layer, lights, fixed_image)
        _synchronize(edit_layer.device)
        t0 = time.perf_counter()
        for _ in range(args.bench):
            composite(edit_layer, lights, fixed_image)
        _synchronize(edit_layer.device)
        ms = (time.perf_counter() - t0) / args.bench * 1000.0
        print(
            f"composite edit latency at {edit_layer.cache.width}x{edit_layer.cache.height}: "
            f"{ms:.2f} ms/edit ({1000.0 / ms:.1f} Hz) over {args.bench} edits"
        )


if __name__ == "__main__":
    main()
