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

from ..gather_light import gather_lights
from ..lights import TexturedQuadLight, light_from_dict
from ..path_cache import PathCache
from .device import resolve_device
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
            rgb = (
                torch.ones(3, dtype=torch.float32, device=device)
                if isinstance(light, TexturedQuadLight)
                else torch.as_tensor(light.rgb, dtype=torch.float32, device=device)
            )
            image += model(xy, aux, params) * rgb
    return image.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)


def relight_tiled(
    model: TorchNRP, cache: PathCache, lights: list, tile_pixels: int = 65536
) -> np.ndarray:
    """Forward relight in pixel chunks to bound activation memory.

    This is equivalent to `relight` for deterministic modules, but avoids running the
    full frame through the MLP at once. It is the inference-side half of the E5
    out-of-core path; cache loading is still monolithic unless paired with sharded
    cache loaders.
    """
    if tile_pixels <= 0:
        raise ValueError("tile_pixels must be positive")
    device = next(model.parameters()).device
    n_px = cache.height * cache.width
    xy, aux = pixel_tensors(cache, device)
    image = torch.zeros((n_px, 3), device=device)
    with torch.no_grad():
        for start in range(0, n_px, tile_pixels):
            end = min(start + tile_pixels, n_px)
            chunk = torch.zeros((end - start, 3), device=device)
            for light in lights:
                params = torch.as_tensor(
                    light_param_vector(light), dtype=torch.float32, device=device
                ).expand(end - start, -1)
                rgb = (
                    torch.ones(3, dtype=torch.float32, device=device)
                    if isinstance(light, TexturedQuadLight)
                    else torch.as_tensor(light.rgb, dtype=torch.float32, device=device)
                )
                chunk += model(xy[start:end], aux[start:end], params) * rgb
            image[start:end] = chunk
    return image.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)


def render_quality_tier(
    model: TorchNRP,
    cache: PathCache,
    lights: list,
    quality: str = "preview",
    tile_pixels: int = 0,
    final_cache: PathCache | None = None,
    residual_lights: list | None = None,
) -> tuple[np.ndarray, dict]:
    """Render one of E9's quality tiers and return image plus metadata.

    - preview: proxy output, optionally corrected by an approval residual.
    - draft: GATHERLIGHT from the current cache.
    - final: GATHERLIGHT from `final_cache` if supplied, else the current cache.
    """
    if quality not in {"preview", "draft", "final"}:
        raise ValueError("quality must be one of preview, draft, final")
    metadata = {
        "quality": quality,
        "cache_resolution": [cache.width, cache.height],
        "light_count": len(lights),
        "residual_applied": residual_lights is not None,
    }
    if quality == "draft":
        metadata["source"] = "gatherlight_cached"
        return gather_lights(cache, lights), metadata
    if quality == "final":
        source_cache = final_cache if final_cache is not None else cache
        metadata["source"] = (
            "gatherlight_final_cache" if final_cache is not None else "gatherlight_cached"
        )
        metadata["final_cache_resolution"] = [source_cache.width, source_cache.height]
        return gather_lights(source_cache, lights), metadata

    image = (
        relight_tiled(model, cache, lights, tile_pixels)
        if tile_pixels
        else relight(model, cache, lights)
    )
    metadata["source"] = "proxy"
    if residual_lights is not None:
        approved_proxy = (
            relight_tiled(model, cache, residual_lights, tile_pixels)
            if tile_pixels
            else relight(model, cache, residual_lights)
        )
        approved_gather = gather_lights(cache, residual_lights)
        image = image + (approved_gather - approved_proxy)
        metadata["source"] = "proxy_plus_cached_residual"
    return image, metadata


def write_image_with_metadata(path: str, image: np.ndarray, metadata: dict) -> None:
    np.save(path, image)
    meta_path = f"{path}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_light_specs(text_or_path: str) -> list:
    try:
        spec = json.loads(text_or_path)
    except json.JSONDecodeError:
        with open(text_or_path) as f:
            spec = json.load(f)
    specs = spec if isinstance(spec, list) else [spec]
    return [light_from_dict(s) for s in specs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="trained model .pt")
    parser.add_argument("--cache", required=True, help="path cache .npz (aux features)")
    parser.add_argument(
        "--light", required=True, help="JSON file or inline JSON: one light spec or a list"
    )
    parser.add_argument("--out", help="output image .npy")
    parser.add_argument(
        "--quality",
        choices=["preview", "draft", "final"],
        default="preview",
        help="E9 quality tier: proxy preview, cached GATHERLIGHT draft, or final-cache GATHERLIGHT",
    )
    parser.add_argument(
        "--final-cache",
        help="higher-spp cache for --quality final; defaults to --cache when omitted",
    )
    parser.add_argument(
        "--residual-light",
        help="approved light JSON used to add cached residual to preview output",
    )
    parser.add_argument("--bench", type=int, default=0, help="benchmark N relight frames")
    parser.add_argument(
        "--device",
        default="cpu",
        help="inference device (cpu/mps/cuda; validated, cuda-unavailable fails clearly)",
    )
    parser.add_argument(
        "--tile-pixels",
        type=int,
        default=0,
        help="use tiled inference with at most this many pixels per chunk",
    )
    args = parser.parse_args()

    model = TorchNRP.load(args.model).to(resolve_device(args.device))
    cache = PathCache.load(args.cache)
    lights = load_light_specs(args.light)
    final_cache = PathCache.load(args.final_cache) if args.final_cache else None
    residual_lights = load_light_specs(args.residual_light) if args.residual_light else None

    image, metadata = render_quality_tier(
        model,
        cache,
        lights,
        quality=args.quality,
        tile_pixels=args.tile_pixels,
        final_cache=final_cache,
        residual_lights=residual_lights,
    )
    if args.out:
        write_image_with_metadata(args.out, image, metadata)
        print(f"wrote {args.out} and {args.out}.json")
    print(f"mean radiance {image.mean():.6f} over {len(lights)} light(s)")

    if args.bench:
        t0 = time.perf_counter()
        for _ in range(args.bench):
            if args.tile_pixels:
                relight_tiled(model, cache, lights, args.tile_pixels)
            else:
                relight(model, cache, lights)
        ms = (time.perf_counter() - t0) / args.bench * 1000.0
        print(
            f"relight rate at {cache.width}x{cache.height}: {ms:.2f} ms/frame "
            f"({1000.0 / ms:.1f} Hz) over {args.bench} frames"
        )


if __name__ == "__main__":
    main()
