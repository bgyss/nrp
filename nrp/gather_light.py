"""Reference GATHERLIGHT: decoupled emission evaluation over a path cache (NRP M2).

Given a light-agnostic `PathCache` and a virtual `SphereLight`, the per-pixel estimate is

    L(p) = (1 / n_paths[p]) * sum over segments s of pixel p:
               throughput[s] * light.rgb * [segment s overlaps the light sphere]

i.e. every pass through the (transparent, purely emissive) light sphere accumulates
emission weighted by the throughput up to that segment; a segment intersecting the
sphere more than once still counts once per segment, but a path whose consecutive
segments each cross the sphere accumulates once per crossing segment. Pixels with zero
cached paths return 0 (undersampled — reported, not interpolated).

Known differences from the paper's GPU/Triton implementation are recorded in the
project README (no MIS/next-event estimation, no re-check of occlusion when the light
radius grows, CPU/numpy instead of Triton).
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from .lights import SphereLight, segment_hits_sphere
from .path_cache import PathCache


def gather_throughput(cache: PathCache, center: np.ndarray, radius: float) -> np.ndarray:
    """Per-pixel summed throughput of segments hitting the sphere, *before* emission
    scaling — the quantity the neural proxy learns. Returns (H, W, 3)."""
    contrib = np.zeros((cache.height * cache.width, 3), dtype=np.float64)
    if cache.segment_count:
        hits = segment_hits_sphere(cache.seg_origin, cache.seg_dir, cache.seg_tmax, center, radius)
        np.add.at(contrib, cache.seg_pixel[hits], cache.seg_throughput[hits])
    denom = np.maximum(cache.n_paths, 1).astype(np.float64)
    contrib /= denom[:, None]
    return contrib.reshape(cache.height, cache.width, 3)


def gather_light(cache: PathCache, light: SphereLight) -> np.ndarray:
    """Full per-pixel light contribution: gather_throughput scaled by light.rgb."""
    return gather_throughput(cache, light.center, light.radius) * light.rgb


def undersampled_mask(cache: PathCache) -> np.ndarray:
    """(H, W) bool mask of pixels with zero cached paths (contribution is untrusted)."""
    return (cache.n_paths == 0).reshape(cache.height, cache.width)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True, help="path cache .npz")
    parser.add_argument("--light", required=True, help="JSON file or inline JSON light spec")
    parser.add_argument("--out", required=True, help="output image .npy (H,W,3 float64)")
    args = parser.parse_args()

    cache = PathCache.load(args.cache)
    try:
        spec = json.loads(args.light)
    except json.JSONDecodeError:
        with open(args.light) as f:
            spec = json.load(f)
    light = SphereLight.from_dict(spec)
    image = gather_light(cache, light)
    np.save(args.out, image)
    n_under = int(undersampled_mask(cache).sum())
    print(
        f"gathered {cache.width}x{cache.height} image from {cache.segment_count} segments; "
        f"mean radiance {image.mean():.6f}; undersampled pixels: {n_under}"
    )


if __name__ == "__main__":
    main()
