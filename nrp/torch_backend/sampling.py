"""Light-configuration sampling for training (paper §4.4, "Sampling light positions").

Two position strategies, matching the paper:
- "segments": pick a recorded path segment uniformly, then a point uniformly along it.
  This implicitly importance-samples positions that contribute to the image and avoids
  unreachable ones (e.g. inside closed objects). Escape segments (t_max = inf) are
  sampled over a finite range set by the scene's extent.
- "bbox": uniform within the bounding box of camera-visible positions — the paper's
  fallback for gigantic scenes where segment sampling wastes network capacity.

Shape parameters (sphere radius, quad normal/size) are uniform within configured
bounds; the quad normal is uniform on the unit sphere.
"""

from __future__ import annotations

import numpy as np

from ..lights import QuadLight, SphereLight, TexturedQuadLight
from ..path_cache import PathCache


def _scene_extent(cache: PathCache) -> float:
    finite = cache.seg_tmax[np.isfinite(cache.seg_tmax)]
    if finite.size:
        return float(finite.max())
    return 1.0


def sample_position_on_segments(cache: PathCache, rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 3) positions, each uniform along a uniformly chosen recorded segment."""
    if not cache.segment_count:
        raise ValueError("cannot segment-sample an empty path cache")
    idx = rng.integers(0, cache.segment_count, size=n)
    t_max = np.minimum(cache.seg_tmax[idx], _scene_extent(cache))
    t = rng.random(n) * t_max
    return cache.seg_origin[idx] + t[:, None] * cache.seg_dir[idx]


def sample_position_in_bbox(cache: PathCache, rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 3) positions uniform in the bbox of camera-visible first-hit positions."""
    pos = cache.position.reshape(-1, 3)
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    return lo + rng.random((n, 3)) * (hi - lo)


def sample_positions(
    cache: PathCache, rng: np.random.Generator, n: int, strategy: str = "segments"
) -> np.ndarray:
    if strategy == "segments":
        return sample_position_on_segments(cache, rng, n)
    if strategy == "bbox":
        return sample_position_in_bbox(cache, rng, n)
    raise ValueError(f"unknown sampling strategy {strategy!r}")


def sample_light(
    cache: PathCache,
    rng: np.random.Generator,
    light_type: str,
    bounds: dict,
    strategy: str = "segments",
) -> SphereLight | QuadLight | TexturedQuadLight:
    """One random light configuration with unit emission (E is factored out during
    training; the network learns the pre-emission contribution)."""
    center = sample_positions(cache, rng, 1, strategy)[0]
    if light_type == "sphere":
        radius = bounds["radius_min"] + rng.random() * (bounds["radius_max"] - bounds["radius_min"])
        return SphereLight(center=center, radius=float(radius))
    if light_type == "quad":
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        width = bounds["size_min"] + rng.random() * (bounds["size_max"] - bounds["size_min"])
        height = bounds["size_min"] + rng.random() * (bounds["size_max"] - bounds["size_min"])
        return QuadLight(center=center, normal=normal, width=float(width), height=float(height))
    if light_type == "textured_quad":
        tex_h, tex_w = bounds.get("texture_size", [2, 2])
        tex_min = float(bounds.get("texture_min", 0.0))
        tex_max = float(bounds.get("texture_max", 1.0))
        texture = tex_min + rng.random((int(tex_h), int(tex_w), 3)) * (tex_max - tex_min)
        normal = np.asarray(bounds.get("normal", [0.0, 0.0, -1.0]), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        center = np.asarray(bounds.get("center", center), dtype=np.float64)
        width = float(bounds.get("width", bounds.get("size", 1.0)))
        height = float(bounds.get("height", bounds.get("size", 1.0)))
        return TexturedQuadLight(
            center=center,
            normal=normal,
            width=width,
            height=height,
            texture=texture,
        )
    raise ValueError(f"unknown light type {light_type!r}")
