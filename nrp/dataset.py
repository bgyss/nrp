"""Shared feature/dataset assembly for NRP training and inference."""

from __future__ import annotations

import numpy as np

from .gather_light import gather_throughput
from .model import encode_inputs
from .path_cache import PathCache


def pixel_feature_block(cache: PathCache) -> dict[str, np.ndarray]:
    """Per-pixel, light-independent features (computed once per scene)."""
    h, w = cache.height, cache.width
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    return {
        "pixel_xy": xy,
        "albedo": cache.albedo.reshape(-1, 3),
        "depth": cache.depth.reshape(-1),
        "normal": cache.normal.reshape(-1, 3),
        "position": cache.position.reshape(-1, 3),
    }


def build_inputs(px: dict[str, np.ndarray], center: np.ndarray, radius: float) -> np.ndarray:
    """(N_pixels, INPUT_DIM) inputs for one light configuration."""
    n = px["pixel_xy"].shape[0]
    return encode_inputs(
        px["pixel_xy"],
        px["albedo"],
        px["depth"],
        px["normal"],
        px["position"],
        np.broadcast_to(np.asarray(center, dtype=np.float64), (n, 3)),
        np.full(n, float(radius)),
    )


def sample_lights(bounds: dict, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, float]]:
    """Uniformly sample (center, radius) pairs within the configured bounds."""
    lo = np.asarray(bounds["center_min"], dtype=np.float64)
    hi = np.asarray(bounds["center_max"], dtype=np.float64)
    out = []
    for _ in range(n):
        center = lo + rng.random(3) * (hi - lo)
        radius = bounds["radius_min"] + rng.random() * (bounds["radius_max"] - bounds["radius_min"])
        out.append((center, float(radius)))
    return out


def light_targets(cache: PathCache, lights: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """(n_lights, N_pixels, 3) GATHERLIGHT throughput targets (pre-emission-scaling)."""
    n_px = cache.height * cache.width
    targets = np.empty((len(lights), n_px, 3))
    for i, (center, radius) in enumerate(lights):
        targets[i] = gather_throughput(cache, center, radius).reshape(n_px, 3)
    return targets
