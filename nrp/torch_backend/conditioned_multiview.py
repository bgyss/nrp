"""Shared data contracts for the R2 camera-conditioned NRP.

The representation-track R2 experiment trains one world-anchored proxy over several
fixed-camera path caches.  This module owns the camera-aware manifest and geometry
helpers; training, pool construction, and shared inference are added below this
contract so all R2 paths agree on paths, camera directions, and scene bounds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..path_cache import PathCache


def _finite_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"camera {name} must contain exactly three values")
    if not np.isfinite(vector).all():
        raise ValueError(f"camera {name} must be finite")
    return vector


def camera_direction(camera: dict) -> np.ndarray:
    """Return the normalized camera forward direction ``target - origin``."""
    if not isinstance(camera, dict):
        raise ValueError("camera metadata must be an object")
    if "origin" not in camera or "target" not in camera:
        raise ValueError("camera metadata requires origin and target")
    origin = _finite_vector(camera["origin"], "origin")
    target = _finite_vector(camera["target"], "target")
    direction = target - origin
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("camera origin and target must define a non-zero direction")
    return direction / length


@dataclass(frozen=True)
class CameraView:
    """One resolved path cache and its physical camera metadata."""

    name: str
    cache_path: Path
    camera: dict

    @property
    def view_dir(self) -> np.ndarray:
        return camera_direction(self.camera)


def load_camera_manifest(path: str | Path) -> list[CameraView]:
    """Load and validate an R2 manifest with relative cache paths."""
    manifest_path = Path(path).resolve()
    with manifest_path.open() as handle:
        payload = json.load(handle)
    entries = payload.get("views") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("camera manifest must contain a non-empty views list")

    views: list[CameraView] = []
    resolution: tuple[int, int] | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"view {index} must be an object")
        raw_cache = entry.get("cache")
        if not isinstance(raw_cache, str) or not raw_cache:
            raise ValueError(f"view {index} requires a cache path")
        camera = entry.get("camera")
        if not isinstance(camera, dict):
            raise ValueError(f"view {index} requires camera metadata")
        # Validate before loading so malformed camera records fail consistently even
        # when an optional cache is not available.
        camera_direction(camera)
        cache_path = Path(raw_cache)
        if not cache_path.is_absolute():
            cache_path = manifest_path.parent / cache_path
        cache = PathCache.load(str(cache_path))
        current_resolution = (cache.height, cache.width)
        if resolution is None:
            resolution = current_resolution
        elif current_resolution != resolution:
            raise ValueError(
                "all camera-manifest caches must have the same resolution "
                f"({resolution[1]}x{resolution[0]}), got "
                f"{current_resolution[1]}x{current_resolution[0]} for {cache_path}"
            )
        views.append(
            CameraView(
                name=str(entry.get("name", f"view{index}")),
                cache_path=cache_path.resolve(),
                camera=camera,
            )
        )
    return views


def global_world_bounds(caches: Sequence[PathCache]) -> dict:
    """Return one finite, non-degenerate world-position bound for all views."""
    if not caches:
        raise ValueError("at least one cache is required for global world bounds")
    positions = []
    for cache in caches:
        position = np.asarray(cache.position, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(position).all():
            raise ValueError("cache.position contains non-finite values")
        positions.append(position)
    all_positions = np.concatenate(positions, axis=0)
    lower = all_positions.min(axis=0)
    upper = all_positions.max(axis=0)
    if np.any(upper <= lower):
        raise ValueError("global cache positions must span a non-zero range on every axis")
    return {"min": lower.tolist(), "max": upper.tolist()}


def camera_tensor(view: CameraView, n_pixels: int, device) -> torch.Tensor:
    """Broadcast one manifest camera direction to a device-resident pixel batch."""
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive")
    direction = torch.as_tensor(view.view_dir, dtype=torch.float32, device=device)
    return direction.reshape(1, 3).expand(n_pixels, -1)
