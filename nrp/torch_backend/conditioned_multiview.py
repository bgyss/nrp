"""Shared data contracts for the R2 camera-conditioned NRP.

The representation-track R2 experiment trains one world-anchored proxy over several
fixed-camera path caches.  This module owns the camera-aware manifest and geometry
helpers; training, pool construction, and shared inference are added below this
contract so all R2 paths agree on paths, camera directions, and scene bounds.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..gather_light import gather_light
from ..path_cache import PathCache
from .denoise import denoise_image
from .gather import TorchPathCache
from .sampling import sample_light
from .train import light_param_dim_from_cfg, light_param_vector


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


def _render_target(
    cache: PathCache, light, cfg: dict, torch_cache: TorchPathCache | None
) -> np.ndarray:
    """Render one unit-emission training target using the configured gather backend."""
    if torch_cache is None:
        image = gather_light(cache, light)
    else:
        image = torch_cache.gather_light(light).detach().cpu().numpy().astype(np.float64)
    denoise = cfg.get("denoise", {})
    if denoise.get("enabled", True):
        image = denoise_image(
            image,
            cache.albedo,
            cache.normal,
            cache.depth,
            method=denoise.get("method", "bilateral"),
            **{
                key: value
                for key, value in denoise.items()
                if key not in {"enabled", "method"}
            },
        )
    return np.asarray(image, dtype=np.float64).reshape(-1, 3)


class MultiViewImagePool:
    """Shared light-shape pool with one target image per camera view."""

    def __init__(self, caches: Sequence[PathCache], cfg: dict, rng, device, fill: bool = True):
        if not caches:
            raise ValueError("at least one cache is required for a multi-view pool")
        resolution = (caches[0].height, caches[0].width)
        if any((cache.height, cache.width) != resolution for cache in caches[1:]):
            raise ValueError("all multi-view pool caches must have the same resolution")
        self.caches = list(caches)
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.size = int(cfg["pool"]["size"])
        if self.size <= 0:
            raise ValueError("pool.size must be positive")
        self._next_replace = 0
        n_pixels = caches[0].height * caches[0].width
        light_dim = light_param_dim_from_cfg(cfg)
        self.params = torch.empty((self.size, light_dim), dtype=torch.float32, device=device)
        self.targets = torch.empty(
            (len(caches), self.size, n_pixels, 3), dtype=torch.float32, device=device
        )
        self.used_params: list[np.ndarray] = []
        self.supervision_seconds = 0.0
        self._torch_caches = (
            [TorchPathCache(cache, device) for cache in self.caches]
            if cfg.get("gather_backend", "numpy") == "torch"
            else None
        )
        if cfg.get("gather_backend", "numpy") not in {"numpy", "torch"}:
            raise ValueError("gather_backend must be 'numpy' or 'torch'")
        if fill:
            for slot in range(self.size):
                self.fill(slot)

    def fill(self, slot: int) -> None:
        if not 0 <= slot < self.size:
            raise IndexError(f"pool slot {slot} is outside [0, {self.size})")
        started = time.perf_counter()
        light = sample_light(
            self.caches[0],
            self.rng,
            self.cfg["light_type"],
            self.cfg["light_bounds"],
            self.cfg.get("sampling", "segments"),
        )
        vector = np.asarray(light_param_vector(light), dtype=np.float64)
        self.params[slot] = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        for view_index, cache in enumerate(self.caches):
            torch_cache = self._torch_caches[view_index] if self._torch_caches is not None else None
            self.targets[view_index, slot] = torch.as_tensor(
                _render_target(cache, light, self.cfg, torch_cache),
                dtype=torch.float32,
                device=self.device,
            )
        self.used_params.append(vector.copy())
        self.supervision_seconds += time.perf_counter() - started

    def replace_round(self) -> None:
        replace_count = int(self.cfg["pool"]["replace_count"])
        if replace_count < 0:
            raise ValueError("pool.replace_count must be non-negative")
        for _ in range(replace_count):
            self.fill(self._next_replace)
            self._next_replace = (self._next_replace + 1) % self.size

    @property
    def supervision_images(self) -> int:
        return len(self.used_params) * len(self.caches)


def build_validation_sets(
    caches: Sequence[PathCache], cfg: dict, seed: int | None = None
) -> list[list[dict]]:
    """Build independent, per-view held-out light sets and physical references."""
    if not caches:
        raise ValueError("at least one cache is required for validation")
    base_seed = int(cfg.get("seed", 0) if seed is None else seed)
    count = int(cfg.get("n_val_lights", 12))
    if count < 0:
        raise ValueError("n_val_lights must be non-negative")
    result: list[list[dict]] = []
    for view_index, cache in enumerate(caches):
        rng = np.random.default_rng([base_seed, 0x5EED, view_index])
        entries = []
        for _ in range(count):
            light = sample_light(
                cache,
                rng,
                cfg["light_type"],
                cfg["light_bounds"],
                cfg.get("sampling", "segments"),
            )
            raw = gather_light(cache, light).reshape(-1, 3)
            denoised = _render_target(cache, light, cfg, None)
            entries.append(
                {
                    "light": light,
                    "params": np.asarray(light_param_vector(light), dtype=np.float64),
                    "raw": raw,
                    "denoised": denoised,
                }
            )
        result.append(entries)
    return result


def validation_disjointness(
    training_params: Sequence[np.ndarray], validation_sets: Sequence[Sequence[dict]]
) -> list[bool]:
    """Return one exact parameter-disjointness result for each view's validation set."""
    training_keys = {
        tuple(np.asarray(vector, dtype=np.float64).reshape(-1).tolist())
        for vector in training_params
    }
    result = []
    for entries in validation_sets:
        result.append(
            all(
                tuple(np.asarray(entry["params"], dtype=np.float64).reshape(-1).tolist())
                not in training_keys
                for entry in entries
            )
        )
    return result
