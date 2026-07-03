"""Toy dynamic-geometry cache invalidation and splicing (extension E2).

This module handles the conservative primary-visibility slice of E2: when the toy
sphere moves, identify pixels whose first-hit G-buffer changed and replace only those
pixels' cached paths with paths from a freshly traced cache. The implementation is
kept cache-level and deterministic so tests can prove the splice equals a full retrace
outside the invalidation mask for one-bounce caches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .path_cache import PathCache


@dataclass
class SpliceStats:
    invalid_pixels: int
    total_pixels: int
    old_segments_removed: int
    new_segments_inserted: int

    @property
    def invalid_fraction(self) -> float:
        return self.invalid_pixels / max(self.total_pixels, 1)


def primary_visibility_invalidation_mask(
    before: PathCache,
    after: PathCache,
    *,
    depth_atol: float = 1e-9,
    normal_atol: float = 1e-9,
    albedo_atol: float = 1e-12,
) -> np.ndarray:
    """Pixels whose first-hit G-buffer differs between two scene states."""
    if before.width != after.width or before.height != after.height:
        raise ValueError("caches must have matching resolution")
    depth_changed = ~np.isclose(before.depth, after.depth, atol=depth_atol, rtol=0.0)
    normal_changed = np.any(
        ~np.isclose(before.normal, after.normal, atol=normal_atol, rtol=0.0), axis=2
    )
    albedo_changed = np.any(
        ~np.isclose(before.albedo, after.albedo, atol=albedo_atol, rtol=0.0), axis=2
    )
    position_changed = np.any(
        ~np.isclose(before.position, after.position, atol=depth_atol, rtol=0.0), axis=2
    )
    return depth_changed | normal_changed | albedo_changed | position_changed


def splice_invalidated_pixels(
    base: PathCache, fresh: PathCache, invalid_mask: np.ndarray
) -> tuple[PathCache, SpliceStats]:
    """Return `base` with invalid pixels' segments and aux buffers replaced by `fresh`."""
    if base.width != fresh.width or base.height != fresh.height:
        raise ValueError("caches must have matching resolution")
    mask = np.asarray(invalid_mask, dtype=bool)
    if mask.shape != (base.height, base.width):
        raise ValueError(f"invalid_mask must be {(base.height, base.width)}, got {mask.shape}")

    invalid_pixels = np.flatnonzero(mask.reshape(-1))
    invalid_lookup = np.zeros(base.width * base.height, dtype=bool)
    invalid_lookup[invalid_pixels] = True
    keep_old = ~invalid_lookup[base.seg_pixel]
    take_new = invalid_lookup[fresh.seg_pixel]

    n_paths = base.n_paths.copy()
    n_paths[invalid_pixels] = fresh.n_paths[invalid_pixels]
    albedo = base.albedo.copy()
    position = base.position.copy()
    depth = base.depth.copy()
    normal = base.normal.copy()
    albedo[mask] = fresh.albedo[mask]
    position[mask] = fresh.position[mask]
    depth[mask] = fresh.depth[mask]
    normal[mask] = fresh.normal[mask]

    cache = PathCache(
        width=base.width,
        height=base.height,
        n_paths=n_paths,
        seg_pixel=np.concatenate([base.seg_pixel[keep_old], fresh.seg_pixel[take_new]]),
        seg_origin=np.concatenate([base.seg_origin[keep_old], fresh.seg_origin[take_new]]),
        seg_dir=np.concatenate([base.seg_dir[keep_old], fresh.seg_dir[take_new]]),
        seg_tmax=np.concatenate([base.seg_tmax[keep_old], fresh.seg_tmax[take_new]]),
        seg_throughput=np.concatenate(
            [base.seg_throughput[keep_old], fresh.seg_throughput[take_new]]
        ),
        albedo=albedo,
        position=position,
        depth=depth,
        normal=normal,
        medium=dict(base.medium) if base.medium is not None else None,
    )
    cache.validate()
    stats = SpliceStats(
        invalid_pixels=int(invalid_pixels.size),
        total_pixels=base.width * base.height,
        old_segments_removed=int((~keep_old).sum()),
        new_segments_inserted=int(take_new.sum()),
    )
    return cache, stats
