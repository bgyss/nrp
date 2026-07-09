"""Toy dynamic-geometry cache invalidation and splicing (extension E2).

This module covers two invalidation slices. `primary_visibility_invalidation_mask`
is the conservative one-bounce slice: pixels whose first-hit G-buffer changed. It
misses a real multi-bounce failure mode — a moving object can change *indirect*
illumination on a surface whose own first-hit G-buffer never changes (e.g. an object
moving between two other surfaces changes how light bounces between them, without
occluding either surface's own camera ray). `swept_volume_invalidation_mask` covers
that case: any *cached segment at any bounce depth* whose path could have passed
through the moving object's swept volume is treated as invalid, regardless of which
pixel's primary visibility it belongs to. The implementation is kept cache-level and
deterministic so tests can prove the splice equals a full retrace outside each mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lights import segment_hits_sphere
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


def swept_bounding_sphere(
    center_before: np.ndarray, center_after: np.ndarray, radius: float
) -> tuple[np.ndarray, float]:
    """Conservative bounding sphere covering a translating sphere's swept volume.

    The true swept volume of a translating sphere is a capsule; a sphere centered at
    the midpoint of the two positions with radius `object_radius + half the travel
    distance` contains that capsule entirely (every point on the capsule is within
    `radius` of the nearest point on the segment `[center_before, center_after]`,
    which is at most `half_dist` from the midpoint).
    """
    center_before = np.asarray(center_before, dtype=np.float64)
    center_after = np.asarray(center_after, dtype=np.float64)
    midpoint = (center_before + center_after) / 2.0
    half_dist = float(np.linalg.norm(center_after - center_before)) / 2.0
    return midpoint, float(radius) + half_dist


def swept_volume_invalidation_mask(
    cache: PathCache,
    center_before: np.ndarray,
    center_after: np.ndarray,
    radius: float,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    """Pixels with *any* cached segment (any bounce depth) overlapping the moving
    object's swept volume — the multi-bounce generalization of primary-visibility
    invalidation. A segment overlapping the swept region could have had its
    occlusion/transport status changed by the object's motion, even if it is not
    that pixel's first-hit segment; checking all bounce depths (not just each path's
    first segment) is what makes this cover indirect-bounce changes that primary
    visibility alone misses.
    """
    midpoint, swept_radius = swept_bounding_sphere(center_before, center_after, radius)
    mask = np.zeros(cache.height * cache.width, dtype=bool)
    if cache.segment_count:
        hits = segment_hits_sphere(
            cache.seg_origin, cache.seg_dir, cache.seg_tmax, midpoint, swept_radius + margin
        )
        if hits.any():
            mask[np.unique(cache.seg_pixel[hits])] = True
    return mask.reshape(cache.height, cache.width)


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
