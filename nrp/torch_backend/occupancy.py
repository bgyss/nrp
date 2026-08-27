"""Exact per-level queried-vertex sets for world-anchored spatial encodings.

The original R1 experiment sized world-anchored encoding tables against a
parameter budget rather than against the number of grid vertices the cache
actually queries — a real methodological defect (see
`docs/representation-track.md`'s R1 fair-allocation correction for the
measurement showing this mismatch, on its own, does not explain R1's original
negative). This module fixes that defect by making the queried-vertex count a
measured, tested quantity instead of an invisible one, so table allocation can
be sized against actual grid occupancy. It knows about caches; encoders do not.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..path_cache import PathCache


@dataclass(frozen=True)
class LevelOccupancy:
    """The unique grid vertices one level of an encoding actually reads."""

    level: int
    resolution: int
    vertices: np.ndarray  # (V, ndim) int64, unique, lexicographically sorted

    @property
    def count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.vertices.shape[1])


def level_resolutions(levels: int, base_resolution: int, finest_resolution: int) -> list[int]:
    """Reproduce the hashgrid geometric schedule exactly.

    Any divergence here would make the occupancy describe a different grid than the
    encoder queries, which is precisely the class of error this module exists to
    prevent.
    """
    if levels <= 0:
        raise ValueError("levels must be positive")
    if base_resolution <= 0 or finest_resolution <= 0:
        raise ValueError("resolutions must be positive")
    if finest_resolution < base_resolution:
        raise ValueError("finest_resolution must be >= base_resolution")
    growth = (
        math.exp(math.log(finest_resolution / base_resolution) / max(levels - 1, 1))
        if levels > 1
        else 1.0
    )
    return [max(int(math.floor(base_resolution * growth**level)), 1) for level in range(levels)]


def normalize_positions(positions: np.ndarray, bounds: dict) -> np.ndarray:
    """Map world positions into the unit cube using the model's stored bounds."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lo = np.asarray(bounds["min"], dtype=np.float64)
    hi = np.asarray(bounds["max"], dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bounds min and max must each contain three values")
    if np.any(hi <= lo):
        raise ValueError("bounds max must exceed min on every axis")
    return np.clip((positions - lo) / (hi - lo), 0.0, 1.0)


def grid_occupancy(normalized: np.ndarray, resolutions: list[int]) -> list[LevelOccupancy]:
    """Unique vertices touched per level, matching the encoders' corner enumeration."""
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError("normalized coordinates must be 2-D")
    ndim = normalized.shape[1]
    out: list[LevelOccupancy] = []
    for level, res in enumerate(resolutions):
        base = np.floor(normalized * res).astype(np.int64).clip(0, res - 1)
        corners = [
            base + np.asarray(offset, dtype=np.int64)
            for offset in itertools.product((0, 1), repeat=ndim)
        ]
        stacked = np.concatenate(corners, axis=0)
        unique = np.unique(stacked, axis=0)
        out.append(LevelOccupancy(level=level, resolution=res, vertices=unique))
    return out


def cache_occupancy(
    cache: PathCache,
    bounds: dict,
    levels: int,
    base_resolution: int,
    finest_resolution: int,
) -> list[LevelOccupancy]:
    """Occupancy of a cache's first-hit world positions under a level schedule."""
    normalized = normalize_positions(cache.position, bounds)
    return grid_occupancy(normalized, level_resolutions(levels, base_resolution, finest_resolution))


def capacity_report(occupancy: list[LevelOccupancy], index_fn, slots: list[int]) -> dict:
    """Per-level capacity: distinct vertices, used slots, collision fraction, max load.

    `index_fn(vertices, level) -> np.ndarray` maps vertex coordinates to slot indices,
    so this stays agnostic to whether the encoder hashes, indexes densely, or uses an
    exact sparse map.
    """
    if len(slots) != len(occupancy):
        raise ValueError("slots must supply one entry per level")
    levels = []
    for occ, n_slots in zip(occupancy, slots, strict=True):
        idx = np.asarray(index_fn(occ.vertices, occ.level), dtype=np.int64)
        if idx.shape != (occ.count,):
            raise ValueError("index_fn must return one index per vertex")
        counts = np.bincount(idx, minlength=int(n_slots))
        used = int((counts > 0).sum())
        levels.append(
            {
                "level": occ.level,
                "resolution": occ.resolution,
                "distinct_vertices": occ.count,
                "slots": int(n_slots),
                "used_slots": used,
                "collision_fraction": float(1.0 - used / occ.count) if occ.count else 0.0,
                "max_slot_load": int(counts.max()) if counts.size else 0,
                "slots_per_distinct_vertex": float(n_slots / occ.count) if occ.count else 0.0,
            }
        )
    finest = levels[-1] if levels else {}
    return {
        "levels": levels,
        "total_distinct_vertices": int(sum(level["distinct_vertices"] for level in levels)),
        "total_slots": int(sum(level["slots"] for level in levels)),
        "finest_collision_fraction": finest.get("collision_fraction", 0.0),
        "finest_slots_per_distinct_vertex": finest.get("slots_per_distinct_vertex", 0.0),
    }
