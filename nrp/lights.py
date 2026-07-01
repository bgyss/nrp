"""Sphere-light parameterization and segment intersection queries (NRP M1/M2).

A `SphereLight` is a *virtual* light in the paper's sense: it is a pure emitter that
never blocks or scatters cached paths, so the same light-agnostic path cache can be
reused for any light configuration. A path segment "hits" the light iff the segment's
parametric interval [0, t_max] overlaps the ray's sphere-interior interval [t0, t1]
(this counts segments that start inside the sphere, and segments that pass through it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SphereLight:
    """center (3,), radius (scalar, >0), rgb (3,) emitted-radiance scale."""

    center: np.ndarray
    radius: float
    rgb: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.rgb = np.asarray(self.rgb, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"center must be (3,), got {self.center.shape}")
        if self.rgb.shape != (3,):
            raise ValueError(f"rgb must be (3,), got {self.rgb.shape}")
        if not self.radius > 0.0:
            raise ValueError(f"radius must be > 0, got {self.radius}")

    def to_dict(self) -> dict:
        return {
            "center": self.center.tolist(),
            "radius": float(self.radius),
            "rgb": self.rgb.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SphereLight:
        return cls(center=d["center"], radius=d["radius"], rgb=d.get("rgb", [1.0, 1.0, 1.0]))


def segment_hits_sphere(
    origins: np.ndarray,
    dirs: np.ndarray,
    t_max: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Vectorized segment-vs-sphere overlap test.

    origins (S,3), dirs (S,3) unit-length, t_max (S,) possibly np.inf for escape rays.
    Returns bool (S,): True iff [t0, t1] (the ray's interval inside the sphere) overlaps
    [0, t_max].
    """
    origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    t_max = np.atleast_1d(np.asarray(t_max, dtype=np.float64))
    oc = origins - np.asarray(center, dtype=np.float64)
    b = np.einsum("ij,ij->i", oc, dirs)
    c = np.einsum("ij,ij->i", oc, oc) - float(radius) ** 2
    disc = b * b - c
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    return (disc >= 0.0) & (t0 <= t_max) & (t1 >= 0.0)
