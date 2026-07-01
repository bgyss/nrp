"""Virtual-light parameterizations and segment intersection queries (paper §3.1–3.2).

A virtual light in the paper's sense is a pure emitter that never blocks or scatters
cached paths, so the same light-agnostic path cache can be reused for any light
configuration. Two light types are implemented, matching the paper's GATHERtype split:

- `SphereLight` (4 shape params: 3D center + radius). A path segment "hits" the light
  iff the segment's parametric interval [0, t_max] overlaps the ray's sphere-interior
  interval [t0, t1] (this counts segments that start inside the sphere, and segments
  that pass through it).
- `QuadLight` (8 shape params: 3D center + 3D normal + width + height, paper §3.2 and
  Figure 13). A segment hits iff it crosses the quad's plane within [0, t_max] at a
  point inside the rectangle. The in-plane tangent frame is derived deterministically
  from the normal, so (center, normal, width, height) fully determine the light.

Emission `rgb` scales each light's contribution (the E(v) factor of Eq. 1).
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


def quad_tangent_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic orthonormal (u, v) spanning the quad's plane for a unit normal."""
    n = np.asarray(normal, dtype=np.float64)
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


@dataclass
class QuadLight:
    """center (3,), unit normal (3,), width/height (scalars, >0), rgb (3,) emission."""

    center: np.ndarray
    normal: np.ndarray
    width: float
    height: float
    rgb: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.normal = np.asarray(self.normal, dtype=np.float64)
        self.rgb = np.asarray(self.rgb, dtype=np.float64)
        if self.center.shape != (3,):
            raise ValueError(f"center must be (3,), got {self.center.shape}")
        if self.normal.shape != (3,):
            raise ValueError(f"normal must be (3,), got {self.normal.shape}")
        norm = np.linalg.norm(self.normal)
        if norm <= 0.0:
            raise ValueError("normal must be nonzero")
        self.normal = self.normal / norm
        if self.rgb.shape != (3,):
            raise ValueError(f"rgb must be (3,), got {self.rgb.shape}")
        if not (self.width > 0.0 and self.height > 0.0):
            raise ValueError(f"width/height must be > 0, got {self.width}x{self.height}")

    def to_dict(self) -> dict:
        return {
            "type": "quad",
            "center": self.center.tolist(),
            "normal": self.normal.tolist(),
            "width": float(self.width),
            "height": float(self.height),
            "rgb": self.rgb.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> QuadLight:
        return cls(
            center=d["center"],
            normal=d["normal"],
            width=d["width"],
            height=d["height"],
            rgb=d.get("rgb", [1.0, 1.0, 1.0]),
        )


def light_from_dict(d: dict) -> SphereLight | QuadLight:
    """Dispatch on the optional "type" key; specs without one are sphere lights."""
    kind = d.get("type", "quad" if "width" in d else "sphere")
    if kind == "sphere":
        return SphereLight.from_dict(d)
    if kind == "quad":
        return QuadLight.from_dict(d)
    raise ValueError(f"unknown light type {kind!r}")


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


def segment_hits_quad(
    origins: np.ndarray,
    dirs: np.ndarray,
    t_max: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    """Vectorized segment-vs-rectangle test.

    Returns bool (S,): True iff the segment crosses the quad's plane at t in [0, t_max]
    with the crossing point inside the (width x height) rectangle. Segments parallel to
    the plane never hit (a zero-thickness emitter is a measure-zero grazing case).
    """
    origins = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    dirs = np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    t_max = np.atleast_1d(np.asarray(t_max, dtype=np.float64))
    center = np.asarray(center, dtype=np.float64)
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    u, v = quad_tangent_frame(n)

    denom = dirs @ n
    parallel = np.abs(denom) < 1e-12
    safe = np.where(parallel, 1.0, denom)
    t = ((center - origins) @ n) / safe
    p = origins + t[:, None] * dirs
    local = p - center
    lu = local @ u
    lv = local @ v
    return (
        ~parallel
        & (t >= 0.0)
        & (t <= t_max)
        & (np.abs(lu) <= width / 2.0)
        & (np.abs(lv) <= height / 2.0)
    )
