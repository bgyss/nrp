"""Light-agnostic path-cache schema, serialization, and loaders (NRP M1).

The cache stores, for a fixed camera and static scene, everything a decoupled emission
pass (`gather_light.py`) needs to evaluate an arbitrary virtual sphere light without
re-tracing: flattened path segments with per-segment throughput, plus per-pixel
auxiliary features (albedo, depth, normal, first-hit position) for proxy training.

Layout (S = total segment count across all pixels and paths):
  - n_paths (H*W,)        int64   paths traced per pixel (may be 0: undersampled pixel)
  - seg_pixel (S,)        int64   row-major pixel index of each segment
  - seg_origin (S, 3)     float64 segment start point
  - seg_dir (S, 3)        float64 unit direction
  - seg_tmax (S,)         float64 segment length; np.inf marks an escape direction
  - seg_throughput (S, 3) float64 path throughput accumulated *before* this segment
  - albedo (H, W, 3), depth (H, W), normal (H, W, 3), position (H, W, 3)
    auxiliary buffers (position = first-hit world position, standard G-buffer content)

Two serializations: `.npz` for tracer-exported caches, and a JSON dict form
(`to_dict`/`from_dict`) for tiny hand-authored caches in tests. In JSON, an escape
segment's t_max is `null` (JSON has no inf).

The `.npz` form has two layouts. The default stores every float array as float64.
`save(path, compressed=True)` writes the paper's packed layout (§4.2) instead:
geometry (segment origins/directions/t_max and the G-buffer aux) as fp16,
per-segment throughput as shared-exponent rgb9e5 words (`nrp/rgb9e5.py`), and
seg_pixel as int32. `load` auto-detects the layout (packed caches carry a
`packed_layout` key) and always hands back float64 arrays, so everything
downstream of `load` is layout-agnostic. fp16 directions are renormalized on
load to restore unit length. Escape segments survive packing: fp16 represents
inf exactly, and finite t_max values are clamped to the fp16 finite range so
they can never round *to* inf.

Schema versions:
  - v1: surface-only, no version field (all caches written before mid-2026).
  - v2: adds `schema_version` and an optional `medium` metadata dict
    (`{"sigma_t": float, "albedo": float}`) recorded by producers that free-flight
    sample a homogeneous participating medium (§3.1 "Volume rendering"). Segments in
    a medium cache may end at scattering vertices instead of surfaces; nothing about
    the segment arrays themselves changes, so v1 readers of the arrays and GATHERLIGHT
    work unchanged — transmittance is implicit in the recorded segment lengths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rgb9e5 import rgb9e5_decode, rgb9e5_encode

SCHEMA_VERSION = 2

_FP16_MAX = float(np.finfo(np.float16).max)
_FP16_TINY = float(np.finfo(np.float16).smallest_subnormal)


def _to_fp16(arr: np.ndarray) -> np.ndarray:
    """fp16 with finite values clamped into fp16's finite range (inf stays inf)."""
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    return np.where(finite, np.clip(a, -_FP16_MAX, _FP16_MAX), a).astype(np.float16)


@dataclass
class PathCache:
    width: int
    height: int
    n_paths: np.ndarray
    seg_pixel: np.ndarray
    seg_origin: np.ndarray
    seg_dir: np.ndarray
    seg_tmax: np.ndarray
    seg_throughput: np.ndarray
    albedo: np.ndarray
    position: np.ndarray
    depth: np.ndarray
    normal: np.ndarray
    medium: dict | None = field(default=None)

    def validate(self) -> None:
        h, w = self.height, self.width
        if self.n_paths.shape != (h * w,):
            raise ValueError(f"n_paths must be ({h * w},), got {self.n_paths.shape}")
        s = self.seg_pixel.shape[0]
        for name, arr, shape in [
            ("seg_pixel", self.seg_pixel, (s,)),
            ("seg_origin", self.seg_origin, (s, 3)),
            ("seg_dir", self.seg_dir, (s, 3)),
            ("seg_tmax", self.seg_tmax, (s,)),
            ("seg_throughput", self.seg_throughput, (s, 3)),
            ("albedo", self.albedo, (h, w, 3)),
            ("position", self.position, (h, w, 3)),
            ("depth", self.depth, (h, w)),
            ("normal", self.normal, (h, w, 3)),
        ]:
            if arr.shape != shape:
                raise ValueError(f"{name} must be {shape}, got {arr.shape}")
        if s and (self.seg_pixel.min() < 0 or self.seg_pixel.max() >= h * w):
            raise ValueError("seg_pixel indices out of range")
        if s and not np.all(self.seg_tmax > 0.0):
            raise ValueError("seg_tmax must be positive (np.inf for escape segments)")
        norms = np.linalg.norm(self.seg_dir, axis=1)
        if s and not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("seg_dir rows must be unit length")
        if self.medium is not None:
            if not float(self.medium["sigma_t"]) > 0.0:
                raise ValueError("medium sigma_t must be positive")
            if not 0.0 <= float(self.medium["albedo"]) <= 1.0:
                raise ValueError("medium albedo must be in [0, 1]")

    @property
    def segment_count(self) -> int:
        return int(self.seg_pixel.shape[0])

    def save(self, path: str, compressed: bool = False) -> None:
        self.validate()
        extra = {}
        if self.medium is not None:
            extra["medium_sigma_t"] = float(self.medium["sigma_t"])
            extra["medium_albedo"] = float(self.medium["albedo"])
        if compressed:
            # Packed layout (§4.2): fp16 geometry + rgb9e5 throughput. Positive
            # t_max that would round to fp16 zero is pinned to the smallest
            # subnormal so validate()'s positivity invariant survives the trip.
            tmax16 = _to_fp16(self.seg_tmax)
            tmax16 = np.where((tmax16 == 0) & (self.seg_tmax > 0), np.float16(_FP16_TINY), tmax16)
            np.savez_compressed(
                path,
                schema_version=SCHEMA_VERSION,
                packed_layout=1,
                width=self.width,
                height=self.height,
                n_paths=self.n_paths,
                seg_pixel=self.seg_pixel.astype(np.int32),
                seg_origin=_to_fp16(self.seg_origin),
                seg_dir=_to_fp16(self.seg_dir),
                seg_tmax=tmax16,
                seg_throughput_rgb9e5=rgb9e5_encode(self.seg_throughput),
                albedo=_to_fp16(self.albedo),
                position=_to_fp16(self.position),
                depth=_to_fp16(self.depth),
                normal=_to_fp16(self.normal),
                **extra,
            )
            return
        np.savez_compressed(
            path,
            schema_version=SCHEMA_VERSION,
            width=self.width,
            height=self.height,
            n_paths=self.n_paths,
            seg_pixel=self.seg_pixel,
            seg_origin=self.seg_origin,
            seg_dir=self.seg_dir,
            seg_tmax=self.seg_tmax,
            seg_throughput=self.seg_throughput,
            albedo=self.albedo,
            position=self.position,
            depth=self.depth,
            normal=self.normal,
            **extra,
        )

    @classmethod
    def load(cls, path: str) -> PathCache:
        z = np.load(path)
        # v1 caches have no schema_version key; v2 adds it plus optional medium_*.
        medium = None
        if "medium_sigma_t" in z:
            medium = {
                "sigma_t": float(z["medium_sigma_t"]),
                "albedo": float(z["medium_albedo"]),
            }
        packed = "packed_layout" in z
        if packed:
            seg_dir = z["seg_dir"].astype(np.float64)
            norms = np.linalg.norm(seg_dir, axis=1, keepdims=True)
            seg_dir = np.divide(seg_dir, norms, out=seg_dir, where=norms > 0)
            throughput = rgb9e5_decode(z["seg_throughput_rgb9e5"])
        else:
            seg_dir = z["seg_dir"]
            throughput = z["seg_throughput"]
        cache = cls(
            width=int(z["width"]),
            height=int(z["height"]),
            n_paths=z["n_paths"],
            seg_pixel=z["seg_pixel"].astype(np.int64),
            seg_origin=z["seg_origin"].astype(np.float64),
            seg_dir=seg_dir,
            seg_tmax=z["seg_tmax"].astype(np.float64),
            seg_throughput=throughput,
            albedo=z["albedo"].astype(np.float64),
            position=z["position"].astype(np.float64),
            depth=z["depth"].astype(np.float64),
            normal=z["normal"].astype(np.float64),
            medium=medium,
        )
        cache.validate()
        return cache

    def to_dict(self) -> dict:
        self.validate()
        tmax = [None if not np.isfinite(t) else float(t) for t in self.seg_tmax]
        return {
            "schema_version": SCHEMA_VERSION,
            "medium": dict(self.medium) if self.medium is not None else None,
            "width": self.width,
            "height": self.height,
            "n_paths": self.n_paths.tolist(),
            "seg_pixel": self.seg_pixel.tolist(),
            "seg_origin": self.seg_origin.tolist(),
            "seg_dir": self.seg_dir.tolist(),
            "seg_tmax": tmax,
            "seg_throughput": self.seg_throughput.tolist(),
            "albedo": self.albedo.tolist(),
            "position": self.position.tolist(),
            "depth": self.depth.tolist(),
            "normal": self.normal.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> PathCache:
        tmax = np.array(
            [np.inf if t is None else float(t) for t in d["seg_tmax"]], dtype=np.float64
        )
        cache = cls(
            width=int(d["width"]),
            height=int(d["height"]),
            n_paths=np.asarray(d["n_paths"], dtype=np.int64),
            seg_pixel=np.asarray(d["seg_pixel"], dtype=np.int64),
            seg_origin=np.asarray(d["seg_origin"], dtype=np.float64).reshape(-1, 3),
            seg_dir=np.asarray(d["seg_dir"], dtype=np.float64).reshape(-1, 3),
            seg_tmax=tmax,
            seg_throughput=np.asarray(d["seg_throughput"], dtype=np.float64).reshape(-1, 3),
            albedo=np.asarray(d["albedo"], dtype=np.float64),
            position=np.asarray(d["position"], dtype=np.float64),
            depth=np.asarray(d["depth"], dtype=np.float64),
            normal=np.asarray(d["normal"], dtype=np.float64),
            medium=d.get("medium"),
        )
        cache.validate()
        return cache
