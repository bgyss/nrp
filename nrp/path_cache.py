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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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

    @property
    def segment_count(self) -> int:
        return int(self.seg_pixel.shape[0])

    def save(self, path: str) -> None:
        self.validate()
        np.savez_compressed(
            path,
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
        )

    @classmethod
    def load(cls, path: str) -> PathCache:
        z = np.load(path)
        cache = cls(
            width=int(z["width"]),
            height=int(z["height"]),
            n_paths=z["n_paths"],
            seg_pixel=z["seg_pixel"],
            seg_origin=z["seg_origin"],
            seg_dir=z["seg_dir"],
            seg_tmax=z["seg_tmax"],
            seg_throughput=z["seg_throughput"],
            albedo=z["albedo"],
            position=z["position"],
            depth=z["depth"],
            normal=z["normal"],
        )
        cache.validate()
        return cache

    def to_dict(self) -> dict:
        self.validate()
        tmax = [None if not np.isfinite(t) else float(t) for t in self.seg_tmax]
        return {
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
        )
        cache.validate()
        return cache
