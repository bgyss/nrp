"""Batched torch GATHERLIGHT: all segments tested against a light in one op.

The paper fuses GATHERLIGHT into a single Triton kernel; this is the same idea at
PyTorch-op granularity — the segment-vs-light overlap test and the per-pixel
throughput accumulation run as a handful of batched tensor ops over the whole cache,
so the work executes on whatever device the tensors live on (CPU, MPS, CUDA).

The numpy implementation in `nrp/gather_light.py` remains the authoritative
reference; this module mirrors its semantics exactly (same overlap predicates, same
n_paths normalization) and is unit-tested against it. On CPU the default dtype is
float64 and results match numpy to floating-point noise; MPS only supports float32,
where boundary-grazing segments can round differently — fine for training targets,
not for physics validation.
"""

from __future__ import annotations

import numpy as np
import torch

from ..lights import QuadLight, SphereLight, quad_tangent_frame
from ..path_cache import PathCache


class TorchPathCache:
    """Device-resident copy of a PathCache's segment arrays for batched gathering."""

    def __init__(self, cache: PathCache, device: torch.device, dtype: torch.dtype | None = None):
        if dtype is None:
            dtype = torch.float32 if device.type in ("mps",) else torch.float64
        self.width = cache.width
        self.height = cache.height
        self.device = device
        self.dtype = dtype
        to = lambda a: torch.as_tensor(a, dtype=dtype, device=device)  # noqa: E731
        self.origin = to(cache.seg_origin)
        self.dir = to(cache.seg_dir)
        # inf t_max (escape segments) participates in the same comparisons as numpy.
        self.tmax = to(cache.seg_tmax)
        self.throughput = to(cache.seg_throughput)
        self.pixel = torch.as_tensor(cache.seg_pixel, dtype=torch.long, device=device)
        self.inv_paths = to(1.0 / np.maximum(cache.n_paths, 1))

    @property
    def segment_count(self) -> int:
        return int(self.pixel.shape[0])

    def _accumulate(self, hits: torch.Tensor) -> torch.Tensor:
        """Sum throughput of hit segments per pixel, normalized by path count."""
        n_px = self.height * self.width
        contrib = torch.zeros((n_px, 3), dtype=self.dtype, device=self.device)
        # Weight-and-scatter over all segments (fixed shapes, no host sync) rather
        # than boolean indexing.
        weighted = self.throughput * hits.to(self.dtype).unsqueeze(-1)
        contrib.index_add_(0, self.pixel, weighted)
        contrib *= self.inv_paths.unsqueeze(-1)
        return contrib.reshape(self.height, self.width, 3)

    def gather_throughput(self, center, radius: float) -> torch.Tensor:
        """Sphere GATHERtype: (H,W,3) pre-emission contribution (see numpy reference)."""
        if not self.segment_count:
            return torch.zeros((self.height, self.width, 3), dtype=self.dtype, device=self.device)
        c = torch.as_tensor(center, dtype=self.dtype, device=self.device)
        oc = self.origin - c
        b = (oc * self.dir).sum(dim=1)
        cc = (oc * oc).sum(dim=1) - float(radius) ** 2
        disc = b * b - cc
        sq = torch.sqrt(torch.clamp(disc, min=0.0))
        t0 = -b - sq
        t1 = -b + sq
        hits = (disc >= 0.0) & (t0 <= self.tmax) & (t1 >= 0.0)
        return self._accumulate(hits)

    def gather_throughput_quad(self, center, normal, width: float, height: float) -> torch.Tensor:
        """Quad GATHERtype: (H,W,3) pre-emission contribution (see numpy reference)."""
        if not self.segment_count:
            return torch.zeros((self.height, self.width, 3), dtype=self.dtype, device=self.device)
        n = np.asarray(normal, dtype=np.float64)
        n = n / np.linalg.norm(n)
        u, v = quad_tangent_frame(n)
        to = lambda a: torch.as_tensor(a, dtype=self.dtype, device=self.device)  # noqa: E731
        c, n_t, u_t, v_t = to(center), to(n), to(u), to(v)

        denom = self.dir @ n_t
        parallel = denom.abs() < 1e-12
        safe = torch.where(parallel, torch.ones_like(denom), denom)
        t = ((c - self.origin) @ n_t) / safe
        p = self.origin + t.unsqueeze(-1) * self.dir
        local = p - c
        hits = (
            ~parallel
            & (t >= 0.0)
            & (t <= self.tmax)
            & ((local @ u_t).abs() <= width / 2.0)
            & ((local @ v_t).abs() <= height / 2.0)
        )
        return self._accumulate(hits)

    def gather_light(self, light: SphereLight | QuadLight) -> torch.Tensor:
        """Full contribution of one light: GATHERtype scaled by emission rgb."""
        if isinstance(light, SphereLight):
            image = self.gather_throughput(light.center, light.radius)
        else:
            image = self.gather_throughput_quad(
                light.center, light.normal, light.width, light.height
            )
        return image * torch.as_tensor(light.rgb, dtype=self.dtype, device=self.device)
