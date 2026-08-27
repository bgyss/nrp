"""Multiresolution hash encodings (Müller et al. [MESK22], paper §4.3).

The paper path is the original 2D pixel-coordinate encoding. Representation-track
rung R1 adds a selectable 3D world-position encoding with the same dense/hashed table
policy and geometric resolution growth, using trilinear interpolation over eight
corners. Both are plain-PyTorch instant-ngp-style implementations.
"""

from __future__ import annotations

import torch
from torch import nn

from . import sparse_encoding as _sparse_encoding  # noqa: F401  # registers arm B
from .encoder_registry import (  # noqa: F401
    SPATIAL_ENCODERS,
    _floor_cell,
    build_encoder,
    register_encoder,
)
from .occupancy import level_resolutions

_PRIMES = (1, 2654435761, 805459861)


def _grid_capacity_report(encoder) -> dict:
    """Static slot budget per level, shared by the 2D and 3D grids.

    Occupancy-aware numbers come from `nrp.torch_backend.occupancy.capacity_report`,
    which needs a cache; this is the cache-free view.
    """
    return {
        "encoding": type(encoder).__name__,
        "levels": [
            {
                "level": level,
                "resolution": res,
                "dense": bool(encoder._dense[level]),
                "slots": int(encoder.tables[level].shape[0]),
            }
            for level, res in enumerate(encoder.resolutions)
        ],
        "total_slots": int(sum(t.shape[0] for t in encoder.tables)),
    }


@register_encoder("pixel2d")
class HashEncoding2D(nn.Module):
    needs_occupancy = False
    needs_normals = False
    guarantees_zero_collisions = False

    def __init__(
        self,
        levels: int = 8,
        features_per_level: int = 2,
        table_size_log2: int = 14,
        base_resolution: int = 4,
        finest_resolution: int = 256,
    ):
        super().__init__()
        self.levels = levels
        self.features_per_level = features_per_level
        self.table_size = 1 << table_size_log2
        self.resolutions = level_resolutions(levels, base_resolution, finest_resolution)
        # One table per level. Dense when the grid fits (no collisions), hashed otherwise.
        self.tables = nn.ParameterList()
        self._dense = []
        for res in self.resolutions:
            n_vertices = (res + 1) * (res + 1)
            dense = n_vertices <= self.table_size
            self._dense.append(dense)
            size = n_vertices if dense else self.table_size
            self.tables.append(
                nn.Parameter(torch.empty(size, features_per_level).uniform_(-1e-4, 1e-4))
            )

    @property
    def output_dim(self) -> int:
        return self.levels * self.features_per_level

    def capacity_report(self) -> dict:
        return _grid_capacity_report(self)

    def _index(self, ix: torch.Tensor, iy: torch.Tensor, level: int) -> torch.Tensor:
        res = self.resolutions[level]
        if self._dense[level]:
            return iy * (res + 1) + ix
        return ((ix * _PRIMES[0]) ^ (iy * _PRIMES[1])) & (self.table_size - 1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy in [0,1]^2, shape (N, 2) -> (N, levels * features_per_level)."""
        outputs = []
        for level, res in enumerate(self.resolutions):
            pos = xy * res
            pos0, frac = _floor_cell(pos, res)
            x0, y0 = pos0[:, 0], pos0[:, 1]
            x1 = (x0 + 1).clamp(max=res)
            y1 = (y0 + 1).clamp(max=res)
            table = self.tables[level]
            f00 = table[self._index(x0, y0, level) % table.shape[0]]
            f10 = table[self._index(x1, y0, level) % table.shape[0]]
            f01 = table[self._index(x0, y1, level) % table.shape[0]]
            f11 = table[self._index(x1, y1, level) % table.shape[0]]
            wx = frac[:, 0:1]
            wy = frac[:, 1:2]
            outputs.append(
                f00 * (1 - wx) * (1 - wy)
                + f10 * wx * (1 - wy)
                + f01 * (1 - wx) * wy
                + f11 * wx * wy
            )
        return torch.cat(outputs, dim=1)


@register_encoder("world3d")
class HashEncoding3D(nn.Module):
    """3D multiresolution hashgrid with trilinear interpolation.

    Inputs are normalized world positions in ``[0, 1]^3``. A level is dense when
    all ``(resolution + 1)^3`` vertices fit in the configured table and hashed
    otherwise.
    """

    needs_occupancy = False
    needs_normals = False
    guarantees_zero_collisions = False

    def __init__(
        self,
        levels: int = 8,
        features_per_level: int = 2,
        table_size_log2: int = 14,
        base_resolution: int = 4,
        finest_resolution: int = 256,
        allocation: str = "uniform",
        occupancy=None,
        slot_budget: int | None = None,
    ):
        super().__init__()
        if levels <= 0:
            raise ValueError("levels must be positive")
        if features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if table_size_log2 < 0:
            raise ValueError("table_size_log2 must be non-negative")
        if base_resolution <= 0 or finest_resolution <= 0:
            raise ValueError("base_resolution and finest_resolution must be positive")
        if finest_resolution < base_resolution:
            raise ValueError("finest_resolution must be >= base_resolution")
        if allocation not in {"uniform", "occupancy"}:
            raise ValueError("allocation must be 'uniform' or 'occupancy'")
        if allocation == "occupancy" and occupancy is None:
            raise ValueError("allocation='occupancy' requires occupancy")
        self.allocation = allocation
        self.levels = levels
        self.features_per_level = features_per_level
        self.table_size = 1 << table_size_log2
        self.resolutions = level_resolutions(levels, base_resolution, finest_resolution)
        self.tables = nn.ParameterList()
        self._dense = []
        if allocation == "uniform":
            sizes = []
            for res in self.resolutions:
                n_vertices = (res + 1) ** 3
                dense = n_vertices <= self.table_size
                self._dense.append(dense)
                sizes.append(n_vertices if dense else self.table_size)
        else:
            # Validate against the FULL schedule before any truncation -- occupancy
            # built with the wrong base/finest resolution describes a different grid
            # than this encoder queries, and truncation must not hide that mismatch.
            if len(occupancy) != levels:
                raise ValueError(f"occupancy has {len(occupancy)} levels, expected {levels}")
            expected_resolutions = level_resolutions(levels, base_resolution, finest_resolution)
            actual_resolutions = [occ.resolution for occ in occupancy]
            if actual_resolutions != expected_resolutions:
                raise ValueError(
                    f"occupancy resolutions {actual_resolutions} do not match the schedule "
                    f"implied by base_resolution={base_resolution}, "
                    f"finest_resolution={finest_resolution}, levels={levels} "
                    f"(expected {expected_resolutions}); occupancy was built with a "
                    "different resolution schedule than this encoder was configured for"
                )
            # Size each level from the vertices the cache actually reads, coarsest
            # first, and drop levels the budget cannot serve rather than crushing them
            # to a few percent of their occupancy -- the failure R1 measured.
            budget = int(slot_budget) if slot_budget is not None else self.table_size * levels
            sizes = []
            remaining = budget
            for occ in occupancy:
                want = int(occ.count)
                if want > remaining:
                    break
                sizes.append(want)
                self._dense.append(want == (occ.resolution + 1) ** 3)
                remaining -= want
            if not sizes:
                raise ValueError("slot_budget is too small to serve even the coarsest level")
            self.levels = len(sizes)
            self.resolutions = self.resolutions[: self.levels]
        for size in sizes:
            self.tables.append(
                nn.Parameter(torch.empty(size, features_per_level).uniform_(-1e-4, 1e-4))
            )

    @property
    def output_dim(self) -> int:
        return self.levels * self.features_per_level

    def capacity_report(self) -> dict:
        return _grid_capacity_report(self)

    def _index(
        self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, level: int
    ) -> torch.Tensor:
        res = self.resolutions[level]
        if self._dense[level]:
            side = res + 1
            return (iz * side + iy) * side + ix
        hashed = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) ^ (iz * _PRIMES[2])
        if self.allocation == "occupancy":
            return hashed % self.tables[level].shape[0]
        return hashed & (self.table_size - 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz in [0,1]^3, shape (N, 3) -> (N, levels * features_per_level)."""
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        outputs = []
        for level, res in enumerate(self.resolutions):
            pos = xyz * res
            pos0, frac = _floor_cell(pos, res)
            x0, y0, z0 = pos0[:, 0], pos0[:, 1], pos0[:, 2]
            x1 = (x0 + 1).clamp(max=res)
            y1 = (y0 + 1).clamp(max=res)
            z1 = (z0 + 1).clamp(max=res)
            table = self.tables[level]
            wx, wy, wz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
            out = torch.zeros(
                (xyz.shape[0], self.features_per_level), dtype=table.dtype, device=table.device
            )
            for ix, xw in ((x0, 1 - wx), (x1, wx)):
                for iy, yw in ((y0, 1 - wy), (y1, wy)):
                    for iz, zw in ((z0, 1 - wz), (z1, wz)):
                        index = self._index(ix, iy, iz, level) % table.shape[0]
                        out = out + table[index] * xw * yw * zw
            outputs.append(out)
        return torch.cat(outputs, dim=1)


@register_encoder("world_triplane")
class HashEncodingTriPlane(nn.Module):
    """World-anchored tri-plane encoding built from three 2D hashgrids.

    The XY, XZ, and YZ projections share architecture but not parameters. This keeps
    world anchoring while testing whether a surface-dominated scene benefits from the
    lower collision pressure of 2D tables.
    """

    needs_occupancy = False
    needs_normals = False
    guarantees_zero_collisions = False

    def __init__(
        self,
        levels: int = 3,
        features_per_level: int = 2,
        table_size_log2: int = 13,
        base_resolution: int = 4,
        finest_resolution: int = 256,
    ):
        super().__init__()
        config = {
            "levels": levels,
            "features_per_level": features_per_level,
            "table_size_log2": table_size_log2,
            "base_resolution": base_resolution,
            "finest_resolution": finest_resolution,
        }
        self.planes = nn.ModuleList([HashEncoding2D(**config) for _ in range(3)])
        self.levels = levels
        self.features_per_level = features_per_level
        self.table_size = 1 << table_size_log2
        self.resolutions = self.planes[0].resolutions

    @property
    def output_dim(self) -> int:
        return 3 * self.planes[0].output_dim

    def capacity_report(self) -> dict:
        per_plane = [plane.capacity_report() for plane in self.planes]
        return {
            "encoding": type(self).__name__,
            "levels": per_plane[0]["levels"],
            "planes": per_plane,
            "total_slots": int(sum(p["total_slots"] for p in per_plane)),
        }

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        projections = (xyz[:, (0, 1)], xyz[:, (0, 2)], xyz[:, (1, 2)])
        return torch.cat(
            [plane(projected) for plane, projected in zip(self.planes, projections, strict=True)],
            dim=1,
        )


@register_encoder("world_normal_triplane")
class NormalAwareTriPlane(nn.Module):
    """Tri-plane where each point reads only the plane aligned with its normal.

    The concatenating tri-plane reads all three planes, tripling capacity pressure and
    tying quality to the world frame -- the corrected R1C matrix measured a -3.651 dB
    worst-orientation failure. Selecting the plane by surface normal makes the choice
    follow geometry instead of the frame, and reads one plane instead of three.

    The argmax is discontinuous across normal boundaries; gradients flow through the
    features, not the selection. Expected artifacts are seams at sharp surface-orientation
    discontinuities, which the per-camera error maps will show if they matter.
    """

    needs_occupancy = False
    needs_normals = True
    guarantees_zero_collisions = False

    #: dominant normal axis -> the two coordinate axes spanning the plane it reads
    AXIS_TO_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def __init__(
        self,
        levels: int = 3,
        features_per_level: int = 2,
        table_size_log2: int = 13,
        base_resolution: int = 4,
        finest_resolution: int = 256,
    ):
        super().__init__()
        config = {
            "levels": levels,
            "features_per_level": features_per_level,
            "table_size_log2": table_size_log2,
            "base_resolution": base_resolution,
            "finest_resolution": finest_resolution,
        }
        self.planes = nn.ModuleList([HashEncoding2D(**config) for _ in range(3)])
        self.levels = levels
        self.features_per_level = features_per_level
        self.resolutions = self.planes[0].resolutions

    @property
    def output_dim(self) -> int:
        return self.planes[0].output_dim

    def forward(self, xyz: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        if normals is None:
            raise ValueError("NormalAwareTriPlane requires normals")
        if normals.shape != xyz.shape:
            raise ValueError(f"normals must match xyz shape, got {tuple(normals.shape)}")
        magnitude = normals.abs()
        if bool((magnitude.sum(dim=1) <= 1e-8).any()):
            raise ValueError("normals must be non-zero")
        axis = magnitude.argmax(dim=1)
        out = torch.zeros((xyz.shape[0], self.output_dim), dtype=xyz.dtype, device=xyz.device)
        for a in range(3):
            mask = axis == a
            if not bool(mask.any()):
                continue
            u, v = self.AXIS_TO_PLANE[a]
            out[mask] = self.planes[a](xyz[mask][:, (u, v)])
        return out

    def capacity_report(self) -> dict:
        per_plane = [plane.capacity_report() for plane in self.planes]
        return {
            "encoding": type(self).__name__,
            "levels": per_plane[0]["levels"],
            "planes": per_plane,
            "total_slots": int(sum(p["total_slots"] for p in per_plane)),
        }
