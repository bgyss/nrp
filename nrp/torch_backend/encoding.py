"""Multiresolution hash encodings (Müller et al. [MESK22], paper §4.3).

The paper path is the original 2D pixel-coordinate encoding. Representation-track
rung R1 adds a selectable 3D world-position encoding with the same dense/hashed table
policy and geometric resolution growth, using trilinear interpolation over eight
corners. Both are plain-PyTorch instant-ngp-style implementations.
"""

from __future__ import annotations

import math

import torch
from torch import nn

_PRIMES = (1, 2654435761, 805459861)


class HashEncoding2D(nn.Module):
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
        growth = (
            math.exp(math.log(finest_resolution / base_resolution) / max(levels - 1, 1))
            if levels > 1
            else 1.0
        )
        self.resolutions = [
            max(int(math.floor(base_resolution * growth**level)), 1) for level in range(levels)
        ]
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
            # Clamp to res-1 so the cell [x0, x0+1] is always non-degenerate; x0 == res
            # would make all corners identical and the level output constant.
            pos0 = torch.floor(pos).long().clamp_(0, res - 1)
            frac = pos - pos0
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


class HashEncoding3D(nn.Module):
    """3D multiresolution hashgrid with trilinear interpolation.

    Inputs are normalized world positions in ``[0, 1]^3``. A level is dense when
    all ``(resolution + 1)^3`` vertices fit in the configured table and hashed
    otherwise.
    """

    def __init__(
        self,
        levels: int = 8,
        features_per_level: int = 2,
        table_size_log2: int = 14,
        base_resolution: int = 4,
        finest_resolution: int = 256,
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
        self.levels = levels
        self.features_per_level = features_per_level
        self.table_size = 1 << table_size_log2
        growth = (
            math.exp(math.log(finest_resolution / base_resolution) / max(levels - 1, 1))
            if levels > 1
            else 1.0
        )
        self.resolutions = [
            max(int(math.floor(base_resolution * growth**level)), 1) for level in range(levels)
        ]
        self.tables = nn.ParameterList()
        self._dense = []
        for res in self.resolutions:
            n_vertices = (res + 1) ** 3
            dense = n_vertices <= self.table_size
            self._dense.append(dense)
            size = n_vertices if dense else self.table_size
            self.tables.append(
                nn.Parameter(torch.empty(size, features_per_level).uniform_(-1e-4, 1e-4))
            )

    @property
    def output_dim(self) -> int:
        return self.levels * self.features_per_level

    def _index(
        self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor, level: int
    ) -> torch.Tensor:
        res = self.resolutions[level]
        if self._dense[level]:
            side = res + 1
            return (iz * side + iy) * side + ix
        hashed = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) ^ (iz * _PRIMES[2])
        return hashed & (self.table_size - 1)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz in [0,1]^3, shape (N, 3) -> (N, levels * features_per_level)."""
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        outputs = []
        for level, res in enumerate(self.resolutions):
            pos = xyz * res
            # Clamp to res-1 so the cell [x0, x0+1] is always non-degenerate; x0 == res
            # would make all corners identical and the level output constant.
            pos0 = torch.floor(pos).long().clamp_(0, res - 1)
            frac = pos - pos0
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


class HashEncodingTriPlane(nn.Module):
    """World-anchored tri-plane encoding built from three 2D hashgrids.

    The XY, XZ, and YZ projections share architecture but not parameters. This keeps
    world anchoring while testing whether a surface-dominated scene benefits from the
    lower collision pressure of 2D tables.
    """

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

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        projections = (xyz[:, (0, 1)], xyz[:, (0, 2)], xyz[:, (1, 2)])
        return torch.cat(
            [plane(projected) for plane, projected in zip(self.planes, projections, strict=True)],
            dim=1,
        )
