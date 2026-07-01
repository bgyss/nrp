"""2D multiresolution hash encoding (Müller et al. [MESK22], paper §4.3).

Plain-PyTorch implementation of the instant-ngp grid encoding, specialized to two
dimensions (the paper encodes pixel coordinates px only). Each level l has a virtual
grid of resolution N_l = floor(N_min * b^l); a vertex's feature vector lives either in
a dense table (when the level's grid fits) or at a hashed slot (spatial hash with the
instant-ngp primes, XOR combined). Encoded output is the bilinear interpolation of the
four corner features, concatenated across levels: dim = levels * features_per_level.
"""

from __future__ import annotations

import math

import torch
from torch import nn

_PRIMES = (1, 2654435761)


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
        return (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) & (self.table_size - 1)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        """xy in [0,1]^2, shape (N, 2) -> (N, levels * features_per_level)."""
        outputs = []
        for level, res in enumerate(self.resolutions):
            pos = xy * res
            pos0 = torch.floor(pos).long().clamp_(0, res)
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
