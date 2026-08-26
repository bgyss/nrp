"""Arm B of the encoding redesign: an exact sparse occupancy index.

The path cache is static and fully known before training, so the set of grid
vertices any level will ever read can be enumerated up front. Storing exactly those
vertices removes hashing, and therefore collisions, by construction rather than by
tuning -- which is what R1A's +-1.74 dB seed spread was mostly measuring.

A vertex outside the occupied set (only a novel camera produces these) contributes
zero features at that level and is carried by the coarser levels. The fallback is
differentiable and its per-camera frequency is reported, never hidden.
"""

from __future__ import annotations

import dataclasses
import itertools

import torch
from torch import nn

from .encoder_registry import _floor_cell, register_encoder


def _vertex_codes(vertices: torch.Tensor, resolution: int) -> torch.Tensor:
    """Fold integer vertex coordinates into one int64 key. Unique for res <= ~1e6."""
    side = resolution + 1
    return (vertices[:, 2] * side + vertices[:, 1]) * side + vertices[:, 0]


@register_encoder("world_sparse")
class SparseVoxelEncoding(nn.Module):
    needs_occupancy = True
    needs_normals = False

    def __init__(
        self,
        occupancy,
        levels: int = 8,
        features_per_level: int = 2,
        base_resolution: int = 4,
        finest_resolution: int = 128,
        **_ignored,
    ):
        super().__init__()
        if len(occupancy) != levels:
            raise ValueError(f"occupancy has {len(occupancy)} levels, expected {levels}")
        if features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        self.levels = levels
        self.features_per_level = features_per_level
        self.occupancy = list(occupancy)
        self.resolutions = [occ.resolution for occ in self.occupancy]
        self.tables = nn.ParameterList()
        for level, occ in enumerate(self.occupancy):
            if occ.count == 0:
                raise ValueError(f"level {level} has empty occupancy")
            vertices = torch.as_tensor(occ.vertices, dtype=torch.long)
            keys = _vertex_codes(vertices, occ.resolution)
            order = torch.argsort(keys)
            self.register_buffer(f"keys_{level}", keys[order].contiguous())
            self.tables.append(
                nn.Parameter(torch.empty(occ.count, features_per_level).uniform_(-1e-4, 1e-4))
            )
            # Table rows follow sorted key order so searchsorted indexes them directly.
            self.occupancy[level] = dataclasses.replace(occ, vertices=occ.vertices[order.numpy()])

    @property
    def output_dim(self) -> int:
        return self.levels * self.features_per_level

    def _lookup(self, vertices: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (row indices, hit mask) for integer vertex coordinates."""
        keys = getattr(self, f"keys_{level}")
        codes = _vertex_codes(vertices, self.resolutions[level])
        idx = torch.searchsorted(keys, codes).clamp(max=keys.numel() - 1)
        hit = keys[idx] == codes
        return idx, hit

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        outputs = []
        for level, res in enumerate(self.resolutions):
            pos = xyz * res
            pos0, frac = _floor_cell(pos, res)
            table = self.tables[level]
            out = torch.zeros(
                (xyz.shape[0], self.features_per_level), dtype=table.dtype, device=table.device
            )
            for offset in itertools.product((0, 1), repeat=3):
                corner = pos0 + torch.tensor(offset, device=pos0.device, dtype=pos0.dtype)
                weight = torch.ones((xyz.shape[0], 1), dtype=table.dtype, device=table.device)
                for axis, o in enumerate(offset):
                    w = frac[:, axis : axis + 1]
                    weight = weight * (w if o else (1.0 - w))
                idx, hit = self._lookup(corner, level)
                out = out + table[idx] * weight * hit.unsqueeze(1).to(table.dtype)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def out_of_occupancy_fraction(self, xyz: torch.Tensor) -> float:
        """Fraction of query points whose finest-level base cell is unoccupied.

        Reported per held-out camera so a good gate result cannot hide behind a
        favourable fallback (spec: G5).
        """
        level = self.levels - 1
        res = self.resolutions[level]
        pos0, _ = _floor_cell(xyz * res, res)
        _, hit = self._lookup(pos0, level)
        return float((~hit).to(torch.float64).mean())

    def capacity_report(self) -> dict:
        levels = []
        for level, occ in enumerate(self.occupancy):
            levels.append(
                {
                    "level": level,
                    "resolution": occ.resolution,
                    "dense": False,
                    "sparse": True,
                    "distinct_vertices": occ.count,
                    "slots": int(self.tables[level].shape[0]),
                    "used_slots": occ.count,
                    "collision_fraction": 0.0,
                    "max_slot_load": 1,
                    "slots_per_distinct_vertex": 1.0,
                    "key_bytes": int(getattr(self, f"keys_{level}").numel() * 8),
                }
            )
        return {
            "encoding": type(self).__name__,
            "levels": levels,
            "total_slots": int(sum(level["slots"] for level in levels)),
            "total_key_bytes": int(sum(level["key_bytes"] for level in levels)),
        }
