"""Vertex-support diagnostic: how many pixels touch each finest-level grid vertex.

Reproduces, per level of a world-anchored hashgrid ladder, how many of a cache's
first-hit pixels touch each queried grid vertex -- the quantity behind the
"toy 64^2 vs Kitchen 128^2" vertex-support hypothesis in
`docs/representation-track.md`'s R1 correction. That hypothesis previously cited
numbers computed ad hoc in a shell session with nothing committed to reproduce
them; this script makes the measurement regenerable from a cache path and a
level/base/finest ladder.

Reuses `nrp.torch_backend.occupancy.normalize_positions` and `level_resolutions`,
and the same floor-and-offset corner enumeration `occupancy.grid_occupancy` uses,
so this describes exactly the grid the encoders and `occupancy.py` already query
-- not a reimplementation that could quietly diverge from it.

Usage:
    uv run python examples/vertex_support.py --cache out/kitchen/path_cache.npz \\
        --levels 8 --base-resolution 4 --finest-resolution 128 \\
        --out out/vertex-support/kitchen128.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.occupancy import level_resolutions, normalize_positions  # noqa: E402
from nrp.torch_backend.train import world_bounds  # noqa: E402


def level_vertex_support(normalized: np.ndarray, resolution: int) -> dict:
    """Vertex-support distribution for one level's corner enumeration.

    Mirrors `occupancy.grid_occupancy`'s per-pixel corner rule exactly: floor to
    the base cell (clipped to the grid), then the 2**ndim +/-1 offset corners,
    unclipped. Support of a vertex is the number of distinct pixels whose corner
    set includes it -- since a pixel's 8 corner offsets are pairwise distinct,
    each pixel contributes at most one touch per vertex, so counting raw
    occurrences of a vertex across all (pixel, corner) pairs already counts
    distinct pixels.
    """
    ndim = normalized.shape[1]
    n_pixels = normalized.shape[0]
    base = np.floor(normalized * resolution).astype(np.int64).clip(0, resolution - 1)
    corners = [
        base + np.asarray(offset, dtype=np.int64)
        for offset in itertools.product((0, 1), repeat=ndim)
    ]
    stacked = np.concatenate(corners, axis=0)
    vertices, counts = np.unique(stacked, axis=0, return_counts=True)
    n_vertices = int(vertices.shape[0])

    support = counts.astype(np.int64)
    if support.size:
        median = float(np.percentile(support, 50))
        p75 = float(np.percentile(support, 75))
        p90 = float(np.percentile(support, 90))
        frac_le1 = float(np.mean(support <= 1))
        frac_le2 = float(np.mean(support <= 2))
        frac_le4 = float(np.mean(support <= 4))
    else:
        median = p75 = p90 = frac_le1 = frac_le2 = frac_le4 = 0.0

    return {
        "resolution": int(resolution),
        "n_pixels": int(n_pixels),
        "n_vertices": n_vertices,
        "vertices_per_pixel": float(n_vertices / n_pixels) if n_pixels else 0.0,
        "median_support": median,
        "p75_support": p75,
        "p90_support": p90,
        "fraction_touched_by_le1_pixel": frac_le1,
        "fraction_touched_by_le2_pixels": frac_le2,
        "fraction_touched_by_le4_pixels": frac_le4,
    }


def cache_vertex_support(
    cache: PathCache, levels: int, base_resolution: int, finest_resolution: int
) -> dict:
    """Per-level vertex-support distributions for one cache under a level schedule."""
    bounds = world_bounds(cache)
    normalized = normalize_positions(cache.position, bounds)
    resolutions = level_resolutions(levels, base_resolution, finest_resolution)
    per_level = [
        {"level": level, **level_vertex_support(normalized, res)}
        for level, res in enumerate(resolutions)
    ]
    return {
        "levels": levels,
        "base_resolution": base_resolution,
        "finest_resolution": finest_resolution,
        "world_bounds": bounds,
        "per_level": per_level,
        "finest": per_level[-1] if per_level else {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="Path to a PathCache .npz file")
    parser.add_argument("--levels", type=int, required=True)
    parser.add_argument("--base-resolution", type=int, required=True)
    parser.add_argument("--finest-resolution", type=int, required=True)
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()

    cache = PathCache.load(args.cache)
    report = cache_vertex_support(cache, args.levels, args.base_resolution, args.finest_resolution)
    report["cache"] = str(args.cache)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
