"""Single-view occupancy building for `train()` (needed by world_sparse/world3d-occupancy).

`build_single_cache_occupancy` is the single-cache counterpart of
`conditioned_multiview.train_conditioned`'s occupancy build: decides, from
`encoder_wants_occupancy`, whether a spatial encoding needs an occupancy grid built
from a cache's first-hit positions before the model can be constructed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.train import build_single_cache_occupancy  # noqa: E402


class _FakeCache:
    """Duck-typed stand-in exposing only what build_single_cache_occupancy reads."""

    def __init__(self, position: np.ndarray):
        self.position = position


UNIT_BOUNDS = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}


class TestBuildSingleCacheOccupancy(unittest.TestCase):
    def test_pixel2d_never_builds_occupancy(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        result = build_single_cache_occupancy(cache, "pixel2d", {}, UNIT_BOUNDS)
        self.assertIsNone(result)

    def test_pixel2d_ignores_missing_bounds(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        result = build_single_cache_occupancy(cache, "pixel2d", {}, None)
        self.assertIsNone(result)

    def test_world3d_uniform_allocation_builds_nothing(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        result = build_single_cache_occupancy(
            cache, "world3d", {"allocation": "uniform"}, UNIT_BOUNDS
        )
        self.assertIsNone(result)

    def test_world3d_default_config_builds_nothing(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        result = build_single_cache_occupancy(cache, "world3d", {}, UNIT_BOUNDS)
        self.assertIsNone(result)

    def test_world3d_occupancy_allocation_without_bounds_raises(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        with self.assertRaises(ValueError):
            build_single_cache_occupancy(cache, "world3d", {"allocation": "occupancy"}, None)

    def test_world3d_occupancy_allocation_single_cell_returns_cube_corners(self):
        # Two points that both floor into cell (0,0,0) at resolution 1 share exactly
        # the unit cube's 8 corners -- hand-computed, not re-derived from the
        # production formula.
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1], [0.4, 0.2, 0.3]]))
        result = build_single_cache_occupancy(
            cache,
            "world3d",
            {"allocation": "occupancy", "levels": 1, "base_resolution": 1, "finest_resolution": 1},
            UNIT_BOUNDS,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].resolution, 1)
        self.assertEqual(result[0].count, 8)
        expected = {(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)}
        actual = {tuple(row) for row in result[0].vertices.tolist()}
        self.assertEqual(actual, expected)

    def test_world3d_occupancy_allocation_two_cells_union_overlaps_by_one_vertex(self):
        # Resolution 2: point A floors to cell (0,0,0), point B floors to cell
        # (1,1,1) (clamped). Their corner sets are {0,1}^3 and {1,2}^3, overlapping
        # only at (1,1,1), so the union has 8 + 8 - 1 = 15 vertices -- hand-counted.
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]))
        result = build_single_cache_occupancy(
            cache,
            "world3d",
            {"allocation": "occupancy", "levels": 1, "base_resolution": 2, "finest_resolution": 2},
            UNIT_BOUNDS,
        )
        self.assertEqual(result[0].count, 15)

    def test_world_sparse_always_builds_occupancy_even_without_allocation_key(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        result = build_single_cache_occupancy(
            cache,
            "world_sparse",
            {"levels": 1, "base_resolution": 1, "finest_resolution": 1},
            UNIT_BOUNDS,
        )
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].count, 8)

    def test_unknown_spatial_encoding_raises(self):
        cache = _FakeCache(np.array([[0.1, 0.1, 0.1]]))
        with self.assertRaises(ValueError):
            build_single_cache_occupancy(cache, "does_not_exist", {}, UNIT_BOUNDS)


if __name__ == "__main__":
    unittest.main()
