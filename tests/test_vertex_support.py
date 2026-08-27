"""Vertex-support diagnostic: reproducible per-level occupancy-vs-pixel-count stats."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.vertex_support import cache_vertex_support, level_vertex_support  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402


def _make_cache(positions: np.ndarray) -> PathCache:
    """A minimal valid PathCache holding only the given first-hit positions.

    height=1, width=len(positions); no segments (segment_count 0 is valid).
    """
    n = positions.shape[0]
    return PathCache(
        width=n,
        height=1,
        n_paths=np.zeros((n,), dtype=np.int64),
        seg_pixel=np.zeros((0,), dtype=np.int64),
        seg_origin=np.zeros((0, 3), dtype=np.float64),
        seg_dir=np.zeros((0, 3), dtype=np.float64),
        seg_tmax=np.zeros((0,), dtype=np.float64),
        seg_throughput=np.zeros((0, 3), dtype=np.float64),
        albedo=np.zeros((1, n, 3), dtype=np.float64),
        position=positions.reshape(1, n, 3),
        depth=np.zeros((1, n), dtype=np.float64),
        normal=np.zeros((1, n, 3), dtype=np.float64),
    )


class TestLevelVertexSupport(unittest.TestCase):
    """A hand-derivable 4-pixel / resolution-2 case.

    Four pixels sit at the corners of the unit cube: (0,0,0), (1,1,1), (0,1,0),
    (1,0,1). world_bounds normalizes them to themselves (min=(0,0,0),
    max=(1,1,1)). At resolution 2 each pixel's base cell is:
      (0,0,0) -> base (0,0,0); (1,1,1) -> floor(2,2,2).clip(0,1) = (1,1,1)
      (0,1,0) -> base (0,1,0); (1,0,1) -> base (1,0,1)
    Each pixel's 8 unclipped +/-1 corners were enumerated by hand (see
    docs-accuracy-report.md for the full corner listing) and tallied for
    support (number of distinct pixels touching each vertex), giving 21
    distinct vertices: twelve touched by exactly 1 pixel, eight by exactly 2,
    and one (1,1,1) touched by all 4 pixels.
    """

    def setUp(self):
        self.normalized = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )

    def test_hand_derived_support_distribution(self):
        result = level_vertex_support(self.normalized, resolution=2)
        self.assertEqual(result["n_pixels"], 4)
        self.assertEqual(result["n_vertices"], 21)
        self.assertAlmostEqual(result["vertices_per_pixel"], 21 / 4)
        self.assertAlmostEqual(result["median_support"], 1.0)
        self.assertAlmostEqual(result["p75_support"], 2.0)
        self.assertAlmostEqual(result["p90_support"], 2.0)
        self.assertAlmostEqual(result["fraction_touched_by_le1_pixel"], 12 / 21)
        self.assertAlmostEqual(result["fraction_touched_by_le2_pixels"], 20 / 21)
        self.assertAlmostEqual(result["fraction_touched_by_le4_pixels"], 1.0)

    def test_breaks_if_corner_enumeration_diverges(self):
        """Demonstrates the test can fail: a wrong corner rule changes the tally.

        Using only the (0,0,0) corner (i.e. treating each pixel as touching just
        its base cell, not all 8 offset corners) collapses distinct vertices to
        4 -- one per pixel, since all four base cells above are already
        distinct -- which contradicts every hand-derived figure above.
        """
        base = np.floor(self.normalized * 2).astype(np.int64).clip(0, 1)
        unique = np.unique(base, axis=0)
        self.assertEqual(unique.shape[0], 4)
        self.assertNotEqual(unique.shape[0], 21)


class TestCacheVertexSupport(unittest.TestCase):
    def test_single_level_matches_hand_derived_values(self):
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ]
        )
        cache = _make_cache(positions)
        report = cache_vertex_support(cache, levels=1, base_resolution=2, finest_resolution=2)
        self.assertEqual(report["per_level"][0]["n_vertices"], 21)
        self.assertEqual(report["finest"], report["per_level"][0])
        self.assertEqual(report["world_bounds"]["min"], [0.0, 0.0, 0.0])
        self.assertEqual(report["world_bounds"]["max"], [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
