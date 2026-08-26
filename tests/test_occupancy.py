"""Exact per-level queried-vertex sets for world-anchored encodings."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.occupancy import (  # noqa: E402
    capacity_report,
    grid_occupancy,
    level_resolutions,
    normalize_positions,
)


class TestLevelResolutions(unittest.TestCase):
    def test_matches_hashgrid_geometric_growth(self):
        # Must reproduce HashEncoding2D/3D's schedule exactly, or occupancy would
        # describe a different grid than the encoder queries.
        self.assertEqual(level_resolutions(8, 4, 128), [4, 6, 10, 17, 28, 47, 78, 128])

    def test_single_level_is_base_resolution(self):
        self.assertEqual(level_resolutions(1, 7, 64), [7])


class TestNormalizePositions(unittest.TestCase):
    def test_maps_bounds_to_unit_cube(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 8.0]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [2.0, 4.0, 8.0]}
        np.testing.assert_allclose(
            normalize_positions(positions, bounds),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        )

    def test_clamps_out_of_bounds_positions(self):
        positions = np.array([[-1.0, 5.0, 0.5]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        out = normalize_positions(positions, bounds)
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))


class TestGridOccupancy(unittest.TestCase):
    def test_single_point_touches_eight_corners(self):
        normalized = np.array([[0.5, 0.5, 0.5]])
        occ = grid_occupancy(normalized, [4])
        self.assertEqual(occ[0].count, 8)
        self.assertEqual(occ[0].resolution, 4)

    def test_point_on_a_vertex_still_touches_eight_corners(self):
        # The clamp-to-res-1 fix means an exact vertex hit still spans a full cell.
        occ = grid_occupancy(np.array([[0.0, 0.0, 0.0]]), [4])
        self.assertEqual(occ[0].count, 8)

    def test_vertices_are_unique_and_sorted(self):
        rng = np.random.default_rng(0)
        normalized = rng.random((200, 3))
        occ = grid_occupancy(normalized, [8])
        v = occ[0].vertices
        self.assertEqual(len(np.unique(v, axis=0)), len(v))
        order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
        np.testing.assert_array_equal(v, v[order])

    def test_vertices_stay_within_resolution(self):
        rng = np.random.default_rng(1)
        occ = grid_occupancy(rng.random((200, 3)), [8])
        self.assertGreaterEqual(int(occ[0].vertices.min()), 0)
        self.assertLessEqual(int(occ[0].vertices.max()), 8)

    def test_coarse_level_has_no_more_vertices_than_fine_level(self):
        rng = np.random.default_rng(2)
        occ = grid_occupancy(rng.random((500, 3)), [4, 32])
        self.assertLessEqual(occ[0].count, occ[1].count)


class TestCapacityReport(unittest.TestCase):
    def test_perfect_index_reports_zero_collisions(self):
        occ = grid_occupancy(np.array([[0.5, 0.5, 0.5]]), [4])

        def perfect(vertices, level):
            return np.arange(len(vertices))

        report = capacity_report(occ, perfect, slots=[8])
        level = report["levels"][0]
        self.assertEqual(level["distinct_vertices"], 8)
        self.assertEqual(level["collision_fraction"], 0.0)
        self.assertEqual(level["max_slot_load"], 1)

    def test_constant_index_reports_total_collapse(self):
        occ = grid_occupancy(np.array([[0.5, 0.5, 0.5]]), [4])

        def constant(vertices, level):
            return np.zeros(len(vertices), dtype=np.int64)

        report = capacity_report(occ, constant, slots=[8])
        level = report["levels"][0]
        self.assertEqual(level["used_slots"], 1)
        self.assertAlmostEqual(level["collision_fraction"], 7 / 8)
        self.assertEqual(level["max_slot_load"], 8)

    def test_reports_slots_per_distinct_vertex(self):
        occ = grid_occupancy(np.array([[0.5, 0.5, 0.5]]), [4])

        def perfect(vertices, level):
            return np.arange(len(vertices))

        report = capacity_report(occ, perfect, slots=[16])
        self.assertAlmostEqual(report["levels"][0]["slots_per_distinct_vertex"], 2.0)


class TestKitchenDiagnosticIsReproducible(unittest.TestCase):
    """Pins the measurement the redesign is founded on (spec: Diagnosis)."""

    def test_kitchen_world3d_finest_level_is_capacity_starved(self):
        cache_path = Path(__file__).resolve().parent.parent / "out" / "kitchen" / "path_cache.npz"
        if not cache_path.exists():
            self.skipTest("kitchen cache not present; run the Mitsuba export first")
        from nrp.path_cache import PathCache
        from nrp.torch_backend.occupancy import cache_occupancy

        cache = PathCache.load(str(cache_path))
        position = np.asarray(cache.position, dtype=np.float64).reshape(-1, 3)
        bounds = {"min": position.min(axis=0).tolist(), "max": position.max(axis=0).tolist()}
        occ = cache_occupancy(cache, bounds, levels=8, base_resolution=7, finest_resolution=128)
        # Measured 78,080 against a 4096-slot table: a ~19x handicap versus pixel2d.
        self.assertGreater(occ[-1].count, 70_000)
        self.assertLess(4096 / occ[-1].count, 0.1)


if __name__ == "__main__":
    unittest.main()
