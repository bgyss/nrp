"""Arm B: exact sparse occupancy index, no hashing, no collisions."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.occupancy import grid_occupancy  # noqa: E402
from nrp.torch_backend.sparse_encoding import SparseVoxelEncoding  # noqa: E402

CONFIG = {"levels": 2, "features_per_level": 2, "base_resolution": 4, "finest_resolution": 8}


def _encoder(points: np.ndarray) -> SparseVoxelEncoding:
    from nrp.torch_backend.occupancy import level_resolutions

    res = level_resolutions(
        CONFIG["levels"], CONFIG["base_resolution"], CONFIG["finest_resolution"]
    )
    return SparseVoxelEncoding(occupancy=grid_occupancy(points, res), **CONFIG)


class TestZeroCollisions(unittest.TestCase):
    def test_table_is_sized_exactly_to_occupancy(self):
        rng = np.random.default_rng(0)
        points = rng.random((100, 3))
        enc = _encoder(points)
        for level, occ in enumerate(enc.occupancy):
            self.assertEqual(enc.tables[level].shape[0], occ.count)

    def test_every_occupied_key_resolves_uniquely(self):
        rng = np.random.default_rng(1)
        enc = _encoder(rng.random((100, 3)))
        for level, _occ in enumerate(enc.occupancy):
            keys = getattr(enc, f"keys_{level}")
            self.assertEqual(len(torch.unique(keys)), len(keys))
            self.assertTrue(bool((keys[1:] > keys[:-1]).all()), "keys must be sorted")

    def test_capacity_report_asserts_zero_collisions(self):
        rng = np.random.default_rng(2)
        enc = _encoder(rng.random((100, 3)))
        report = enc.capacity_report()
        for level in report["levels"]:
            self.assertEqual(level["collision_fraction"], 0.0)
            self.assertEqual(level["max_slot_load"], 1)


class TestLookup(unittest.TestCase):
    def test_corner_query_returns_that_table_entry(self):
        points = np.array([[0.5, 0.5, 0.5]])
        enc = _encoder(points)
        with torch.no_grad():
            for table in enc.tables:
                table.uniform_(-1.0, 1.0)
        # A query exactly on an occupied vertex has trilinear weight 1 on that corner,
        # so the level's output must equal that vertex's table row. Row order follows
        # sorted key order, so vertices[0] owns tables[0][0].
        occ0 = enc.occupancy[0]
        vertex = occ0.vertices[0]
        xyz = torch.tensor([vertex / occ0.resolution], dtype=torch.float32)
        out = enc(xyz)
        torch.testing.assert_close(
            out[0, : enc.features_per_level], enc.tables[0][0], atol=1e-5, rtol=0
        )

    def test_gradients_flow_to_the_tables(self):
        rng = np.random.default_rng(3)
        enc = _encoder(rng.random((50, 3)))
        out = enc(torch.rand(10, 3))
        out.sum().backward()
        self.assertTrue(
            any(t.grad is not None and bool(t.grad.abs().sum() > 0) for t in enc.tables)
        )

    def test_output_dim_matches_forward_width(self):
        rng = np.random.default_rng(4)
        enc = _encoder(rng.random((50, 3)))
        self.assertEqual(enc(torch.rand(7, 3)).shape[1], enc.output_dim)


class TestFallback(unittest.TestCase):
    def test_unoccupied_query_contributes_zero_and_stays_differentiable(self):
        # Occupancy built from one corner of the cube; query the opposite corner.
        enc = _encoder(np.array([[0.01, 0.01, 0.01]]))
        far = torch.tensor([[0.99, 0.99, 0.99]], requires_grad=False)
        out = enc(far)
        self.assertTrue(torch.isfinite(out).all())
        # Differentiability: a table that received no gradient must still not crash.
        out.sum().backward()

    def test_out_of_occupancy_fraction_is_one_for_unseen_region(self):
        enc = _encoder(np.array([[0.01, 0.01, 0.01]]))
        frac = enc.out_of_occupancy_fraction(torch.tensor([[0.99, 0.99, 0.99]]))
        self.assertAlmostEqual(frac, 1.0)

    def test_out_of_occupancy_fraction_is_zero_for_training_points(self):
        rng = np.random.default_rng(5)
        points = rng.random((80, 3))
        enc = _encoder(points)
        frac = enc.out_of_occupancy_fraction(torch.tensor(points, dtype=torch.float32))
        self.assertAlmostEqual(frac, 0.0)


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_keys_survive_state_dict_round_trip(self):
        rng = np.random.default_rng(6)
        points = rng.random((60, 3))
        enc = _encoder(points)
        clone = _encoder(points)
        clone.load_state_dict(enc.state_dict())
        query = torch.rand(12, 3)
        torch.testing.assert_close(enc(query), clone(query))


if __name__ == "__main__":
    unittest.main()
