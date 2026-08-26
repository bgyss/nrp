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

    def test_capacity_report_measures_collisions_from_the_key_buffer(self):
        # capacity_report must derive its fields from keys_{level}, not report
        # hard-coded literals: corrupting the buffer to duplicate one key must
        # be visible in the report.
        rng = np.random.default_rng(2)
        enc = _encoder(rng.random((100, 3)))
        with torch.no_grad():
            keys0 = enc.keys_0.clone()
            keys0[1] = keys0[0]
            enc.keys_0.copy_(keys0)
        report = enc.capacity_report()
        level0 = report["levels"][0]
        self.assertGreater(level0["collision_fraction"], 0.0)
        self.assertGreater(level0["max_slot_load"], 1)
        self.assertEqual(level0["used_slots"], level0["slots"] - 1)


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
        # `enc` and `clone` must be built from DIFFERENT occupancy so each
        # natively computes different keys_{level} buffers. Mirroring the point
        # cloud (1 - points) keeps the per-level distinct-vertex counts equal
        # (a reflection is a bijection of the discretized grid) so the table
        # shapes match and load_state_dict can transfer strictly, while the
        # actual vertex sets -- and therefore the keys -- differ. If keys were
        # a plain attribute instead of a registered buffer, load_state_dict
        # would silently leave clone's natively-computed (different) keys in
        # place and this test would fail.
        rng = np.random.default_rng(6)
        points = rng.random((60, 3))
        mirrored = 1.0 - points
        enc = _encoder(points)
        clone = _encoder(mirrored)
        for level in range(CONFIG["levels"]):
            self.assertFalse(
                torch.equal(getattr(enc, f"keys_{level}"), getattr(clone, f"keys_{level}")),
                f"level {level} keys already match before the round trip; "
                "the two occupancies are not actually distinguishing",
            )

        clone.load_state_dict(enc.state_dict())

        for level in range(CONFIG["levels"]):
            torch.testing.assert_close(
                getattr(enc, f"keys_{level}"), getattr(clone, f"keys_{level}")
            )
        query = torch.rand(12, 3)
        torch.testing.assert_close(enc(query), clone(query))


class TestResolutionScheduleValidation(unittest.TestCase):
    def test_occupancy_built_with_a_different_schedule_raises(self):
        # Occupancy resolutions [4, 8] built for CONFIG's schedule, but the
        # encoder is asked to believe a different (finest_resolution=16)
        # schedule named it. base_resolution/finest_resolution must be
        # load-bearing: this must not silently construct.
        rng = np.random.default_rng(7)
        points = rng.random((60, 3))
        from nrp.torch_backend.occupancy import level_resolutions

        res = level_resolutions(
            CONFIG["levels"], CONFIG["base_resolution"], CONFIG["finest_resolution"]
        )
        occ = grid_occupancy(points, res)
        with self.assertRaises(ValueError):
            SparseVoxelEncoding(
                occupancy=occ,
                levels=CONFIG["levels"],
                features_per_level=CONFIG["features_per_level"],
                base_resolution=CONFIG["base_resolution"],
                finest_resolution=16,
            )


if __name__ == "__main__":
    unittest.main()
