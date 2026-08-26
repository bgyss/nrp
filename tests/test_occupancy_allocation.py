"""Arm A control: size hash tables from measured occupancy, not a parameter budget."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import HashEncoding3D  # noqa: E402
from nrp.torch_backend.occupancy import grid_occupancy, level_resolutions  # noqa: E402

BASE = {"levels": 4, "features_per_level": 2, "base_resolution": 4, "finest_resolution": 32}


def _occ(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    res = level_resolutions(BASE["levels"], BASE["base_resolution"], BASE["finest_resolution"])
    return grid_occupancy(rng.random((n, 3)), res)


class TestUniformAllocationUnchanged(unittest.TestCase):
    def test_default_matches_committed_behaviour(self):
        enc = HashEncoding3D(table_size_log2=8, **BASE)
        for level, res in enumerate(enc.resolutions):
            expected = (res + 1) ** 3 if (res + 1) ** 3 <= 256 else 256
            self.assertEqual(enc.tables[level].shape[0], expected)

    def test_allocation_attribute_defaults_to_uniform(self):
        enc = HashEncoding3D(table_size_log2=8, **BASE)
        self.assertEqual(enc.allocation, "uniform")


class TestOccupancyAllocation(unittest.TestCase):
    def test_slots_cover_measured_occupancy_when_budget_allows(self):
        occ = _occ()
        budget = sum(o.count for o in occ) * 2
        enc = HashEncoding3D(
            allocation="occupancy", occupancy=occ, slot_budget=budget, table_size_log2=8, **BASE
        )
        for level, o in enumerate(occ):
            self.assertGreaterEqual(enc.tables[level].shape[0], o.count)

    def test_levels_the_budget_cannot_serve_are_dropped_not_crushed(self):
        occ = _occ()
        # Budget large enough for the two coarsest levels only.
        budget = occ[0].count + occ[1].count
        enc = HashEncoding3D(
            allocation="occupancy", occupancy=occ, slot_budget=budget, table_size_log2=8, **BASE
        )
        self.assertLess(enc.levels, BASE["levels"])
        self.assertEqual(enc.output_dim, enc.levels * BASE["features_per_level"])

    def test_rejects_occupancy_allocation_without_occupancy(self):
        with self.assertRaises(ValueError):
            HashEncoding3D(allocation="occupancy", table_size_log2=8, **BASE)

    def test_rejects_unknown_allocation(self):
        with self.assertRaises(ValueError):
            HashEncoding3D(allocation="nonsense", table_size_log2=8, **BASE)

    def test_truncated_encoder_forward_does_not_index_past_end(self):
        """Forward on a truncated encoder must run and only use the surviving levels."""
        occ = _occ()
        budget = occ[0].count + occ[1].count
        enc = HashEncoding3D(
            allocation="occupancy", occupancy=occ, slot_budget=budget, table_size_log2=8, **BASE
        )
        xyz = np.random.default_rng(1).random((16, 3))
        import torch

        out = enc(torch.as_tensor(xyz, dtype=torch.float32))
        self.assertEqual(out.shape, (16, enc.output_dim))
        # Truncation must have actually happened and be reflected consistently
        # everywhere: levels, resolutions, and per-level bookkeeping all shrink.
        self.assertEqual(len(enc.resolutions), enc.levels)
        self.assertEqual(len(enc.tables), enc.levels)
        self.assertEqual(len(enc._dense), enc.levels)

    def test_rejects_occupancy_whose_resolutions_mismatch_full_schedule(self):
        """Validate against the FULL schedule before truncation, not after.

        Occupancy computed for the wrong base/finest resolution has the wrong
        per-level resolutions; that must be caught even though the budget below
        would otherwise truncate the encoder down to levels where, by coincidence
        of construction, the mismatch might not be scrutinized.
        """
        wrong_res = level_resolutions(BASE["levels"], base_resolution=5, finest_resolution=32)
        rng = np.random.default_rng(0)
        occ = grid_occupancy(rng.random((400, 3)), wrong_res)
        budget = sum(o.count for o in occ) * 2
        with self.assertRaises(ValueError):
            HashEncoding3D(
                allocation="occupancy",
                occupancy=occ,
                slot_budget=budget,
                table_size_log2=8,
                **BASE,
            )


if __name__ == "__main__":
    unittest.main()
