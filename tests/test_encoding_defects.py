"""Regression tests for three defects in the inherited hash-grid code.

The precedence defect is numerically inert in every committed config because
_PRIMES[0] == 1 and ix <= res < table_size, so masking commutes with the XOR.
It only diverges once res >= table_size, which is what test_hash_matches_reference
_at_high_resolution exercises. A test using a committed config would pass on the bug.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import _PRIMES, HashEncoding2D, HashEncoding3D  # noqa: E402


class TestHashPrecedence(unittest.TestCase):
    def test_hash_matches_reference_at_high_resolution(self):
        """res >= table_size is where the '&' / '^' precedence defect diverges."""
        enc = HashEncoding2D(
            levels=1,
            features_per_level=2,
            table_size_log2=4,
            base_resolution=64,
            finest_resolution=64,
        )
        self.assertFalse(enc._dense[0], "level must be hashed for this test to mean anything")
        ix = torch.arange(0, 64, dtype=torch.long)
        iy = torch.full_like(ix, 33)
        got = enc._index(ix, iy, 0)
        want = ((ix * _PRIMES[0]) ^ (iy * _PRIMES[1])) & (enc.table_size - 1)
        torch.testing.assert_close(got, want)

    def test_index_never_exceeds_table(self):
        enc = HashEncoding2D(
            levels=1,
            features_per_level=2,
            table_size_log2=4,
            base_resolution=64,
            finest_resolution=64,
        )
        grid = torch.arange(0, 65, dtype=torch.long)
        ix, iy = torch.meshgrid(grid, grid, indexing="ij")
        idx = enc._index(ix.reshape(-1), iy.reshape(-1), 0)
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), enc.table_size)


class TestBoundaryClamp(unittest.TestCase):
    """floor(pos).clamp(0, res) makes x0 == res a degenerate constant cell."""

    def test_2d_interpolates_at_upper_boundary(self):
        enc = HashEncoding2D(
            levels=1,
            features_per_level=2,
            table_size_log2=14,
            base_resolution=4,
            finest_resolution=4,
        )
        with torch.no_grad():
            enc.tables[0].uniform_(-1.0, 1.0)
        just_inside = enc(torch.tensor([[0.999, 0.999]]))
        at_corner = enc(torch.tensor([[1.0, 1.0]]))
        torch.testing.assert_close(just_inside, at_corner, atol=2e-2, rtol=0)

    def test_3d_interpolates_at_upper_boundary(self):
        enc = HashEncoding3D(
            levels=1,
            features_per_level=2,
            table_size_log2=14,
            base_resolution=4,
            finest_resolution=4,
        )
        with torch.no_grad():
            enc.tables[0].uniform_(-1.0, 1.0)
        just_inside = enc(torch.tensor([[0.999, 0.999, 0.999]]))
        at_corner = enc(torch.tensor([[1.0, 1.0, 1.0]]))
        torch.testing.assert_close(just_inside, at_corner, atol=2e-2, rtol=0)


class TestStreamedWorldSupport(unittest.TestCase):
    def test_streamed_module_exposes_spatial_tensors_for(self):
        from nrp.torch_backend import streamed_train

        self.assertTrue(hasattr(streamed_train, "spatial_tensors_for"))


if __name__ == "__main__":
    unittest.main()
