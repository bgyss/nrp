"""Regression tests for defects (and one guaranteed invariant) in the inherited
hash-grid code, plus a wiring check for streamed world-position support.

The precedence defect is numerically inert in every committed config because
_PRIMES[0] == 1 and ix <= res < table_size, so masking commutes with the XOR.
It only diverges once res >= table_size, which is what test_hash_matches_reference
_at_high_resolution exercises. A test using a committed config would pass on the bug.

The `clamp(0, res - 1)` boundary change (vs. the prior `clamp(0, res)`) is a
no-op for every in-range input: at xy=1.0, `clamp(0, res)` gives pos0=res,
frac=0, selecting the corner entry, while `clamp(0, res - 1)` gives
pos0=res-1, frac=1.0, whose interpolation weights collapse onto that same
corner entry. Output is bit-identical either way. The change is retained not
because it fixes an observable defect, but because it *guarantees* frac in
[0, 1] and a non-degenerate interpolation cell (x0 != x1) for every input,
an invariant a later sparse-index encoder relies on. `TestBoundaryClamp`
pins that invariant directly rather than comparing values across the change.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoder_registry import _floor_cell  # noqa: E402
from nrp.torch_backend.encoding import (  # noqa: E402
    _PRIMES,
    HashEncoding2D,
    HashEncoding3D,
)


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
    """`clamp(0, res - 1)` guarantees frac in [0, 1] and a non-degenerate cell.

    The old value-comparison tests here (just-inside vs. exactly-at-the-corner
    interpolation) passed identically before AND after the clamp change, since
    `clamp(0, res)` at xy=1.0 selects the corner entry directly (frac=0) while
    `clamp(0, res - 1)` selects the same entry via frac=1.0 interpolation
    weights that collapse onto it — bit-identical output, so those tests could
    never fail. What the clamp change actually guarantees is the invariant
    checked below: frac always lands in [0, 1], and the interpolation cell's
    upper corner index is always strictly greater than its lower corner index
    (never degenerate), for every axis and every in-range input including the
    exact boundaries 0.0 and 1.0.
    """

    # Inputs spanning exactly the lower boundary, an interior point, and
    # exactly the upper boundary on every axis.
    _COORDS_1D = [0.0, 0.37, 1.0]

    def test_floor_cell_invariant(self):
        """Directly exercise the production `_floor_cell` helper across a range of
        resolutions and coordinates, including the exact boundaries 0.0 and 1.0."""
        for res in (1, 4, 16, 256):
            coords = torch.tensor(self._COORDS_1D, dtype=torch.float32)
            pos = coords * res
            pos0, frac = _floor_cell(pos, res)
            x1 = (pos0 + 1).clamp(max=res)
            self.assertTrue(bool((frac >= 0.0).all()), f"frac < 0 at res={res}")
            self.assertTrue(bool((frac <= 1.0).all()), f"frac > 1 at res={res}")
            self.assertTrue(
                bool((x1 > pos0).all()), f"degenerate cell at res={res}: pos0={pos0}, x1={x1}"
            )

    def test_2d_clamp_invariant_and_finite_output(self):
        enc = HashEncoding2D(
            levels=2,
            features_per_level=2,
            table_size_log2=14,
            base_resolution=4,
            finest_resolution=16,
        )
        with torch.no_grad():
            enc.tables[0].uniform_(-1.0, 1.0)
            enc.tables[1].uniform_(-1.0, 1.0)
        coords = [(x, y) for x in self._COORDS_1D for y in self._COORDS_1D]
        xy = torch.tensor(coords, dtype=torch.float32)
        for level, res in enumerate(enc.resolutions):
            pos0, frac = _floor_cell(xy * res, res)
            x1 = (pos0[:, 0] + 1).clamp(max=res)
            y1 = (pos0[:, 1] + 1).clamp(max=res)
            self.assertTrue(bool((frac >= 0.0).all()), f"frac < 0 at level {level}")
            self.assertTrue(bool((frac <= 1.0).all()), f"frac > 1 at level {level}")
            self.assertTrue(bool((x1 > pos0[:, 0]).all()), f"degenerate x cell at level {level}")
            self.assertTrue(bool((y1 > pos0[:, 1]).all()), f"degenerate y cell at level {level}")

        inputs = torch.tensor(coords, dtype=torch.float32)
        out = enc(inputs)
        self.assertEqual(tuple(out.shape), (len(coords), enc.output_dim))
        self.assertTrue(torch.isfinite(out).all())

    def test_3d_clamp_invariant_and_finite_output(self):
        enc = HashEncoding3D(
            levels=2,
            features_per_level=2,
            table_size_log2=14,
            base_resolution=4,
            finest_resolution=16,
        )
        with torch.no_grad():
            enc.tables[0].uniform_(-1.0, 1.0)
            enc.tables[1].uniform_(-1.0, 1.0)
        coords = [
            (x, y, z) for x in self._COORDS_1D for y in self._COORDS_1D for z in self._COORDS_1D
        ]
        xyz = torch.tensor(coords, dtype=torch.float32)
        for level, res in enumerate(enc.resolutions):
            pos0, frac = _floor_cell(xyz * res, res)
            x1 = (pos0[:, 0] + 1).clamp(max=res)
            y1 = (pos0[:, 1] + 1).clamp(max=res)
            z1 = (pos0[:, 2] + 1).clamp(max=res)
            self.assertTrue(bool((frac >= 0.0).all()), f"frac < 0 at level {level}")
            self.assertTrue(bool((frac <= 1.0).all()), f"frac > 1 at level {level}")
            self.assertTrue(bool((x1 > pos0[:, 0]).all()), f"degenerate x cell at level {level}")
            self.assertTrue(bool((y1 > pos0[:, 1]).all()), f"degenerate y cell at level {level}")
            self.assertTrue(bool((z1 > pos0[:, 2]).all()), f"degenerate z cell at level {level}")

        inputs = torch.tensor(coords, dtype=torch.float32)
        out = enc(inputs)
        self.assertEqual(tuple(out.shape), (len(coords), enc.output_dim))
        self.assertTrue(torch.isfinite(out).all())


class TestStreamedWorldSupport(unittest.TestCase):
    def test_streamed_module_exposes_spatial_tensors_for(self):
        from nrp.torch_backend import streamed_train

        self.assertTrue(hasattr(streamed_train, "spatial_tensors_for"))

    def test_train_streamed_wires_world3d_encoding_into_model(self):
        """`train_streamed` previously built TorchNRP without spatial_encoding, so
        a world-anchored config silently trained through the dead pixel2d branch
        of `spatial_tensors_for`. Prove the config now reaches the constructor by
        running the smallest possible streamed job and checking the model it
        hands back reports the selected encoding."""
        import tempfile

        from nrp.torch_backend.streamed_train import train_streamed
        from nrp.toy_tracer import trace_path_cache

        cache = trace_path_cache(6, 6, 4, 2, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            shard_dir = Path(tmp) / "shards"
            cache.save_sharded(str(shard_dir), tile_size=3)
            cfg = {
                "seed": 0,
                "light_type": "sphere",
                "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
                "sampling": "segments",
                "denoise": {"enabled": False},
                "pool": {"size": 2, "replace_count": 1, "replace_every": 2},
                "model": {
                    "hidden_width": 8,
                    "hidden_layers": 1,
                    "encoding": {
                        "levels": 1,
                        "features_per_level": 2,
                        "finest_resolution": 8,
                    },
                    "spatial_encoding": "world3d",
                },
                "lr": 5e-3,
                "batch_pixels": 16,
                "iters": 1,
            }
            model, _ = train_streamed(shard_dir, cache, cfg)

        self.assertEqual(model.spatial_encoding, "world3d")


if __name__ == "__main__":
    unittest.main()
