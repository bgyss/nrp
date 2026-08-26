"""Cross-cutting guarantees: corner exactness, checkpoint back-compat, streamed parity."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import HashEncoding2D, HashEncoding3D  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402

SMALL = {
    "levels": 2,
    "features_per_level": 2,
    "table_size_log2": 14,
    "base_resolution": 4,
    "finest_resolution": 8,
}


class TestCornerExactness(unittest.TestCase):
    """A query exactly on a grid vertex must return that vertex's feature."""

    def test_2d_corner_returns_table_entry(self):
        enc = HashEncoding2D(**SMALL)
        with torch.no_grad():
            for table in enc.tables:
                table.uniform_(-1.0, 1.0)
        res = enc.resolutions[0]
        xy = torch.tensor([[2.0 / res, 3.0 / res]], dtype=torch.float32)
        want = enc.tables[0][
            enc._index(torch.tensor([2]), torch.tensor([3]), 0) % enc.tables[0].shape[0]
        ]
        got = enc(xy)[:, : enc.features_per_level]
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)

    def test_3d_corner_returns_table_entry(self):
        enc = HashEncoding3D(**SMALL)
        with torch.no_grad():
            for table in enc.tables:
                table.uniform_(-1.0, 1.0)
        res = enc.resolutions[0]
        xyz = torch.tensor([[2.0 / res, 3.0 / res, 1.0 / res]], dtype=torch.float32)
        idx = enc._index(torch.tensor([2]), torch.tensor([3]), torch.tensor([1]), 0)
        want = enc.tables[0][idx % enc.tables[0].shape[0]]
        got = enc(xyz)[:, : enc.features_per_level]
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


class TestCheckpointBackCompat(unittest.TestCase):
    def test_committed_pixel2d_checkpoint_still_loads(self):
        path = Path(__file__).resolve().parent.parent / "out" / "toy-torch" / "model.pt"
        if not path.exists():
            self.skipTest("committed toy torch checkpoint not present")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        self.assertIsInstance(state, dict)
        self.assertEqual(set(state.keys()), {"config", "state_dict"})
        cfg = state["config"]
        self.assertIsInstance(cfg, dict)
        self.assertEqual(cfg.get("spatial_encoding", "pixel2d"), "pixel2d")
        # The checkpoint must actually construct a working model from its own config.
        model = TorchNRP(
            light_type=cfg["light_type"],
            hidden_width=cfg["hidden_width"],
            hidden_layers=cfg["hidden_layers"],
            encoding=cfg["encoding"],
            spatial_encoding=cfg.get("spatial_encoding", "pixel2d"),
        )
        model.load_state_dict(state["state_dict"])

    def test_round_trip_preserves_predictions(self):
        model = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=SMALL)
        clone = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=SMALL)
        clone.load_state_dict(model.state_dict())
        xy, aux, params = torch.rand(4, 2), torch.rand(4, 7), torch.rand(4, 4)
        torch.testing.assert_close(model(xy, aux, params), clone(xy, aux, params))


class TestStreamedParity(unittest.TestCase):
    """S1 convention: the streamed path must agree with the in-memory path."""

    def test_spatial_tensors_for_matches_train_spatial_tensors(self):
        from nrp.torch_backend.streamed_train import spatial_tensors_for
        from nrp.torch_backend.train import spatial_tensors
        from tests.test_camera_machinery_audit import _tiny_cache

        cache = _tiny_cache()
        device = torch.device("cpu")
        for name in ("pixel2d", "world3d"):
            model = TorchNRP(
                light_type="sphere",
                hidden_width=8,
                hidden_layers=1,
                encoding=SMALL,
                spatial_encoding=name,
                world_bounds=None
                if name == "pixel2d"
                else {"min": [-1.0, -1.0, -1.0], "max": [2.0, 2.0, 2.0]},
            )
            streamed, streamed_aux = spatial_tensors_for(cache, model, device)
            direct, direct_aux = spatial_tensors(cache, device, name)
            torch.testing.assert_close(streamed, direct)
            torch.testing.assert_close(streamed_aux, direct_aux)


if __name__ == "__main__":
    unittest.main()
