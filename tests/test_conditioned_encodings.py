"""Camera-conditioned training must accept every world-anchored encoding."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.conditioned_multiview import train_conditioned  # noqa: E402
from nrp.torch_backend.train import spatial_tensors, world_encodings  # noqa: E402


class TestWorldEncodingSet(unittest.TestCase):
    def test_contains_every_world_arm(self):
        self.assertLessEqual(
            {"world3d", "world_triplane", "world_sparse", "world_normal_triplane"},
            set(world_encodings()),
        )

    def test_excludes_pixel2d(self):
        self.assertNotIn("pixel2d", world_encodings())


class TestSpatialTensors(unittest.TestCase):
    def test_returns_positions_for_every_world_encoding(self):
        import torch

        from tests.test_camera_machinery_audit import _tiny_cache

        cache = _tiny_cache()
        for name in world_encodings():
            spatial, aux = spatial_tensors(cache, torch.device("cpu"), name)
            self.assertEqual(spatial.shape[1], 3, name)
            self.assertEqual(aux.shape[1], 7, name)


class TestGuardRejectsPixel2d(unittest.TestCase):
    def test_pixel2d_is_still_rejected(self):
        cfg = {"model": {"camera_conditioned": True, "spatial_encoding": "pixel2d"}}
        with self.assertRaises(ValueError):
            train_conditioned(cfg)

    def test_unregistered_encoding_is_rejected(self):
        cfg = {"model": {"camera_conditioned": True, "spatial_encoding": "not_an_arm"}}
        with self.assertRaises(ValueError):
            train_conditioned(cfg)


if __name__ == "__main__":
    unittest.main()
