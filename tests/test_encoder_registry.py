"""Uniform interface and registry for spatial encoders."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoder_registry import SPATIAL_ENCODERS, build_encoder  # noqa: E402
from nrp.torch_backend.encoding import HashEncoding2D  # noqa: E402

CONFIG = {
    "levels": 3,
    "features_per_level": 2,
    "table_size_log2": 8,
    "base_resolution": 4,
    "finest_resolution": 16,
}


class TestRegistry(unittest.TestCase):
    def test_registry_contains_the_committed_encoders(self):
        self.assertLessEqual({"pixel2d", "world3d", "world_triplane"}, set(SPATIAL_ENCODERS))

    def test_build_encoder_returns_the_registered_class(self):
        enc = build_encoder("pixel2d", CONFIG)
        self.assertIsInstance(enc, HashEncoding2D)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            build_encoder("does_not_exist", CONFIG)

    def test_occupancy_encoder_without_occupancy_raises(self):
        for name, cls in SPATIAL_ENCODERS.items():
            if getattr(cls, "needs_occupancy", False):
                with self.assertRaises(ValueError, msg=name):
                    build_encoder(name, CONFIG, occupancy=None)


class TestUniformInterface(unittest.TestCase):
    def test_every_encoder_declares_the_interface(self):
        for name, cls in SPATIAL_ENCODERS.items():
            self.assertIsInstance(getattr(cls, "needs_occupancy", None), bool, name)
            self.assertIsInstance(getattr(cls, "needs_normals", None), bool, name)
            self.assertTrue(hasattr(cls, "capacity_report"), name)

    def test_capacity_report_shape(self):
        enc = build_encoder("pixel2d", CONFIG)
        report = enc.capacity_report()
        self.assertIn("levels", report)
        self.assertEqual(len(report["levels"]), CONFIG["levels"])
        for level in report["levels"]:
            self.assertIn("slots", level)
            self.assertIn("resolution", level)


class TestModelUsesRegistry(unittest.TestCase):
    def test_pixel2d_model_is_unchanged(self):
        from nrp.torch_backend.model import TorchNRP

        model = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=CONFIG)
        out = model(torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 4))
        self.assertEqual(tuple(out.shape), (5, 3))


if __name__ == "__main__":
    unittest.main()
