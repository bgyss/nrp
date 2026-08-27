"""Uniform interface and registry for spatial encoders."""

import subprocess
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
REPO_ROOT = Path(__file__).resolve().parent.parent

from nrp.torch_backend.encoder_registry import (  # noqa: E402
    SPATIAL_ENCODERS,
    build_encoder,
    encoder_schedule_params,
)
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


class TestEncoderScheduleParams(unittest.TestCase):
    def test_world3d_defaults_match_the_class_constructor(self):
        self.assertEqual(encoder_schedule_params("world3d", {}), (8, 4, 256))

    def test_world_sparse_defaults_match_the_class_constructor(self):
        self.assertEqual(encoder_schedule_params("world_sparse", {}), (8, 4, 128))

    def test_explicit_config_values_override_class_defaults(self):
        self.assertEqual(
            encoder_schedule_params(
                "world3d", {"levels": 3, "base_resolution": 2, "finest_resolution": 32}
            ),
            (3, 2, 32),
        )

    def test_none_config_falls_back_to_class_defaults(self):
        self.assertEqual(encoder_schedule_params("world3d", None), (8, 4, 256))

    def test_unknown_encoder_raises(self):
        with self.assertRaises(ValueError):
            encoder_schedule_params("does_not_exist", {})


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


class TestColdImportPopulatesRegistry(unittest.TestCase):
    """FIX 2: `build_encoder` must not depend on import order.

    Previously `SPATIAL_ENCODERS` was populated only as a side effect of some
    other module (`encoding` or `model`) having been imported first -- correct in
    this repo's normal test order only by accident. A genuinely fresh interpreter
    process (no `encoding`/`model` import anywhere in its history) is the only way
    to prove `build_encoder` is self-sufficient; a `sys.modules`-clearing trick
    inside this same process would not exercise the real failure mode, since other
    already-imported modules (e.g. `nrp.torch_backend.sparse_encoding`, imported
    transitively by earlier tests in this file) can leave registrations behind in
    the shared `SPATIAL_ENCODERS` dict even after `del sys.modules[...]`.
    """

    def _run_cold(self, code: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_build_encoder_resolves_without_importing_encoding_or_model_first(self):
        code = (
            "import sys\n"
            "assert 'nrp.torch_backend.encoding' not in sys.modules\n"
            "assert 'nrp.torch_backend.model' not in sys.modules\n"
            "from nrp.torch_backend.encoder_registry import build_encoder\n"
            "enc = build_encoder('pixel2d', {'levels': 1, 'features_per_level': 2, "
            "'table_size_log2': 8, 'base_resolution': 4, 'finest_resolution': 8})\n"
            "print(type(enc).__name__)\n"
        )
        result = self._run_cold(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "HashEncoding2D")

    def test_encoder_schedule_params_resolves_cold_too(self):
        code = (
            "from nrp.torch_backend.encoder_registry import encoder_schedule_params\n"
            "print(encoder_schedule_params('world3d', {}))\n"
        )
        result = self._run_cold(code)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "(8, 4, 256)")


class TestModelUsesRegistry(unittest.TestCase):
    def test_pixel2d_model_is_unchanged(self):
        from nrp.torch_backend.model import TorchNRP

        model = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=CONFIG)
        out = model(torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 4))
        self.assertEqual(tuple(out.shape), (5, 3))


if __name__ == "__main__":
    unittest.main()
