"""Unit tests for nrp/torch_backend/train.py helpers that don't need a full
training run (see tests/test_torch_backend.py's TrainingSmokeTests for those)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.train import build_val_set  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def _tiny_cache():
    return trace_path_cache(5, 4, 2, 1, seed=9)


class BuildValSetTests(unittest.TestCase):
    def test_build_val_set_larger_count_is_a_superset(self):
        """A bigger gate set must extend the committed one, not resample it --
        this is what makes a re-read comparable to the run it re-reads."""
        cache = _tiny_cache()
        cfg = {
            "seed": 3,
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
            "sampling": "segments",
            "denoise": {"enabled": False},
            "n_val_lights": 4,
        }
        small = build_val_set(cache, cfg, 4)
        large = build_val_set(cache, cfg, 10)
        self.assertEqual(len(small), 4)
        self.assertEqual(len(large), 10)
        for a, b in zip(small, large, strict=False):
            # SphereLight's dataclass-derived __eq__ raises on numpy-array fields
            # (ambiguous truth value) even for equal values, so compare via to_dict().
            self.assertEqual(a["light"].to_dict(), b["light"].to_dict())

    def test_build_val_set_defaults_to_config_count(self):
        cache = _tiny_cache()
        cfg = {
            "seed": 3,
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
            "sampling": "segments",
            "denoise": {"enabled": False},
            "n_val_lights": 4,
        }
        self.assertEqual(len(build_val_set(cache, cfg)), 4)


if __name__ == "__main__":
    unittest.main()
