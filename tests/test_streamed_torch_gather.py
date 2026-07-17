"""S1: streamed torch-backend gather parity and batched multi-light shard passes."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_throughput  # noqa: E402
from nrp.torch_backend.streamed_train import (  # noqa: E402
    gather_sphere_streamed,
    gather_spheres_streamed,
    train_streamed,
)
from nrp.toy_tracer import trace_path_cache  # noqa: E402

try:
    import torch

    HAVE_MPS = torch.backends.mps.is_available()
except Exception:  # pragma: no cover
    HAVE_MPS = False

LIGHTS = [
    (np.array([0.1, 0.6, 0.0]), 0.2),
    (np.array([0.75, 0.75, 0.35]), 0.12),
    (np.array([-0.4, 0.3, -0.2]), 0.3),
]


class StreamedTorchGatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(12, 12, 6, 3, seed=17)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.shards = Path(cls.tmp.name) / "shards"
        cls.packed = Path(cls.tmp.name) / "packed"
        cls.cache.save_sharded(str(cls.shards), tile_size=4)
        cls.cache.save_sharded(str(cls.packed), tile_size=4, packed=True)
        cls.n_paths = cls.cache.n_paths.reshape(-1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_torch_backend_matches_numpy_backend(self):
        for center, radius in LIGHTS:
            ref, ref_stats = gather_sphere_streamed(
                self.shards, self.n_paths, center, radius, backend="numpy"
            )
            got, stats = gather_sphere_streamed(
                self.shards, self.n_paths, center, radius, backend="torch", device="cpu"
            )
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)
            # bounded residency: identical decoded-numpy accounting in both backends
            self.assertEqual(stats["peak_segment_bytes"], ref_stats["peak_segment_bytes"])
            self.assertGreater(stats["peak_device_tensor_bytes"], 0)

    def test_torch_backend_matches_numpy_on_packed_shards(self):
        for center, radius in LIGHTS:
            ref, _ = gather_sphere_streamed(
                self.packed, self.n_paths, center, radius, backend="numpy"
            )
            got, _ = gather_sphere_streamed(
                self.packed, self.n_paths, center, radius, backend="torch", device="cpu"
            )
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)

    def test_streamed_matches_in_memory_gather(self):
        for center, radius in LIGHTS:
            ref = gather_throughput(self.cache, center, radius)
            got, _ = gather_sphere_streamed(
                self.shards, self.n_paths, center, radius, backend="torch", device="cpu"
            )
            np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)

    def test_multi_light_pass_bit_identical_to_single_passes(self):
        batched, stats = gather_spheres_streamed(self.shards, self.n_paths, LIGHTS, backend="numpy")
        self.assertEqual(stats["lights_per_pass"], len(LIGHTS))
        for (center, radius), image in zip(LIGHTS, batched, strict=True):
            single, _ = gather_sphere_streamed(
                self.shards, self.n_paths, center, radius, backend="numpy"
            )
            np.testing.assert_array_equal(image, single)

    @unittest.skipUnless(HAVE_MPS, "MPS not available")
    def test_torch_mps_close_to_numpy(self):
        # fp32 on MPS: aggregate-error tolerance, matching test_torch_gather's
        # MPS convention (boundary-grazing segments may round differently).
        for center, radius in LIGHTS:
            ref, _ = gather_sphere_streamed(
                self.shards, self.n_paths, center, radius, backend="numpy"
            )
            got, _ = gather_sphere_streamed(
                self.shards, self.n_paths, center, radius, backend="torch", device="mps"
            )
            rel_l1 = np.abs(got - ref).sum() / max(ref.sum(), 1e-9)
            self.assertLess(rel_l1, 1e-4)

    def test_train_streamed_torch_backend_matches_numpy_targets(self):
        cfg = {
            "seed": 0,
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
            "sampling": "segments",
            "denoise": {"enabled": False},
            "pool": {"size": 4, "replace_count": 1, "replace_every": 4},
            "model": {
                "hidden_width": 16,
                "hidden_layers": 2,
                "encoding": {"levels": 2, "features_per_level": 2, "finest_resolution": 10},
            },
            "lr": 5e-3,
            "batch_pixels": 64,
            "iters": 8,
        }
        _, np_stats = train_streamed(self.shards, self.cache, cfg)
        cfg_torch = dict(cfg, gather_backend="torch", gather_device="cpu")
        _, t_stats = train_streamed(self.shards, self.cache, cfg_torch)
        # same rng stream + cpu float64 torch gather: loss curves agree closely
        np.testing.assert_allclose(t_stats["loss_curve"], np_stats["loss_curve"], rtol=1e-4)
        self.assertEqual(
            t_stats["peak_segment_bytes_loaded"], np_stats["peak_segment_bytes_loaded"]
        )
        self.assertEqual(t_stats["gather_backend"], "torch")


if __name__ == "__main__":
    unittest.main()
