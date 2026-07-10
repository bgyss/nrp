import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.residual_dynamic import (  # noqa: E402
    ResidualNRP,
    composite_predict,
    invalidated_shards,
    pixel_features,
    train_residual,
)
from nrp.toy_tracer import SPHERE_CENTER, trace_path_cache  # noqa: E402


class InvalidatedShardsTests(unittest.TestCase):
    def test_invalidated_shards_cover_mask_and_align_to_tile_grid(self):
        mask = np.zeros((16, 16), dtype=bool)
        mask[3, 5] = True  # tile (0, 0) with shard_size 8? no: (0, 0) since 3<8,5<8
        mask[10, 12] = True  # tile (1, 1)
        region, tiles = invalidated_shards(mask, shard_size=8)
        self.assertEqual(tiles, [(0, 0), (1, 1)])
        # every invalid pixel is inside the region
        self.assertTrue(region[mask].all())
        # the region is exactly the union of the two full 8x8 tiles
        expected = np.zeros((16, 16), dtype=bool)
        expected[0:8, 0:8] = True
        expected[8:16, 8:16] = True
        np.testing.assert_array_equal(region, expected)

    def test_invalidated_shards_empty_mask_returns_empty(self):
        mask = np.zeros((16, 16), dtype=bool)
        region, tiles = invalidated_shards(mask, shard_size=8)
        self.assertEqual(tiles, [])
        self.assertFalse(region.any())

    def test_invalidated_shards_handles_non_divisible_resolution(self):
        mask = np.zeros((18, 18), dtype=bool)
        mask[17, 17] = True  # edge tile (2, 2) covers rows/cols 16..17 only
        region, tiles = invalidated_shards(mask, shard_size=8)
        self.assertEqual(tiles, [(2, 2)])
        expected = np.zeros((18, 18), dtype=bool)
        expected[16:18, 16:18] = True
        np.testing.assert_array_equal(region, expected)


class ResidualProxyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(16, 16, 2, max_bounces=1, seed=7, sphere_center=SPHERE_CENTER)
        cls.light = SphereLight(center=[0.35, 0.28, 0.62], radius=0.2, rgb=[1.3, 1.0, 0.8])
        cls.light_params = np.array([0.35, 0.28, 0.62, 0.2], dtype=np.float32)
        torch.manual_seed(0)
        cls.base = TorchNRP(
            light_type="sphere", hidden_width=16, hidden_layers=2, use_encoding=False
        )
        torch.manual_seed(1)
        cls.residual = ResidualNRP(hidden_width=16, hidden_layers=2)

    def test_residual_head_is_signed(self):
        xy, aux = pixel_features(self.cache)
        with torch.no_grad():
            out = self.residual(
                torch.as_tensor(xy),
                torch.as_tensor(aux),
                torch.as_tensor(self.light_params).expand(xy.shape[0], -1),
            ).numpy()
        self.assertTrue((out < 0).any(), "linear head should produce signed outputs")

    def test_composite_equals_base_outside_region_exactly(self):
        region = np.zeros((16, 16), dtype=bool)
        region[4:8, 4:8] = True
        composite = composite_predict(
            self.base, self.residual, self.cache, self.light_params, region
        )
        xy, aux = pixel_features(self.cache)
        with torch.no_grad():
            base_pred = (
                self.base(
                    torch.as_tensor(xy),
                    torch.as_tensor(aux),
                    torch.as_tensor(self.light_params).expand(xy.shape[0], -1),
                )
                .numpy()
                .reshape(16, 16, 3)
            )
        outside = ~region
        np.testing.assert_array_equal(composite[outside], base_pred[outside])
        self.assertTrue(
            np.any(composite[region] != base_pred[region]),
            "random-init residual should change region pixels",
        )

    def test_train_residual_reduces_region_error(self):
        target = gather_light(self.cache, self.light)
        region = np.zeros((16, 16), dtype=bool)
        region[2:14, 2:14] = True
        torch.manual_seed(2)
        residual = ResidualNRP(hidden_width=16, hidden_layers=2)
        base_only_psnr_in_region = None
        xy, aux = pixel_features(self.cache)
        with torch.no_grad():
            base_pred = (
                self.base(
                    torch.as_tensor(xy),
                    torch.as_tensor(aux),
                    torch.as_tensor(self.light_params).expand(xy.shape[0], -1),
                )
                .numpy()
                .reshape(16, 16, 3)
            )
        base_only_psnr_in_region = psnr(base_pred[region], target[region])
        losses = train_residual(
            residual,
            self.base,
            self.cache,
            target,
            region,
            self.light_params,
            iters=150,
            lr=5e-3,
        )
        self.assertEqual(len(losses), 150)
        first_window = float(np.mean(losses[:20]))
        last_window = float(np.mean(losses[-20:]))
        self.assertLess(last_window, first_window)
        composite = composite_predict(self.base, residual, self.cache, self.light_params, region)
        composite_psnr_in_region = psnr(composite[region], target[region])
        self.assertGreater(composite_psnr_in_region, base_only_psnr_in_region)

    def test_residual_save_load_roundtrip(self):
        xy, aux = pixel_features(self.cache)
        params = torch.as_tensor(self.light_params).expand(xy.shape[0], -1)
        with torch.no_grad():
            before = self.residual(torch.as_tensor(xy), torch.as_tensor(aux), params).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "residual.pt")
            self.residual.save(path)
            loaded = ResidualNRP.load(path)
        with torch.no_grad():
            after = loaded(torch.as_tensor(xy), torch.as_tensor(aux), params).numpy()
        np.testing.assert_array_equal(before, after)


if __name__ == "__main__":
    unittest.main()
