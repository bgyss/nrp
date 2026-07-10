import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.export_webgpu_demo import build_link_mask, export_g1_frame  # noqa: E402
from examples.export_webgpu_runtime import export_mlp, numpy_forward  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.residual_dynamic import ResidualNRP  # noqa: E402
from nrp.toy_tracer import SPHERE_CENTER, trace_path_cache  # noqa: E402


def cache_gbuffer(cache) -> dict:
    return {
        "width": cache.width,
        "height": cache.height,
        "albedo": cache.albedo,
        "depth": cache.depth,
        "normal": cache.normal,
        "position": cache.position,
    }


class LinkMaskTests(unittest.TestCase):
    def test_mask_flags_exactly_the_in_box_first_hits(self):
        position = np.zeros((2, 2, 3))
        position[0, 0] = [0.5, 0.5, 0.5]  # inside
        position[0, 1] = [2.0, 0.5, 0.5]  # outside (x)
        position[1, 0] = [1.0, 1.0, 1.0]  # boundary counts as inside
        position[1, 1] = [-0.1, 0.5, 0.5]  # outside (x)
        mask = build_link_mask(position, np.zeros(3), np.ones(3))
        np.testing.assert_array_equal(mask, np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32))
        self.assertEqual(mask.dtype, np.float32)


class ExportFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(8, 8, 2, max_bounces=1, seed=3, sphere_center=SPHERE_CENTER)
        cls.gbuf = cache_gbuffer(cls.cache)
        cls.light_params = np.array([0.35, 0.28, 0.62, 0.2], dtype=np.float32)
        torch.manual_seed(0)
        cls.base = TorchNRP(
            light_type="sphere", hidden_width=8, hidden_layers=1, use_encoding=False
        )
        torch.manual_seed(1)
        cls.residual = ResidualNRP(hidden_width=8, hidden_layers=1)

    def test_linear_head_numpy_replica_matches_torch(self):
        flat, dims = export_mlp(self.residual)
        n = 64
        xy = np.random.default_rng(0).random((n, 2)).astype(np.float32)
        aux = np.random.default_rng(1).random((n, 7)).astype(np.float32)
        replica = numpy_forward(
            xy, aux, self.light_params, flat, dims, None, None, 0, 0, output_activation="linear"
        )
        with torch.no_grad():
            ref = self.residual(
                torch.as_tensor(xy),
                torch.as_tensor(aux),
                torch.as_tensor(self.light_params).expand(n, -1),
            ).numpy()
        self.assertLess(float(np.max(np.abs(replica - ref))), 1e-5)
        self.assertTrue((replica < 0).any(), "linear replica must carry signed values")

    def test_g1_frame_export_self_check_and_mask_gating(self):
        region = np.zeros((8, 8), dtype=bool)
        region[2:6, 2:6] = True
        blobs = export_g1_frame(self.base, self.residual, self.gbuf, region, self.light_params)
        self.assertLess(blobs["self_check_max_abs_diff"], 1e-3)
        self.assertEqual(blobs["pixels"].shape, (64, 9))
        np.testing.assert_array_equal(blobs["mask"], region.reshape(-1).astype(np.float32))
        self.assertEqual(blobs["base_dims"][0], 13)
        self.assertEqual(blobs["residual_dims"][-1], 3)


if __name__ == "__main__":
    unittest.main()
