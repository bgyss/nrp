"""Per-layer compositing NRPs (roadmap item 8, §6.1): layer path partition, the
GATHERLIGHT linearity property compositing relies on, ownership masks, and the
composite CLI."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.torch_backend import composite as composite_cli  # noqa: E402
from nrp.torch_backend.composite import composite  # noqa: E402
from nrp.torch_backend.train import train  # noqa: E402
from nrp.toy_tracer import (  # noqa: E402
    SPHERE_ALBEDO,
    layer_ownership_mask,
    trace_path_cache,
)

W, H, SPP, BOUNCES, SEED = (16, 16, 4, 2, 3)


class LayerPartitionTests(unittest.TestCase):
    """The same seed traces the same paths; a layer cache keeps exactly the paths
    whose first hit is on the layer's geometry."""

    @classmethod
    def setUpClass(cls):
        cls.full = trace_path_cache(W, H, SPP, BOUNCES, SEED)
        cls.sphere = trace_path_cache(W, H, SPP, BOUNCES, SEED, layer="sphere")
        cls.box = trace_path_cache(W, H, SPP, BOUNCES, SEED, layer="box")
        cls.light = SphereLight(center=[0.3, 0.7, 0.5], radius=0.15, rgb=[1.0, 1.0, 1.0])

    def test_layers_partition_the_full_segment_set(self):
        self.assertEqual(
            self.sphere.segment_count + self.box.segment_count, self.full.segment_count
        )
        self.assertGreater(self.sphere.segment_count, 0)
        self.assertGreater(self.box.segment_count, 0)

    def test_n_paths_keeps_the_full_estimator_denominator(self):
        # Both layer caches divide by the full spp so their gathers sum to the
        # full-scene estimate; a layer-local count would break linearity.
        np.testing.assert_array_equal(self.sphere.n_paths, self.full.n_paths)
        np.testing.assert_array_equal(self.box.n_paths, self.full.n_paths)

    def test_gather_linearity_layer_sum_equals_full(self):
        full_img = gather_light(self.full, self.light)
        layer_sum = gather_light(self.sphere, self.light) + gather_light(self.box, self.light)
        np.testing.assert_allclose(layer_sum, full_img, rtol=1e-12, atol=1e-14)

    def test_layer_caches_validate_and_share_full_scene_aux(self):
        for cache in (self.sphere, self.box):
            cache.validate()
            np.testing.assert_array_equal(cache.albedo, self.full.albedo)
            np.testing.assert_array_equal(cache.depth, self.full.depth)

    def test_layer_with_medium_raises(self):
        with self.assertRaises(ValueError):
            trace_path_cache(W, H, SPP, BOUNCES, SEED, medium={"sigma_t": 4.0}, layer="sphere")

    def test_unknown_layer_raises(self):
        with self.assertRaises(ValueError):
            trace_path_cache(W, H, SPP, BOUNCES, SEED, layer="teapot")


class OwnershipMaskTests(unittest.TestCase):
    def setUp(self):
        self.sphere_mask = layer_ownership_mask(W, H, "sphere")
        self.box_mask = layer_ownership_mask(W, H, "box")

    def test_masks_are_disjoint_and_cover_every_pixel(self):
        self.assertFalse(np.any(self.sphere_mask & self.box_mask))
        self.assertTrue(np.all(self.sphere_mask | self.box_mask))

    def test_sphere_mask_is_a_nonempty_minority(self):
        n_sphere = int(self.sphere_mask.sum())
        self.assertGreater(n_sphere, 0, "the sphere must be visible")
        self.assertLess(n_sphere, W * H // 2, "the box fills most of the frame")

    def test_masks_agree_with_the_aux_gbuffer(self):
        # Ownership uses the same deterministic pixel-center primaries as the aux
        # buffers, so the sphere mask is exactly where aux albedo is the sphere's.
        cache = trace_path_cache(W, H, 1, 1, SEED)
        is_sphere_albedo = np.all(np.isclose(cache.albedo, SPHERE_ALBEDO), axis=2)
        np.testing.assert_array_equal(self.sphere_mask, is_sphere_albedo)

    def test_unknown_layer_raises(self):
        with self.assertRaises(ValueError):
            layer_ownership_mask(W, H, "teapot")


def tiny_layer_cfg(tmp: str, layer: str) -> dict:
    return {
        "cache": str(Path(tmp) / f"{layer}.npz"),
        "out_dir": str(Path(tmp) / layer),
        "trace": {"width": 12, "height": 12, "spp": 4, "bounces": 2, "seed": 2, "layer": layer},
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.08, "radius_max": 0.25},
        "sampling": "segments",
        "pool": {"size": 8, "replace_every": 5, "replace_count": 1},
        "denoise": {"enabled": True, "radius": 1},
        "iters": 40,
        "batch_pixels": 256,
        "lr": 0.005,
        "model": {
            "hidden_width": 16,
            "hidden_layers": 2,
            "encoding": {
                "levels": 2,
                "features_per_level": 2,
                "table_size_log2": 6,
                "base_resolution": 4,
                "finest_resolution": 12,
            },
        },
        "n_val_lights": 2,
        "seed": 0,
        "device": "cpu",
    }


class CompositeCLITests(unittest.TestCase):
    """End-to-end: train a tiny sphere-layer proxy (traced via the config's
    trace.layer key), hold a box-layer GATHERLIGHT image fixed, composite."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        tmp = cls._tmp.name
        train(tiny_layer_cfg(tmp, "sphere"))
        box_cache = trace_path_cache(12, 12, 4, 2, 2, layer="box")
        fixed_light = SphereLight(center=[0.7, 0.8, 0.5], radius=0.12, rgb=[3.0, 2.0, 1.0])
        cls.fixed_image = gather_light(box_cache, fixed_light)
        cls.fixed_path = Path(tmp) / "box_fixed.npy"
        np.save(cls.fixed_path, cls.fixed_image)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_cli_smoke_composites_edit_plus_fixed(self):
        tmp = self._tmp.name
        out = Path(tmp) / "composite.npy"
        argv = [
            "composite",
            "--edit-model",
            str(Path(tmp) / "sphere" / "model.pt"),
            "--edit-cache",
            str(Path(tmp) / "sphere.npz"),
            "--light",
            json.dumps(
                {"type": "sphere", "center": [0.3, 0.7, 0.4], "radius": 0.1, "rgb": [1, 2, 3]}
            ),
            "--fixed-image",
            str(self.fixed_path),
            "--out",
            str(out),
            "--bench",
            "2",
        ]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), redirect_stdout(buf):
            composite_cli.main()
        image = np.load(out)
        self.assertEqual(image.shape, (12, 12, 3))
        # The composite is the fixed layer plus a non-negative proxy term.
        self.assertTrue(np.all(image >= self.fixed_image - 1e-12))
        self.assertGreater(float(image.sum()), float(self.fixed_image.sum()))
        self.assertIn("ms/edit", buf.getvalue())

    def test_composite_rejects_mismatched_shapes(self):
        import torch

        from nrp.path_cache import PathCache
        from nrp.torch_backend.model import TorchNRP
        from nrp.torch_backend.relight_multiview import ViewProxy

        tmp = self._tmp.name
        proxy = ViewProxy(
            "sphere",
            TorchNRP.load(str(Path(tmp) / "sphere" / "model.pt")),
            PathCache.load(str(Path(tmp) / "sphere.npz")),
            torch.device("cpu"),
        )
        light = SphereLight(center=[0.3, 0.7, 0.4], radius=0.1, rgb=[1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
            composite(proxy, [light], np.zeros((8, 8, 3)))


if __name__ == "__main__":
    unittest.main()
