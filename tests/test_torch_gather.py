"""Batched torch GATHERLIGHT parity with the numpy reference (roadmap item 3).

The numpy gather stays authoritative; the torch gather must match it allclose
(rtol 1e-5) for 50 random sphere and 50 random quad lights per cache, on the toy
cache and (when the extra is installed) a Mitsuba-exported cache.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import QuadLight, SphereLight  # noqa: E402
from nrp.torch_backend.gather import TorchPathCache  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

HAVE_MITSUBA = importlib.util.find_spec("mitsuba") is not None
HAVE_MPS = torch.backends.mps.is_available()


def random_lights(rng: np.random.Generator, n: int, scale: float, offset: float):
    """n sphere + n quad lights spanning roughly [offset, offset+scale]^3."""
    lights = []
    for _ in range(n):
        lights.append(
            SphereLight(
                center=offset + scale * rng.random(3),
                radius=float(scale * rng.uniform(0.05, 0.3)),
                rgb=rng.uniform(0.5, 2.0, 3),
            )
        )
        lights.append(
            QuadLight(
                center=offset + scale * rng.random(3),
                normal=rng.normal(size=3),
                width=float(scale * rng.uniform(0.1, 0.5)),
                height=float(scale * rng.uniform(0.1, 0.5)),
                rgb=rng.uniform(0.5, 2.0, 3),
            )
        )
    return lights


class ToyCacheParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(24, 24, spp=12, max_bounces=3, seed=2)
        cls.torch_cache = TorchPathCache(cls.cache, torch.device("cpu"))

    def test_matches_numpy_for_50_sphere_and_50_quad_lights(self):
        rng = np.random.default_rng(0)
        for light in random_lights(rng, 50, scale=0.8, offset=0.1):
            with self.subTest(light=type(light).__name__):
                ref = gather_light(self.cache, light)
                got = self.torch_cache.gather_light(light).numpy()
                np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)

    def test_volume_cache_and_empty_cache(self):
        vol = trace_path_cache(
            12, 12, spp=8, max_bounces=3, seed=4, medium={"sigma_t": 2.0, "albedo": 0.8}
        )
        tvol = TorchPathCache(vol, torch.device("cpu"))
        light = SphereLight(center=[0.5, 0.5, 0.5], radius=0.2, rgb=[1.0, 1.0, 1.0])
        np.testing.assert_allclose(
            tvol.gather_light(light).numpy(), gather_light(vol, light), rtol=1e-5
        )
        vol.seg_pixel = vol.seg_pixel[:0]
        vol.seg_origin = vol.seg_origin[:0]
        vol.seg_dir = vol.seg_dir[:0]
        vol.seg_tmax = vol.seg_tmax[:0]
        vol.seg_throughput = vol.seg_throughput[:0]
        tempty = TorchPathCache(vol, torch.device("cpu"))
        self.assertEqual(float(tempty.gather_light(light).abs().sum()), 0.0)

    @unittest.skipUnless(HAVE_MPS, "MPS not available")
    def test_mps_float32_close_to_numpy(self):
        # fp32 on MPS: boundary-grazing segments may round differently, so this is a
        # tolerance check on aggregate error, not exact parity.
        tc = TorchPathCache(self.cache, torch.device("mps"))
        rng = np.random.default_rng(1)
        for light in random_lights(rng, 5, scale=0.8, offset=0.1):
            ref = gather_light(self.cache, light)
            got = tc.gather_light(light).cpu().numpy()
            rel_l1 = np.abs(got - ref).sum() / max(ref.sum(), 1e-9)
            self.assertLess(rel_l1, 1e-4, f"{type(light).__name__}: rel L1 {rel_l1}")


@unittest.skipUnless(HAVE_MITSUBA, "mitsuba extra not installed")
class MitsubaCacheParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nrp.mitsuba_exporter import _load_mitsuba, _load_scene, export_path_cache

        mi = _load_mitsuba()
        scene = _load_scene(mi, "builtin:cornell-box", 8, 8)
        cls.cache = export_path_cache(scene, mi, 8, 8, spp=8, max_bounces=4, seed=1)
        cls.torch_cache = TorchPathCache(cls.cache, torch.device("cpu"))

    def test_matches_numpy_for_50_sphere_and_50_quad_lights(self):
        # The Mitsuba cornell box spans roughly [-1,1]^3 and has escape segments
        # (t_max = inf), which must flow through the same comparisons as in numpy.
        rng = np.random.default_rng(0)
        for light in random_lights(rng, 50, scale=2.0, offset=-1.0):
            with self.subTest(light=type(light).__name__):
                ref = gather_light(self.cache, light)
                got = self.torch_cache.gather_light(light).numpy()
                np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)


class PoolBackendEquivalenceTests(unittest.TestCase):
    def test_pool_targets_identical_across_gather_backends(self):
        # Same seed, denoise off: pool params and targets must agree between the
        # numpy and torch gather backends (fp32 storage in both).
        from nrp.torch_backend.train import ImagePool

        cache = trace_path_cache(12, 12, spp=8, max_bounces=2, seed=3)
        base = {
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
            "sampling": "segments",
            "pool": {"size": 6, "replace_every": 5, "replace_count": 1},
            "denoise": {"enabled": False},
        }
        pools = {}
        for backend in ("numpy", "torch"):
            cfg = {**base, "gather_backend": backend}
            pools[backend] = ImagePool(cache, cfg, np.random.default_rng(0), torch.device("cpu"))
        torch.testing.assert_close(pools["numpy"].params, pools["torch"].params)
        torch.testing.assert_close(pools["numpy"].targets, pools["torch"].targets)


if __name__ == "__main__":
    unittest.main()
