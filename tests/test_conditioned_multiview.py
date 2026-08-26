"""R2 camera-conditioned multi-view training and inference contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.conditioned_multiview import (  # noqa: E402
    MultiViewImagePool,
    build_validation_sets,
    camera_direction,
    global_world_bounds,
    load_camera_manifest,
    validation_disjointness,
)


def tiny_cache(width: int = 2, offset: float = 0.0) -> PathCache:
    """A deterministic one-segment-per-pixel cache for R2 unit tests."""
    positions = np.stack(
        [
            np.array([float(i) + offset, 0.5 + 0.1 * i, 1.0 + offset + 0.05 * i])
            for i in range(width)
        ],
        axis=0,
    ).reshape(1, width, 3)
    return PathCache(
        width=width,
        height=1,
        n_paths=np.ones(width, dtype=np.int64),
        seg_pixel=np.arange(width, dtype=np.int64),
        seg_origin=np.stack(
            [np.array([float(i) + offset, 0.0, 0.0]) for i in range(width)], axis=0
        ),
        seg_dir=np.tile(np.array([0.0, 0.0, 1.0]), (width, 1)),
        seg_tmax=np.ones(width, dtype=np.float64),
        seg_throughput=np.tile(np.array([0.8, 0.6, 0.4]), (width, 1)),
        albedo=np.full((1, width, 3), 0.5 + offset * 0.01),
        position=positions,
        depth=np.full((1, width), 1.0 + offset),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, width, 1)),
    )


class ManifestTests(unittest.TestCase):
    def test_camera_manifest_resolves_paths_and_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_cache(offset=0.0).save(root / "front.npz")
            tiny_cache(offset=0.2).save(root / "side.npz")
            manifest = root / "views.json"
            manifest.write_text(
                json.dumps(
                    {
                        "views": [
                            {
                                "name": "front",
                                "cache": "front.npz",
                                "camera": {
                                    "origin": [0.0, 0.0, 2.0],
                                    "target": [0.0, 0.0, 0.0],
                                },
                            },
                            {
                                "name": "side",
                                "cache": "side.npz",
                                "camera": {
                                    "origin": [2.0, 0.0, 0.0],
                                    "target": [0.0, 0.0, 0.0],
                                },
                            },
                        ]
                    }
                )
            )
            views = load_camera_manifest(manifest)

        self.assertEqual([view.name for view in views], ["front", "side"])
        self.assertEqual(views[0].cache_path, (root / "front.npz").resolve())
        np.testing.assert_allclose(views[0].view_dir, [0.0, 0.0, -1.0])
        np.testing.assert_allclose(views[1].view_dir, [-1.0, 0.0, 0.0])

    def test_camera_direction_rejects_missing_and_zero_length_metadata(self):
        with self.assertRaisesRegex(ValueError, "origin"):
            camera_direction({"target": [0.0, 0.0, 0.0]})
        with self.assertRaisesRegex(ValueError, "non-zero"):
            camera_direction(
                {"origin": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0]}
            )

    def test_manifest_rejects_mixed_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tiny_cache(width=2).save(root / "small.npz")
            tiny_cache(width=3).save(root / "wide.npz")
            manifest = root / "views.json"
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "name": "small",
                            "cache": "small.npz",
                            "camera": {
                                "origin": [0.0, 0.0, 2.0],
                                "target": [0.0, 0.0, 0.0],
                            },
                        },
                        {
                            "name": "wide",
                            "cache": "wide.npz",
                            "camera": {
                                "origin": [2.0, 0.0, 0.0],
                                "target": [0.0, 0.0, 0.0],
                            },
                        },
                    ]
                )
            )
            with self.assertRaisesRegex(ValueError, "same resolution"):
                load_camera_manifest(manifest)

    def test_global_world_bounds_spans_all_caches(self):
        first = tiny_cache(width=2, offset=0.0)
        second = tiny_cache(width=2, offset=2.0)
        bounds = global_world_bounds([first, second])
        np.testing.assert_allclose(bounds["min"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(bounds["max"], [3.0, 0.6, 3.05])


class PoolTests(unittest.TestCase):
    def _config(self):
        return {
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.1, "radius_max": 0.2},
            "sampling": "segments",
            "pool": {"size": 3, "replace_count": 1},
            "denoise": {"enabled": False},
            "n_val_lights": 3,
        }

    def test_shared_pool_shapes_and_replacement(self):
        caches = [tiny_cache(offset=0.0), tiny_cache(offset=0.2)]
        pool = MultiViewImagePool(
            caches, self._config(), np.random.default_rng(7), torch.device("cpu")
        )
        self.assertEqual(tuple(pool.params.shape), (3, 4))
        self.assertEqual(tuple(pool.targets.shape), (2, 3, 2, 3))
        self.assertEqual(pool.supervision_images, 6)
        self.assertFalse(np.array_equal(pool.targets[0, 0].numpy(), pool.targets[1, 0].numpy()))
        before = pool.params[0].clone()
        pool.replace_round()
        self.assertFalse(torch.equal(before, pool.params[0]))
        self.assertEqual(pool.supervision_images, 8)

    def test_validation_sets_are_per_view_and_disjoint(self):
        caches = [tiny_cache(offset=0.0), tiny_cache(offset=0.2)]
        config = self._config()
        pool = MultiViewImagePool(
            caches, config, np.random.default_rng(11), torch.device("cpu")
        )
        validation = build_validation_sets(caches, config, seed=11)
        self.assertEqual(len(validation), 2)
        self.assertTrue(all(len(entries) == 3 for entries in validation))
        disjoint = validation_disjointness(pool.used_params, validation)
        self.assertEqual(disjoint, [True, True])
        duplicate = [[{"params": pool.used_params[0]}], []]
        self.assertEqual(validation_disjointness(pool.used_params, duplicate), [False, True])


if __name__ == "__main__":
    unittest.main()
