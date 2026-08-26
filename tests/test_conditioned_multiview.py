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

from examples.r2_conditioned import camera_pairs_present, quality_gate  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.conditioned_multiview import (  # noqa: E402
    MultiViewImagePool,
    build_validation_sets,
    camera_direction,
    global_world_bounds,
    load_camera_manifest,
    train_conditioned,
    validation_disjointness,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight_multiview import (  # noqa: E402
    conditioned_edit_latency_ms,
    load_conditioned_views,
    relight_conditioned_all,
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
            camera_direction({"origin": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0]})

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
        pool = MultiViewImagePool(caches, config, np.random.default_rng(11), torch.device("cpu"))
        validation = build_validation_sets(caches, config, seed=11)
        self.assertEqual(len(validation), 2)
        self.assertTrue(all(len(entries) == 3 for entries in validation))
        disjoint = validation_disjointness(pool.used_params, validation)
        self.assertEqual(disjoint, [True, True])
        duplicate = [[{"params": pool.used_params[0]}], []]
        self.assertEqual(validation_disjointness(pool.used_params, duplicate), [False, True])


class TrainingTests(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        tiny_cache(width=2, offset=0.0).save(root / "front.npz")
        tiny_cache(width=2, offset=0.2).save(root / "side.npz")
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
        return manifest

    def test_train_conditioned_two_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "conditioned"
            report = train_conditioned(
                {
                    "manifest": str(self._manifest(root)),
                    "out_dir": str(out_dir),
                    "light_type": "sphere",
                    "light_bounds": {"radius_min": 0.1, "radius_max": 0.2},
                    "sampling": "segments",
                    "pool": {"size": 3, "replace_every": 2, "replace_count": 1},
                    "denoise": {"enabled": False},
                    "iters": 6,
                    "batch_pixels": 16,
                    "lr": 0.005,
                    "model": {
                        "camera_conditioned": True,
                        "spatial_encoding": "world3d",
                        "hidden_width": 8,
                        "hidden_layers": 1,
                        "encoding": {
                            "levels": 1,
                            "features_per_level": 2,
                            "finest_resolution": 4,
                        },
                    },
                    "n_val_lights": 2,
                    "seed": 7,
                    "device": "cpu",
                }
            )
            self.assertEqual(report["view_count"], 2)
            self.assertEqual(len(report["views"]), 2)
            self.assertEqual(report["validation_disjoint_by_view"], [True, True])
            self.assertEqual(len(report["shared_training_light_params"]), 6)
            self.assertTrue(np.isfinite(report["loss_first"]))
            self.assertTrue(np.isfinite(report["loss_last"]))
            self.assertTrue(
                all(np.isfinite(row["val_psnr_db_vs_raw_mean"]) for row in report["views"])
            )
            self.assertTrue((out_dir / "model.pt").exists())
            self.assertTrue((out_dir / "conditioned_train_report.json").exists())


class OccupancyScheduleRegressionTests(unittest.TestCase):
    """Guard the schedule-mismatch bug found in review of 25f41aa.

    `train_conditioned` used to build the occupancy resolution schedule with its
    own hard-coded fallback defaults instead of the encoder class's actual
    defaults. `world3d` with `allocation: "occupancy"` and no explicit
    `finest_resolution` diverged (occupancy built at 128, `HashEncoding3D`
    defaults to 256) and construction raised `ValueError`. Both arms below must
    train with an otherwise-empty `encoding` config.
    """

    def _manifest(self, root: Path) -> Path:
        tiny_cache(width=2, offset=0.0).save(root / "front.npz")
        tiny_cache(width=2, offset=0.2).save(root / "side.npz")
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
        return manifest

    def _base_config(self, root: Path, spatial_encoding: str, encoding_cfg: dict) -> dict:
        return {
            "manifest": str(self._manifest(root)),
            "out_dir": str(root / "conditioned"),
            "light_type": "sphere",
            "light_bounds": {"radius_min": 0.1, "radius_max": 0.2},
            "sampling": "segments",
            "pool": {"size": 2, "replace_every": 2, "replace_count": 1},
            "denoise": {"enabled": False},
            "iters": 1,
            "batch_pixels": 8,
            "lr": 0.005,
            "model": {
                "camera_conditioned": True,
                "spatial_encoding": spatial_encoding,
                "hidden_width": 4,
                "hidden_layers": 1,
                "encoding": encoding_cfg,
            },
            "n_val_lights": 1,
            "seed": 3,
            "device": "cpu",
        }

    def test_world3d_occupancy_allocation_trains_with_empty_encoding_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = train_conditioned(
                self._base_config(root, "world3d", {"allocation": "occupancy"})
            )
            self.assertEqual(report["view_count"], 2)
            self.assertTrue(np.isfinite(report["loss_last"]))

    def test_world_sparse_trains_with_empty_encoding_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = train_conditioned(self._base_config(root, "world_sparse", {}))
            self.assertEqual(report["view_count"], 2)
            self.assertTrue(np.isfinite(report["loss_last"]))


class InferenceTests(unittest.TestCase):
    def _manifest_and_model(self, root: Path) -> tuple[Path, Path]:
        tiny_cache(width=2, offset=0.0).save(root / "front.npz")
        tiny_cache(width=2, offset=0.2).save(root / "side.npz")
        manifest = root / "views.json"
        manifest.write_text(
            json.dumps(
                [
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
            )
        )
        model = TorchNRP(
            hidden_width=8,
            hidden_layers=1,
            encoding={"levels": 1, "features_per_level": 2, "finest_resolution": 4},
            spatial_encoding="world3d",
            world_bounds=global_world_bounds(
                [tiny_cache(width=2, offset=0.0), tiny_cache(width=2, offset=0.2)]
            ),
            camera_conditioned=True,
        )
        model_path = root / "model.pt"
        model.save(model_path)
        return manifest, model_path

    def test_shared_conditioned_relight_uses_one_model_and_resident_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest, model_path = self._manifest_and_model(Path(tmp))
            model, views = load_conditioned_views(manifest, model_path, device="cpu")
            self.assertTrue(all(view.spatial.shape == (2, 3) for view in views))
            self.assertIs(model, views[0].model)
            self.assertIs(model, views[1].model)
            lights = [SphereLight(center=[0.2, 0.0, 0.5], radius=0.2)]
            before = relight_conditioned_all(model, views, lights)
            for view in views:
                view.cache.seg_origin[:] = 999.0
            after = relight_conditioned_all(model, views, lights)
            for name in before:
                np.testing.assert_allclose(before[name], after[name])
            latency = conditioned_edit_latency_ms(model, views[:2], lights, frames=2, warmup=1)
            self.assertTrue(np.isfinite(latency))
            self.assertGreater(latency, 0.0)


class ReportTests(unittest.TestCase):
    def test_camera_pair_check_requires_origin_and_target(self):
        self.assertTrue(camera_pairs_present([{"origin": [0, 0, 1], "target": [0, 0, 0]}]))
        self.assertFalse(camera_pairs_present([{"origin": [0, 0, 1]}]))

    def test_quality_gate_reports_per_view_deltas(self):
        rows = [
            {"view": "front", "baseline_psnr_db": 20.0, "conditioned_psnr_db": 19.4},
            {"view": "side", "baseline_psnr_db": 21.0, "conditioned_psnr_db": 20.5},
        ]
        result = quality_gate(rows, tolerance_db=1.0)
        self.assertTrue(result["passed"])
        self.assertEqual(result["per_view"][0]["delta_db"], -0.6)

    def test_quality_gate_fails_when_one_view_exceeds_tolerance(self):
        result = quality_gate(
            [{"view": "front", "baseline_psnr_db": 20.0, "conditioned_psnr_db": 18.99}],
            tolerance_db=1.0,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["per_view"][0]["within_tolerance"], False)


if __name__ == "__main__":
    unittest.main()
