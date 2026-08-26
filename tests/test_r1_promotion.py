"""R1C/R1E promotion-gate helper contracts."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.r1_promotion import (  # noqa: E402
    aggregate_reports,
    out_of_bounds_fraction,
    percentile_bounds,
    promotion_gate,
    r1a_seed_rows,
    rotation_matrix_y,
    transform_cache,
)
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.train import configured_world_bounds  # noqa: E402


def tiny_cache() -> PathCache:
    return PathCache(
        width=2,
        height=1,
        n_paths=np.ones(2, dtype=np.int64),
        seg_pixel=np.array([0, 1], dtype=np.int64),
        seg_origin=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        seg_dir=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        seg_tmax=np.array([1.0, 2.0]),
        seg_throughput=np.ones((2, 3)),
        albedo=np.full((1, 2, 3), 0.5),
        position=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
        depth=np.ones((1, 2)),
        normal=np.array([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]),
    )


class R1PromotionHelperTests(unittest.TestCase):
    def test_rotation_matrix_is_a_proper_y_rotation(self):
        rotation = rotation_matrix_y(90.0)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        np.testing.assert_allclose(
            rotation @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-12
        )

    def test_transform_cache_rotates_world_geometry_and_preserves_shapes(self):
        transformed = transform_cache(tiny_cache(), rotation_matrix_y(90.0))
        np.testing.assert_allclose(transformed.seg_origin[1], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(transformed.seg_dir[0], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(transformed.position[0, 0], [0.0, 0.0, -1.0], atol=1e-12)
        np.testing.assert_allclose(transformed.normal[0, 1], [0.0, 1.0, 0.0])
        transformed.validate()

    def test_percentile_bounds_and_out_of_bounds_fraction(self):
        positions = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [100.0, 100.0, 100.0]]
        )
        bounds = percentile_bounds(positions, lower=0.0, upper=75.0)
        np.testing.assert_allclose(bounds["min"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(bounds["max"], [26.5, 26.5, 26.5])
        self.assertAlmostEqual(out_of_bounds_fraction(positions, bounds), 0.25)

    def test_promotion_gate_requires_r1a_r1c_and_r1e(self):
        passing = [{"seed": seed, "delta_db": -0.1} for seed in range(5)]
        failing = passing[:-1] + [{"seed": 4, "delta_db": -0.6}]
        result = promotion_gate(True, passing, passing)
        self.assertTrue(result["promoted"])
        self.assertTrue(result["r1c_pass"])
        self.assertTrue(result["r1e_pass"])
        self.assertFalse(promotion_gate(False, passing, passing)["promoted"])
        self.assertFalse(promotion_gate(True, passing, failing)["promoted"])
        self.assertFalse(promotion_gate(True, failing, passing)["promoted"])
        self.assertFalse(
            promotion_gate(True, passing, passing, r1c_complete=False)["promoted"]
        )

    def test_configured_world_bounds_override_cache_aabb(self):
        cache = tiny_cache()
        cfg = {"model": {"world_bounds": {"min": [-2, -3, -4], "max": [2, 3, 4]}}}
        self.assertEqual(configured_world_bounds(cache, cfg), cfg["model"]["world_bounds"])

    def test_r1a_seed_rows_preserve_the_paired_gate(self):
        report = {
            "comparisons": {
                "world_triplane": {
                    "target_scale": {
                        "per_seed": [
                            {"seed": 0, "summary": {"mean_db": -0.1}, "gate_pass": True},
                            {"seed": 1, "summary": {"mean_db": -0.6}, "gate_pass": False},
                        ]
                    }
                }
            }
        }
        rows = r1a_seed_rows(report)
        self.assertEqual([row["seed"] for row in rows], [0, 1])
        self.assertEqual([row["delta_db"] for row in rows], [-0.1, -0.6])
        self.assertEqual([row["gate_pass"] for row in rows], [True, False])

    def test_aggregate_report_does_not_promote_partial_r1c_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            r1a = {
                "gate": {"passing_world_anchored_arms": ["world_triplane/target_scale"]},
            }
            r1c = {
                "r1c": {
                    "runs": [
                        {
                            "seed": seed,
                            "rotation_degrees": 0.0,
                            "bounds_mode": "aabb",
                            "delta_db": -0.1,
                        }
                        for seed in range(5)
                    ]
                }
            }
            r1e = {
                "r1e": {"scene": "Bedroom", "runs": []},
                "promotion": {"r1e_complete": False},
            }
            for name, payload in (("r1a.json", r1a), ("r1c.json", r1c), ("r1e.json", r1e)):
                (root / name).write_text(json.dumps(payload))
            report = aggregate_reports(
                root / "r1a.json", root / "r1c.json", root / "r1e.json", root / "out.json", root
            )
        self.assertFalse(report["promotion"]["promoted"])
        self.assertFalse(report["promotion"]["r1c_complete"])


if __name__ == "__main__":
    unittest.main()
