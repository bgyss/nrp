"""Arm C: each point reads only the plane aligned with its surface normal."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import NormalAwareTriPlane  # noqa: E402

CONFIG = {
    "levels": 2,
    "features_per_level": 2,
    "table_size_log2": 8,
    "base_resolution": 4,
    "finest_resolution": 16,
}


class TestPlaneSelection(unittest.TestCase):
    def test_output_dim_is_one_plane_not_three(self):
        enc = NormalAwareTriPlane(**CONFIG)
        self.assertEqual(enc.output_dim, CONFIG["levels"] * CONFIG["features_per_level"])

    def test_axis_aligned_normal_selects_the_expected_plane(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with torch.no_grad():
            for i, plane in enumerate(enc.planes):
                for table in plane.tables:
                    table.fill_(float(i + 1))
        xyz = torch.tensor([[0.5, 0.5, 0.5]])
        for axis in range(3):
            normal = torch.zeros(1, 3)
            normal[0, axis] = 1.0
            out = enc(xyz, normal)
            self.assertAlmostEqual(float(out[0, 0]), float(axis + 1), places=5)

    def test_sign_of_normal_does_not_change_plane_choice(self):
        enc = NormalAwareTriPlane(**CONFIG)
        xyz = torch.tensor([[0.3, 0.4, 0.5]])
        up = enc(xyz, torch.tensor([[0.0, 1.0, 0.0]]))
        down = enc(xyz, torch.tensor([[0.0, -1.0, 0.0]]))
        torch.testing.assert_close(up, down)

    def test_batch_with_mixed_normals_matches_per_row_evaluation(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with torch.no_grad():
            for plane in enc.planes:
                for table in plane.tables:
                    table.uniform_(-1.0, 1.0)
        xyz = torch.rand(6, 3)
        normals = torch.eye(3).repeat(2, 1)
        batched = enc(xyz, normals)
        for i in range(6):
            single = enc(xyz[i : i + 1], normals[i : i + 1])
            torch.testing.assert_close(batched[i : i + 1], single)

    def test_axis_to_plane_projects_the_expected_coordinate_pair(self):
        # Constant-filled tables (as above) make output independent of which two
        # coordinates are projected, so they cannot catch a wrong AXIS_TO_PLANE
        # entry. Use position-varying tables and compare against a direct call
        # into the single plane with the coordinate pair the geometry requires
        # (spelled out here, not read off enc.AXIS_TO_PLANE), so a wrong pair
        # -- even an in-range permutation -- projects a different point and
        # changes the output.
        expected_pairs = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        enc = NormalAwareTriPlane(**CONFIG)
        with torch.no_grad():
            for plane in enc.planes:
                for table in plane.tables:
                    table.uniform_(-1.0, 1.0)
        xyz = torch.tensor([[0.15, 0.55, 0.85]])
        for axis in range(3):
            normal = torch.zeros(1, 3)
            normal[0, axis] = 1.0
            out = enc(xyz, normal)
            u, v = expected_pairs[axis]
            expected = enc.planes[axis](xyz[:, (u, v)])
            torch.testing.assert_close(out, expected)


class TestGradients(unittest.TestCase):
    def test_gradients_flow_to_the_selected_plane(self):
        enc = NormalAwareTriPlane(**CONFIG)
        xyz = torch.rand(8, 3)
        normals = torch.zeros(8, 3)
        normals[:, 1] = 1.0
        enc(xyz, normals).sum().backward()
        selected = [t.grad for t in enc.planes[1].tables if t.grad is not None]
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in selected))


class TestValidation(unittest.TestCase):
    def test_missing_normals_raises(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with self.assertRaises(ValueError):
            enc(torch.rand(4, 3), None)

    def test_zero_normal_raises(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with self.assertRaises(ValueError):
            enc(torch.rand(1, 3), torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
