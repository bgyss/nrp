"""Look-at camera orientation for the toy tracer."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.toy_tracer import CAM_POS, _camera_rays, layer_ownership_mask  # noqa: E402


class TestBackwardCompatibility(unittest.TestCase):
    def test_default_is_bit_identical_to_the_committed_plus_z_camera(self):
        # Committed toy caches and their tests depend on this exact ray set. The
        # pre-lookat formula is spelled out here literally (not read off any helper
        # in toy_tracer.py), so a change to the lookat code path -- even one that
        # still happens to leave a center ray with z > 0.9 -- would be caught by
        # comparing every ray's direction, not just one pixel's z-component.
        width = height = 8
        fov_deg = 68.0
        ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        px = xs.reshape(-1).astype(np.float64)
        py = ys.reshape(-1).astype(np.float64)
        half = np.tan(np.radians(fov_deg) / 2.0)
        u = ((px + 0.5) / width * 2.0 - 1.0) * half
        v = -((py + 0.5) / height * 2.0 - 1.0) * half * (height / width)
        expected_dirs = np.stack([u, v, np.ones_like(u)], axis=1)
        expected_dirs /= np.linalg.norm(expected_dirs, axis=1, keepdims=True)

        origins, dirs = _camera_rays(width, height, None)
        self.assertTrue(np.allclose(origins, CAM_POS))
        np.testing.assert_allclose(dirs, expected_dirs, atol=1e-12)

    def test_explicit_target_along_plus_z_reproduces_the_default(self):
        a_o, a_d = _camera_rays(8, 8, None)
        b_o, b_d = _camera_rays(
            8, 8, None, camera_pos=CAM_POS, camera_target=CAM_POS + np.array([0.0, 0.0, 1.0])
        )
        np.testing.assert_allclose(a_o, b_o, atol=1e-12)
        np.testing.assert_allclose(a_d, b_d, atol=1e-12)


class TestLookAt(unittest.TestCase):
    def test_centre_ray_points_at_the_target(self):
        pos = np.array([0.5, 0.5, 0.1])
        target = np.array([0.9, 0.5, 0.9])
        _, dirs = _camera_rays(9, 9, None, camera_pos=pos, camera_target=target)
        centre = dirs[9 * 4 + 4]
        want = (target - pos) / np.linalg.norm(target - pos)
        np.testing.assert_allclose(centre, want, atol=2e-2)

    def test_all_directions_are_unit_length(self):
        _, dirs = _camera_rays(
            6,
            6,
            None,
            camera_pos=np.array([0.2, 0.5, 0.2]),
            camera_target=np.array([0.8, 0.5, 0.8]),
        )
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)

    def test_rejects_degenerate_target(self):
        pos = np.array([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            _camera_rays(4, 4, None, camera_pos=pos, camera_target=pos)

    def test_rejects_target_parallel_to_up_axis(self):
        with self.assertRaises(ValueError):
            _camera_rays(
                4,
                4,
                None,
                camera_pos=np.array([0.5, 0.1, 0.5]),
                camera_target=np.array([0.5, 0.9, 0.5]),
            )


class TestLayerOwnershipMaskCameraBug(unittest.TestCase):
    def test_mask_changes_when_the_camera_moves(self):
        # Before the fix, layer_ownership_mask always used the default camera,
        # so a trace with --camera-pos/--camera-target got a mask computed for
        # a different camera than the one that produced the segments.
        default_mask = layer_ownership_mask(16, 16, "sphere")
        moved_mask = layer_ownership_mask(
            16,
            16,
            "sphere",
            camera_pos=np.array([0.85, 0.85, 0.08]),
            camera_target=np.array([0.35, 0.28, 0.62]),
        )
        self.assertFalse(np.array_equal(default_mask, moved_mask))


if __name__ == "__main__":
    unittest.main()
