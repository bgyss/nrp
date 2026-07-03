import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.time_conditioned_camera import camera_at, interpolate_frames  # noqa: E402
from nrp.toy_tracer import CAM_POS, trace_path_cache  # noqa: E402


class TimeConditionedCameraTests(unittest.TestCase):
    def test_interpolate_frames_endpoints_and_midpoint(self):
        times = np.array([0.0, 1.0])
        frames = np.stack(
            [
                np.zeros((2, 2, 3), dtype=np.float64),
                np.ones((2, 2, 3), dtype=np.float64) * 4.0,
            ],
            axis=0,
        )
        np.testing.assert_allclose(interpolate_frames(times, frames, -1.0), frames[0])
        np.testing.assert_allclose(interpolate_frames(times, frames, 2.0), frames[1])
        np.testing.assert_allclose(interpolate_frames(times, frames, 0.25), 1.0)

    def test_interpolate_frames_rejects_unsorted_times(self):
        with self.assertRaises(ValueError):
            interpolate_frames(np.array([0.0, 0.0]), np.zeros((2, 1, 1, 3)), 0.5)

    def test_camera_at_symmetric_offsets(self):
        np.testing.assert_allclose(camera_at(0.0, 0.04), CAM_POS + [-0.04, 0.0, 0.0])
        np.testing.assert_allclose(camera_at(0.5, 0.04), CAM_POS)
        np.testing.assert_allclose(camera_at(1.0, 0.04), CAM_POS + [0.04, 0.0, 0.0])

    def test_trace_path_cache_camera_override_changes_aux_but_stays_valid(self):
        base = trace_path_cache(8, 8, spp=2, max_bounces=1, seed=9)
        shifted = trace_path_cache(
            8,
            8,
            spp=2,
            max_bounces=1,
            seed=9,
            camera_pos=CAM_POS + np.array([0.04, 0.0, 0.0]),
        )
        self.assertEqual(base.segment_count, shifted.segment_count)
        self.assertGreater(float(np.mean(np.abs(base.position - shifted.position))), 1e-4)
        shifted.validate()


if __name__ == "__main__":
    unittest.main()
