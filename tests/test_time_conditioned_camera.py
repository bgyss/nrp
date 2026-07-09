import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.time_conditioned_camera import camera_at, interpolate_frames  # noqa: E402
from examples.time_conditioned_proxy import _aux, _light_time_params, _pixel_xy  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
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

    def test_light_time_params_appends_time_scalar(self):
        light = SphereLight(center=np.array([0.1, 0.2, 0.3]), radius=0.05)
        params = _light_time_params(light, 0.75, n=4)
        self.assertEqual(params.shape, (4, 5))
        np.testing.assert_allclose(params[:, :3], np.tile(light.center, (4, 1)), atol=1e-6)
        np.testing.assert_allclose(params[:, 3], light.radius)
        np.testing.assert_allclose(params[:, 4], 0.75)

    def test_pixel_xy_and_aux_shapes(self):
        cache = trace_path_cache(4, 3, spp=2, max_bounces=1, seed=1)
        xy = _pixel_xy(4, 3)
        aux = _aux(cache)
        self.assertEqual(xy.shape, (12, 2))
        self.assertEqual(aux.shape, (12, 7))
        self.assertTrue(np.all((xy >= 0.0) & (xy <= 1.0)))


if __name__ == "__main__":
    unittest.main()
