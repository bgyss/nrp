import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.animate import (  # noqa: E402
    frame_times,
    interpolate_light_spec,
    lights_at,
    mean_frame_delta,
    render_sequence,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


class AnimatedLightTests(unittest.TestCase):
    def test_frame_times_include_endpoints(self):
        np.testing.assert_allclose(frame_times(1), [0.0])
        np.testing.assert_allclose(frame_times(3), [0.0, 0.5, 1.0])

    def test_interpolates_numeric_light_fields(self):
        spec = [
            {
                "time": 0.0,
                "light": {
                    "type": "sphere",
                    "center": [0.0, 1.0, 2.0],
                    "radius": 0.1,
                    "rgb": [1.0, 0.0, 0.0],
                },
            },
            {
                "time": 1.0,
                "light": {
                    "type": "sphere",
                    "center": [2.0, 3.0, 4.0],
                    "radius": 0.3,
                    "rgb": [0.0, 0.0, 1.0],
                },
            },
        ]
        out = interpolate_light_spec(spec, 0.25)
        self.assertEqual(out["type"], "sphere")
        np.testing.assert_allclose(out["center"], [0.5, 1.5, 2.5])
        self.assertAlmostEqual(out["radius"], 0.15)
        np.testing.assert_allclose(out["rgb"], [0.75, 0.0, 0.25])

    def test_mean_frame_delta(self):
        a = np.zeros((2, 2, 3))
        b = np.ones((2, 2, 3))
        c = np.ones((2, 2, 3)) * 3.0
        self.assertAlmostEqual(mean_frame_delta([a, b, c]), 1.5)

    def test_rendered_frame_matches_direct_relight(self):
        cache = trace_path_cache(4, 4, spp=1, max_bounces=1, seed=2)
        model = TorchNRP(
            hidden_width=8,
            hidden_layers=1,
            encoding={"levels": 1, "finest_resolution": 4},
        )
        spec = {
            "frames": 3,
            "latency_frame_counts": [1, 3],
            "lights": [
                {
                    "keyframes": [
                        {
                            "time": 0.0,
                            "light": {
                                "type": "sphere",
                                "center": [-0.2, 0.4, 0.0],
                                "radius": 0.15,
                                "rgb": [1.0, 1.0, 1.0],
                            },
                        },
                        {
                            "time": 1.0,
                            "light": {
                                "type": "sphere",
                                "center": [0.2, 0.4, 0.0],
                                "radius": 0.25,
                                "rgb": [2.0, 0.5, 1.0],
                            },
                        },
                    ]
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            report = render_sequence(model, cache, spec, Path(tmp), measure_reference=True)
            frame = np.load(Path(tmp) / "frame_0001.npy")
            with open(Path(tmp) / "report.json") as f:
                written = json.load(f)
        direct = relight(model, cache, lights_at(spec, 0.5))
        np.testing.assert_array_equal(frame, direct)
        self.assertEqual(report["cache_access"], "aux_features_only_no_gatherlight")
        self.assertEqual([r["frames"] for r in report["latency_vs_frame_count"]], [1, 3])
        self.assertIn("reference_mean_frame_delta", report)
        self.assertIn("proxy_vs_reference_delta_ratio", report)
        self.assertEqual(written["frames"], 3)


if __name__ == "__main__":
    unittest.main()
