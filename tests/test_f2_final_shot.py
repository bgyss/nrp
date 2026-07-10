import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.f2_final_shot import (  # noqa: E402
    encode_mp4,
    render_final_shot,
    store_residual,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None


def toy_shot_spec(frames=3):
    return {
        "frames": frames,
        "interpolation": "linear",
        "lights": [
            {
                "keyframes": [
                    {
                        "time": 0.0,
                        "light": {
                            "type": "sphere",
                            "center": [-0.3, 0.6, 0.0],
                            "radius": 0.2,
                            "rgb": [1.2, 1.0, 0.8],
                        },
                    },
                    {
                        "time": 1.0,
                        "light": {
                            "type": "sphere",
                            "center": [0.3, 0.6, 0.0],
                            "radius": 0.2,
                            "rgb": [0.8, 1.0, 1.2],
                        },
                    },
                ]
            }
        ],
    }


class StoreResidualTests(unittest.TestCase):
    def test_round_trip_is_fp16_exact(self):
        residual = np.linspace(-0.5, 0.5, 48).reshape(4, 4, 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.npz"
            nbytes = store_residual(path, residual)
            self.assertEqual(nbytes, path.stat().st_size)
            loaded = np.load(path)["residual"].astype(np.float64)
        np.testing.assert_allclose(loaded, residual.astype(np.float16).astype(np.float64))


class RenderFinalShotTests(unittest.TestCase):
    def _run(self, tmp, encode=False):
        cache = trace_path_cache(12, 12, 4, 2, seed=3)
        model = TorchNRP(
            hidden_width=16,
            hidden_layers=2,
            encoding={"levels": 2, "features_per_level": 2, "finest_resolution": 12},
        )
        return render_final_shot(
            model,
            cache,
            toy_shot_spec(),
            Path(tmp) / "shot",
            denoise_method="bilateral",
            encode=encode,
        )

    def test_residual_identity_and_final_gate_per_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run(tmp)
            out_dir = Path(tmp) / "shot"
            self.assertTrue((out_dir / "report.json").exists())
            on_disk = json.loads((out_dir / "report.json").read_text())
            self.assertEqual(on_disk["frames"], 3)
            for idx in range(report["frames"]):
                self.assertTrue((out_dir / "residuals" / f"frame_{idx:04d}.npz").exists())
        self.assertEqual(report["rung"], "F2")
        self.assertEqual(report["frames"], 3)
        for row in report["per_frame"]:
            # float64 identity is exact by construction up to rounding (one ulp)
            self.assertLessEqual(row["exact_identity_max_abs"], 1e-12)
            # fp16-stored reconstruction stays within the stated tolerance
            self.assertTrue(row["stored_identity_within_tolerance"])
            # fp16 quantization error clears the final tier
            self.assertTrue(row["quality_gate"]["passed"], row.get("flag"))
            self.assertIsNone(row["flag"])
        self.assertTrue(report["all_frames_pass_final_gate"])
        storage = report["storage"]
        self.assertGreater(storage["model_bytes"], 0)
        self.assertGreater(storage["residual_bytes_total"], 0)
        self.assertGreater(storage["raw_frames_bytes_total"], 0)
        self.assertAlmostEqual(
            storage["proxy_plus_residuals_bytes"],
            storage["model_bytes"] + storage["residual_bytes_total"],
        )
        self.assertIn("shot_total_seconds", report["wall_clock"])

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not on PATH")
    def test_mp4_encoded(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self._run(tmp, encode=True)
            mp4 = Path(tmp) / "shot" / "shot.mp4"
            self.assertTrue(mp4.exists())
            self.assertGreater(report["storage"]["mp4_bytes"], 0)

    @unittest.skipUnless(HAVE_FFMPEG, "ffmpeg not on PATH")
    def test_encode_mp4_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb = Path(tmp) / "frames.rgb"
            frames = (np.random.default_rng(0).random((3, 16, 16, 3)) * 255).astype(np.uint8)
            rgb.write_bytes(frames.tobytes())
            out = Path(tmp) / "clip.mp4"
            encode_mp4(rgb, 16, 16, 24, out)
            self.assertGreater(out.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
