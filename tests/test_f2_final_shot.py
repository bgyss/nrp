import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.f2_final_shot import (  # noqa: E402
    dequantize_residual_int8,
    encode_mp4,
    load_residual,
    quantize_residual_int8,
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


class Int8QuantizationTests(unittest.TestCase):
    def test_round_trip_within_one_quantum(self):
        rng = np.random.default_rng(0)
        residual = rng.uniform(-0.4, 0.4, size=(6, 6, 3))
        q, scale = quantize_residual_int8(residual)
        self.assertEqual(q.dtype, np.int8)
        back = dequantize_residual_int8(q, scale)
        quantum = scale / 127.0
        np.testing.assert_allclose(back, residual, atol=quantum + 1e-9)

    def test_zero_residual_has_nonzero_scale_and_round_trips_to_zero(self):
        residual = np.zeros((3, 3, 3))
        q, scale = quantize_residual_int8(residual)
        self.assertGreater(scale, 0.0)
        np.testing.assert_allclose(dequantize_residual_int8(q, scale), residual)

    def test_store_residual_int8_is_smaller_than_fp16(self):
        rng = np.random.default_rng(1)
        residual = rng.uniform(-0.1, 0.1, size=(32, 32, 3))
        with tempfile.TemporaryDirectory() as tmp:
            fp16_path = Path(tmp) / "fp16.npz"
            int8_path = Path(tmp) / "int8.npz"
            fp16_bytes = store_residual(fp16_path, residual, "fp16")
            int8_bytes = store_residual(int8_path, residual, "int8")
            self.assertLess(int8_bytes, fp16_bytes)
            loaded = load_residual(int8_path, "int8")
        np.testing.assert_allclose(loaded, residual, atol=residual.max() / 127.0 + 1e-9)

    def test_unknown_precision_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "r.npz"
            with self.assertRaises(ValueError):
                store_residual(path, np.zeros((2, 2, 3)), "fp8")


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

    def test_approval_frames_only_stores_residual_for_named_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = trace_path_cache(12, 12, 4, 2, seed=3)
            model = TorchNRP(
                hidden_width=16,
                hidden_layers=2,
                encoding={"levels": 2, "features_per_level": 2, "finest_resolution": 12},
            )
            report = render_final_shot(
                model,
                cache,
                toy_shot_spec(frames=4),
                Path(tmp) / "shot",
                denoise_method="bilateral",
                encode=False,
                approval_frames={0, 3},
            )
        rows = {row["index"]: row for row in report["per_frame"]}
        self.assertEqual(report["n_approval_frames"], 2)
        self.assertEqual(report["n_proxy_only_frames"], 2)
        for idx in (0, 3):
            self.assertTrue(rows[idx]["is_approval_frame"])
            self.assertGreater(rows[idx]["residual_bytes"], 0)
            self.assertEqual(rows[idx]["gate_tier"], "final")
        for idx in (1, 2):
            self.assertFalse(rows[idx]["is_approval_frame"])
            self.assertEqual(rows[idx]["residual_bytes"], 0)
            self.assertEqual(rows[idx]["gate_tier"], "preview")
            self.assertIsNone(rows[idx]["stored_identity_max_abs"])
        # approval frames still hit exact residual-identity reconstruction and pass
        # final tier regardless of proxy quality (structural, by construction) --
        # this untrained toy model just isn't good enough for proxy-only frames to
        # clear even preview tier, which is exactly the quality/storage tradeoff H6
        # documents, not a bug in the gating mechanism itself
        for idx in (0, 3):
            self.assertTrue(rows[idx]["quality_gate"]["passed"], rows[idx]["flag"])
        # proxy-only frames contribute nothing to residual storage
        self.assertLess(
            report["storage"]["residual_bytes_total"],
            report["storage"]["raw_frames_bytes_total"],
        )

    def test_int8_precision_plumbed_into_storage_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = trace_path_cache(12, 12, 4, 2, seed=3)
            model = TorchNRP(
                hidden_width=16,
                hidden_layers=2,
                encoding={"levels": 2, "features_per_level": 2, "finest_resolution": 12},
            )
            report_fp16 = render_final_shot(
                model,
                cache,
                toy_shot_spec(),
                Path(tmp) / "fp16",
                denoise_method="bilateral",
                encode=False,
                residual_precision="fp16",
            )
            report_int8 = render_final_shot(
                model,
                cache,
                toy_shot_spec(),
                Path(tmp) / "int8",
                denoise_method="bilateral",
                encode=False,
                residual_precision="int8",
            )
        self.assertLess(
            report_int8["storage"]["residual_bytes_total"],
            report_fp16["storage"]["residual_bytes_total"],
        )

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
