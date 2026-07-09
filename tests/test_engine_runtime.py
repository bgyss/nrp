import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.engine_runtime import runtime_resolution_sweep, synthetic_runtime_cache  # noqa: E402
from nrp.lights import QuadLight, SphereLight  # noqa: E402
from nrp.torch_backend.engine_runtime import (  # noqa: E402
    export_artifact,
    load_runtime,
    runtime_relight,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


class EngineRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache = trace_path_cache(6, 5, 2, 1, seed=12)

    def check_runtime_parity(self, model, lights):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = str(Path(tmp) / "runtime.pt")
            metadata = export_artifact(model, artifact)
            runtime = load_runtime(artifact)
            exported = runtime_relight(runtime, self.cache, lights)
        direct = relight(model, self.cache, lights)
        np.testing.assert_allclose(exported, direct, rtol=1e-4, atol=1e-6)
        self.assertEqual(metadata["format"], "torchscript_nrp_runtime")
        self.assertGreater(metadata["artifact_bytes"], 0)

    def test_sphere_runtime_matches_module(self):
        model = TorchNRP(
            light_type="sphere",
            hidden_width=8,
            hidden_layers=1,
            encoding={"levels": 1, "finest_resolution": 6},
        )
        self.check_runtime_parity(
            model,
            [SphereLight(center=[0.0, 0.55, 0.0], radius=0.2, rgb=[1.0, 0.8, 0.6])],
        )

    def test_quad_runtime_matches_module(self):
        model = TorchNRP(
            light_type="quad",
            hidden_width=8,
            hidden_layers=1,
            encoding={"levels": 1, "finest_resolution": 6},
        )
        self.check_runtime_parity(
            model,
            [
                QuadLight(
                    center=[0.0, 0.5, 0.0],
                    normal=[0.0, -1.0, 0.0],
                    width=0.5,
                    height=0.25,
                    rgb=[0.8, 1.0, 1.2],
                )
            ],
        )

    def test_synthetic_runtime_cache_is_valid_and_segment_free(self):
        cache = synthetic_runtime_cache(9, 7)
        self.assertEqual(cache.segment_count, 0)
        self.assertEqual(cache.n_paths.shape, (63,))
        cache.validate()

    def test_resolution_sweep_records_device_availability(self):
        model = TorchNRP(
            light_type="sphere",
            hidden_width=4,
            hidden_layers=1,
            encoding={"levels": 1, "finest_resolution": 4},
        )
        with tempfile.TemporaryDirectory() as tmp:
            artifact = str(Path(tmp) / "runtime.pt")
            export_artifact(model, artifact)
            runtime = load_runtime(artifact)
            rows = runtime_resolution_sweep(
                runtime,
                [SphereLight(center=[0.0, 0.55, 0.0], radius=0.2, rgb=[1.0, 0.8, 0.6])],
                frames=1,
                devices=("cpu", "not_real"),
            )
        self.assertEqual(len(rows), 6)
        cpu_rows = [row for row in rows if row["device"] == "cpu"]
        missing_rows = [row for row in rows if row["device"] == "not_real"]
        self.assertTrue(all(row["available"] for row in cpu_rows))
        self.assertTrue(all(row["fps"] is not None for row in cpu_rows))
        self.assertTrue(all(not row["available"] for row in missing_rows))
        self.assertTrue(all(row["ms_per_frame"] is None for row in missing_rows))


if __name__ == "__main__":
    unittest.main()
