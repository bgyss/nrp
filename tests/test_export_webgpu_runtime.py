import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.export_webgpu_runtime import (  # noqa: E402
    export_encoding,
    export_mlp,
    load_gbuffer,
    numpy_forward,
    pixel_arrays,
)
from nrp.torch_backend.model import TorchNRP
from nrp.toy_tracer import trace_path_cache

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBGPU_DIR = REPO_ROOT / "webgpu"
HAVE_NODE = shutil.which("node") is not None
HAVE_PLAYWRIGHT = (WEBGPU_DIR / "node_modules" / "playwright").exists()
HAVE_T4_EXPORT = (REPO_ROOT / "out" / "t4-runtime" / "export" / "manifest.json").exists()


def _tiny_model(use_encoding=True, seed=3):
    torch.manual_seed(seed)
    encoding = {
        "levels": 4,
        "features_per_level": 2,
        "table_size_log2": 6,  # small enough that upper levels are hashed
        "base_resolution": 2,
        "finest_resolution": 32,
    }
    return TorchNRP(
        light_type="sphere",
        hidden_width=16,
        hidden_layers=2,
        encoding=encoding if use_encoding else None,
        use_encoding=use_encoding,
    )


class NumpyReplicaTests(unittest.TestCase):
    """The exported flat-blob format's semantic contract: `numpy_forward` (which the
    WGSL shader in webgpu/bench_t4.mjs mirrors) reproduces the torch module."""

    def _check(self, model):
        model.eval()
        rng = np.random.default_rng(0)
        n = 257
        xy = rng.uniform(0.001, 0.999, size=(n, 2)).astype(np.float32)
        aux = rng.uniform(-1, 1, size=(n, 7)).astype(np.float32)
        light = np.array([0.4, 0.6, 0.3, 0.15], dtype=np.float32)

        mlp_flat, mlp_dims = export_mlp(model)
        if model.encoding is not None:
            tables_flat, level_meta = export_encoding(model)
            fpl, ts = model.encoding.features_per_level, model.encoding.table_size
            self.assertTrue(
                any(not m["dense"] for m in level_meta), "test should cover hashed levels"
            )
            self.assertTrue(any(m["dense"] for m in level_meta))
        else:
            tables_flat, level_meta, fpl, ts = None, None, 0, 0

        replica = numpy_forward(
            xy, aux, light, mlp_flat, mlp_dims, tables_flat, level_meta, fpl, ts
        )
        with torch.no_grad():
            ref = model(
                torch.as_tensor(xy),
                torch.as_tensor(aux),
                torch.as_tensor(light).expand(n, -1),
            ).numpy()
        self.assertLess(float(np.max(np.abs(replica - ref))), 1e-5)

    def test_hashgrid_model_matches_torch(self):
        self._check(_tiny_model(use_encoding=True))

    def test_no_encoding_model_matches_torch(self):
        self._check(_tiny_model(use_encoding=False))

    def test_texture_kernel_model_matches_torch(self):
        # H3 kernel head (S4): MLP consumes only the 8 quad-geometry params;
        # softplus output is a per-texel kernel contracted with the texture.
        torch.manual_seed(7)
        texels = 4 * 4
        model = TorchNRP(
            light_type="textured_quad",
            light_param_dim=8 + 3 * texels,
            hidden_width=16,
            hidden_layers=2,
            encoding={
                "levels": 4,
                "features_per_level": 2,
                "table_size_log2": 6,
                "base_resolution": 2,
                "finest_resolution": 32,
            },
            texture_kernel=True,
        )
        model.eval()
        rng = np.random.default_rng(0)
        n = 129
        xy = rng.uniform(0.001, 0.999, size=(n, 2)).astype(np.float32)
        aux = rng.uniform(-1, 1, size=(n, 7)).astype(np.float32)
        light = rng.uniform(0.1, 1.0, size=(8 + 3 * texels,)).astype(np.float32)
        mlp_flat, mlp_dims = export_mlp(model)
        tables_flat, level_meta = export_encoding(model)
        fpl, ts = model.encoding.features_per_level, model.encoding.table_size
        replica = numpy_forward(
            xy,
            aux,
            light,
            mlp_flat,
            mlp_dims,
            tables_flat,
            level_meta,
            fpl,
            ts,
            texture_kernel=True,
        )
        with torch.no_grad():
            ref = model(
                torch.as_tensor(xy),
                torch.as_tensor(aux),
                torch.as_tensor(light).expand(n, -1),
            ).numpy()
        self.assertEqual(replica.shape, (n, 3))
        self.assertLess(float(np.max(np.abs(replica - ref))), 1e-5)


class ExporterEndToEndTests(unittest.TestCase):
    def test_export_from_saved_cache(self):
        cache = trace_path_cache(8, 8, spp=2, max_bounces=2, seed=5)
        model = _tiny_model()
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = str(Path(tmp) / "cache.npz")
            model_path = str(Path(tmp) / "model.pt")
            cache.save(cache_path, compressed=True)  # packed layout: aux stored fp16
            model.save(model_path)
            out_dir = Path(tmp) / "export"
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "examples" / "export_webgpu_runtime.py"),
                    "--model",
                    model_path,
                    "--cache",
                    cache_path,
                    "--out-dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            manifest = json.loads((out_dir / "manifest.json").read_text())
            self.assertEqual(manifest["resolution"], [8, 8])
            self.assertLess(manifest["numpy_replica_vs_torch_max_abs_diff"], 1e-3)
            for name, floats in manifest["files"].items():
                if floats:
                    self.assertEqual((out_dir / name).stat().st_size, floats * 4, name)
            gbuf = load_gbuffer(cache_path)
            xy, aux = pixel_arrays(gbuf)
            pixels = np.fromfile(out_dir / "pixels.bin", dtype=np.float32).reshape(64, 9)
            np.testing.assert_allclose(pixels[:, :2], xy, rtol=1e-6)
            np.testing.assert_allclose(pixels[:, 2:], aux, rtol=1e-6)


@unittest.skipUnless(
    HAVE_NODE and HAVE_PLAYWRIGHT and HAVE_T4_EXPORT,
    "requires node, `npm install` in webgpu/, and mise run t4-export",
)
class T4BaselineCheckTests(unittest.TestCase):
    """T4 integration: the WGSL backend matches PyTorch on the real exported T1-scene
    proxy and the committed runtime baseline holds. Spawns real Chrome; skips cleanly
    when the browser stack or export artifacts are absent (repo convention)."""

    def test_bench_t4_check_passes(self):
        result = subprocess.run(
            ["node", "bench_t4.mjs", "--check"],
            cwd=str(WEBGPU_DIR),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr or result.stdout)
        summary = json.loads(result.stdout[: result.stdout.rindex("}") + 1])
        self.assertLess(summary["parity_vs_pytorch_max_abs_diff"], 2e-4)


if __name__ == "__main__":
    unittest.main()
