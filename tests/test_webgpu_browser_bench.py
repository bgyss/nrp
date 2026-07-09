import json
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBGPU_DIR = REPO_ROOT / "webgpu"

HAVE_NODE = shutil.which("node") is not None
HAVE_PLAYWRIGHT = (WEBGPU_DIR / "node_modules" / "playwright").exists()
HAVE_WEIGHTS = (REPO_ROOT / "out" / "engine-runtime" / "js_viewer" / "model_weights.json").exists()


@unittest.skipUnless(
    HAVE_NODE and HAVE_PLAYWRIGHT and HAVE_WEIGHTS,
    "requires node, `npm install` in webgpu/, and mise run js-viewer to have exported the proxy",
)
class WebgpuBrowserBenchTests(unittest.TestCase):
    """E6: the completed WebGPU backend runs the real exported proxy inside real
    Chrome and matches the PyTorch reference. This is an end-to-end integration
    test (spawns Chrome via Playwright), not a unit test — it skips cleanly when
    the browser/dependencies aren't set up, matching this repo's convention for
    optional-dependency tests."""

    def test_browser_webgpu_matches_pytorch_reference(self):
        result = subprocess.run(
            ["node", "bench_browser.mjs"],
            cwd=str(WEBGPU_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout)
        self.assertLess(report["parity_vs_pytorch_max_abs_diff"], 1e-4)
        self.assertEqual(len(report["latency_sweep"]), 3)
        for row in report["latency_sweep"]:
            self.assertGreater(row["fps"], 0)


if __name__ == "__main__":
    unittest.main()
