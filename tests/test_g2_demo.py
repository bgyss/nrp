import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WEBGPU_DIR = REPO_ROOT / "webgpu"

HAVE_NODE = shutil.which("node") is not None
HAVE_PLAYWRIGHT = (WEBGPU_DIR / "node_modules" / "playwright").exists()
HAVE_EXPORT = (REPO_ROOT / "out" / "g2-demo" / "export" / "manifest.json").exists()
HAVE_G1_EXPORT = (REPO_ROOT / "out" / "g2-demo" / "export-g1" / "manifest.json").exists()

# Correctness-only probe: loads the demo page in real Chrome, renders one kitchen
# frame (checking the output buffer is finite and non-trivial), and verifies the
# G1 panel's GPU composite against the exporter's torch composite per frame. No
# frame-time assertion — perf lives in demo_g2.mjs / the committed G2 report,
# because wall-clock under an arbitrary test-time machine load is not evidence.
PROBE = """
import { chromium } from "playwright";
import { createDemoServer } from "./demo/server.mjs";
const server = createDemoServer();
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const browser = await chromium.launch({ channel: "chrome", headless: true,
  args: ["--headless=new", "--no-sandbox", "--ignore-gpu-blocklist",
         "--use-angle=metal", "--enable-unsafe-webgpu"] });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/webgpu/demo/index.html`);
await page.waitForFunction(() => window.__demo && window.__demo.ready !== undefined,
                           null, { timeout: 60000 });
const result = await page.evaluate(async () => {
  const d = window.__demo;
  if (!d.ready) return { ready: false, error: d.error };
  d.setDriven(true);
  d.setTime(1.0);
  const ms = await d.renderFrame();
  const b64 = await d.readFrame();
  const bytes = atob(b64);
  const f32 = new Float32Array(new Uint8Array([...bytes].map((c) => c.charCodeAt(0))).buffer);
  let finite = true, nonzero = 0;
  for (const v of f32) { if (!Number.isFinite(v)) finite = false; if (v !== 0) nonzero++; }
  const parities = [];
  if (d.hasG1) for (let f = 0; f < 10; f++) parities.push(await d.g1Parity(f));
  return { ready: true, ms, finite, nonzeroFrac: nonzero / f32.length, hasG1: d.hasG1, parities };
});
await browser.close(); server.close();
console.log(JSON.stringify(result));
"""


@unittest.skipUnless(
    HAVE_NODE and HAVE_PLAYWRIGHT and HAVE_EXPORT,
    "requires node, `npm install` in webgpu/, and mise run g2-export",
)
class G2DemoBrowserTests(unittest.TestCase):
    """G2: the demo page initializes in real Chrome, renders finite non-trivial
    kitchen frames, and the G1 panel's compute pipeline matches the exporter's
    torch composite numerically. Skips cleanly without the browser toolchain or
    exported blobs, matching this repo's optional-dependency convention."""

    def test_demo_page_renders_and_g1_panel_matches_torch(self):
        import json

        result = subprocess.run(
            ["node", "--input-type=module", "-e", PROBE],
            cwd=str(WEBGPU_DIR),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(report["ready"], msg=report.get("error"))
        self.assertTrue(report["finite"])
        self.assertGreater(report["nonzeroFrac"], 0.5)
        if HAVE_G1_EXPORT:
            self.assertTrue(report["hasG1"])
            for parity in report["parities"]:
                self.assertLess(parity, 2e-4)


if __name__ == "__main__":
    unittest.main()
