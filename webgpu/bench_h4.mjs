// H4 (docs/hardening-track.md): N-light rig compositing on the T4/G2 WGSL runtime.
//
// V1's slider loop is CPU torch: 950 ms mean per adjustment for the 8-light kitchen
// rig (~111 ms/light, linear in N), ~29x off real time, while T4 proved the
// identical proxy architecture at 20.9 ms/frame at 512^2 in real Chrome. This ports
// N-light rig compositing to that WGSL runtime: one compute dispatch per active
// rig light, each `light_rgb`-multiplied (Eq. 3) and additively composited into one
// shared output buffer (Eq. 1's linearity) -- shader_gen.mjs's `rigLight`/
// composite:"add" mode. Solo/mute is resolved on the JS side (which lights get
// dispatched this frame), not a per-shader uniform.
//
// The actual compositor (`h4_page.mjs`) runs entirely inside the browser page and
// fetches its own blobs via fetch() against a local static server (same pattern as
// `demo/main.mjs`/`demo_g2.mjs`) -- an earlier version shipped the same data through
// page.evaluate()'s argument channel instead, and the ~100 MB combined payload for
// 8 real lights reproducibly killed the whole Chrome process before any page code
// ran. Every page.evaluate call from this driver only ever crosses small JSON.
//
//   node bench_h4.mjs                     # run, write out/h4-rig/report.json
//
// Measures: GPU-vs-CPU composite parity against LightRig.render (all 8 lights
// active); a scripted slider-session latency (mean/p95) at 512^2 nudging rgb for
// each colorable light in turn (mirrors V2's `slider_loop` methodology); and
// per-light marginal cost (soloing the first k lights, k=1..N, linear fit),
// mirroring V1's compositing-overhead-vs-light-count methodology.
import { chromium } from "playwright";
import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { createDemoServer } from "./demo/server.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const exportDir = path.join(repoRoot, "out", "h4-rig-export");
const reportPath = path.join(repoRoot, "out", "h4-rig", "report.json");

const PARITY_TOLERANCE = 2e-4; // f32 op-order drift, same basis as T4

function stats(times) {
  const sorted = [...times].sort((a, b) => a - b);
  const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  return { mean_ms: mean, p50_ms: q(0.5), p95_ms: q(0.95), min_ms: sorted[0], max_ms: sorted[sorted.length - 1] };
}

async function main() {
  if (!existsSync(path.join(exportDir, "manifest.json"))) {
    console.error(`missing ${exportDir}/manifest.json — run: mise run h4-export`);
    process.exit(2);
  }

  const server = createDemoServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;

  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: [
      "--headless=new", "--no-sandbox", "--ignore-gpu-blocklist",
      "--use-angle=metal", "--enable-unsafe-webgpu",
    ],
  });
  const page = await browser.newPage();
  page.on("console", (msg) => { if (msg.type() === "error") console.error("page:", msg.text()); });
  await page.goto(`http://127.0.0.1:${port}/webgpu/h4.html`);
  await page.waitForFunction(() => window.__h4 && window.__h4.ready !== undefined, null, { timeout: 60000 });
  const ready = await page.evaluate(() => window.__h4.ready);
  if (!ready) {
    const error = await page.evaluate(() => window.__h4.error);
    console.error("h4 page init failed:", error);
    await browser.close();
    server.close();
    process.exit(1);
  }
  const meta = await page.evaluate(() => ({
    adapter: window.__h4.adapter,
    resolution: window.__h4.manifest.resolution,
    lightNames: window.__h4.lightNames,
    lights: window.__h4.manifest.lights.map((l) => ({ name: l.name, rgb: l.rgb })),
    lightParamsByName: window.__h4.lightParamsByName,
  }));

  const parityMaxAbsDiff = await page.evaluate(() => window.__h4.parity());

  const overheadRows = await page.evaluate(() => window.__h4.overhead(3, 20));

  // Scripted per-light rgb nudges, round-robin across colorable lights (matches
  // examples/v2_art_loop.py::default_adjustments's methodology, fixed seed 0).
  const colorable = meta.lights.filter((l) => l.rgb).map((l) => l.name);
  const adjustments = [];
  let seed = 0;
  const rand = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  };
  for (let i = 0; i < 10 && colorable.length; i++) {
    adjustments.push({
      light: colorable[i % colorable.length],
      rgb: [0.3 + rand() * 3.7, 0.3 + rand() * 3.7, 0.3 + rand() * 3.7],
    });
  }
  const sliderLatencyMs = await page.evaluate((adjustments) => window.__h4.sliderLoop(adjustments), adjustments);

  // Worst-case slider session: light *shape* param nudges (position jitter),
  // round-robin across all lights -- each invalidates one light's cached raw
  // contribution, so the cost is one MLP dispatch + composite per nudge.
  const paramAdjustments = [];
  for (let i = 0; i < 10; i++) {
    const name = meta.lightNames[i % meta.lightNames.length];
    const base = meta.lightParamsByName[name];
    const jittered = [...base];
    for (let k = 0; k < Math.min(3, jittered.length); k++) jittered[k] += (rand() - 0.5) * 0.1;
    paramAdjustments.push({ light: name, light_params: jittered });
  }
  const paramEditLatencyMs = await page.evaluate(
    (adjustments) => window.__h4.sliderLoopParamEdit(adjustments),
    paramAdjustments
  );

  await browser.close();
  server.close();

  const ns = overheadRows.map((r) => r.n_lights);
  const mss = overheadRows.map((r) => r.ms);
  const n = ns.length;
  const sumX = ns.reduce((a, b) => a + b, 0);
  const sumY = mss.reduce((a, b) => a + b, 0);
  const sumXY = ns.reduce((a, x, i) => a + x * mss[i], 0);
  const sumXX = ns.reduce((a, x) => a + x * x, 0);
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  const sliderStats = stats(sliderLatencyMs);
  const paramEditStats = stats(paramEditLatencyMs);

  const report = {
    rung: "H4",
    scope: "N-light rig compositing on the T4/G2 WGSL runtime, all lights active",
    backend: "webgpu (Chrome, via Playwright)",
    adapter: meta.adapter,
    resolution: meta.resolution,
    n_lights: meta.lightNames.length,
    parity_vs_cpu_max_abs_diff: parityMaxAbsDiff,
    parity_tolerance: PARITY_TOLERANCE,
    parity_passed: parityMaxAbsDiff <= PARITY_TOLERANCE,
    compositing_overhead_ms: {
      rows: overheadRows,
      fit_slope_ms_per_light: slope,
      fit_intercept_ms: intercept,
    },
    slider_loop: {
      n_adjustments: sliderLatencyMs.length,
      latency_ms: sliderLatencyMs,
      ...sliderStats,
    },
    // Worst case per nudge: a shape-param edit invalidates one light's cached
    // raw contribution (one MLP dispatch + composite); rgb nudges above reuse
    // all cached contributions (composite pass only).
    slider_loop_param_edit: {
      n_adjustments: paramEditLatencyMs.length,
      latency_ms: paramEditLatencyMs,
      ...paramEditStats,
      meets_p95_100ms: paramEditStats.p95_ms <= 100,
      meets_p95_33ms_stretch: paramEditStats.p95_ms <= 33,
    },
    v1_cpu_baseline_ms_per_light: 111,
    v2_cpu_baseline_slider_mean_ms: 950,
    meets_p95_100ms: sliderStats.p95_ms <= 100,
    meets_p95_33ms_stretch: sliderStats.p95_ms <= 33,
  };
  mkdirSync(path.dirname(reportPath), { recursive: true });
  writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
  console.log(
    `parity ${report.parity_passed ? "PASS" : "FAIL"} (${report.parity_vs_cpu_max_abs_diff.toExponential(2)}), ` +
    `rgb-slider p95 ${sliderStats.p95_ms.toFixed(2)} ms, param-edit p95 ${paramEditStats.p95_ms.toFixed(2)} ms, ` +
    `marginal cost ${slope.toFixed(3)} ms/light -- wrote ${reportPath}`
  );
}

main();
