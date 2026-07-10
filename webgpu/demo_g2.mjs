// G2 scripted trace runner: replays the committed interaction trace
// (webgpu/demo/trace.json) through the live demo page in real Chrome, records
//   - the frame-time histogram under interaction (T4 methodology; the rung's
//     criterion is p95 <= 33 ms at 512^2),
//   - a screen recording (out/g2-demo/recording.webm),
//   - the gate-sample frames (raw linear HDR buffers + their light/control
//     states) that examples/g2_gate.py gates at preview tier.
// Exits 1 if the p95 criterion fails.
import { chromium } from "playwright";
import { mkdirSync, writeFileSync, readFileSync, existsSync, renameSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createDemoServer } from "./demo/server.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");
const outDir = path.join(repoRoot, "out", "g2-demo");
const framesDir = path.join(outDir, "frames");
const P95_LIMIT_MS = 33;

async function main() {
  if (!existsSync(path.join(repoRoot, "out", "g2-demo", "export", "manifest.json"))) {
    console.error("missing out/g2-demo/export — run: mise run g2-export");
    process.exit(2);
  }
  const trace = JSON.parse(readFileSync(path.join(__dirname, "demo", "trace.json"), "utf8"));
  mkdirSync(framesDir, { recursive: true });

  const server = createDemoServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;

  const browser = await chromium.launch({
    channel: "chrome",
    headless: true,
    args: ["--headless=new", "--no-sandbox", "--ignore-gpu-blocklist", "--use-angle=metal", "--enable-unsafe-webgpu"],
  });
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: outDir, size: { width: 1280, height: 800 } },
  });
  const page = await context.newPage();
  page.on("console", (msg) => { if (msg.type() === "error") console.error("page:", msg.text()); });
  await page.goto(`http://127.0.0.1:${port}/webgpu/demo/index.html`);
  await page.waitForFunction(() => window.__demo && window.__demo.ready !== undefined, null, { timeout: 60000 });
  const ready = await page.evaluate(() => window.__demo.ready);
  if (!ready) {
    console.error("demo init failed:", await page.evaluate(() => window.__demo.error));
    process.exit(1);
  }
  const meta = await page.evaluate(() => ({
    adapter: window.__demo.adapter,
    resolution: window.__demo.resolution,
    hasG1: window.__demo.hasG1,
  }));

  // Phase 1: replay the interaction timeline, timing every frame.
  const replay = await page.evaluate(async ({ trace, fps }) => {
    const d = window.__demo;
    d.setDriven(true);
    const frames = Math.round(trace.duration_s * fps);
    const times = [];
    const warmup = 10;
    for (let i = 0; i < warmup; i++) await d.renderFrame();
    for (let i = 0; i <= frames; i++) {
      const t = (i / fps) % (trace.duration_s + 1e-9);
      d.setTime(Math.min(t, trace.duration_s));
      let controls = null;
      for (const ev of trace.events) if (ev.t <= t + 1e-9) controls = ev.controls;
      if (controls) d.setControls(controls);
      if (d.hasG1) d.setG1Frame(Math.floor((t * 2.5) % 10));
      times.push(await d.renderFrame());
    }
    return { times };
  }, { trace, fps: 60 });

  // Phase 2: gate samples — render each sampled state and read the raw buffer back.
  const states = [];
  for (let i = 0; i < trace.gate_samples.length; i++) {
    const sample = trace.gate_samples[i];
    const { b64, light } = await page.evaluate(async ({ sample }) => {
      const d = window.__demo;
      d.setTime(Math.min(sample.t, d.trace.duration_s));
      d.setControls(sample.controls);
      await d.renderFrame();
      return { b64: await d.readFrame(), light: d.lightAt(Math.min(sample.t, d.trace.duration_s)) };
    }, { sample });
    const buf = Buffer.from(b64, "base64");
    const name = `frame_${String(i).padStart(4, "0")}.bin`;
    writeFileSync(path.join(framesDir, name), buf);
    states.push({ index: i, t: sample.t, file: `frames/${name}`, light, controls: sample.controls });
  }
  writeFileSync(path.join(framesDir, "states.json"), JSON.stringify({
    resolution: meta.resolution,
    tier: "preview",
    states,
  }, null, 2) + "\n");

  const video = page.video();
  await context.close();
  const videoPath = await video.path();
  const finalVideo = path.join(outDir, "recording.webm");
  renameSync(videoPath, finalVideo);
  await browser.close();
  server.close();

  const times = replay.times;
  const sorted = [...times].sort((a, b) => a - b);
  const q = (f) => sorted[Math.min(sorted.length - 1, Math.floor(f * sorted.length))];
  const mean = times.reduce((a, b) => a + b, 0) / times.length;
  const histogramBins = {};
  for (const t of times) {
    const bin = `${Math.floor(t / 2) * 2}-${Math.floor(t / 2) * 2 + 2}ms`;
    histogramBins[bin] = (histogramBins[bin] || 0) + 1;
  }
  const report = {
    rung: "G2",
    scope: "summit demo: T1-scene proxy relit live in Chrome with animated lights (E1), light linking + artist attenuation (E8), and the G1 toy moving-object panel; frame times measured under the committed interaction trace",
    backend: "webgpu (Chrome, via Playwright)",
    adapter: meta.adapter,
    resolution: meta.resolution,
    g1_panel_included: meta.hasG1,
    trace: "webgpu/demo/trace.json",
    timed_frames: times.length,
    frame_time_ms: {
      mean: mean,
      p50: q(0.5),
      p95: q(0.95),
      min: sorted[0],
      max: sorted[sorted.length - 1],
    },
    histogram: histogramBins,
    fps_mean: 1000 / mean,
    fps_p95: 1000 / q(0.95),
    p95_limit_ms: P95_LIMIT_MS,
    meets_p95_33ms: q(0.95) <= P95_LIMIT_MS,
    gate_samples: states.length,
    notes: [
      "Each timed frame: light-uniform + control-uniform upload, the 409k-param hashgrid proxy at 512^2, the G1 panel's base+residual composite at 32^2, and canvas presentation of all three panels (onSubmittedWorkDone, no readback).",
      "The interaction timeline drives light animation, a linking toggle, attenuation, and emission tint exactly as committed in trace.json.",
    ],
    frame_times_ms: times.map((t) => Math.round(t * 1000) / 1000),
  };
  writeFileSync(path.join(outDir, "report.json"), JSON.stringify(report, null, 2) + "\n");
  const { frame_times_ms, histogram, ...summary } = report;
  console.log(JSON.stringify(summary, null, 2));
  if (!report.meets_p95_33ms) {
    console.error(`FAIL: p95 ${q(0.95).toFixed(1)} ms > ${P95_LIMIT_MS} ms`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
