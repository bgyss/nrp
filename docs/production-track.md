# Production Track — From Verdict to Summit Demos

This is the post-E10 research track. `docs/pipeline-feasibility.md` rendered all
three target audiences "partly viable" and named the blockers; this track attacks
those named blockers directly, as a **balanced ladder**: every rung pairs one
scale/robustness proof with one performance target, so neither axis gets ahead of
the evidence.

Structure: a shared **trunk** (T1–T4) of rungs every summit needs, then three
short independent **branches** — one per E10 verdict row — each ending in a
summit demo. A failed branch does not stall the others.

Each rung is a self-contained goal prompt (designed for `/goal <prompt>` or as a
session brief) with baked-in **verification** and **measurement**.

Design rationale and decision record: `docs/plans/2026-07-09-production-track-design.md`.

## Status

| rung | title | status | evidence |
|---|---|---|---|
| T1 | Real-scene ingestion at scale | **done** | `out/export-bench.json` (39–59× ≥ 20× target); `out/kitchen-512-torch/torch_train_report.json` (21.01 dB / SSIM 0.30 / ꟻLIP 0.146); `docs/performance.md` |
| T2 | Streaming and memory discipline at real-scene scale | **done** | `out/t2-streaming/report.json`; `out/t2-streaming/export_512x512_64spp.json`; `out/t2-streaming/export_128x128_64spp.json` (0.000 dB parity; 0.80 GiB streamed training peak vs 8.45 GiB monolithic; 44.90 dB minimum packed-cache fidelity; export ceiling documented); `docs/performance.md` |
| T3 | Perceptual quality gates | **done** | `nrp/quality/gate.py` + `tests/test_quality_gate.py` (19 tests, incl. degraded-render failure); `out/quality/report.json` + `out/ablation/report_gated.json` re-emitted, conclusions unchanged; `out/quality/gate_overhead.json` (gate = 4.3% of the draft+reference render pair at 512², < 5% target); `docs/performance.md` |
| T4 | Runtime baseline lock | **done** | `out/t4-runtime/baseline.json` + `out/t4-runtime/report.json` (T1-scene proxy, hashgrid in WGSL, real Chrome: parity 1.2e-6; 512² p95 23.3 ms = 43 fps p95, 30 fps floor holds; `mise run t4-check` regression gate); `docs/performance.md` |
| G1 | Dynamic geometry, second attempt | **done (honest negative, changed failure mode)** | `out/g1-residual/report.json` (regime (d) 37.15 dB, gap 7.7 dB vs the 1 dB target — but out-of-mask fidelity 54.9 dB vs E2's 22.95 dB forgetting, per-frame floor 25.3 dB vs 10.6 dB, median gap −1.17 dB, 0.60× wall-clock); `docs/performance.md` |
| G2 | Summit: live relight + controls in the browser | **done** | `out/g2-demo/report.json` (481 frames under interaction: p95 30.7 ms ≤ 33 ms at 512²; G1-panel parity 4.8e-7); `out/g2-demo/gate.json` (12/12 trace frames pass preview tier vs OIDN-denoised controlled GATHERLIGHT, SSIM ≥ 0.883); `out/g2-demo/recording.webm` + `webgpu/demo/trace.json`; moving object included at G1's toy scale; `docs/performance.md` |
| F1 | Shot harness with temporal stability | **done** | `out/f1-shot/report.json` (120/120 frames pass preview tier, 31.6–37.8 dB / SSIM ≥ 0.881; proxy temporal FLIP-delta check passes at worst excess 0.0019 ≤ 0.02 while a 30 dB/frame flickering baseline fails at 0.165); `nrp/torch_backend/shot.py` + `tests/test_shot.py`; `docs/performance.md` |
| F2 | Summit: final-tier shot | **done** | `out/f2-shot/report.json` (120/120 frames pass final tier via fp16-stored residual-identity reconstruction, ≥ 105.9 dB; 118× wall-clock amortization vs per-frame re-export; storage honest negative: residuals 1.17× raw frames); `out/f2-shot/shot.mp4` committed; `docs/performance.md` |
| V1 | Production light rig | **done (honest negative)** | `out/v1-rig/report.json` (8-light rig — 3 sphere + 3 quad + 2 textured-quad — each an independent 800-iter proxy; additivity gate at preview tier: PSNR 25.6 dB pass, SSIM 0.622 and FLIP 0.352 both fail vs 0.80/0.15 thresholds; sizes 24.1× per-light-total vs matched-budget monolithic baseline; compositing overhead ≈111 ms/light, linear 1–8 lights); `nrp/torch_backend/rig.py` + `tests/test_rig.py`; `docs/performance.md` |
| V2 | Summit: art-direction loop | **done (partial recovery, honestly caveated)** | `out/v2-artloop/report.json` (art-direction target on 6 colorable V1 lights, 500-step Adam `optimize_colors`; convergence gate passes at draft tier: PSNR 154.5 dB / SSIM 0.9999999999999992 / FLIP 8.4e-7; but only 3 of 6 colorable lights — key/fill/rim — are genuinely gradient-recovered, the other 3 — window/ceiling_panel/practical — have zero-gradient proxies inherited from V1's reduced-budget training and their reported recovery is the untouched neutral guess; reload-identical rig round-trip; slider-loop latency 950.1 ms mean / 1011.4 ms p95; wall-clock 1320.7 s with a noted avoidable per-step inefficiency); `nrp/torch_backend/art_loop.py` + `tests/test_art_loop.py`; `docs/performance.md` |

Ordering: T1 unblocks everything; T2–T4 are parallelizable after T1. Branches
depend only on the trunk (G1 additionally revisits E2's fixtures and may run at
toy scale). Within a branch, rungs are sequential.

## Shared conventions (same as `docs/roadmap.md` / `docs/extensions.md`)

- All new/changed code passes `mise run test` and `mise run lint`; new features
  get unit tests that skip cleanly when optional dependencies are absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context; never quote the paper's numbers
  as ours. `mise run pipeline-audit` (or a successor) verifies referenced
  report paths exist.
- Honest negative results are deliverables.
- Single Apple Silicon laptop; no CUDA assumed.
- Update `docs/paper-mapping.md` for any row a change affects; commit in the
  repo's usual message style.

---

## Trunk

### T1. Real-scene ingestion at scale

> Finish the vectorized (drjit wavefront) Mitsuba exporter under whichever JIT
> variant is available (`llvm_ad_rgb` or `metal_ad_rgb`), keeping the scalar
> path as fallback and the CLI unchanged (roadmap item 1's remaining scope).
> Then export at least one real academic scene (kitchen/bedroom-class from the
> official Mitsuba scene repository, checked into `examples/scenes/` only as a
> download script — never vendor assets) at ≥ 512×512 and ≥ 64 spp, and train
> the torch proxy on it end to end. This scene becomes "the T1 scene" that
> every later rung builds on.
> **Verify:** existing exporter tests pass against the vectorized path; a
> fixed-seed equivalence test showing scalar and vectorized exports of the 8×8
> cornell box produce statistically compatible GATHERLIGHT images (mean
> radiance within 2%); an end-to-end training report exists for the new scene.
> **Measure:** exporter throughput (segments/s) scalar vs vectorized at 48×48
> and 128×128, target ≥ 20×; training wall-clock and held-out PSNR/SSIM/FLIP
> on the T1 scene. All in `docs/performance.md`.

### T2. Streaming and memory discipline at real-scene scale

> Re-prove E5's sharded-cache / streamed-target-table / streamed-TorchNRP
> machinery on the T1 scene, using the packed fp16 + shared-exponent cache
> format. The point is that the out-of-core proof, currently
> 512×512-cornell-box-only, survives real scene complexity within an explicit
> memory budget.
> **Verify:** streamed-vs-monolithic training parity within 0.1 dB held-out
> PSNR on the T1 scene (or monolithic infeasible — then parity vs a documented
> subsampled monolithic reference); a cache-size-vs-quality curve committed.
> **Measure:** peak RSS during export, training, and inference against an
> explicit ≤ 8 GB budget; shard-streaming throughput (segments/s). If even the
> streamed path exceeds budget, reduce spp/resolution and record the ceiling
> honestly — that is a track finding, not a failure to report. All in
> `docs/performance.md`.

### T3. Perceptual quality gates

> Promote SSIM/FLIP from ablation tooling to pass/fail gates: a reusable
> `nrp.quality.gate` CLI that generalizes the E9 trust-verdict ladder, is
> invokable by any report script, and carries named thresholds per tier
> (preview / draft / final).
> **Verify:** unit tests for the gate logic; at least two existing reports
> re-emitted through the gate with unchanged conclusions; a deliberately
> degraded render demonstrably fails the gate.
> **Measure:** gate evaluation overhead < 5% of render time at 512×512,
> recorded in `docs/performance.md`.

### T4. Runtime baseline lock

> Re-run the browser WebGPU bench (`webgpu/bench_browser.mjs`) on the exported
> T1-scene proxy and freeze the result as a regression baseline checkable by a
> mise task. The E10 games row cleared 30/60 fps at 128–512² on the cornell
> box; this rung locks that floor against a real scene and against future
> changes.
> **Verify:** parity vs the PyTorch reference within the existing tolerance;
> the baseline JSON committed; a mise task that fails when frame time
> regresses beyond a stated threshold.
> **Measure:** frame-time **histogram** (mean and p95, not just mean) at
> 128/256/512². Floor: 30 fps at 512² sustained, p95-verified. All in
> `docs/performance.md`.

---

## Games branch — Summit: interactive real-scene demo

### G1. Dynamic geometry, second attempt (partitioned residual retraining)

> E2's settled negative result: segment-local TorchNRP weight fine-tuning,
> even replay-regularized, fails its 1 dB recovery target by 11–20 dB. This
> rung tests a *changed hypothesis*, not a rerun: use E2's swept-volume
> invalidation to mark affected cache shards, keep the base proxy frozen, and
> train a **residual proxy over only the invalidated region**, composited at
> inference. Evaluate against E2's exact failed recovery target so the
> comparison is apples-to-apples. An honest failure is acceptable — but it
> must be a *different failure mode* than E2's, and the report must say which.
> **Verify:** a recovery-target comparison table (E2 fine-tune vs G1 residual)
> produced by one script; unit tests for shard invalidation and residual
> compositing.
> **Measure:** invalidate-and-recover wall-clock vs full retrain; quality at
> matched budget. Toy scale acceptable; run on the T1 scene if feasible. All
> in `docs/performance.md`.

### G2. Summit demo: live relight + controls in the browser

> The T1 scene in the WebGPU viewer with animated lights (E1 harness) plus at
> least two E8 production controls live, sustained at 30 fps at 512². If G1
> succeeded, include one moving object; if it failed, the demo ships without
> it and says so.
> **Verify:** the T3 gate passes at preview tier per frame; a scripted
> interaction trace committed alongside a screen-recording artifact.
> **Measure:** frame-time histogram under interaction (T4 methodology),
> p95 ≤ 33 ms, in `docs/performance.md`.

---

## Film branch — Summit: shot-relight demo

### F1. Shot harness with temporal stability

> A ≥ 120-frame keyframed-light shot on the T1 scene through the E9
> quality-tier ladder with a per-frame trust verdict — plus a temporal metric
> the ladder currently lacks: frame-to-frame FLIP delta (flicker is the
> failure mode PSNR can't see).
> **Verify:** per-frame verdict JSON committed; a deliberately flickering
> baseline (e.g. per-frame independent noise) fails the temporal check while
> the proxy sequence passes — or the report explains why not.
> **Measure:** per-frame render time at each tier; the temporal FLIP-delta
> distribution. All in `docs/performance.md`.

### F2. Summit demo: final-tier shot

> The F1 shot rendered at final tier with residual-identity frames, encoded as
> a committed MP4 plus a per-frame report.
> **Verify:** every frame passes the T3 final-tier gate or is flagged with its
> cause; residual identity holds per frame.
> **Measure:** end-to-end shot wall-clock (cache-reuse amortization vs
> re-rendering every frame); storage cost of proxy+residuals vs raw frames.
> All in `docs/performance.md`.

---

## VFX branch — Summit: rig-relight demo

### V1. Production light rig

> A rig of ≥ 8 lights, including textured/area emitters, on the T1 scene, with
> per-light layered proxies (scaling the E1/roadmap compositing machinery),
> solo/mute per light, and verified additivity.
> **Verify:** sum-of-layers vs full-rig render within a stated tolerance
> (report the tolerance, don't assume one); unit tests for rig serialization
> and per-light compositing.
> **Measure:** per-light proxy size vs a monolithic rig proxy; compositing
> overhead per added light at 512². All in `docs/performance.md`.

### V2. Summit demo: art-direction loop

> E7's image-target optimization driving the full V1 rig: inverse-recover
> per-light intensities/colors toward a hand-authored target, then interactive
> per-light grading in the viewer (or the headless slider loop).
> **Verify:** the optimization converges to the target within a stated image
> metric; the recovered rig is exported and re-loadable.
> **Measure:** optimizer iterations and wall-clock to convergence; interactive
> grading latency per adjustment. All in `docs/performance.md`.
