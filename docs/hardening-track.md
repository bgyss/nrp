# Hardening Track — From Summit Demos to Trustworthy Claims

This is the post-production-track phase. `docs/production-track.md` completed
all 10 rungs and shipped one summit demo per target audience — but three rungs
closed as honest negatives or partial results, and the track surfaced the
project's first genuinely undiagnosed model-behavior bug. This phase **fixes
the diagnosed failures, re-earns the caveated summit claims, and refreshes the
E10 verdict** with the accumulated evidence.

Audit that motivated this track: `docs/status/2026-07-11.md`. Each rung is a
self-contained goal prompt with baked-in verification and measurement, same as
every prior phase.

## Status

| rung | title | status | evidence |
|---|---|---|---|
| H1 | Diagnose and fix the QuadLight zero-collapse | done | `out/h1-quad-fix/report.json`, `docs/performance.md#quadlight-zero-collapse-diagnosis-and-fix-hardening-track-rung-h1` |
| H2 | Re-earn the V1/V2 summit claims | done (partial: 8/8 proxies nonzero, additivity gate still fails preview tier; 5/6 colorable lights genuinely recovered in V2) | `out/h2-rig/report.json`, `out/h2-v2-artloop/report.json`, `docs/performance.md#re-earning-the-v1v2-summit-claims-hardening-track-rung-h2` |
| H3 | Textured-quad proxy quality | done (honest negative: neither 4x iteration budget nor 2x capacity closes the gap to the sphere/quad envelope; input-representation finding documented) | `out/h3-textured-quad/report.json`, `docs/performance.md#textured-quad-proxy-quality-hardening-track-rung-h3` |
| H4 | Rig compositing on the WebGPU runtime | done (partial: parity passes clean, 5x per-light speedup over CPU; slider p95 181ms misses the <=100ms target) | `out/h4-rig/report.json`, `docs/performance.md#rig-compositing-on-the-webgpu-runtime-hardening-track-rung-h4` |
| H5 | Real-scene dynamic geometry (exporter retrace path) | done (honest negative: neither regime (b) nor (d) meets the 1dB recovery target at real scale; mask machinery verified correct) | `out/h5-kitchen-fixed/report.json`, `docs/performance.md#real-scene-dynamic-geometry-hardening-track-rung-h5` |
| H6 | Flip the F2 storage negative | done (approval-frame gating flips the negative at 0.589x raw with zero quality cost; int8 quantization is a documented floor, not a win) | `out/h6-storage/sweep_report.json`, `docs/performance.md#storage-vs-quality-sweep-flipping-the-f2-negative-hardening-track-rung-h6` |
| H7 | Feasibility-verdict refresh | done | `docs/pipeline-feasibility.md#revision-2026-07-16-production-track--hardening-track-evidence-update`, `out/pipeline-feasibility/audit.json` |

Ordering: H1 → H2 → H4; H3, H5, H6 independent; H7 after everything else.

## Shared conventions (same as all prior phases)

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

### H1. Diagnose and fix the QuadLight zero-collapse

> V1/V2 established that the 3 `QuadLight` proxies on the kitchen cache emit
> exactly-zero raw output (`out/v2-artloop/report.json`,
> `colorable_light_raw_output_magnitude`), and ruled out training budget. The
> 2026-07-11 audit narrowed it further: numpy `gather_throughput_quad` at the
> authored quad params gives **nonzero** targets (mean 0.0028–0.0053, 13–30%
> of pixels lit — same order as `rim`'s working sphere), so the failure is
> training-side; and quad proxies trained fine on the toy and Cornell caches,
> so it is cache × type specific. Leading hypothesis, to be tested not
> assumed: §4.4 pool sampling draws quad normals uniformly on the sphere
> (`sampling.py`), so most sampled pool targets on this cache may be
> near-black, and Eq. 4's relative MSE has a zero-collapse basin (stop-grad
> denominator, ε = 0.01) that softplus cannot escape once predictions go
> dark. Run a discriminating experiment matrix: (a) measure the pool-target
> brightness distribution for sampled quads vs sampled spheres on this cache;
> (b) log prediction magnitude and loss-gradient norms over training to catch
> the collapse happening (or not); (c) ablate one variable at a time —
> normal sampling restricted near the authored normal, ε, output-head bias
> init, target-brightness-weighted sampling — until the collapse reproduces
> and un-reproduces on demand. Then fix it the minimal way the diagnosis
> supports.
> **Verify:** a kitchen-cache `QuadLight` proxy with nonzero raw output at
> the authored params; a unit test that fails on the collapse (e.g. asserts
> nonzero output magnitude after a short training run on a fixture that
> previously collapsed); existing sphere/toy-quad results unchanged.
> **Measure:** per-hypothesis evidence table in the report; fixed-quad val
> PSNR vs the rig's spheres; any training wall-clock delta from the fix.
> Mechanism write-up in `docs/performance.md` — this is the rung's real
> deliverable even if the fix is one line.

### H2. Re-earn the V1/V2 summit claims

> With H1 fixed, V1's additivity gate and V2's recovery claim were both
> measured against a rig where only 5 of 8 proxies contributed. Retrain the
> full 8-light rig at a budget informed by G2's finding (the same
> architecture reaches 29.4 dB at 15k iters vs 21.0 dB at 3k; V1's 800-iter
> budget was the floor) — pick the budget deliberately and record the
> tradeoff. Re-run the additivity gate (OIDN-denoised reference, per the V1
> methodology fix) and the full V2 art-direction loop. Fix the known V2
> optimizer inefficiency while in there: hoist constant (zero-gradient)
> light contributions out of the per-step loop instead of re-running their
> forward pass 500 times.
> **Verify:** 8/8 proxies with nonzero output; V2 recovery table regenerated
> — target is 6/6 colorable lights genuinely gradient-recovered, and the
> report must keep the genuine/vacuous distinction machinery so a regression
> is visible; rig round-trip still bit-identical.
> **Measure:** additivity PSNR/SSIM/FLIP vs the preview-tier thresholds
> (pass or an honest fail with the residual error decomposed); per-light val
> PSNR at the new budget; optimizer wall-clock vs V2's 1320.7 s with the
> hoist quantified separately from the budget change. All in
> `docs/performance.md`.

### H3. Textured-quad proxy quality

> V1's genuinely-contributing low scorers are the two `TexturedQuadLight`
> proxies (12.35 / 13.35 dB at 800 iters, 3.4× slower per iter). Sweep
> iteration budget (and, if budget alone plateaus, model capacity and
> texture-conditioning inputs) to bring them inside the rig's quality
> envelope, or land a documented finding about why texture conditioning
> needs a different input scheme.
> **Verify:** sweep report committed; unit tests for any conditioning change.
> **Measure:** val PSNR/SSIM/FLIP vs budget curve; train wall-clock; the
> chosen operating point justified in `docs/performance.md`.

### H4. Rig compositing on the WebGPU runtime

> V2's slider loop is 950 ms mean per adjustment (CPU torch, ≈ 111 ms/light,
> linear in N) — ~29× off real time, while T4 proved the identical proxy
> architecture at 20.9 ms/frame at 512² in real Chrome. Port N-light rig
> compositing to the T4/G2 WGSL runtime: N proxy dispatches (or one batched
> dispatch) summed in-shader, solo/mute as uniforms, per-light rgb as the
> Eq. 3 emission multiply.
> **Verify:** GPU composite parity vs `LightRig.render` within T4's
> tolerance; the G2 interaction-trace methodology reused for a scripted
> slider session.
> **Measure:** slider-loop latency mean/p95 at 512² for the 8-light rig,
> target ≤ 100 ms p95 (stretch: 33 ms); per-light marginal cost vs V1's
> 111 ms CPU figure; frame-time histogram per T4 methodology. All in
> `docs/performance.md`.

### H5. Real-scene dynamic geometry (exporter retrace path)

> G1's residual result is toy-scale only because the Mitsuba exporter cannot
> re-trace an edited scene (it records paths for a fixed scene), so
> invalidation targets cannot be produced for the kitchen. Add a
> scene-edit/retrace path — minimally: accept an object transform override,
> re-export only, and let the existing swept-volume machinery compute the
> invalidation mask between the two caches. Then run G1's regime table on
> the kitchen.
> **Verify:** exporter tests for the transform override; mask correctness
> spot-check (edited-region pixels differ between caches, out-of-mask pixels
> statistically compatible).
> **Measure:** G1's regime comparison (frozen base + residual vs full
> retrace/retrain vs stale) at kitchen scale; invalidate-and-recover
> wall-clock vs full re-export. An honest negative at real scale is a
> deliverable. All in `docs/performance.md`.

### H6. Flip the F2 storage negative

> F2's residual-identity shot stores 1.17× the bytes of the raw frames —
> fp16 residuals are noise-dominated. Two levers the F2 report itself named:
> keep residuals only for approval frames (proxy-only elsewhere, gated at
> preview/draft tier), and quantize residuals more aggressively (sweep fp16
> → int8/shared-exponent against the final-tier gate).
> **Verify:** every stored-reconstruction frame still passes its declared
> tier gate; the storage accounting script extended, unit-tested.
> **Measure:** storage-vs-quality curve; the crossover point where
> proxy+residual beats raw frames, or a documented floor showing it cannot
> at this proxy quality. All in `docs/performance.md`.

### H7. Feasibility-verdict refresh

> `docs/pipeline-feasibility.md` predates the production track: its verdict
> rows still say the scale proof is cornell-box-only and light rigs are
> unproven. Re-issue the per-audience verdicts against the full evidence
> base (T1–V2 plus H1–H6), keeping the original text intact as history and
> adding a dated revision — the same convention the V1 correction used.
> **Verify:** `mise run pipeline-audit` passes over the updated doc; every
> changed verdict cites its report.
> **Measure:** n/a — this is a decision document. Update
> `docs/tracks.md` and the README's phase table to close the track.
