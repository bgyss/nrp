# Pipeline Feasibility

This is the E10 decision document for the extension program in `docs/extensions.md`.
E1-E9 have each produced measured evidence (toy scale, and 512x512 real-Mitsuba scale
for E5/E9); the one genuinely open item is not a missing measurement but a settled
negative/structural finding: E2's TorchNRP fine-tune, even with replay
regularization, fails its recovery target. E6's WebGPU compute-shader criterion is
now fully closed — a native-binding-only attempt (`webgpu/bench.mjs`) reproducibly
crashed on real trained-model data, was rigorously bisected (`webgpu/README.md`) to
a third-party defect, and then genuinely resolved by running the identical shader
in a production WebGPU implementation (real Chrome via Playwright,
`webgpu/bench_browser.mjs`) — see below. E7's hand-authored-fixture gap is also
closed with a genuinely authored (not render-derived) pixel-art target — see below.
Every
quantitative claim below cites a report under `out/` or a `docs/performance.md` row;
`mise run pipeline-audit` verifies every `out/` path referenced here exists. Claims
are measured at toy scale unless stated otherwise (E5 and E9 additionally have
512x512 real-Mitsuba cornell-box measurements); no production-scale (real film/game
asset complexity) extrapolation is treated as achieved — cornell-box-scale results
are explicitly labeled as such throughout, not as a stand-in for production content.

## Evidence Base

Measured reports used here:

| extension | report | measured scope |
|---|---|---|
| E1 animated lights | `out/animate/report.json` | static-camera animated-light sequence |
| E1 animated camera | `out/time-camera/report.json` | camera-keyframe cache scaling and image-space interpolation baseline |
| E1 time-conditioned proxy | `out/time-camera/proxy_report.json` | single TorchNRP proxy conditioned on time, held-out camera PSNR gap |
| E2 dynamic geometry | `out/dynamic-geometry/report.json` | one-bounce primary-visibility cache splicing, TorchNRP fine-tune regimes, multi-bounce swept-volume invalidation |
| E3 light-aware sampling | `out/light-aware-sampling/report.json` | spherical guide-region sampling density |
| E3 proxy A/B | `out/light-aware-proxy-ab/report.json` | toy standard-vs-guided proxy training comparison |
| E4 environment light | `out/environment-fit/report.json` | degree-2 SH inverse recovery |
| E4 textured quad | `out/textured-quad-fit/report.json` | textured-quad reference inverse recovery |
| E5 out-of-core | `out/out-of-core/report.json` | sharded cache, streamed target table, tiled inference |
| E5 streamed TorchNRP | `out/out-of-core/streamed_torchnrp_report.json` | streamed vs monolithic TorchNRP training parity |
| E5 production scale | `out/out-of-core/mitsuba_512_report.json` | 512x512/128spp streamed TorchNRP end-to-end report |
| E6 exported runtime | `out/engine-runtime/report.json` | TorchScript artifact, CPU/MPS resolution sweep, headless slider loop |
| E6 JS GUI viewer | `out/engine-runtime/js_viewer/report.json` | self-contained HTML/JS GUI slider, JS-vs-PyTorch parity |
| E6 WebGPU smoke test | `out/engine-runtime/webgpu_smoke.json` | native-binding pipeline sanity check with synthetic weights (see `webgpu/README.md` for the real-weight crash bisection on that specific binding) |
| E6 WebGPU (completed) | `out/engine-runtime/webgpu_browser_report.json` | real WebGPU compute shader in real Chrome, running the actual exported proxy, parity + 128/256/512 latency sweep |
| E7 image target loop | `out/generative/report.json` | synthesized scribble and stylized target optimization |
| E7 hand-authored target | `out/generative/hand_authored_report.json` | genuinely hand-authored (not render-derived) pixel-art target inverse recovery |
| E8 production controls | `out/production-controls/report.json` | gather-time controls and conditioned control proxies |
| E9 quality tiers | `out/quality/report.json` | preview/draft/final PSNR/SSIM/FLIP and residual identity |
| E9 production scale | `out/quality/production_report.json` | 512x512 real-Mitsuba trust verdict with trained proxy |

The original torch toy proxy report at `out/toy-torch/torch_train_report.json` is used
only as baseline context. Claims below are measured at toy scale unless stated
otherwise. No production-scale extrapolation is treated as achieved.

## Summary Table

| target | interactivity met? | quality tier reached | hardest structural blocker | estimated engineering distance |
|---|---|---|---|---|
| Games | Partly, for static scenes and exported proxy loops | Preview only | Dynamic geometry and frozen transport | Large: multi-bounce invalidation, live controls (engine backend now proven: real WebGPU, real hardware, 30/60 fps at 128-512²) |
| Animated film | Partly, for animated lights and final-frame residual identity | Full ladder + trust verdict proven at 512x512 cornell-box scale | Animated characters and true production-shot complexity | Medium-large: per-shot caches plausible, scale proof still cornell-box only |
| Feature VFX | Partly, for per-shot relighting and art-direction loops | Full ladder + trust verdict proven at 512x512 cornell-box scale | Production light rigs and layered controls | Medium: useful component, not a full renderer primitive |

## Games

**Verdict: not the right primitive as a core renderer yet; viable component for
static-scene relighting.**

Measured support:

- The exported runtime path in `out/engine-runtime/report.json` reports 0.46 ms/frame
  exported inference and 1.19 ms/frame mean headless slider-loop latency at 32x32.
  That meets a 16 ms frame budget at toy resolution. The same report now includes
  real CPU/MPS 128/256/512 rows: CPU reaches 29.9 fps at 512x512 (misses 30 fps); MPS
  is 9.4 fps at 128x128 (dispatch overhead dominates below 256x256), 35.0 fps at
  256x256, 26.2 fps at 512x512. Neither backend hits 30 fps at 512x512.
- E1 animated-light evaluation in `out/animate/report.json` reports 4.81 ms/frame at
  48x48, with proxy frame-to-frame delta 1.05x the GATHERLIGHT reference. Static-scene
  animated lights are aligned with a game-style live lighting edit.
- E2 in `out/dynamic-geometry/report.json` reports one-bounce cache splicing at
  0.71 ms/frame, plus a warm-start image-proxy repair at 0.30 ms/frame. Together
  they use 6.3% of a 16 ms frame, recover exactly versus full retrace for the
  spliced cache, and leave the image proxy at 65.86 dB minimum PSNR versus full
  retrace for that constrained case. Real TorchNRP warm-started weight fine-tuning
  (regime b) has now been measured against a full-retrain reference (regime a) and a
  stale baseline (regime c): regime b reaches only 25.20 dB vs the 44.88 dB
  full-retrain reference (19.7 dB short of the 1 dB recovery target), barely above
  the 22.13 dB stale baseline. Multi-bounce invalidation is also now proven: the same
  report's `multi_bounce_invalidation` field shows primary-visibility-only
  invalidation misses 534 pixels with changed indirect bounces at 2 bounces (33.43 dB
  vs a full retrace — a real, measured artifact), while adding a swept-volume mask
  (any cached segment at any bounce depth passing through the moving object's
  conservative swept bound) recovers an exact match.

Blocking evidence:

- TorchNRP weight fine-tuning is proven *not* to meet the 1 dB recovery target with
  segment-local fine-tuning alone — a structural finding, not a missing measurement.
- The swept-volume multi-bounce mask is conservative (over-invalidates: 896 pixels
  vs 362 for primary-only, on a scene where only 534 pixels actually needed it) and
  is not yet integrated into the per-frame regime (a)/(b)/(c) timing loop, which
  remains one-bounce only.
- E8 in `out/production-controls/report.json` shows production controls survive at
  gather time, that one binary linking toggle can be precomputed into a table proxy,
  that a learned linear image proxy predicts a held-out attenuation setting at
  333.65 dB PSNR, that a soft 4-basis mask proxy predicts a held-out mask at
  331.25 dB PSNR, and that a quadratic attenuation proxy predicts a held-out curve at
  323.22 dB PSNR with 158x speedup versus polynomial gather.
- E3 in `out/light-aware-proxy-ab/report.json` improves the fixed in-region proxy
  result by 10.10 dB on a geometric open-top-box occluder fixture and does not
  regress the fixed open-region light. This is still a toy lampshade-style fixture,
  not a production lighting scene.

Engineering versus structural:

- Engineering blockers: exported backend beyond TorchScript (real MPS timings,
  streamed optimizer training, and a real WebGPU compute-shader backend running the
  actual proxy are all now measured — none of these remain blockers).
- Structural blockers: frozen transport under dynamic geometry and per-light-type
  proxy boundaries; free-form production controls scale with the chosen conditioning
  parameterization.

Costed gap list (what a production team would still have to build):

- A non-segment-local incremental update mechanism for dynamic geometry (measured:
  segment-local TorchNRP fine-tuning fails its 1 dB target by 11-20 dB even with
  replay regularization) — this is the largest, least-understood item; likely a
  research-scale effort (per-region adapter weights, or a different caching
  granularity), not an engineering sprint.
- A compiled engine backend beyond the browser-hosted WebGPU compute shader already
  measured (`webgpu/bench_browser.mjs`: real proxy, real Chrome, 30/60 fps at
  128–512² on real hardware) — a native (non-browser) engine integration is a
  smaller remaining step now that the shader and data path are proven correct end
  to end; either way the hashgrid encoding still needs porting for the paper's actual
  architecture.
- Multi-bounce invalidation integrated into a live per-frame budget, including its
  conservative-mask over-invalidation cost — medium effort; the mask is proven
  correct, the remaining work is tightening it and wiring it into a real frame loop.
- Live per-object/per-light production controls (linking, attenuation) at
  arbitrary, free-form parameterization, not just the measured control bases —
  medium effort, mostly conditioning-vector design and training data coverage.

## Animated Film

**Verdict: viable component for interactive lighting preview on mostly static
sets, not yet a complete preview-to-final pipeline.**

Measured support:

- E1 animated-light playback is already flat enough at toy scale to support live
  keyframed light review (`out/animate/report.json`).
- E1 camera-keyframe baseline in `out/time-camera/report.json` traces K=3 camera
  caches totaling 798,180 bytes and reaches 26.64 dB mean held-out image-space
  interpolation PSNR. `out/time-camera/proxy_report.json` goes further: a single
  TorchNRP proxy conditioned on time trained jointly on those K=3 keyframes reaches
  26.68 dB mean held-out PSNR against freshly traced ground truth at unseen
  intermediate camera poses, a 2.06 dB gap from its 28.74 dB mean training-keyframe
  PSNR — inside the 3 dB criterion.
- E9 in `out/quality/report.json` proves the approval-frame residual identity: proxy
  plus cached residual matches cached GATHERLIGHT at the approved config to max
  absolute error 5.6e-17. Its toy trust verdict is to trust only the approved frame
  and re-bake after any measured light move because dx=0.05 drops to 24.67 dB.
  `out/quality/production_report.json` reproduces this at 512x512 on real Mitsuba
  cornell-box caches (32spp export, 128spp converged reference) with a real
  streamed-trained proxy (not intentionally untrained): 33.76 dB preview / 35.72 dB
  draft PSNR vs the 128spp final, exact residual identity at approval, and the same
  qualitative trust verdict. **E9 is fully closed** at this scale.
- E5 in `out/out-of-core/report.json` shows streamed fixed-light target construction
  matches monolithic targets to max error 3.33e-16 while loading only 11.1% of cache
  segments and 11.1% of resident segment bytes at once, a 9.0x resident segment-memory
  reduction estimate at toy scale. `out/out-of-core/streamed_torchnrp_report.json`
  trains a real TorchNRP sphere-light proxy from streamed shards with a
  bitwise-identical loss curve to the monolithic run (11.1x lower peak segment
  memory). `out/out-of-core/mitsuba_512_report.json` scales this to
  512x512/128spp/94.2M segments: 3.35 GB monolithic cache / 8.29 GB resident segment
  bytes vs 574 MB peak streamed (14.4x lower), ~16.7 min streamed train, 32.57 dB
  held-out PSNR, 118.3 ms tiled full-frame inference. **E5 is fully closed** at this
  scale.

Blocking evidence:

- E1's time-conditioned proxy is only proven at K=3 keyframes and a small ±0.04
  camera range; larger K or motion range is untested.
- E5's remaining caveat is not a missing measurement but a performance one: the
  streamed path is I/O/Python-loop bound (16.7 min at 512x512), not compute bound —
  an engineering distance (batched streaming gather), not a structural blocker.

Engineering versus structural:

- Engineering blockers: high-resolution cache streaming, GPU/export backend, final
  quality metrics.
- Structural blockers: animated characters still require invalidation/retraining
  machinery, not just cached relighting.

Costed gap list:

- Validating E1's time-conditioned proxy and E5/E9's streamed-scale results on real
  production-shot complexity (character models, complex materials), not just a
  512x512 cornell-box — medium-large effort, primarily engineering (more scenes,
  more compute) rather than new algorithms, since the underlying mechanisms are
  proven.
- Character animation support (E2's dynamic-geometry findings apply directly: the
  measured negative result on segment-local fine-tuning means per-shot proxies, not
  live incremental updates, are the near-term path for animated characters) —
  large effort, shares the games-target research gap above.
- Production denoiser/pool-training throughput at real shot resolution — the E5
  streamed path is currently I/O/Python-loop bound (16.7 min at 512x512); closing
  this is an engineering distance (a batched streaming gather, analogous to
  `TorchPathCache`), not a structural one.

## Feature VFX

**Verdict: viable component for per-shot relighting and physically grounded
art-direction loops, not a standalone production renderer.**

Measured support:

- E7 in `out/generative/report.json` demonstrates the image-target-to-physical-light
  loop and reports proxy-space versus GATHERLIGHT errors separately. The synthesized
  scribble fixture passes its mask/protect thresholds, and
  `out/generative/provenance.json` records deterministic fixture recipes and SHA-256
  hashes for the current toy targets. The proxy is now pretrained on 24 random
  sphere lights (800 steps, windowed mean loss 0.330 → 0.169) before inversion,
  rather than differentiating through an untrained network.
  `out/generative/hand_authored_report.json` adds a genuinely hand-authored target
  (an explicit hand-picked stroke list, no algorithmic connection to any render or
  light) through the same pipeline: 12.56 dB target-vs-realized PSNR, same
  physical-realization-gap finding as the stylized target. **E7 is now fully
  closed** at toy scale.
- E4 in `out/environment-fit/report.json` recovers degree-2 SH environment
  coefficients with 1.26e-15 relative coefficient error, proving that at least this
  richer-light inverse slice is well-conditioned.
- E4 textured-quad fitting in `out/textured-quad-fit/report.json` recovers 2x2, 4x4,
  and 8x8 RGB textures with <= 2.34e-16 relative texture error on full-rank reference
  fixtures. Its equal-observation linear proxy-scaling baseline shows 2x2/4x4 remain
  full-rank at 48 observations while 8x8 is underdetermined and drops to 10.66 dB
  held-out PSNR. The same report now includes a compact learned texture-embedding
  torch proxy with mean held-out PSNR of 20.08, 21.19, and 22.27 dB for 2x2, 4x4,
  and 8x8 textures, plus a first-class `textured_quad` TorchNRP train/relight smoke
  with 20 light parameters and 12.31 dB held-out relight PSNR for a 2x2 texture.
- E8 in `out/production-controls/report.json` proves exact gather-time light linking
  algebra for the toy layer partition, measures attenuation controls, keeps one
  precomputed binary linking toggle live through a table proxy, keeps one fixed-family
  continuous attenuation control live through a learned linear proxy, and keeps
  measured soft-mask and quadratic attenuation controls live through learned
  basis-conditioned proxies.

Blocking evidence:

- E4 is now complete at toy scale, but the first-class TorchNRP smoke only covers
  2x2 textures; higher-resolution first-class runs remain a scale/quality follow-up.
- E7 stylized and hand-authored target realization both remain physically limited:
  the reports are useful precisely because they expose the gap between an arbitrary
  image target and a physically realizable lighting setup — that gap is the finding,
  not a shortcoming of the demo.
- E8 is satisfied at toy scale for the measured control bases; broader production
  control rigs remain a scale and UX problem, not a missing proof-of-concept.

Engineering versus structural:

- Engineering blockers: richer light embeddings, compositor/export integration,
  larger-scene reports.
- Structural blockers: inverse optimization can only realize targets within the
  chosen physical light family; arbitrary generative edits remain constrained.

Costed gap list:

- A true hand-authored or external generative-model image fixture with documented
  provenance — small effort in principle, but requires an actual external asset or
  paint-tool session this environment cannot produce unprompted; the algorithmic
  pipeline (scribble/generative-target optimization, now through a pretrained
  proxy) is otherwise complete at toy scale.
- Richer light families beyond sphere/quad/textured-quad/environment for real
  production rigs — medium effort per family, following E4's established pattern
  (differentiable GATHERLIGHT term + proxy conditioning + inverse-recovery test).
- Larger-scene, higher-resolution art-direction loop reports (the current loop is
  14x14) — mostly an engineering/compute-time question given the underlying
  optimizer and mask/protect machinery are proven.

## Current Verdicts

| target | verdict | why |
|---|---|---|
| Games | Not the right primitive as a core renderer yet | Dynamic everything is the core requirement; multi-bounce invalidation correctness is now proven, but its wall-clock cost is unmeasured and the TorchNRP weight-fine-tune path that would keep a proxy live under geometry changes has failed its recovery target. |
| Animated film | Viable component | Static-set animated-light preview, residual identity, and the final-frame trust verdict are now proven at 512x512 cornell-box scale; neural animated-camera support at that scale is unproven. |
| Feature VFX | Viable component | Per-shot caches, art-direction loops, and the final-frame trust verdict fit VFX workflows at 512x512 cornell-box scale; production-scale light rigs on real shot complexity remain incomplete. |

## Revision (2026-07-16): production-track + hardening-track evidence update

The table above predates the production track (`docs/production-track.md`,
T1-V2) and the hardening track (`docs/hardening-track.md`, H1-H6) entirely —
it was written and marked complete before either phase ran (see
`docs/tracks.md`'s phase ordering). Both tracks have since produced a
substantial amount of real-scene (not cornell-box) evidence that changes the
"cornell-box scale only" caveat throughout the table above, plus several new
honest negatives that sharpen rather than soften the remaining blockers.
Per this program's convention (see the V1 additivity "Correction:" above and
in `docs/performance.md`), the original table is left intact as history;
this section supersedes it.

**What's new since the original verdict, per audience:**

*Games.* T1's exported real WebGPU compute-shader runtime (E6, already
counted above) now has a *rig*-scale counterpart: H4 ports N-light
compositing to the same WGSL runtime — GPU-vs-CPU parity clean (1.05e-5,
`out/h4-rig/report.json`), ~22.5 ms/light cold-render marginal cost (~5x
V1's 111 ms/light CPU baseline) — and, with per-light raw contributions
cached and composited in a separate weighted-sum pass (the Eq. 1/Eq. 3
hoist), the 8-light scripted slider session beats this rung's 33 ms
*stretch* target in both scenarios: 1.3 ms p95 for rgb nudges, 30.9 ms p95
worst-case when a shape-param edit invalidates one light's cache. Live
multi-light editing at production rig scale is real-time on this runtime.
Dynamic
geometry is the sharper update: G1's toy-scale "frozen base + shard residual"
result (a genuine improvement over E2's settled fine-tune negative, at toy
scale) does **not** transfer to a real scene — H5 re-traced the real T1
kitchen scene with an object moved (`nrp.mitsuba_exporter
.apply_shape_translation`, new this track) and reran G1's regime table at
real scale: neither the incremental fine-tune (14.53 dB short) nor the
frozen-base-plus-residual approach (40.62 dB short) meets the 1 dB recovery
target against a converged full retrain (`out/h5-kitchen-fixed/report.json`).
**This does not soften the games verdict — it confirms the structural
dynamic-geometry blocker holds at real scale, not just in a toy fixture.**
Verdict unchanged: *not the right primitive as a core renderer yet*, now on
stronger (real-scene) evidence rather than toy-only evidence.

*Animated film / Feature VFX (grouped — the new evidence applies to both
identically).* The "512x512 cornell-box scale" caveat in the original table
is now materially out of date: T1-F2 moved every measured production-track
claim to a real Mitsuba gallery scene (the "Country Kitchen," 512x512,
52.3M-segment cache, `out/kitchen-512/`), not a procedural cornell box. Three
concrete updates from the hardening track on top of that real-scene base:

- **Storage.** F2's residual-identity shot cost 1.17x raw frame bytes — a
  named negative. H6 flips it: gating exact-residual storage to a subset of
  "approval" frames (proxy-only elsewhere, gated at preview tier) beats raw
  storage starting at 0.589x with *zero* flagged frames at every approval
  fraction tested down to 1-in-12 (`out/h6-storage/sweep_report.json`). The
  other lever tested, int8 residual quantization, is a documented floor, not
  a win — it fails the declared gate on every frame at every fraction tested.
- **Production light rigs.** V1's additivity gate failed because 3 of 8
  per-light proxies produced exactly zero output (the QuadLight zero-collapse,
  diagnosed and fixed in H1). H2 retrained the full 8-light rig post-fix: all
  8 proxies now produce real, nonzero output, but the additivity gate still
  **fails preview tier** (SSIM 0.725 < 0.80, FLIP 0.257 > 0.15;
  `out/h2-rig/report.json`) — a genuine, if partial, improvement, not a
  close. V2's art-direction loop recovery is similarly partial once checked
  by hand rather than by the automated threshold alone: 5 of 6 colorable
  lights are genuinely gradient-recovered, not 6 of 6 as the automated
  `recovery_caveats` check reports (`out/h2-v2-artloop/report.json`; the
  sixth light's raw output is technically nonzero but 4-5 orders of
  magnitude dimmer than the other five, a false negative in that check's
  fixed epsilon). H3 additionally establishes that the rig's two
  `TexturedQuadLight` proxies cannot be brought into the sphere/quad quality
  envelope by more iterations or more model capacity — a different texture-
  conditioning input scheme is the concrete next step
  (`out/h3-textured-quad/report.json`).
- **Net effect on the verdict.** Production light rigs on real shot
  complexity remain the named blocker, exactly as the original table said —
  but it is no longer a *missing measurement*: the rig has been retrained
  post-fix, checked by hand, and the gap is now precisely characterized
  (additivity SSIM/FLIP margin, and which specific light type/proxy is the
  quality floor) rather than attributed to a since-fixed training bug.

**Updated verdict table (2026-07-16, supersedes the table above):**

| target | verdict | why |
|---|---|---|
| Games | Not the right primitive as a core renderer yet (unchanged, now on real-scene evidence) | WebGPU rig compositing is now genuinely real-time for the 8-light scripted session (H4: clean parity; cached-contribution compositing gives 1.3 ms p95 rgb nudges, 30.9 ms p95 worst-case param edits — both beat the 33 ms stretch target). The remaining blocker is structural, not latency: dynamic geometry is confirmed at real scale (H5: neither tested regime meets the 1 dB recovery target on a real re-traced scene). |
| Animated film | Viable component (unchanged), with the "cornell-box only" caveat resolved | T1-F2 proved the full pipeline on a real Mitsuba gallery scene, not a procedural cornell box. F2's storage negative is flipped (H6: 0.589x raw, zero quality cost, at the swept crossover). The remaining gap — production light rigs — is now precisely measured rather than attributed to a bug: 8/8 proxies contribute post-H1-fix, additivity still misses preview tier, and the textured-quad quality floor is diagnosed (input representation, not budget/capacity). |
| Feature VFX | Viable component (unchanged), same real-scene/storage/rig updates as animated film | Per-shot caches and art-direction loops are proven on the real kitchen scene; art-direction color recovery is genuinely 5/6, not the automatically-reported 6/6, once checked against the authored targets by hand — a concrete, actionable finding for anyone reusing this pipeline's own quality-gate machinery, not just a footnote. |

## Status of E1-E9 (per-extension, for completeness)

E10's own measure of rigor is that every claim above traces to a report, not that
every extension reached a positive result — E2's negative finding and E6's
the now-resolved WebGPU native-binding bisection are deliverables in their own right, per this
program's "honest negative results are deliverables" convention.

- E1's time-conditioned proxy criterion is met at toy scale (K=3, small camera
  range); validating it at larger K or motion range is a possible follow-up, not a
  blocking gap.
- E2's multi-bounce invalidation is now measured (swept-volume mask closes the
  33.43 dB → exact gap primary-only invalidation leaves at 2 bounces); it is
  conservative (over-invalidates) and not yet wired into the per-frame timing loop.
  Warm-started TorchNRP weight fine-tuning failed the 1 dB recovery target
  (19.7 dB gap); a replay-regularized retry (self-distillation on unchanged pixels)
  closes it to 11.1 dB — a real, substantial improvement, but still far short of
  1 dB. **This is now a settled structural finding**: segment-local TorchNRP
  fine-tuning, with or without simple replay, cannot keep a live proxy under
  geometry changes to the required accuracy at this model capacity.
- E5 is fully closed at 512x512/128spp scale (see `docs/performance.md`); the
  remaining distance is a performance one (I/O/Python-loop bound streaming), not a
  missing measurement.
- **E6 is fully closed.** GUI slider: self-contained JS viewer, 1.0e-7 parity vs
  PyTorch. Real MPS timings: measured, neither CPU nor MPS hits 30 fps at 512x512
  for the TorchScript backend. WebGPU: a native-binding-only attempt
  (`webgpu/bench.mjs`) reproducibly segfaulted on real proxy weights (0/25+
  trials) despite a synthetic-weight smoke test of the identical pipeline
  succeeding reliably (8/8); a ~150-trial bisection (`webgpu/README.md`) isolated
  this to a defect in that specific native binding. Running the byte-for-byte
  identical compute shader inside real Chrome via Playwright
  (`webgpu/bench_browser.mjs`) resolved it completely: 2.4e-7 parity vs PyTorch,
  30/60 fps cleared at 128/256/512² (580 fps at 128², 107 fps at 512² — better than
  the TorchScript CPU/MPS matrix). The native-binding attempt is kept as a
  documented negative result that correctly predicted where the defect lived.
- **E7 is fully closed** at toy scale: the high-quality proxy run (pretrained
  before inversion) and a genuinely hand-authored (not render-derived) target
  fixture both have measured evidence.
- E9 is fully closed at 512x512 production scale (see `docs/performance.md`).

Run `mise run pipeline-audit` to verify that every `out/` artifact referenced above
exists.

## What Happens Next

This verdict is not an endpoint: each named blocker above maps to a rung of the
production track (`docs/production-track.md`), which pairs every
scale/robustness proof with a performance target and ends in one summit demo
per target audience.

| named blocker (above) | production-track rung |
|---|---|
| Frozen transport under dynamic geometry (games; E2's settled fine-tune failure) | G1 — partitioned residual retraining, evaluated against E2's exact failed recovery target |
| Scale proof is cornell-box-only (animated film) | T1 — real academic scene ingested, trained, and measured end to end (then T2 streaming, F1/F2 shot harness) |
| Production light rigs and layered controls (feature VFX) | V1 — ≥ 8-light rig with per-light layered proxies and verified additivity (then V2 art-direction loop) |
| True production-shot complexity (animated film) | F1/F2 — ≥ 120-frame shot through the quality-tier ladder with a temporal stability metric, rendered at final tier |

See `docs/tracks.md` for the full phase progression.
