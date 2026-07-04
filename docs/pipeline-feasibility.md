# Pipeline Feasibility

This is the current E10 decision document for the extension program in
`docs/extensions.md`. It is intentionally an interim verdict: several E1-E9 completion
criteria remain open, so the table below answers the north-star question only from the
measurements currently present in local reports.

## Evidence Base

Measured reports used here:

| extension | report | measured scope |
|---|---|---|
| E1 animated lights | `out/animate/report.json` | static-camera animated-light sequence |
| E1 animated camera | `out/time-camera/report.json` | camera-keyframe cache scaling and image-space interpolation baseline |
| E2 dynamic geometry | `out/dynamic-geometry/report.json` | one-bounce primary-visibility cache splicing |
| E3 light-aware sampling | `out/light-aware-sampling/report.json` | spherical guide-region sampling density |
| E3 proxy A/B | `out/light-aware-proxy-ab/report.json` | toy standard-vs-guided proxy training comparison |
| E4 environment light | `out/environment-fit/report.json` | degree-2 SH inverse recovery |
| E4 textured quad | `out/textured-quad-fit/report.json` | textured-quad reference inverse recovery |
| E5 out-of-core | `out/out-of-core/report.json` | sharded cache, streamed target table, tiled inference |
| E6 exported runtime | `out/engine-runtime/report.json` | TorchScript artifact, CPU resolution sweep, headless slider loop |
| E7 image target loop | `out/generative/report.json` | synthesized scribble and stylized target optimization |
| E8 production controls | `out/production-controls/report.json` | gather-time controls and binary toggle proxy |
| E9 quality tiers | `out/quality/report.json` | preview/draft/final PSNR/SSIM/FLIP and residual identity |

The original torch toy proxy report at `out/toy-torch/torch_train_report.json` is used
only as baseline context. Claims below are measured at toy scale unless stated
otherwise. No production-scale extrapolation is treated as achieved.

## Summary Table

| target | interactivity met? | quality tier reached | hardest structural blocker | estimated engineering distance |
|---|---|---|---|---|
| Games | Partly, for static scenes and exported proxy loops | Preview only | Dynamic geometry and frozen transport | Large: multi-bounce invalidation, live controls, engine backend |
| Animated film | Partly, for animated lights and final-frame residual identity | Draft/final plumbing exists, not final-frame trust | Animated camera/characters and high-spp validation | Medium-large: per-shot caches plausible, scale proof missing |
| Feature VFX | Partly, for per-shot relighting and art-direction loops | Draft/final plumbing exists, no supervisor-trust verdict | Production light rigs and layered controls | Medium: useful component, not a full renderer primitive |

## Games

**Interim verdict: not the right primitive as a core renderer yet; viable component for
static-scene relighting.**

Measured support:

- The exported runtime path in `out/engine-runtime/report.json` reports 0.46 ms/frame
  exported inference and 1.19 ms/frame mean headless slider-loop latency at 32x32.
  That meets a 16 ms frame budget at toy resolution. The same report now includes
  CPU/MPS 128/256/512 rows; CPU reaches 32.2 fps at 512x512, while MPS is recorded as
  unavailable in this PyTorch build.
- E1 animated-light evaluation in `out/animate/report.json` reports 4.81 ms/frame at
  48x48, with proxy frame-to-frame delta 1.05x the GATHERLIGHT reference. Static-scene
  animated lights are aligned with a game-style live lighting edit.
- E2 in `out/dynamic-geometry/report.json` reports one-bounce cache splicing at
  0.71 ms/frame, plus a warm-start image-proxy repair at 0.30 ms/frame. Together
  they use 6.3% of a 16 ms frame, recover exactly versus full retrace for the
  spliced cache, and leave the image proxy at 65.86 dB minimum PSNR versus full
  retrace for that constrained case.

Blocking evidence:

- The same E2 report is explicitly one-bounce and primary-visibility only. It does not
  prove secondary transport invalidation or TorchNRP weight fine-tuning.
- E8 in `out/production-controls/report.json` shows production controls survive at
  gather time, that one binary linking toggle can be precomputed into a table proxy,
  and that a learned linear image proxy can predict a held-out attenuation setting at
  333.65 dB PSNR with 103x speedup versus attenuated gather.
  It does not prove arbitrary live control masks or arbitrary attenuation curves at
  neural proxy speed.
- E3 in `out/light-aware-proxy-ab/report.json` improves the fixed in-region proxy
  result by 10.10 dB on a geometric open-top-box occluder fixture and does not
  regress the fixed open-region light. This is still a toy lampshade-style fixture,
  not a production lighting scene.

Engineering versus structural:

- Engineering blockers: exported backend beyond TorchScript, WebGPU runtime matrix,
  a host with available MPS for actual MPS timings, streamed optimizer training.
- Structural blockers: frozen transport under dynamic geometry, per-light-type proxy
  boundaries, arbitrary controls that still require cache access.

## Animated Film

**Interim verdict: viable component for interactive lighting preview on mostly static
sets, not yet a complete preview-to-final pipeline.**

Measured support:

- E1 animated-light playback is already flat enough at toy scale to support live
  keyframed light review (`out/animate/report.json`).
- E1 camera-keyframe baseline in `out/time-camera/report.json` traces K=3 camera
  caches totaling 798,180 bytes and reaches 26.64 dB mean held-out image-space
  interpolation PSNR. This is useful camera-cache scaling evidence, not a trained
  time-conditioned proxy.
- E9 in `out/quality/report.json` proves the approval-frame residual identity: proxy
  plus cached residual matches cached GATHERLIGHT at the approved config to max
  absolute error 5.6e-17. Its toy trust verdict is to trust only the approved frame
  and re-bake after any measured light move because dx=0.05 drops to 24.67 dB.
- E5 in `out/out-of-core/report.json` shows streamed fixed-light target construction
  matches monolithic targets to max error 3.33e-16 while loading only 11.1% of cache
  segments and 11.1% of resident segment bytes at once, a 9.0x resident segment-memory
  reduction estimate at toy scale. The same report now trains a tiny streamed
  per-pixel image-proxy optimizer to the same result as the monolithic optimizer
  (max diff 3.33e-16).

Blocking evidence:

- E1 animated camera has only an image-space interpolation baseline; the requested
  single neural proxy conditioned on time/camera inputs is not implemented.
- E9 has no high-spp production-scale cache and no production supervisor trust verdict.
- E5 has no 512x512 / 128 spp Mitsuba run, no production-scale peak RSS comparison,
  and no streamed TorchNRP optimizer yet.

Engineering versus structural:

- Engineering blockers: high-resolution cache streaming, GPU/export backend, final
  quality metrics.
- Structural blockers: animated characters still require invalidation/retraining
  machinery, not just cached relighting.

## Feature VFX

**Interim verdict: viable component for per-shot relighting and physically grounded
art-direction loops, not a standalone production renderer.**

Measured support:

- E7 in `out/generative/report.json` demonstrates the image-target-to-physical-light
  loop and reports proxy-space versus GATHERLIGHT errors separately. The synthesized
  scribble fixture passes its mask/protect thresholds, and
  `out/generative/provenance.json` records deterministic fixture recipes and SHA-256
  hashes for the current toy targets.
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
  precomputed binary linking toggle live through a table proxy, and keeps one
  fixed-family continuous attenuation control live through a learned linear proxy.

Blocking evidence:

- E4 is now complete at toy scale, but the first-class TorchNRP smoke only covers
  2x2 textures; higher-resolution first-class runs remain a scale/quality follow-up.
- E7 stylized target realization remains physically limited: the report is useful
  precisely because it exposes the gap between an arbitrary image target and a
  physically realizable lighting setup. E7 still has no high-quality proxy run and no
  true hand-authored or external generative image fixture.
- E8 arbitrary proxy-conditioned masks and arbitrary custom attenuation curves are
  not implemented.

Engineering versus structural:

- Engineering blockers: richer light embeddings, compositor/export integration,
  larger-scene reports.
- Structural blockers: inverse optimization can only realize targets within the
  chosen physical light family; arbitrary generative edits remain constrained.

## Current Verdicts

| target | verdict | why |
|---|---|---|
| Games | Not the right primitive as a core renderer yet | Dynamic everything is the core requirement, and only one-bounce cache splicing is proven. |
| Animated film | Viable component | Static-set animated-light preview and residual identity are promising, but final-frame trust and neural animated-camera support are unproven. |
| Feature VFX | Viable component | Per-shot caches and art-direction loops fit VFX workflows, but richer lights and learned proxy-conditioned controls remain incomplete. |

## Open Work Before Final E10 Completion

- Finish E1 animated camera/time-conditioned proxy validation.
- Finish E2 multi-bounce invalidation and warm-started TorchNRP weight fine-tuning.
- Finish E5 streamed TorchNRP optimizer training plus 512x512 / 128 spp Mitsuba
  report.
- Finish E6 WebGPU 128/256/512 exported-runtime matrix, real MPS timings on an
  MPS-enabled PyTorch build, and GUI slider.
- Finish E7 high-quality proxy run and a true hand-authored or external generative
  image fixture with provenance.
- Finish E8 learned proxy-conditioned control comparison for arbitrary masks and
  arbitrary attenuation curves.
- Finish E9 high-spp production-scale final-frame trust verdict.

Run `mise run pipeline-audit` to verify that every `out/` artifact referenced above
exists.
