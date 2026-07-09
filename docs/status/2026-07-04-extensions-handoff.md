# Extensions Handoff - 2026-07-04

Scope: continuation notes for the `docs/extensions.md` goal: work through E1-E10
until each completion criterion is satisfied and the final pipeline-feasibility
verdict is evidence-backed. This handoff is intentionally repo-local and uses
project-relative paths only.

## Current Branch State

- Branch: `codex/nrp-extensions`.
- At handoff time the branch is ahead of `origin/codex/nrp-extensions`.
- Repo-local skills have been pulled in from `main`:
  - `skills/nrp-experiment-reporting/SKILL.md`
  - `skills/nrp-path-cache-and-gather/SKILL.md`
  - `skills/nrp-torch-proxy-workflow/SKILL.md`
- Primary tracker: `docs/pipeline-feasibility.md`, especially
  "Open Work Before Final E10 Completion".

## Verification Commands Used

Use the same cache env for sandboxed runs:

```sh
UV_CACHE_DIR=.uv-cache uv run ruff check .
UV_CACHE_DIR=.uv-cache uv run python -m unittest discover -s tests
UV_CACHE_DIR=.uv-cache uv run python examples/audit_pipeline_claims.py \
  --doc docs/pipeline-feasibility.md \
  --out out/pipeline-feasibility/audit.json
```

Useful targeted checks from this cycle:

```sh
UV_CACHE_DIR=.uv-cache uv run python -m unittest tests.test_dynamic_geometry
UV_CACHE_DIR=.uv-cache uv run python -m unittest tests.test_light_aware_sampling
UV_CACHE_DIR=.uv-cache uv run python examples/dynamic_geometry.py \
  --out out/dynamic-geometry/report.json
UV_CACHE_DIR=.uv-cache uv run python examples/light_aware_proxy_ab.py \
  --out out/light-aware-proxy-ab/report.json
```

Last broad verification after the E3 occluder work:

- `ruff check .`: passed.
- `tests.test_light_aware_sampling`: passed, 4 tests.
- Full unit suite: passed, 191 tests, 16 skipped.
- Pipeline claims audit: passed.
- Path-hygiene scan over touched docs/code: passed.

## Completed or Advanced Extension Slices

### E2 Dynamic Geometry

Current evidence: `out/dynamic-geometry/report.json`.

What exists:

- One-bounce primary-visibility invalidation and segment splicing.
- `WarmStartImageProxy` image-space repair baseline in
  `examples/dynamic_geometry.py`.
- Targeted coverage in `tests/test_dynamic_geometry.py`.
- Docs updated in `docs/performance.md`, `docs/pipeline-feasibility.md`, and
  `docs/paper-mapping.md`.

Key numbers:

- Mean full retrace: 4.45 ms/frame.
- Mean splice pass: 0.71 ms/frame.
- Mean image-proxy fine-tune: 0.30 ms/frame.
- Splice plus image-proxy: 6.3% of a 16 ms frame.
- Mean invalid pixels: 29.7%.
- Spliced cache vs full retrace: exact for the one-bounce fixture.
- Image-proxy PSNR after repair: 65.86 dB minimum, 69.69 dB mean.

Update (2026-07-08): warm-started TorchNRP weight fine-tuning is now implemented
and measured (`TorchNRPWarmStartProxy` in `examples/dynamic_geometry.py`), with a
negative result — see `docs/performance.md` "TorchNRP weight fine-tune regimes".
Regime (a) full-retrace-retrain reaches 44.88 dB mean PSNR vs the fully retraced
frame; regime (b) incremental splice + masked-pixel fine-tune only reaches 25.20 dB
(19.7 dB short of the 1 dB criterion), barely ahead of doing nothing at all (regime
(c) stale: 22.13 dB). Root cause: fine-tuning only against the invalidated pixels'
loss has no regularizer holding the shared MLP's fit on unchanged pixels steady, so
the network drifts. Test coverage:
`tests/test_dynamic_geometry.py::test_torchnrp_warm_start_fine_tune_only_touches_masked_pixels`,
`::test_torchnrp_warm_start_no_op_on_empty_mask`.

Important limitation:

- This is not multi-bounce invalidation. Unchanged first-hit pixels can still see
  changed indirect transport once secondary bounces matter.

Update (2026-07-08): multi-bounce invalidation is now implemented and measured.
`nrp.dynamic_geometry.swept_volume_invalidation_mask` flags any pixel with a cached
segment at any bounce depth passing through the moving object's conservative
swept-volume bound. `examples/dynamic_geometry.py::multi_bounce_invalidation_comparison`
(`out/dynamic-geometry/report.json`, field `multi_bounce_invalidation`) shows the
failure-then-fix: at 32x32/8spp/2-bounce, primary-visibility-only invalidation
misses 534 pixels with changed indirect bounces (33.43 dB vs a full retrace, not
exact); adding the swept-volume mask catches them and recovers an exact match. See
`docs/performance.md` "Multi-bounce invalidation: swept-volume masking". Test
coverage: `tests/test_dynamic_geometry.py::test_swept_bounding_sphere_contains_endpoints_and_object_extent`,
`::test_swept_volume_mask_flags_pixels_with_any_bounce_depth_segment_in_region`,
`::test_multi_bounce_spliced_cache_matches_full_retrace_with_swept_mask`.

Update (2026-07-08): the replay-regularized retry is done.
`TorchNRPWarmStartProxy.fine_tune_with_replay` mixes self-distillation targets
(the model's own pre-update predictions on unchanged pixels) into the incremental
fine-tune batch. Result: regime (b2) reaches 33.76 dB, closing the gap from 19.7 dB
to **11.1 dB** — a real, substantial improvement, but still far outside the 1 dB
criterion. **Conclusion: the 1 dB target is not reachable with segment-local
fine-tuning (with or without simple replay) at this model capacity** — this is now
a settled negative result, not an open question. See `docs/performance.md`
"Replay-regularized retry (regime b2)". Test coverage:
`tests/test_dynamic_geometry.py::test_torchnrp_fine_tune_with_replay_runs_and_produces_finite_loss`,
`::test_torchnrp_fine_tune_with_replay_no_op_on_empty_mask`.

Remaining E2 criterion:

- The swept-volume mask is conservative (over-invalidates); it is not yet integrated
  into the per-frame regime (a)/(b)/(c) loop above, which is still one-bounce only.

### E3 Light-Aware Sampling

Current evidence:

- `out/light-aware-sampling/report.json`
- `out/light-aware-proxy-ab/report.json`

What exists:

- Light-aware cone/cosine mixture sampling in the toy tracer via `light_region` and
  `guide_probability`.
- Optional `open_top_box` occluder fixture in `nrp/toy_tracer.py`.
- `trace.occluder` forwarding through `nrp/train.py`.
- E3 standard-vs-guided proxy A/B in `examples/light_aware_proxy_ab.py`.
- Targeted coverage in `tests/test_light_aware_sampling.py`.

Key A/B numbers from `out/light-aware-proxy-ab/report.json`:

- Fixture: `open_top_box`.
- Resolution/spp/bounces: 20x20 / 8 spp / 3 bounces.
- Iterations: 350.
- Region-hit fraction: standard 0.47%, guided 15.73%.
- Fixed in-region PSNR: standard -6.72 dB, guided 3.38 dB.
- Guided in-region gain: 10.10 dB.
- Fixed open-region PSNR: standard 7.35 dB, guided 16.44 dB.
- Open-region delta: +9.09 dB.
- Equal segment budget: yes.
- Target met fields:
  - `inside_region_target_met: true`
  - `open_region_regression_within_0p5db: true`

Status:

- The explicit E3 target of at least 3 dB guided-proxy improvement is now satisfied
  on a literal toy open-top-box occluder fixture.
- The caveat is realism: this is a toy lampshade-style box, not a production asset.

### E4 Richer Light Models

Current evidence: `out/textured-quad-fit/report.json` and
`out/environment-fit/report.json`.

What exists:

- Reference inverse recovery for SH environment lighting.
- Textured-quad reference inverse recovery through 8x8.
- Linear texture-proxy scaling baseline showing the equal-observation 8x8 case is
  underdetermined and collapses on held-out samples.
- Compact learned texture-embedding proxy scaling for 2x2, 4x4, and 8x8 textures.
- First-class `textured_quad` support in the main TorchNRP train/relight path for a
  small 2x2 texture smoke.

Key learned-proxy numbers:

- Mean held-out PSNR: 20.08 dB for 2x2, 21.19 dB for 4x4, and 22.27 dB for 8x8.
- First-class TorchNRP textured-quad smoke: 20 light parameters, 2,005 model
  parameters, 13.27 dB validation PSNR, 12.31 dB held-out relight PSNR.

Status:

- E4 is complete at toy scale against the explicit extension criteria.
- Remaining follow-up is quality/scale, not basic API coverage: the first-class
  TorchNRP smoke covers 2x2 textures, while the broader 8x8 result is in the
  learned texture-proxy experiment harness.

### E5 Out-of-Core

Current evidence: `out/out-of-core/report.json`.

What exists:

- Sharded cache foundation.
- Streamed fixed-light target table.
- Streamed per-pixel image-proxy optimizer that matches the monolithic optimizer.
- Tiled inference.
- Reported resident segment-byte ratio: 9.0x lower peak streamed segment bytes
  than the monolithic resident segment bytes for the current toy run.

Update (2026-07-08): streamed TorchNRP optimizer training is now closed at toy
scale. `nrp/torch_backend/streamed_train.py` trains a real sphere-light `TorchNRP`
from a `StreamedImagePool` that renders pool targets by visiting shard tiles
(`out/out-of-core/streamed_torchnrp_report.json`, `mise run streamed-torchnrp`):
bitwise-identical loss curve and held-out PSNR (9.12 dB) to the monolithic
in-memory run at 200 iterations/seed 0, with 11.1x lower peak resident segment
bytes. See `docs/performance.md` "Streamed TorchNRP pool training". Test coverage:
`tests/test_out_of_core.py::test_streamed_torchnrp_training_matches_monolithic`.

Update (2026-07-08): the 512x512/128spp Mitsuba end-to-end report is done —
`out/out-of-core/mitsuba_512_report.json` via `mise run export-mitsuba-512` +
`mise run mitsuba-512-streamed`. 94.2M segments, 3.35 GB monolithic cache / 8.29 GB
resident segment bytes vs 574 MB peak streamed (14.4x lower), streamed train
~16.7 min (150 iters), held-out PSNR 32.57 dB, tiled full-frame inference 118.3 ms.
See `docs/performance.md` "512x512 / 128 spp Mitsuba end-to-end report". **E5 is now
closed at this scale** — the caveat is that streaming cost is I/O/Python-loop bound,
not compute bound (a batched streaming gather would speed it up, engineering not
structural).

### E6 Engine-Shaped Runtime

Current evidence: `out/engine-runtime/report.json` and `docs/engine-integration.md`.

What exists:

- TorchScript-shaped exported runtime path.
- CPU 128/256/512 matrix.
- Headless slider-loop style runtime measurement.

Update (2026-07-08): this machine's PyTorch build now has a working MPS backend
(previously reported unavailable). Re-ran `mise run viewer` for real MPS numbers:
128x128 106.1 ms/frame (9.4 fps, slower than CPU — dispatch overhead dominates at
this size), 256x256 28.6 ms/frame (35.0 fps), 512x512 38.15 ms/frame (26.2 fps).
Neither CPU nor MPS hits 30 fps at 512x512 for this exported artifact on this
hardware. See `docs/performance.md` "Exported engine-shaped runtime" for the full
table.

Update (2026-07-08): the GUI slider gap is closed.
`examples/export_js_viewer.py` (`out/engine-runtime/js_viewer/viewer.html`) trains a
`use_encoding=False` proxy, exports raw weights, and writes a self-contained HTML
page with 4 real range-input sliders (light center xyz + radius) that recompute the
full image through a ~40-line JS reimplementation of the forward pass on every move
— a genuine interactive GUI (no display/GUI toolkit exists in this environment for a
native app, so a browser page is the honest choice). JS-vs-PyTorch parity: 1.0e-7 max
abs diff, verified both in-page and by
`tests/test_export_js_viewer.py::test_js_forward_pass_matches_python_reference`
(runs the JS under Node, skips cleanly if absent). See `docs/performance.md` "A real
GUI slider, and a step toward the WebGPU/engine-backend criterion".

Update (2026-07-08): a genuine WebGPU compute-shader backend was implemented and
**closes E6's last criterion**. First attempt: `webgpu/bench.mjs` (native Dawn
bindings via the `webgpu` npm package, no browser) reproducibly segfaulted when
running the real exported proxy's weights (0/25+ trials). A ~150-trial bisection
(table in `webgpu/README.md`) narrowed this to a defect in that specific native
binding's handling of real trained-model float32 data — ruled out magnitude,
distribution, buffer size, memory layout, and three independent package versions.
Second attempt, `webgpu/bench_browser.mjs` (`mise run webgpu-bench`, report:
`out/engine-runtime/webgpu_browser_report.json`): the byte-for-byte identical
compute shader, executed inside real Google Chrome via Playwright against the same
real exported proxy weights, has **no such issue** — parity 2.4e-7 vs the PyTorch
reference, and a 128/256/512 latency sweep clearing 30 and 60 fps at every
resolution (580 fps at 128², 107 fps at 512²). This confirms the bisection's
diagnosis (the defect was specific to the experimental Node-only Dawn binding, not
to WebGPU, this project's shader, or its JS code) and delivers the actual E6
criterion via a production WebGPU implementation. See `docs/performance.md` "A real
WebGPU compute-shader backend (closes E6's last criterion)". Test coverage:
`tests/test_webgpu_browser_bench.py` (end-to-end, spawns real Chrome).

**E6 is now fully closed**: exported TorchScript runtime with real CPU/MPS timings,
a real interactive GUI slider (JS/canvas), and a real WebGPU compute-shader backend
running the actual exported proxy, all measured on this machine.

### E7 Generative Loop

Current evidence:

- `out/generative/report.json`
- `out/generative/provenance.json`

What exists:

- Synthesized scribble/stylized target loop.
- Deterministic fixture provenance with SHA256s.
- Physical-realization gap is reported.

Update (2026-07-08): the "high-quality proxy run" gap is closed. `pretrain_proxy` in
`examples/generative_loop.py` trains the model on 24 random sphere lights (800 Adam
steps) before it is used for inverse optimization — previously `optimize()`
differentiated through a randomly initialized, never-trained network. Windowed mean
loss drops 0.330 → 0.169. Physical re-render quality (13.19 dB best restart) is
essentially unchanged from before pretraining (14.13 dB), because a single sphere
light's expressiveness — not proxy quality — is the binding constraint on this
fixture. Test coverage: `tests/test_generative_loop.py::test_pretrain_proxy_reduces_windowed_loss`.

Update (2026-07-08): the hand-authored fixture gap is closed.
`examples/hand_authored_target.py` (`out/generative/hand_authored_report.json`) is a
target with no algorithmic connection to any render or light — an explicit,
hand-picked `(row, col, RGB)` stroke list (a plus-sign pixel-art shape) typed
directly into the file, distinct from the other fixtures' render-derived content.
Same pipeline as the rest of E7 (pretrained proxy, 2-light optimization, 3
restarts): 12.56 dB target-vs-realized PSNR, same physical-realization-gap finding
as the stylized target. Provenance sets `hand_authored: true,
derived_from_render: false`. See `docs/performance.md` "A genuinely hand-authored
target". Test coverage: `tests/test_hand_authored_target.py` (4 tests).

**E7 is now fully closed at toy scale** — every criterion in `docs/extensions.md`
for E7 (scribble recovery, generative-target workflow, latency sweep,
proxy-space-vs-re-rendered reporting, high-quality proxy run, hand-authored
fixture) has measured evidence.

### E8 Production Controls

Current evidence: `out/production-controls/report.json`.

What exists:

- Gather-time light linking.
- Gather-time linear attenuation.
- Proxy/table baselines for one binary linking toggle, one linear attenuation
  setting, a soft mask-basis control, and a quadratic attenuation curve.

Key numbers:

- Binary linking proxy: exact active/inactive images, 0 max abs versus gather.
- Linear attenuation proxy: 333.65 dB held-out PSNR versus gather.
- Soft 4-basis mask proxy: 331.25 dB held-out PSNR versus reference mask
  application, 5.55e-17 max abs.
- Quadratic attenuation proxy: 323.22 dB held-out PSNR versus gather, 158x faster
  than polynomial gather in the current toy run.

Status:

- E8 is satisfied at toy scale for the measured parameterized controls.
- Follow-up remains scale/UX: fully free-form masks or arbitrary artist curves need
  a correspondingly expressive conditioning vector and training coverage.

### E9 Final-Frame Quality

Current evidence: `out/quality/report.json`.

What exists:

- `--quality preview|draft|final` plumbing.
- Metadata sidecars.
- Cached residual identity at the approved light config.
- Toy residual-validity trust verdict: trust the approved frame only; re-bake after
  any measured light move.

Update (2026-07-08): the production-scale criterion is closed.
`out/quality/production_report.json` (`mise run quality-tiers-production`) reuses
E5's 512x512 Mitsuba caches (32spp export, 128spp converged reference) with a real
streamed-trained proxy: preview 33.76 dB / draft 35.72 dB PSNR vs the 128spp final,
exact residual identity at the approval config, and a trust verdict of "trust the
approved frame only; re-bake after any measured light move" — same qualitative
verdict as the toy report, now backed by a genuinely converged reference and trained
proxy. See `docs/performance.md` "Production-scale trust verdict". **E9 is now
closed at this scale.**

### E1 Animated Lights and Camera

Current evidence:

- `out/animate/report.json`
- `out/time-camera/report.json`

What exists:

- Animated-light static-camera harness.
- Camera-keyframe cache scaling and image-space interpolation baseline.

Update (2026-07-08): the single time-conditioned neural proxy criterion is now
closed at toy scale. `examples/time_conditioned_proxy.py`
(`out/time-camera/proxy_report.json`) trains one `TorchNRP` (light params + a time
scalar) jointly on the K=3 camera keyframes and evaluates held-out intermediate
camera times (t=0.25, 0.75) against freshly traced ground truth at those exact
poses: mean training-keyframe PSNR 28.74 dB, mean held-out PSNR 26.68 dB, gap
**2.06 dB** (criterion: within 3 dB — met). See `docs/performance.md`
"Time-conditioned TorchNRP proxy". Test coverage:
`tests/test_time_conditioned_camera.py::test_light_time_params_appends_time_scalar`,
`::test_pixel_xy_and_aux_shapes`.

E1 is now closed at toy scale on both axes (animated lights, static camera; and a
single time-conditioned proxy across camera keyframes). Caveat: K=3 keyframes and a
small ±0.04 camera range — untested whether the gap holds at larger K or motion
range.

## Suggested Next Work

Recommended next slice: E2 multi-bounce invalidation, or a replay-regularized retry
of the TorchNRP fine-tune to see if the 1 dB criterion is reachable at all.

Reason: E5 is now fully closed (toy streamed optimizer + 512x512/128spp production
report both done). E1 is now closed on both axes (animated lights, and the
time-conditioned proxy within its 3 dB criterion at K=3). E6's real-MPS-timings gap
is closed. E2's TorchNRP fine-tune criterion is now measured but failed (19.7 dB
short) — the next useful experiment is whether mixing a sample of unchanged-pixel
targets into the incremental fine-tune batch (cheap replay regularization) closes
that gap, before concluding it's a structural limit. Multi-bounce invalidation is
still untouched and is the highest structural-risk game-pipeline item remaining.
Start by reading:

- `examples/dynamic_geometry.py`
- `nrp/dynamic_geometry.py`
- `nrp/torch_backend/train.py`
- `tests/test_dynamic_geometry.py`

## Reporting Rules To Preserve

- Every new quantitative claim should land in an `out/.../report.json` file and in
  `docs/performance.md`.
- Update `docs/paper-mapping.md` whenever a change alters paper coverage or known
  deviations.
- Keep `docs/pipeline-feasibility.md` current; do not remove an open item unless
  the current evidence satisfies the actual `docs/extensions.md` criterion.
- Run the pipeline claims audit after changing `docs/pipeline-feasibility.md`.
- Keep generated docs, skills, and handoffs free of local machine paths.
