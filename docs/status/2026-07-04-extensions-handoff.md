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

Important limitation:

- This is not TorchNRP weight fine-tuning.
- This is not multi-bounce invalidation. Unchanged first-hit pixels can still see
  changed indirect transport once secondary bounces matter.

Remaining E2 criterion:

- Implement and measure multi-bounce invalidation.
- Replace or augment the image-space baseline with warm-started TorchNRP weight
  fine-tuning.

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

Remaining E4 criterion:

- TorchNRP must be conditioned on learned texture embeddings.
- The proxy held-out quality study should use that learned TorchNRP conditioning,
  not only the reference linear baseline.

### E5 Out-of-Core

Current evidence: `out/out-of-core/report.json`.

What exists:

- Sharded cache foundation.
- Streamed fixed-light target table.
- Tiled inference.
- Reported resident segment-byte ratio: 9.0x lower peak streamed segment bytes
  than the monolithic resident segment bytes for the current toy run.

Remaining E5 criterion:

- Streamed optimizer training.
- 512x512 / 128 spp Mitsuba report with peak RSS, throughput, and held-out PSNR.

### E6 Engine-Shaped Runtime

Current evidence: `out/engine-runtime/report.json` and `docs/engine-integration.md`.

What exists:

- TorchScript-shaped exported runtime path.
- CPU 128/256/512 matrix.
- MPS rows are recorded as unavailable in the current PyTorch build.
- Headless slider-loop style runtime measurement.

Remaining E6 criterion:

- WebGPU or engine-backend matrix.
- Real MPS timing on an MPS-enabled PyTorch build.
- GUI slider/viewer evidence.

### E7 Generative Loop

Current evidence:

- `out/generative/report.json`
- `out/generative/provenance.json`

What exists:

- Synthesized scribble/stylized target loop.
- Deterministic fixture provenance with SHA256s.
- Physical-realization gap is reported.

Remaining E7 criterion:

- High-quality proxy run.
- True hand-authored or external generative image fixture with provenance.

### E8 Production Controls

Current evidence: `out/production-controls/report.json`.

What exists:

- Gather-time light linking.
- Gather-time linear attenuation.
- Narrow proxy/table baselines for one binary linking toggle and one linear
  attenuation setting.

Remaining E8 criterion:

- Learned proxy-conditioned comparison for arbitrary masks.
- Learned proxy-conditioned comparison for arbitrary attenuation curves.

### E9 Final-Frame Quality

Current evidence: `out/quality/report.json`.

What exists:

- `--quality preview|draft|final` plumbing.
- Metadata sidecars.
- Cached residual identity at the approved light config.
- Toy residual-validity trust verdict: trust the approved frame only; re-bake after
  any measured light move.

Remaining E9 criterion:

- Fresh high-spp production-scale cache.
- Production-scale final-frame trust verdict.

### E1 Animated Lights and Camera

Current evidence:

- `out/animate/report.json`
- `out/time-camera/report.json`

What exists:

- Animated-light static-camera harness.
- Camera-keyframe cache scaling and image-space interpolation baseline.

Remaining E1 criterion:

- Single neural proxy conditioned on time/camera inputs.
- Held-out intermediate-camera proxy PSNR within 3 dB of training-keyframe PSNR, or
  a report documenting the gap.

## Suggested Next Work

Recommended next slice: E4 learned texture conditioning.

Reason: E3 is now materially advanced, E2's remaining work is larger and touches
training dynamics, and E4 has a clear boundary: add a learned or compact texture
conditioning path to the TorchNRP workflow and report held-out PSNR vs texture
resolution. Start by reading:

- `skills/nrp-torch-proxy-workflow/SKILL.md`
- `nrp/torch_backend/model.py`
- `nrp/torch_backend/train.py`
- `nrp/texture_fit.py`
- `examples/textured_quad_fit.py`
- `tests/test_texture_fit.py`

Alternative next slice: E5 streamed optimizer training.

Reason: E5 is a direct production-scale blocker. Start by reading:

- `skills/nrp-path-cache-and-gather/SKILL.md`
- `examples/out_of_core.py`
- `nrp/path_cache.py`
- `nrp/torch_backend/train.py`

## Reporting Rules To Preserve

- Every new quantitative claim should land in an `out/.../report.json` file and in
  `docs/performance.md`.
- Update `docs/paper-mapping.md` whenever a change alters paper coverage or known
  deviations.
- Keep `docs/pipeline-feasibility.md` current; do not remove an open item unless
  the current evidence satisfies the actual `docs/extensions.md` criterion.
- Run the pipeline claims audit after changing `docs/pipeline-feasibility.md`.
- Keep generated docs, skills, and handoffs free of local machine paths.
