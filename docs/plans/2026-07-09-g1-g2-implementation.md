# G1 + G2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement production-track rungs G1 (dynamic geometry, second attempt: partitioned residual retraining) and G2 (summit demo: live relight + controls in the browser at 30 fps / 512²).

**Architecture:** G1 keeps E2's frozen base `TorchNRP`, marks invalidated cache shards via E2's swept-volume + primary-visibility masks aggregated to a fixed tile grid, and trains a signed-output *residual* MLP over only the invalidated region, composited additively at inference — evaluated against E2's exact failed 1 dB recovery target with the same fixture, budget, and script style. G2 extends the T4 WebGPU stack (exporter + generated WGSL) into an interactive viewer for the T1 kitchen proxy: animated lights (E1 keyframe format ported to JS), two E8 production controls (per-layer light linking, artist attenuation curve) applied identically in the shader and in the Python GATHERLIGHT reference so the T3 preview gate measures proxy fidelity apples-to-apples, plus (G1 succeeding) a toy-scale moving-object panel driven by exported base+residual proxies. A Playwright runner replays a committed interaction trace, records frame times (T4 methodology, p95 ≤ 33 ms at 512²) and a screen recording, and dumps sampled frames that `examples/g2_gate.py` gates at preview tier against controlled GATHERLIGHT references.

**Tech Stack:** Python (numpy, torch, unittest, ruff), WebGPU/WGSL via generated shaders, Node + Playwright + Chrome, mise tasks, uv.

## Global Constraints

- All new/changed code passes `mise run test` and `mise run lint`; new tests skip cleanly when optional dependencies (node, playwright, exported artifacts, the 2 GiB kitchen cache) are absent.
- Every measured claim lands in a JSON report under `out/` **and** in `docs/performance.md` with hardware context.
- Honest negative results are deliverables; G1's report must state its failure mode if it fails, and it must differ from E2's.
- Single Apple Silicon laptop; no CUDA assumed. Browser work uses real Chrome via Playwright (same flags as `webgpu/bench_t4.mjs`).
- Statistical assertions compare windowed means, never single minibatch losses.
- Commit style: `G1: <summary>` / `G2: <summary>` matching repo history; evidence JSONs under `out/` are committed with `git add -f` (as T2/T4 did).
- T1 artifacts (`out/kitchen-512/path_cache.npz`, `out/kitchen-512-torch/model.pt`, `out/t4-runtime/export/`) live in the main checkout `/Users/briangyss/src/nrp/out/`; symlink them into this worktree's `out/` — never regenerate (830 s train) or copy (2 GiB).
- `mise run t4-check` must still pass after the shader-generator refactor (T4 regression gate).

## E2 fixture lock (G1 must reuse exactly)

From `examples/dynamic_geometry.py::main`: 32×32, 8 spp, `max_bounces=1`, seed 21, 10 frames, sphere moving `dx ∈ linspace(0, 0.16, 10)` along x from `SPHERE_CENTER`; light `SphereLight(center=[0.35, 0.28, 0.62], radius=0.2, rgb=[1.3, 1.0, 0.8])`; proxy `TorchNRP(light_type="sphere", hidden_width=32, hidden_layers=2, use_encoding=False)`, 300 Adam iterations, lr 5e-3, `torch.manual_seed(0)`. E2's measured result: regime (a) 44.88 dB, (b) 25.20 dB (gap 19.7 dB), (b2) 33.76 dB (gap 11.1 dB), (c) 22.13 dB; recovery target: gap ≤ 1 dB.

---

### Task 1: G1 core — shard invalidation + signed residual proxy + compositing (`nrp/torch_backend/residual_dynamic.py`)

**Files:**
- Create: `nrp/torch_backend/residual_dynamic.py`
- Test: `tests/test_residual_dynamic.py`

**Interfaces (produces):**
- `invalidated_shards(mask: np.ndarray[bool H,W], shard_size: int) -> tuple[np.ndarray, list[tuple[int,int]]]` — region mask (union of whole tiles touching any invalid pixel) + sorted tile coords `(ty, tx)`.
- `pixel_features(cache) -> tuple[np.ndarray, np.ndarray]` — `(xy (N,2) float32, aux (N,7) float32)`, same convention as `nrp.torch_backend.train.pixel_tensors` / `examples/dynamic_geometry._pixel_xy/_aux`.
- `class ResidualNRP(nn.Module)` — same `(xy, aux, light_params)` forward signature as `TorchNRP`, `use_encoding=False`-style raw-xy input, **linear head** (signed output); `save(path)` / `ResidualNRP.load(path)` round-trip; `config` dict with `hidden_width`, `hidden_layers`, `light_param_dim`.
- `train_residual(residual, base: TorchNRP, cache, target_image (H,W,3), region_mask, light_params (4,) np.float32, iters, lr) -> list[float]` — fits `residual` to `target - base_pred` on region pixels only; base frozen (`no_grad` for base predictions).
- `composite_predict(base, residual, cache, light_params, region_mask) -> np.ndarray (H,W,3)` — `base_pred` everywhere, `+ residual_pred` on region pixels only.

Tests (grouped in `ResidualDynamicTests`, toy caches from `trace_path_cache` at 16×16 / 2 spp for speed):
1. `test_invalidated_shards_cover_mask_and_align_to_tile_grid` — every invalid pixel is in the region; region is exactly a union of full `shard_size` tiles (each returned tile fully set, everything else clear); tiles without invalid pixels excluded.
2. `test_invalidated_shards_empty_mask_returns_empty` — all-false mask → empty region, no tiles.
3. `test_invalidated_shards_handles_non_divisible_resolution` — 18×18 with shard_size 8 → edge tiles clipped to image bounds.
4. `test_composite_equals_base_outside_region_exactly` — random-init base + residual: composite == base prediction bitwise outside region; differs somewhere inside (residual init is nonzero).
5. `test_residual_head_is_signed` — random-init `ResidualNRP` produces at least one negative output component on a batch of inputs (softplus head cannot).
6. `test_train_residual_reduces_region_error` — windowed-mean loss (first 20 vs last 20 of 150 iters) decreases; composite PSNR-in-region vs target improves over base-only.
7. `test_residual_save_load_roundtrip` — predictions identical after save/load.

**Steps:**
- [ ] Write the failing tests (imports will fail).
- [ ] Run `uv run python -m unittest tests.test_residual_dynamic -v` → import errors.
- [ ] Implement `nrp/torch_backend/residual_dynamic.py`.
- [ ] Tests pass; `uv run ruff check . && uv run ruff format .`.
- [ ] Commit: `G1: shard invalidation, signed residual proxy, and compositing`.

### Task 2: G1 report script — E2-vs-G1 recovery comparison (`examples/residual_dynamic.py`)

**Files:**
- Create: `examples/residual_dynamic.py`
- Modify: `mise.toml` (task `g1-residual`)

**Interfaces:**
- Consumes: Task 1 API; `examples.dynamic_geometry.TorchNRPWarmStartProxy` (regimes a/b/b2/c unchanged); `nrp.dynamic_geometry` masks/splice.
- Produces: `out/g1-residual/report.json` with `recovery_comparison` table (per regime: mean PSNR vs full, gap vs (a), `within_1db_of_a`, mean ms/frame), per-frame detail including **in-mask and out-of-mask PSNR for regimes (b) and (d)** (the failure-mode differentiator), wall-clock `invalidate_and_recover_ms` vs `full_retrace_and_retrain_ms`, `failure_mode` narrative fields, and `t1_scene_feasibility` note. Also saves, for Task 5's browser export: `out/g1-residual/models/base.pt`, `models/residual_frame_XXXX.pt`, `frames/gbuffer_frame_XXXX.npz` (spliced aux buffers), `frames/region_mask_XXXX.npy`, `frames/target_frame_XXXX.npy` (full-retrace GATHERLIGHT image), and `manifest.json` (fixture params).

Regime (d) — G1: base `TorchNRP` trained once on the base frame (`train_full`, 300 iters, identical to E2 base); per frame: mask = `primary_visibility_invalidation_mask | swept_volume_invalidation_mask(radius=0.25, margin=0.05)`, shards via `invalidated_shards(mask, shard_size=8)`, splice, fresh `ResidualNRP(hidden_width=32, hidden_layers=2)` trained 300 iters lr 5e-3 (matched budget vs regime b) against the spliced image, composited. Evaluated vs the full-retrace GATHERLIGHT image, same as E2's regimes.

**Steps:**
- [ ] Implement script (argparse defaults matching the E2 fixture lock above); add mise task.
- [ ] Run `mise run g1-residual`; inspect report; iterate if regime (d) behaves pathologically (budget stays matched; any change documented in the report).
- [ ] Add a smoke test to `tests/test_residual_dynamic.py` exercising the regime-(d) loop at 12×12 / 2 spp / 2 frames / 40 iters (asserts report keys + out-of-mask PSNR of (d) equals base's, within float tolerance).
- [ ] Tests + lint pass. Commit: `G1: recovery-target comparison report (E2 fine-tune vs residual compositing)`; `git add -f out/g1-residual/report.json` (models/frames stay untracked).

### Task 3: G1 docs

**Files:**
- Modify: `docs/performance.md` (new section after the E2 sections), `docs/production-track.md` (status row), `docs/pipeline-feasibility.md` only if its G1 pointer needs a result note.

- [ ] Write the G1 section: hypothesis, table (a/b/b2/c/d), gap, failure-mode comparison (out-of-mask PSNR evidence), wall-clock, T1-scene feasibility statement, hardware context. Update status row with evidence paths.
- [ ] Commit: `G1: dynamic geometry second attempt — results and docs`.

### Task 4: shader generator refactor + demo extensions (`webgpu/shader_gen.mjs`)

**Files:**
- Create: `webgpu/shader_gen.mjs` (extract `buildShader` + `repackMlp` from `bench_t4.mjs`)
- Modify: `webgpu/bench_t4.mjs` (import from shader_gen; behavior unchanged)

`buildShader(manifest, opts = {})` gains:
- `opts.outputActivation`: `"softplus"` (default, current behavior) or `"linear"` (residual proxies).
- `opts.demo`: when true, adds bindings `positions: array<f32>` (@5), `linkMask: array<f32>` (@6), and extends the uniform struct with `light_rgb: vec4<f32>` (xyz = emission), `controls: vec4<f32>` (x = link_enabled 0/1, y = attenuation_k, z/w unused), `light_center` already in `params.light.xyz`. Final store becomes:
  `contribution = softplus(out) * light_rgb.xyz`, then `if (controls.x > 0.5 && linkMask[pixel] > 0.5) { contribution = vec3(0); }`, then `w = max(0.0, 1.0 - controls.y * distance(position, params.light.xyz)); contribution *= w;` (attenuation_k = 0 ⇒ w = 1, control off).
- Default opts must generate **byte-identical WGSL** to today's bench_t4 output (assert via a Node snapshot check in the verification step).

**Steps:**
- [ ] Extract module; bench_t4 imports it; `node -e` snapshot check that `buildShader(manifest)` output is unchanged for the committed T4 manifest.
- [ ] Implement `outputActivation` + `demo` options.
- [ ] Symlink T1 artifacts into worktree `out/` (`kitchen-512`, `kitchen-512-torch`, `t4-runtime/export`); run `mise run t4-check` → passes.
- [ ] Commit: `G2: extract WGSL generator with control and linear-head variants`.

### Task 5: demo exporters (`examples/export_webgpu_demo.py`, extends T4 exporter)

**Files:**
- Create: `examples/export_webgpu_demo.py`
- Test: `tests/test_export_webgpu_demo.py`

Two modes:
- **kitchen** (default): reuses `examples.export_webgpu_runtime` helpers; writes to `out/g2-demo/export/` the T4 blobs **plus** `positions.bin` (H·W·3 f32 first-hit positions), `link_mask.bin` (H·W f32, 1.0 where first-hit position is inside `--link-box` `xmin ymin zmin xmax ymax zmax`, the demo's "layer"), and a `manifest.json` superset: `demo: {link_box, link_pixel_fraction, attenuation: {type: "first_hit_linear_distance", k_default}}`. Default link box chosen from the kitchen G-buffer (a visible object region covering 5–30% of pixels; pick empirically, record in manifest).
- **g1** (`--g1`): reads `out/g1-residual/` artifacts; writes `out/g2-demo/export-g1/`: base model blobs (linear=false), per-frame residual blobs (linear head), per-frame pixel blobs (spliced G-buffers), per-frame region masks, per-frame reference images (`target_frame_XXXX.npy` → `reference_frame_XXXX.bin`), `manifest.json` with frame count + fixture light.

Test (unit-level, no kitchen cache needed): build a tiny `TorchNRP(use_encoding=False, hidden_width=8, hidden_layers=1)` + 8×8 toy cache in-test; call the exporter's functions directly; assert blob sizes, mask semantics (pixels inside box are 1.0), manifest self-check < 1e-3, and linear-head export parity via the exporter's numpy replica (extend `numpy_forward` usage with a `linear_head` flag or a local replica). Skips nothing (pure CPU).

- [ ] Failing tests → implement → pass → lint.
- [ ] Run kitchen export (`uv run python examples/export_webgpu_demo.py`) and g1 export; verify manifests.
- [ ] Commit: `G2: WebGPU demo exporters (kitchen controls + G1 residual frames)`.

### Task 6: the demo viewer (`webgpu/demo/`)

**Files:**
- Create: `webgpu/demo/index.html`, `webgpu/demo/main.mjs`, `webgpu/demo/server.mjs`, `webgpu/demo/trace.json`
- Modify: `mise.toml` (task `g2-serve`)

`main.mjs`: fetches `/out/g2-demo/export/*` blobs, builds compute pipeline from `shader_gen.mjs` (`demo: true`), renders to a 512² canvas via a fullscreen-triangle fragment pass reading the output storage buffer (Reinhard tonemap + gamma 2.2). UI: play/pause + scrub for the light animation (E1 keyframe JSON embedded in `trace.json`, linear interpolation ported from `nrp/torch_backend/animate.py::interpolate_light_spec` semantics), sliders for emission RGB intensity and attenuation k, link toggle; a status line with rolling mean/p95 frame time (measured around compute submit → `onSubmittedWorkDone`, T4 methodology). If `/out/g2-demo/export-g1/manifest.json` exists, a second panel: 32² (canvas-upscaled) toy scene, frame scrubber over the 10 G1 frames, compositing base + per-frame residual in the shader, labeled "toy scale (G1)". Exposes `window.__demo = { ready, setTime(t), setControls({rgb, attenuation_k, link}), setG1Frame(i), renderFrame() -> ms, readFrame() -> Float32Array, frameStats() }` for the Playwright runner.

`trace.json` (committed): `{ "keyframes": {...E1 format, authored between two well-fitting T1 val lights...}, "duration_s": 8, "events": [{"t": ..., "controls": {...}}...], "gate_samples": [{"t": ..., "controls": {...}}, ...] }` — 12 gate samples spanning plain relight, link-on, and attenuation-on states.

- [ ] Implement viewer + server + trace; manual check via `mise run g2-serve` in Chrome (verify picture, controls, moving-object panel, fps counter).
- [ ] Commit: `G2: interactive WebGPU demo viewer with animated lights and E8 controls`.

### Task 7: scripted trace runner (`webgpu/demo_g2.mjs`)

**Files:**
- Create: `webgpu/demo_g2.mjs`
- Modify: `mise.toml` (task `g2-demo`)
- Test: extend `tests/test_webgpu_browser_bench.py` pattern in a new `tests/test_g2_demo.py` (skips without node/playwright/export blobs)

Runner: starts `server.mjs`, launches Chrome (same flags as bench_t4) with `recordVideo`, waits `__demo.ready`, replays `trace.json` — for each animation frame over the duration at the display cadence: `setTime`, apply due control events, `renderFrame()` collecting ms. Then for each `gate_samples[i]`: set state, render, `readFrame()` → write `out/g2-demo/frames/frame_i.bin` + append `{light params (resolved from keyframes), controls}` to `out/g2-demo/frames/states.json`. Writes `out/g2-demo/report.json`: frame-time histogram (mean/p50/p95/min/max + raw times) at 512² under interaction, `p95_le_33ms` flag, adapter info, notes; saves the video to `out/g2-demo/recording.webm`. Exits 1 if p95 > 33 ms.

- [ ] Implement; run `mise run g2-demo`; verify report + video + frames.
- [ ] Commit: `G2: scripted interaction trace runner with frame-time histogram and recording`.

### Task 8: per-frame preview gate (`examples/g2_gate.py`)

**Files:**
- Create: `examples/g2_gate.py`
- Test: `tests/test_g2_gate.py`
- Modify: `mise.toml` (task `g2-gate`)

For each sampled frame in `out/g2-demo/frames/states.json`: reference = `gather_light(kitchen_cache, light)` (torch gather backend, cache loaded once), then apply the same control modulations the shader applied: `* light_rgb` is already inside `gather_light` via `light.rgb` (set the trace's rgb on the light), link ⇒ `reference[link_mask] = 0`, attenuation ⇒ `reference *= max(0, 1 - k·dist(first_hit_position, light_center))[..., None]`. Gate browser frame vs reference with `nrp.quality.gate.evaluate_gate(pred, ref, tier="preview")`. Output `out/g2-demo/gate.json` (per-frame verdicts + all-pass flag); exit 1 unless every frame passes.

Unit tests (toy cache, no kitchen dependency): control-modulation helpers — linking zeroes exactly the masked pixels and matches the layered algebra on a toy cache with `layer_ownership_mask`; attenuation weight formula fixture; a fabricated pred==ref frame passes the gate wiring end to end.

- [ ] Failing tests → implement → pass → lint.
- [ ] Run `mise run g2-gate` against the Task 7 frames. If any frame fails, re-author the trace light path (keep to the proxy's competent region — T1 val lights 1 and 3 score 26.7/28.8 dB, SSIM ≈ 0.99) and re-run Tasks 7→8. Frames must pass honestly; if a control state cannot pass, the report says which and why, and the trace keeps it **only** if it passes — otherwise document the excluded state in performance.md.
- [ ] Commit: `G2: per-frame preview-tier gate against controlled GATHERLIGHT references`.

### Task 9: G2 evidence + docs + integration

**Files:**
- Modify: `docs/performance.md` (G2 section), `docs/production-track.md` (G1 + G2 status rows), `webgpu/README.md` (demo pointer), `docs/quickstart.md` if it lists CLIs.

- [ ] Commit evidence: `git add -f out/g2-demo/report.json out/g2-demo/gate.json out/g2-demo/recording.webm out/g1-residual/report.json` (recording only if < ~10 MB; else commit a downsampled GIF and note the source).
- [ ] performance.md: frame-time histogram table under interaction, p95 vs 33 ms, gate results (per-state), controls semantics statement (pixel-level linking is E8's gather-time algebra; attenuation is the first-hit pixel-level variant, applied identically on both sides), moving-object panel status tied to G1's outcome, hardware context.
- [ ] Full `mise run test` + `mise run lint` + `mise run t4-check` green.
- [ ] Commit: `G2: summit demo — live relight + controls in the browser, gated and measured`.

## Self-Review Notes

- G1 verify items covered: comparison table from one script (Task 2), unit tests for shard invalidation + residual compositing (Task 1). Measure items: wall-clock + matched-budget quality (Task 2), toy scale with explicit T1 feasibility statement (Task 3).
- G2 verify items: T3 gate at preview per frame (Task 8), scripted trace + recording committed (Tasks 6/7/9). Measure: histogram under interaction, p95 ≤ 33 ms (Task 7). Moving object conditional on G1 (Tasks 5/6). ≥2 E8 controls: linking + attenuation (Tasks 4–6).
- Type consistency: `invalidated_shards`, `ResidualNRP`, `train_residual`, `composite_predict`, `pixel_features` names used consistently across Tasks 1/2/5; `window.__demo` API consistent across Tasks 6/7.
