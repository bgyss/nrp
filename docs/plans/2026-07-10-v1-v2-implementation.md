# V1 + V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement production-track rungs V1 (production light rig: >=8 lights,
per-light layered proxies, solo/mute, verified additivity) and V2 (summit demo:
E7 inverse-recovery of per-light intensities/colors toward a hand-authored target,
then a headless interactive-grading loop).

**Architecture:** V1 generalizes `composite.py`'s hardcoded 2-term add to an
N-light `LightRig`: each light gets its own independently-trained `TorchNRP`
proxy (mirroring `examples/layers.py`'s one-proxy-per-layer pattern, scaled from
geometric layers to light instances), rendered by summing active (non-muted,
solo-respecting) per-light proxy predictions. Additivity is checked against a
physical multi-light GATHERLIGHT reference (`gather_lights`), gated at T3's
preview tier — the "stated tolerance." A monolithic baseline (one baked,
non-relightable proxy trained directly on the combined 8-light image) is
measured against the sum of 8 per-light proxy sizes. V2 reuses
`optimize_lights.py`'s logit/inverse-softplus reparameterization machinery but
narrows it to *only* per-light RGB (the rung's explicit scope: "intensities/
colors", not full geometry), holding each light's shape fixed and routing its
gradient through its own frozen per-light proxy from V1; the recovered rig is
exported as a `LightRig` JSON and re-loaded, then a scripted sequence of manual
per-light color nudges is timed through the cheap N-way composite (no cache
access), which is the "interactive grading" deliverable (headless, per your
approval — no browser/WebGPU work).

**Tech Stack:** Python (numpy, torch, unittest, ruff), existing `nrp.torch_backend`
modules (`model.py`, `train.py`, `optimize_lights.py`, `relight_multiview.py`,
`gather_light.py`), mise tasks, uv. T1 kitchen scene artifacts reused from the
main checkout via symlink.

## Global Constraints

- All new/changed code passes `mise run test` and `mise run lint`; new tests
  skip cleanly when optional dependencies (the 2 GiB kitchen cache, `out/`
  artifacts) are absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context.
- Single Apple Silicon laptop; no CUDA assumed.
- Statistical assertions compare windowed means, never single minibatch losses.
- Commit style: `V1: <summary>` / `V2: <summary>` matching repo history;
  evidence JSONs under `out/` are committed with `git add -f` (as T2/T4/G1 did).
- T1 artifacts (`out/kitchen-512/path_cache.npz`, 2 GiB) live in the main
  checkout `/Users/briangyss/src/nrp/out/`; symlink into this worktree's `out/`
  — never regenerate (854 s/light train) or copy.
- Per-light proxy training uses a reduced iteration budget (800 iters vs T1's
  3000) to keep 8-light V1 training under ~35 min total; this is stated
  honestly in the report as a documented reduced budget, not silently passed
  off as full T1-scene convergence — same honesty convention as T2's streamed
  ceiling.
- `TorchNRP` is locked to one `light_type`/`light_param_dim` per instance
  (`nrp/torch_backend/model.py:51-58`) — this is why each rig light needs its
  own proxy rather than one proxy handling mixed types.

---

### Task 1: `LightRig` core — serialization, solo/mute, N-way composite (`nrp/torch_backend/rig.py`)

**Files:**
- Create: `nrp/torch_backend/rig.py`
- Test: `tests/test_rig.py`

**Interfaces (produces):**
- `light_type_of(light) -> str` — `light.to_dict()["type"]`.
- `class RigLight` (dataclass): `name: str`, `light: SphereLight | QuadLight |
  TexturedQuadLight`, `mute: bool = False`, `solo: bool = False`.
  `to_dict() -> dict` (flattens `light.to_dict()` plus `name`/`mute`/`solo`);
  `RigLight.from_dict(d: dict) -> RigLight`.
- `class LightRig`:
  - `__init__(self, lights: list[RigLight], models: dict[str, TorchNRP])` —
    `models` is keyed by **`RigLight.name`**, one independently-trained model
    per named light (never shared across lights, even if two lights share a
    type) — this is the literal "per-light" (not "per-type") proxy the rung
    asks for, mirroring `examples/layers.py`'s one-proxy-per-layer pattern.
  - `active_lights() -> list[RigLight]` — if any light has `solo=True`, return
    only solo'd (and non-muted) lights; else return all non-muted lights.
  - `render(cache: PathCache, device=torch.device("cpu")) -> np.ndarray (H,W,3)`
    — sums `self.models[rl.name](xy, aux, light_param_vector(rl.light)) *
    rl.light.rgb` over `active_lights()`, no_grad, reusing
    `nrp.torch_backend.train.pixel_tensors` and
    `nrp.torch_backend.train.light_param_vector`.
  - `render_per_light(cache, device) -> dict[str, np.ndarray]` — same, but one
    image per active light (used for the additivity check and per-light size
    reporting).
  - `to_dict() -> dict` — `{"lights": [rl.to_dict() for rl in self.lights]}`.
  - `LightRig.from_dict(d: dict, models: dict[str, TorchNRP]) -> LightRig`.
  - `save(path: str) -> None` / `LightRig.load(path: str, models: dict[str,
    TorchNRP]) -> LightRig` — JSON round-trip (models aren't serialized into
    the JSON, only referenced by name; the report script separately records
    each light's model path in a sibling `manifest.json`).
- `def train_monolithic(cache: PathCache, target_image: np.ndarray,
  hidden_width: int, hidden_layers: int, iters: int, lr: float, seed: int=0)
  -> tuple[TorchNRP, list[float]]` — a **non-relightable baseline**: a
  `TorchNRP(light_type="sphere", light_param_dim=0, hidden_width=...,
  hidden_layers=..., use_encoding=True, encoding=...)` (light_param_dim=0 means
  the light-conditioning input vanishes; `light_type` is nominal since
  `SUPPORTED_LIGHT_TYPES` still requires a valid string) trained directly
  against the single fixed `target_image` via `nrp.torch_backend.model`'s
  `relative_mse_loss`, plain full-batch Adam, no pool/sampling machinery
  (there is only one training target — the whole 8-light composite).

**Tests (`tests/test_rig.py`, toy caches from `nrp.toy_tracer.trace_path_cache`
at 12x12/4spp for speed, models trained with tiny configs — mirror
`tests/test_multiview.py::tiny_view_cfg`):**
1. `test_rig_light_roundtrip` — `RigLight.to_dict()` -> `from_dict()` produces
   an equal light (compare `.to_dict()` of the inner light) and preserves
   name/mute/solo.
2. `test_light_rig_json_roundtrip` — build a 3-light rig (sphere, quad,
   textured_quad), `save`/`load`, assert `to_dict()` equality (models passed
   back in explicitly, as documented).
3. `test_mute_excludes_light_from_render` — 2-light rig, mute one, assert
   `render()` equals the other light's `render_per_light()` entry (within
   float tolerance).
4. `test_solo_overrides_mute_and_other_lights` — 3 lights, solo one (also
   muting a different one): `active_lights()` == exactly the solo'd light.
5. `test_render_equals_sum_of_render_per_light` — `render()` ==
   `sum(render_per_light().values())` elementwise (this is the trivial
   composite identity; the *meaningful* additivity-vs-physical-truth check
   lives in Task 2's report, since it needs `gather_lights`).
6. `test_train_monolithic_reduces_loss_and_is_not_light_conditioned` — windowed
   mean loss (first 20 vs last 20 of 150 iters) decreases; the returned
   model's `light_param_dim == 0`; calling it with two different light param
   vectors of the same batch produces identical output (proof it's baked, not
   relightable).

**Steps:**
- [ ] Write `tests/test_rig.py` with the 6 tests above (imports will fail).
- [ ] Run `uv run python -m unittest tests.test_rig -v` -> confirm import
      errors only.
- [ ] Implement `nrp/torch_backend/rig.py` per the interfaces above.
- [ ] Run `uv run python -m unittest tests.test_rig -v` -> all pass.
- [ ] `uv run ruff check . && uv run ruff format .`.
- [ ] Commit: `V1: LightRig core — per-light proxies, solo/mute, monolithic baseline`.

---

### Task 2: V1 report script — 8-light T1-scene rig, additivity, size/overhead (`examples/v1_rig.py`)

**Files:**
- Create: `examples/v1_rig.py`
- Modify: `mise.toml` (new `[tasks.v1-rig]`)

**Interfaces:**
- Consumes: Task 1's `RigLight`, `LightRig`, `train_monolithic`;
  `nrp.torch_backend.train.train` (existing per-light-type training entrypoint,
  config shape as in `examples/kitchen_512_torch.json`);
  `nrp.gather_light.gather_lights`; `nrp.quality.gate.evaluate_gate`.
- Produces: `out/v1-rig/report.json`, `out/v1-rig/models/<light-name>.pt` (8
  files), `out/v1-rig/rig.json` (the `LightRig.to_dict()` + a `models` map of
  name -> relative model path, since `LightRig.save` itself doesn't persist
  paths), `out/v1-rig/monolithic.pt`.

**Rig definition (hardcoded in the script, 8 lights over the T1 kitchen
bounding volume — reuse `examples/kitchen_512_torch.json`'s implicit scene
bounds, i.e. sample within the existing `light_bounds` used for that scene):**
- 3 `SphereLight`s: `key`, `fill`, `rim` — varying center/radius/rgb.
- 3 `QuadLight`s: `window`, `ceiling_panel`, `practical` — varying
  center/normal/width/height/rgb.
- 2 `TexturedQuadLight`s: `neon_sign`, `tv_glow` — small procedurally
  generated RGB textures (e.g. a 8x8 checkerboard and an 8x8 radial gradient,
  built with numpy in the script, no external assets).

**Steps:**
- [ ] Symlink the T1 scene into this worktree if not already present:
      `ln -s /Users/briangyss/src/nrp/out/kitchen-512 out/kitchen-512` (the
      `out/` directory itself already exists in this worktree from earlier
      rungs' artifacts; only add the missing `kitchen-512` symlink).
- [ ] Implement `examples/v1_rig.py`:
  - Define the 8 `RigLight`s above with fixed, hand-chosen params (not
    randomly sampled — a production rig is authored, not sampled).
  - For each light, build a per-light-type training config from
    `examples/kitchen_512_torch.json` (same `cache`, `pool`, `denoise`,
    `model` blocks) but `"iters": 800` (documented reduced budget per the
    Global Constraints), `"light_type"` set to that light's type, and
    `"light_bounds"` narrowed around that specific light's own params (so the
    proxy specializes to render *this* light well, not an arbitrary light of
    its type) — write each config to `out/v1-rig/configs/<name>.json` and call
    `nrp.torch_backend.train.train(cfg)`, writing `out/v1-rig/models/<name>.pt`.
  - Build a `LightRig` from the 8 trained models (keyed by light name).
  - Full-rig GATHERLIGHT reference: `gather_lights(cache, [rl.light for rl in
    rig.active_lights()])`.
  - Additivity check: `evaluate_gate(rig.render(cache), reference, tier=
    "preview")` (the stated tolerance: T3's preview tier, PSNR >= 20 dB / SSIM
    >= 0.80 / FLIP <= 0.15 — reported verbatim in the JSON along with the
    actual measured values, per the rung's "report the tolerance, don't
    assume one").
  - Monolithic baseline: `train_monolithic(cache, reference, hidden_width=128,
    hidden_layers=4, iters=800*8, lr=0.005)` (matched *total* iteration budget
    to the 8 per-light proxies combined, for a fair size-vs-quality
    comparison), save to `out/v1-rig/monolithic.pt`.
  - Sizes: `sum(os.path.getsize("out/v1-rig/models/*.pt"))` vs
    `os.path.getsize("out/v1-rig/monolithic.pt")`.
  - Compositing overhead: for `k` in `1..8`, solo the first `k` lights (via
    `RigLight.solo`) and time `rig.render(cache)` (warmup + N repeats,
    `time.perf_counter`, matching `edit_latency_ms`'s style in
    `relight_multiview.py`) at 512x512 — report ms per active light count, and
    the marginal ms/added-light (linear-fit slope).
  - Write `out/v1-rig/report.json` with: per-light training summaries (val
    PSNR from each `train()` report), `additivity_gate` (the `evaluate_gate`
    result + reference/tier), `sizes_bytes` (`per_light_total`,
    `monolithic`, `ratio`), `compositing_overhead_ms` (list of `{n_lights,
    ms}` + fitted slope), hardware context (`platform.platform()`,
    `torch.get_num_threads()`).
- [ ] Add `[tasks.v1-rig]` to `mise.toml`:
  ```toml
  [tasks.v1-rig]
  description = "V1: 8-light production rig on the T1 kitchen — per-light layered proxies, solo/mute, additivity vs monolithic baseline"
  run = "uv run python examples/v1_rig.py --cache out/kitchen-512/path_cache.npz --out-dir out/v1-rig"
  ```
- [ ] Run `mise run v1-rig`; inspect `out/v1-rig/report.json`; if the
      additivity gate fails at preview tier, either raise per-light iters
      modestly (document the change) or move to `draft` tier and say so
      explicitly in the report — do not silently loosen without recording it.
- [ ] Add a fast smoke test to `tests/test_rig.py` (new test class
      `V1ReportSmokeTests`) that runs the same script's core function (factor
      the per-script logic that isn't argparse/IO into an importable
      `build_and_evaluate_rig(cache, lights, iters, hidden_width,
      hidden_layers, ...) -> dict` in `examples/v1_rig.py`, called both by
      `main()` and by the test) at 12x12/tiny iters/2 lights, asserting the
      report's top-level keys exist and the additivity PSNR is finite.
- [ ] `uv run ruff check . && uv run ruff format .`; full test suite passes.
- [ ] Commit: `V1: 8-light production rig report — additivity, sizes, compositing overhead`;
      `git add -f out/v1-rig/report.json out/v1-rig/rig.json`.

---

### Task 3: V1 docs (`docs/performance.md`, `docs/production-track.md`, `docs/paper-mapping.md`)

**Files:**
- Modify: `docs/performance.md` (new section), `docs/production-track.md`
  (status row), `docs/paper-mapping.md` (§6.1 compositing row).

- [ ] Write the V1 section in `docs/performance.md`: the 8-light rig
      composition (types + names), per-light training wall-clock and val
      PSNR table, the additivity gate result (measured PSNR/SSIM/FLIP vs the
      preview-tier thresholds, pass/fail), the sizes table (per-light total
      vs monolithic, ratio), the compositing-overhead-vs-light-count curve,
      hardware context.
- [ ] Update `docs/production-track.md`'s V1 status row: `not started` ->
      `done`, evidence: `out/v1-rig/report.json`; `nrp/torch_backend/rig.py` +
      `tests/test_rig.py`; `docs/performance.md`.
- [ ] Update `docs/paper-mapping.md`'s §6.1 compositing row to note the
      generalization from the 2-layer `composite.py` to N-light `LightRig`
      (V1), citing `nrp/torch_backend/rig.py`.
- [ ] Commit: `V1: performance write-up and paper-mapping/production-track updates`.

---

### Task 4: mixed-light-type color recovery + slider loop (`nrp/torch_backend/art_loop.py`)

**Files:**
- Create: `nrp/torch_backend/art_loop.py`
- Test: `tests/test_art_loop.py`

**Interfaces (produces):**
- `class RigColorReparam`: holds, for each active `RigLight`, its **fixed**
  geometry (center/radius or center/normal/width/height, and for
  `TexturedQuadLight` the fixed texture) plus one **trainable** inverse-softplus
  `rgb` parameter (`u_rgb`, shape `(3,)`, reusing `inv_softplus`/`softplus`
  from `optimize_lights.py`). `__init__(self, rig: LightRig, init_rgbs:
  dict[str, np.ndarray], device: torch.device)`. `parameters -> list[Tensor]`
  (one `u_rgb` per light, each `requires_grad_(True)`).
  `constrained_rgbs() -> dict[str, Tensor]` (name -> softplus(u_rgb)).
  `to_rig() -> LightRig` — returns a new `LightRig` (same models, same
  geometry) with each light's `rgb` replaced by the current
  `constrained_rgbs()` value (detached, as `.cpu().numpy()`).
- `def predicted_image(rig: LightRig, reparam: RigColorReparam, xy: Tensor,
  aux: Tensor) -> Tensor` — differentiable analogue of `LightRig.render`:
  sums `rig.models[name](xy, aux, light_param_vector(light_with_fixed_geometry))
  * reparam.constrained_rgbs()[name]` over active lights, **with grad**
  (unlike `LightRig.render`'s `no_grad`).
- `def optimize_colors(rig: LightRig, cache: PathCache, target: np.ndarray,
  steps: int, lr: float, seed: int=0) -> dict` — Adam over
  `RigColorReparam.parameters`, loss = Reinhard-tonemapped MSE (reuses
  `reinhard` from `optimize_lights.py`) between `predicted_image(...)` and
  `target`, full-batch (rig-scale images are small enough; no pixel-fraction
  needed at toy/report scale here). Returns a report dict: `optimized_rig:
  LightRig` (not JSON-serialized here — caller serializes),
  `proxy_loss_first`, `proxy_loss_last`, `proxy_loss_curve` (subsampled like
  `optimize_lights.py`), `proxy_vs_target_psnr_db` and `proxy_vs_target_ssim`
  computed via `from ..metrics import psnr, ssim, tonemap_srgb` (PSNR on
  linear HDR, SSIM on `tonemap_srgb`'d images — the same convention as
  `nrp.quality.gate.evaluate_gate`).
- `def slider_loop(rig: LightRig, cache: PathCache, adjustments: list[dict],
  device=torch.device("cpu")) -> dict` — `adjustments` is a list of
  `{"light": name, "rgb": [r,g,b]}` nudges applied one at a time (mutating a
  working copy of the rig's `RigLight.light.rgb`), each followed by a timed
  `rig.render(cache, device)` (warmup excluded from timing, first call is
  warmup); returns `{"n_adjustments": ..., "latency_ms": [...],
  "latency_ms_mean": ..., "latency_ms_p95": ...}`.

**Tests (`tests/test_art_loop.py`, toy scale, 2-3 lights, tiny per-light
proxies trained via `tests/test_multiview.py`-style `tiny_view_cfg` configs
adapted to mixed types):**
1. `test_rig_color_reparam_roundtrip` — `RigColorReparam` initialized from a
   rig's own current rgbs; `to_rig()` immediately (no optimization steps)
   reproduces the same rgbs within float32 tolerance.
2. `test_predicted_image_matches_render_before_optimization` — before any
   Adam steps, `predicted_image(...)` (converted to numpy) matches
   `rig.render(cache)` closely (both should be the same forward pass, one
   with grad one without).
3. `test_optimize_colors_reduces_loss_and_recovers_target_rgb` — build a rig,
   render it as the "hand-authored target" with different rgbs than the
   initial guess, run `optimize_colors` ~150 steps, assert
   `proxy_loss_last < proxy_loss_first` (windowed-mean-safe since it's a
   single scalar per call, compare first vs last directly as the existing
   `optimize_lights.py` tests do) and `proxy_vs_target_psnr_db` improves over
   the pre-optimization PSNR.
4. `test_optimized_rig_is_reloadable` — `optimize_colors(...)["optimized_rig"]
   .save(path)`; `LightRig.load(path, models)` round-trips geometry and
   recovered rgbs.
5. `test_slider_loop_applies_adjustments_and_measures_latency` — 3
   adjustments on a 2-light rig; `len(latency_ms) == 3`; each render actually
   used the nudged rgb (compare output brightness monotonically tracks an
   increasing rgb nudge sequence); `latency_ms_mean > 0`.

**Steps:**
- [ ] Write `tests/test_art_loop.py` with the 5 tests above.
- [ ] Run `uv run python -m unittest tests.test_art_loop -v` -> import errors only.
- [ ] Implement `nrp/torch_backend/art_loop.py`.
- [ ] Run tests -> all pass.
- [ ] `uv run ruff check . && uv run ruff format .`.
- [ ] Commit: `V2: mixed-rig color recovery (RigColorReparam) and headless slider loop`.

---

### Task 5: V2 report script — art-direction loop on the V1 rig (`examples/v2_art_loop.py`)

**Files:**
- Create: `examples/v2_art_loop.py`
- Modify: `mise.toml` (new `[tasks.v2-artloop]`)

**Interfaces:**
- Consumes: Task 4's `RigColorReparam`/`optimize_colors`/`slider_loop`; Task
  1/2's `LightRig.load`/`out/v1-rig/rig.json` + `out/v1-rig/models/*.pt`.
- Produces: `out/v2-artloop/report.json`, `out/v2-artloop/recovered_rig.json`,
  `out/v2-artloop/target.npy`.

**Steps:**
- [ ] Implement `examples/v2_art_loop.py`:
  - Load the V1 rig (`out/v1-rig/rig.json` + the 8 `out/v1-rig/models/*.pt`).
  - Hand-author a target: copy the loaded rig, multiply/replace each light's
    `rgb` with explicit different values (e.g. `key` brighter+warmer, `rim`
    cooler, `neon_sign` fully off, etc. — a small hardcoded dict in the
    script, documented in a comment as the "art direction" target), render it
    with `LightRig.render` to get `target.npy`.
  - Reset the working rig's rgbs to a neutral initial guess (e.g. all-white
    unit intensity) and run `optimize_colors(rig, cache, target, steps=500,
    lr=0.05)` (paper-default step count/lr, matching `optimize_lights.py`'s
    CLI defaults).
  - Verify convergence: gate `predicted`-vs-`target` via
    `nrp.quality.gate.evaluate_gate(..., tier="draft")` (a plain color-only
    optimization on top of an already-good proxy should reach the tighter
    draft tier — the rung's "stated image metric"); record pass/fail and
    values regardless.
  - Save `optimized_rig` to `out/v2-artloop/recovered_rig.json`, reload it
    (`LightRig.load`) and assert (in the script, not just tests) the reload
    renders identically to the pre-save render — the rung's "recovered rig is
    exported and re-loadable" verification, recorded as a boolean in the
    report.
  - Run `slider_loop` with a scripted sequence of ~10 per-light nudges
    (alternating which of the 8 lights gets adjusted) on the recovered rig at
    512x512; record `latency_ms_mean`/`p95` as the "interactive grading
    latency per adjustment" measurement.
  - Write `out/v2-artloop/report.json`: `steps`, `lr`,
    `proxy_loss_first`/`last`, `convergence_gate` (tier + measured
    PSNR/SSIM/FLIP + pass/fail), `wall_clock_seconds` to convergence,
    `reload_identical: bool`, `slider_loop` (from Task 4's return dict),
    hardware context.
- [ ] Add `[tasks.v2-artloop]` to `mise.toml`:
  ```toml
  [tasks.v2-artloop]
  description = "V2 summit: E7 color recovery driving the full V1 rig toward a hand-authored target, then a headless interactive-grading loop"
  run = "uv run python examples/v2_art_loop.py --rig out/v1-rig/rig.json --models-dir out/v1-rig/models --cache out/kitchen-512/path_cache.npz --out-dir out/v2-artloop"
  ```
- [ ] Run `mise run v2-artloop`; inspect the report; if convergence misses
      `draft` tier, fall back to reporting against `preview` tier explicitly
      (same honesty convention as Task 2) rather than silently passing.
- [ ] Add a fast smoke test to `tests/test_art_loop.py` (factor the
      non-argparse core into `run_art_loop(rig, cache, target_rgbs: dict,
      steps, lr, adjustments) -> dict` called by both `main()` and the test)
      at toy scale/2 lights/tiny steps, asserting report keys + finite metrics.
- [ ] `uv run ruff check . && uv run ruff format .`; full test suite passes.
- [ ] Commit: `V2: art-direction loop report — target recovery, reload check, slider-loop latency`;
      `git add -f out/v2-artloop/report.json out/v2-artloop/recovered_rig.json`.

---

### Task 6: V2 docs + final production-track status

**Files:**
- Modify: `docs/performance.md`, `docs/production-track.md`,
  `docs/paper-mapping.md`.

- [ ] Write the V2 section in `docs/performance.md`: the hand-authored target
      (table of before/after rgbs per light), convergence gate result and
      wall-clock, reload-identical confirmation, slider-loop latency
      histogram/mean/p95, hardware context.
- [ ] Update `docs/production-track.md`'s V2 status row: `not started` ->
      `done`, evidence: `out/v2-artloop/report.json`;
      `nrp/torch_backend/art_loop.py` + `tests/test_art_loop.py`;
      `docs/performance.md`.
- [ ] Update `docs/paper-mapping.md`'s E7 row: note the mixed-light-type
      color-only recovery closes the "hand-authored target image" gap flagged
      there, citing `nrp/torch_backend/art_loop.py`.
- [ ] Run `mise run test && mise run lint` once more end-to-end; run `mise run
      pipeline-audit` (or note if it's pre-existing-broken per prior worktree
      memory) to confirm referenced report paths resolve.
- [ ] Commit: `V2: performance write-up and paper-mapping/production-track updates`.
