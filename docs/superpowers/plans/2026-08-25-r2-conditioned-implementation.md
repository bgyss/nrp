# R2 Camera-Conditioned NRP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the R2 one-network/multiple-camera pilot with camera-conditioned TorchNRP training, shared-model relighting, per-view baseline comparison, and committed-quality evidence.

**Architecture:** Add an opt-in `camera_conditioned` input to `TorchNRP`, then build a focused multi-view trainer around camera-aware manifests, a shared light pool, global world bounds, and per-view validation. Add shared-model inference and an R2 experiment runner while preserving the existing per-view `relight_multiview` and all legacy checkpoints.

**Tech Stack:** Python 3.12, NumPy, PyTorch, existing NRP path-cache/GATHERLIGHT and metrics code, `unittest`, Ruff, Mitsuba when available, and `mise` tasks.

---

## Scope and invariants

- `TorchNRP(camera_conditioned=False)` remains behaviorally and serialization compatible with current models.
- R2 models use `spatial_encoding: "world3d"` with one global bound across all views and a normalized camera forward direction.
- One shared pool of light-shape vectors is rendered through every view cache; validation lights are per-view and disjoint from training vectors.
- Every R2 claim is written to `out/r2-conditioned/report.json` and reflected in `docs/performance.md`; R1's failed promotion gate remains visible in `docs/representation-track.md`.
- No cloud execution, CUDA requirement, or vendored scene asset is introduced.

## File map

- Modify `nrp/torch_backend/model.py`: optional camera-direction input, config serialization, input validation.
- Modify `nrp/torch_backend/train.py`: optional camera direction in evaluation and shared global-bound helper.
- Create `nrp/torch_backend/conditioned_multiview.py`: manifest records, camera/global-bound helpers, multi-view pool, validation, training, and report construction.
- Modify `nrp/torch_backend/relight_multiview.py`: camera-aware manifest parsing and shared-model inference/latency functions.
- Modify `examples/multiview.py`: include camera metadata in the established manifest and expose reusable manifest-writing behavior.
- Create `examples/r2_conditioned.py`: Cornell-box export/reuse, per-view baselines, one conditioned model, gate/memory/latency report.
- Create `tests/test_conditioned_multiview.py`: model, manifest, pool, trainer, inference, and gate tests using tiny hand-authored caches.
- Modify `tests/test_torch_backend.py`: direct camera-conditioned model contract tests.
- Modify `tests/test_multiview.py`: manifest camera metadata and shared-model inference tests that do not require Mitsuba.
- Modify `mise.toml`: add `r2-conditioned` with the exact experiment command.
- Modify `docs/representation-track.md` and `docs/performance.md`: distinguish implementation/pilot evidence from promotion and record measured results.

---

### Task 1: Add the opt-in camera-conditioned model input

**Files:**

- Modify: `nrp/torch_backend/model.py:50-205`
- Test: `tests/test_torch_backend.py`

- [ ] **Step 1: Write failing model tests.** Add tests that construct
  `TorchNRP(camera_conditioned=True, spatial_encoding="world3d", world_bounds=...)`
  and assert that:

  ```python
  model = TorchNRP(
      hidden_width=8,
      hidden_layers=1,
      encoding={"levels": 1, "features_per_level": 2, "finest_resolution": 4},
      spatial_encoding="world3d",
      world_bounds={"min": [0, 0, 0], "max": [1, 1, 1]},
      camera_conditioned=True,
  )
  xyz = torch.rand(5, 3)
  aux = torch.rand(5, 7)
  params = torch.rand(5, 4)
  direction = torch.tensor([0.0, 0.0, -1.0])
  broadcast = model(xyz, aux, params, direction)
  repeated = model(xyz, aux, params, direction.expand(5, -1))
  self.assertEqual(broadcast.shape, (5, 3))
  torch.testing.assert_close(broadcast, repeated)
  with self.assertRaises(ValueError):
      model(xyz, aux, params)
  with self.assertRaises(ValueError):
      model(xyz, aux, params, torch.ones(5, 2))
  ```

  Also test that a default unconditioned model accepts the existing three arguments,
  ignores no required input, and saves/loads with identical output. Assert the
  conditioned model's config contains `camera_conditioned: true` and that a zero
  direction is rejected rather than silently normalized to NaN.

- [ ] **Step 2: Run only the new tests to verify the intended failure.**

  Run: `uv run python -m unittest tests.test_torch_backend.ModelTests.test_camera_conditioned_forward -v`

  Expected: FAIL because `TorchNRP.__init__` does not accept
  `camera_conditioned` and the current `forward` has no fourth input.

- [ ] **Step 3: Implement the minimal model change.**

  Add `camera_conditioned: bool = False` after `world_bounds` in `TorchNRP.__init__`,
  store it in `self.config`, and add `CAMERA_DIRECTION_DIM = 3`. Increase the MLP
  input dimension by three only when enabled. Change the call signature to
  `forward(self, spatial_coords, aux, light_params, view_dir=None)` and use this
  validation/broadcast logic before concatenating the MLP parts:

  ```python
  camera = None
  if self.camera_conditioned:
      if view_dir is None:
          raise ValueError("camera-conditioned model requires view_dir")
      if view_dir.ndim == 1:
          if view_dir.shape != (3,):
              raise ValueError("view_dir must have shape (3,) or (N, 3)")
          view_dir = view_dir.reshape(1, 3).expand(spatial_coords.shape[0], -1)
      elif view_dir.ndim != 2 or view_dir.shape != (spatial_coords.shape[0], 3):
          raise ValueError(f"view_dir must have shape (3,) or (N, 3), got {tuple(view_dir.shape)}")
      if not bool(torch.isfinite(view_dir).all()):
          raise ValueError("view_dir must be finite")
      norm = torch.linalg.vector_norm(view_dir, dim=1, keepdim=True)
      if bool((norm <= 1e-8).any()):
          raise ValueError("view_dir must be non-zero")
      camera = view_dir / norm
  ```

  Use `[spatial, aux, camera, light_params]` for conditioned models and the current
  `[spatial, aux, light_params]` for all others. `TorchNRP.load` will remain
  compatible because the new config key has a default.

- [ ] **Step 4: Run the focused model tests and existing torch backend tests.**

  Run: `uv run python -m unittest tests.test_torch_backend -v`

  Expected: all existing tests plus the new camera tests pass.

- [ ] **Step 5: Commit the model seam.**

  Run: `git add nrp/torch_backend/model.py tests/test_torch_backend.py && git commit -m "R2: add opt-in camera conditioning"`

---

### Task 2: Define camera-aware manifests and shared world bounds

**Files:**

- Create: `nrp/torch_backend/conditioned_multiview.py`
- Test: `tests/test_conditioned_multiview.py`
- Modify: `nrp/torch_backend/train.py:88-118,249-275`

- [ ] **Step 1: Write failing manifest and geometry tests.** Create two tiny valid
  `PathCache` fixtures with the same resolution and a manifest like:

  ```json
  {
    "views": [
      {"name": "front", "cache": "front.npz", "camera": {
        "origin": [0.0, 0.0, 2.0], "target": [0.0, 0.0, 0.0]
      }},
      {"name": "side", "cache": "side.npz", "camera": {
        "origin": [2.0, 0.0, 0.0], "target": [0.0, 0.0, 0.0]
      }}
    ]
  }
  ```

  Assert that `load_camera_manifest` resolves cache paths relative to the manifest,
  produces directions `[0,0,-1]` and `[-1,0,0]`, rejects missing camera fields,
  rejects a zero-length direction, and rejects mixed cache resolutions. Assert that
  `global_world_bounds` returns the componentwise bound over both caches.

- [ ] **Step 2: Run the new manifest tests and observe the missing-module failure.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.ManifestTests -v`

  Expected: FAIL because `conditioned_multiview.py` and its public helpers do not exist.

- [ ] **Step 3: Implement manifest and geometry helpers.** Add:

  ```python
  @dataclass(frozen=True)
  class CameraView:
      name: str
      cache_path: Path
      camera: dict

      @property
      def view_dir(self) -> np.ndarray: ...

  def load_camera_manifest(path: str | Path) -> list[CameraView]: ...
  def camera_direction(camera: dict) -> np.ndarray: ...
  def global_world_bounds(caches: Sequence[PathCache]) -> dict: ...
  def camera_tensor(view: CameraView, n_pixels: int, device) -> torch.Tensor: ...
  ```

  Accept only a list or `{ "views": [...] }`; resolve relative paths against the
  manifest directory. Require finite three-vectors for `origin` and `target`, require
  nonzero `target - origin`, and normalize the result. Load/validate every cache and
  require identical `(height, width)`. Compute global finite min/max over all
  `cache.position` arrays and reject any zero extent.

  Add `global_world_bounds` to the existing training utility surface without changing
  the old single-cache `world_bounds` behavior. Extend `evaluate` with an optional
  `view_dir=None` argument and pass it as the model's fourth argument only when supplied.

- [ ] **Step 4: Run the manifest/geometry tests.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.ManifestTests -v`

  Expected: PASS, including malformed-manifest and mixed-resolution error cases.

- [ ] **Step 5: Commit the manifest seam.**

  Run: `git add nrp/torch_backend/conditioned_multiview.py nrp/torch_backend/train.py tests/test_conditioned_multiview.py && git commit -m "R2: add camera-aware multiview manifest"`

---

### Task 3: Implement shared-light multi-view pooling and validation accounting

**Files:**

- Modify: `nrp/torch_backend/conditioned_multiview.py`
- Test: `tests/test_conditioned_multiview.py`

- [ ] **Step 1: Write failing pool and validation tests.** Use two tiny caches and a
  small config (`pool.size = 3`, `replace_count = 1`) to assert:

  - `MultiViewImagePool.params` has shape `(pool_size, light_param_dim)`;
  - `targets` has shape `(view_count, pool_size, pixels, 3)`;
  - one pool slot has identical light parameters for every view but distinct target
    images when the caches differ;
  - replacing a slot changes the recorded shared light vector and all corresponding
    view targets;
  - `build_validation_sets` returns one list per view and
    `validation_disjointness` reports true for normal random inputs and false for an
    explicitly duplicated vector.

- [ ] **Step 2: Run the pool tests to confirm the missing implementation.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.PoolTests -v`

  Expected: FAIL because `MultiViewImagePool`, `build_validation_sets`, and
  `validation_disjointness` are not implemented.

- [ ] **Step 3: Implement the minimal shared pool.** Mirror the existing `ImagePool`
  conventions but keep one `params` table and one target image per view:

  ```python
  class MultiViewImagePool:
      def __init__(self, caches, cfg, rng, device, fill=True):
          self.params = torch.empty((pool_size, dim), device=device)
          self.targets = torch.empty((len(caches), pool_size, n_pixels, 3), device=device)
          self.used_params = []

      def fill(self, slot):
          light = sample_light(self.caches[0], self.rng, ...)
          vector = light_param_vector(light)
          self.params[slot] = torch.as_tensor(vector, device=self.device)
          for view_index, cache in enumerate(self.caches):
              self.targets[view_index, slot] = torch.as_tensor(
                  render_denoised_target(cache, light, self.cfg), device=self.device
              ).reshape(-1, 3)
          self.used_params.append(vector.copy())

      def replace_round(self):
          for _ in range(self.cfg["pool"]["replace_count"]):
              self.fill(self._next_replace)
              self._next_replace = (self._next_replace + 1) % self.size
  ```

  `render_denoised_target` must use `TorchPathCache` only when
  `gather_backend == "torch"`; otherwise use authoritative NumPy `gather_light`,
  then the configured denoiser. Keep `used_params` unique per replacement event and
  expose `supervision_images = len(used_params) * len(caches)`.

  Build validation lights with a dedicated RNG seed tuple `[seed, 0x5EED, view_index]`,
  compute both raw and denoised targets, record each exact parameter vector, and
  implement `validation_disjointness(training_vectors, validation_entries)` using
  exact tuple comparison of float64 vectors. The random generator must never reuse
  the training RNG.

- [ ] **Step 4: Run pool tests and the shared gather tests.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.PoolTests tests.test_torch_gather -v`

  Expected: PASS with NumPy/Torch gather behavior within the existing tolerance.

- [ ] **Step 5: Commit the shared pool.**

  Run: `git add nrp/torch_backend/conditioned_multiview.py tests/test_conditioned_multiview.py && git commit -m "R2: add shared multiview supervision pool"`

---

### Task 4: Add one-network conditioned training and per-view evaluation

**Files:**

- Modify: `nrp/torch_backend/conditioned_multiview.py`
- Test: `tests/test_conditioned_multiview.py`

- [ ] **Step 1: Write the failing training test.** Add a tiny end-to-end test that
  writes two 6x5 caches and a two-view manifest, runs:

  ```python
  report = train_conditioned(
      {
          "manifest": str(manifest),
          "out_dir": str(out_dir),
          "light_type": "sphere",
          "light_bounds": {"radius_min": 0.1, "radius_max": 0.2},
          "sampling": "segments",
          "pool": {"size": 3, "replace_every": 2, "replace_count": 1},
          "denoise": {"enabled": False},
          "iters": 6,
          "batch_pixels": 16,
          "lr": 0.005,
          "model": {
              "camera_conditioned": True,
              "spatial_encoding": "world3d",
              "hidden_width": 8,
              "hidden_layers": 1,
              "encoding": {"levels": 1, "features_per_level": 2,
                            "finest_resolution": 4},
          },
          "n_val_lights": 2,
          "seed": 7,
          "device": "cpu",
      }
  )
  ```

  Assert exactly one `model.pt`, one report, two per-view metric rows, recorded
  training/validation vectors, `validation_disjoint: true`, and finite loss/PSNR
  values. Assert the report's model parameter count is not multiplied by view count.

- [ ] **Step 2: Run the training test and verify it fails for the missing trainer.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.TrainingTests.test_train_conditioned_two_views -v`

  Expected: FAIL because `train_conditioned` is not defined.

- [ ] **Step 3: Implement the trainer.** Add `train_conditioned(cfg, resume=False)`:

  1. Validate `cfg["model"]["camera_conditioned"]` is true and load the camera
     manifest/caches.
  2. Compute one global bound, build `TorchNRP` with `camera_conditioned=True`, and
     stack `spatial`, `aux`, and broadcast `view_dirs` as `(V, pixels, columns)`.
  3. Build the shared pool and per-view validation sets before the training loop.
  4. Sample `view_ids`, `pool_ids`, and `pixel_ids` with a CPU-seeded
     `torch.Generator`; gather rows with `spatial[view_ids, pixel_ids]`,
     `aux[view_ids, pixel_ids]`, `pool.params[pool_ids]`, and
     `view_dirs[view_ids, pixel_ids]`.
  5. Apply the existing `relative_mse_loss`, Adam, optional autocast, and periodic
     shared-pool replacement. Initialize output scale from the pooled targets using
     the same `init_output_scale` convention as `train.py`.
  6. Evaluate every view with the same held-out entries and the optional `view_dir`
     argument, then save one model and `conditioned_train_report.json`.

  The report must include `view_count`, `views`, `parameter_count`, `model_bytes`,
  `shared_training_light_params`, `validation_light_params`,
  `validation_disjoint_by_view`, `loss_first`, `loss_last`, `train_seconds`,
  `pool_build_seconds`, and per-view `val_psnr_db_vs_raw_mean`/SMAPE metrics.
  Paths inside the report should be relative to `out_dir` where practical.

- [ ] **Step 4: Run the focused training test and existing training smoke tests.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.TrainingTests tests.test_training_smoke -v`

  Expected: PASS without Mitsuba, with the new one-model/two-view report present.

- [ ] **Step 5: Commit the trainer.**

  Run: `git add nrp/torch_backend/conditioned_multiview.py tests/test_conditioned_multiview.py && git commit -m "R2: train one camera-conditioned multiview proxy"`

---

### Task 5: Add shared-model relighting and latency measurement

**Files:**

- Modify: `nrp/torch_backend/relight_multiview.py`
- Test: `tests/test_conditioned_multiview.py`
- Modify: `tests/test_multiview.py`

- [ ] **Step 1: Write failing shared-inference tests.** Build a conditioned model
  and two `CameraView` records from tiny caches. Assert that:

  - `load_conditioned_views(manifest, model_path, device="cpu")` loads one model
    object and N resident per-view feature tensors;
  - `relight_conditioned_all` equals direct calls to the same model with each
    view's direction and light parameters;
  - `conditioned_edit_latency_ms` returns finite positive values for one and two
    views and invokes no `gather_reference`/GATHERLIGHT path during rendering;
  - the existing `load_views`/`relight_all` per-view path remains unchanged.

- [ ] **Step 2: Run the new inference tests to see the expected missing-symbol failure.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.InferenceTests -v`

  Expected: FAIL because the shared-model loader and latency functions are absent.

- [ ] **Step 3: Implement the shared inference path.** Add a resident record with
  `name`, `cache`, `spatial`, `aux`, and `view_dir`, plus:

  ```python
  def load_conditioned_views(manifest_path, model_path, device="cpu") -> tuple[TorchNRP, list[ConditionedView]]: ...
  def relight_conditioned_all(model, views, lights) -> dict[str, np.ndarray]: ...
  def conditioned_edit_latency_ms(model, views, lights, frames=10, warmup=2) -> float: ...
  ```

  Precompute all feature tensors at load time. In the timed path, pass the shared
  model and each resident direction to `model(...)`; do not call `gather_lights`,
  access segment arrays, or reload a cache. Use `_synchronize` before/after the
  timed loop and match `edit_latency_ms`'s warmup convention. Keep baseline
  `ViewProxy`, `load_views`, `relight_all`, and `edit_latency_ms` behavior intact.

- [ ] **Step 4: Run inference and regression tests.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.InferenceTests tests.test_multiview -v`

  Expected: PASS, with identical proxy images from direct and shared paths.

- [ ] **Step 5: Commit shared inference.**

  Run: `git add nrp/torch_backend/relight_multiview.py tests/test_conditioned_multiview.py tests/test_multiview.py && git commit -m "R2: add shared conditioned multiview relighting"`

---

### Task 6: Extend the established multiview manifest and add the R2 experiment runner

**Files:**

- Modify: `examples/multiview.py:180-220`
- Create: `examples/r2_conditioned.py`
- Modify: `mise.toml` near the existing `multiview` task
- Test: `tests/test_conditioned_multiview.py`

- [ ] **Step 1: Write failing runner/report tests.** Test pure report helpers before
  invoking Mitsuba:

  ```python
  rows = [
      {"view": "front", "baseline_psnr_db": 20.0, "conditioned_psnr_db": 19.4},
      {"view": "side", "baseline_psnr_db": 21.0, "conditioned_psnr_db": 20.5},
  ]
  result = quality_gate(rows, tolerance_db=1.0)
  self.assertTrue(result["passed"])
  self.assertEqual(result["per_view"][0]["delta_db"], -0.6)
  ```

  Add a failing case at `-1.01 dB`, and assert a report records
  `r1_prerequisite.promoted == false` separately from `checks.per_view_quality_gate`.

- [ ] **Step 2: Run the report-helper tests and verify the missing runner module.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.ReportTests -v`

  Expected: FAIL because `examples/r2_conditioned.py` does not exist.

- [ ] **Step 3: Add camera metadata to the existing `views.json` writer.** Include
  the existing pose as:

  ```json
  "camera": {
    "origin": [3.9, 0.0, 0.0],
    "target": [0.0, 0.0, 0.0]
  }
  ```

  without changing the existing `name`, `model`, or `cache` keys. Keep paths
  relative to the manifest directory.

- [ ] **Step 4: Implement `examples/r2_conditioned.py`.** Reuse `view_poses`,
  `export_view`, and `train_view_config` from `examples/multiview.py`; do not copy
  the Mitsuba export logic. Add CLI flags matching the established runner plus
  `--manifest`, `--skip-export`, `--skip-baseline`, `--skip-conditioned`, and
  `--devices`.

  The runner must:

  1. create/reuse N caches and a camera-aware manifest;
  2. create/reuse N baseline `model.pt` artifacts with the existing per-view config;
  3. run `train_conditioned` once with `camera_conditioned=true` and global
     `world3d` bounds;
  4. build the same per-view validation sets used by the conditioned report and
     evaluate each baseline on those exact lights;
  5. compute rows with baseline/conditioned PSNR, SMAPE, and `delta_db`, then apply
     `quality_gate(rows, tolerance_db=1.0)`;
  6. measure `model_bytes` for one conditioned checkpoint versus the sum of N
     baseline checkpoints;
  7. measure conditioned and baseline all-view edit latency for every requested
     device with warmup/synchronization;
  8. write `out/r2-conditioned/report.json` and exit nonzero only when a local
     R2 check fails.

  The JSON must include command/config context, `view_count`, `views`,
  `checks.manifest_camera_pairs`, `checks.validation_disjoint`,
  `checks.per_view_quality_gate`, `r1_prerequisite` with the current unpromoted
  status, `memory_mb`, and `latency_ms_per_edit`. A quality failure is a recorded
  R2 negative, not a reason to alter validation or hide the row.

- [ ] **Step 5: Add the canonical mise task.** Add:

  ```toml
  [tasks.r2-conditioned]
  description = "R2: one camera-conditioned NRP over the Cornell-box view manifest"
  run = "uv run python examples/r2_conditioned.py --out out/r2-conditioned/report.json"
  ```

- [ ] **Step 6: Run report-helper tests and lint the runner.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview.ReportTests -v && uv run ruff check examples/r2_conditioned.py nrp/torch_backend/conditioned_multiview.py nrp/torch_backend/relight_multiview.py`

  Expected: PASS and no Ruff errors.

- [ ] **Step 7: Commit the runner and manifest changes.**

  Run: `git add examples/multiview.py examples/r2_conditioned.py mise.toml tests/test_conditioned_multiview.py && git commit -m "R2: add conditioned multiview experiment runner"`

---

### Task 7: Run the representative R2 pilot and record evidence

**Files:**

- Create/force-add when produced: `out/r2-conditioned/report.json`
- Modify: `docs/representation-track.md`
- Modify: `docs/performance.md`

- [ ] **Step 1: Run the no-Mitsuba synthetic R2 trainer smoke.**

  Run: `uv run python -m unittest tests.test_conditioned_multiview -v`

  Expected: all pure-Python and tiny-cache R2 tests pass.

- [ ] **Step 2: Check optional Mitsuba availability before the full pilot.**

  Run: `uv run python -c "import mitsuba; print(mitsuba.__version__)"`

  If it succeeds, run the exact experiment command:

  ```sh
  mise run r2-conditioned
  ```

  If it fails because the optional extra is absent, do not invent Cornell-box numbers;
  retain the synthetic report and record the exact missing prerequisite in the final
  handoff and documentation.

- [ ] **Step 3: Inspect the JSON artifact directly.** Verify that the report contains
  N views, camera metadata, one conditioned model, per-view validation-disjoint
  evidence, quality deltas, memory totals, latency for each tested device, and the
  explicit R1 prerequisite status. Run:

  ```sh
  uv run python -c 'import json; p=json.load(open("out/r2-conditioned/report.json")); assert p["view_count"] >= 2; assert p["checks"]["manifest_camera_pairs"]; assert p["r1_prerequisite"]["promoted"] is False'
  ```

- [ ] **Step 4: Update the status docs surgically.** In `docs/representation-track.md`,
  change only the R2 row/section needed to say that the R2 implementation and pilot
  evidence exist while promotion remains blocked by R1. Add the report link and keep
  the existing R1 negative tables unchanged. In `docs/performance.md`, add a section
  with the exact command, hardware/device, per-view baseline/conditioned deltas,
  one-model-versus-N-model bytes, all-view latency, validation convention, and the
  R1 caveat. Use `out/r2-conditioned/report.json` as the only source for measured
  values.

- [ ] **Step 5: Add the evidence path to the pipeline audit.** Run the existing
  `mise run pipeline-audit` after the report exists. If the audit task needs a narrow
  extension to include the R2 report, add only that path and test the claim scanner.

- [ ] **Step 6: Commit evidence/docs.**

  Run: `git add out/r2-conditioned/report.json docs/representation-track.md docs/performance.md && git commit -m "R2: record camera-conditioned pilot evidence"`

---

### Task 8: Full verification and completion audit

**Files:**

- No new implementation files; inspect all R2 diffs and committed artifacts.

- [ ] **Step 1: Run focused backend tests.**

  Run: `uv run python -m unittest tests.test_torch_backend tests.test_conditioned_multiview tests.test_multiview -v`

  Expected: zero failures and only documented optional-dependency skips.

- [ ] **Step 2: Run the full repository tests.**

  Run: `mise run test`

  Expected: exit code 0 with no test failures.

- [ ] **Step 3: Run Ruff.**

  Run: `mise run lint`

  Expected: exit code 0.

- [ ] **Step 4: Run diff/document checks.**

  Run: `git diff --check` and inspect `git status --short`, the final plan/spec, the
  R2 JSON report, and all changed Markdown links. Confirm no private machine paths,
  placeholders, downloaded assets, or claims unsupported by the JSON remain.

- [ ] **Step 5: Perform the requirement-by-requirement audit before completion.**

  Confirm current evidence for: one conditioned checkpoint, N cache/camera loading,
  view-direction input, global world bounds, disjoint per-view validation, the 1 dB
  per-view gate, one-model/N-model memory, all-view latency, report path, docs update,
  legacy regression tests, and the explicit R1 promotion caveat. If any item is
  missing or only indirectly implied, keep the goal active and implement/verify it.

- [ ] **Step 6: Mark the goal complete only after fresh evidence.**

  After all checks above pass and the completion audit finds no missing requirement,
  call `update_goal` with `status: "complete"`; otherwise report the actual remaining
  gap and continue working.
