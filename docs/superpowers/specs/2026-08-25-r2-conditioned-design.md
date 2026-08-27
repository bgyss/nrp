# R2 Camera-Conditioned NRP Design

**Date:** 2026-08-25

**Status:** Approved for implementation in the active R2 goal

## Goal

Implement representation-track R2: train one PyTorch NRP checkpoint over N camera
views, condition the prediction on each camera's view direction, compare it with
separate per-view proxy baselines, and record the per-view quality, memory, and
light-edit latency evidence under `out/r2-conditioned/`.

R2 implementation is intentionally separate from R2 promotion. The repository's
R1 world-anchoring prerequisite is currently an honest negative, so the R2 report
must state that prerequisite status and must not promote the representation track
even if the pilot's local checks pass.

## Constraints and acceptance

- Keep the existing `TorchNRP` 2D model and all existing checkpoints backward
  compatible. The camera-conditioned input is opt-in.
- Use the existing relative-MSE training objective, path-cache/GATHERLIGHT targets,
  denoiser configuration, CPU/MPS device conventions, and relative manifest paths.
- A conditioned run accepts N cache/camera records in one training invocation. All
  views in one run must have the same resolution; their world-position bounds are
  normalized with one global scene bound so the shared spatial representation has
  one coordinate system.
- Each manifest camera record contains `origin` and `target`. The model input uses
  the normalized forward direction `normalize(target - origin)`, broadcast over the
  view's pixels. This is a camera-direction condition, not a learned view ID.
- Training uses one shared light pool across views. Validation lights are generated
  by a dedicated per-view RNG and are checked for parameter-vector disjointness from
  every training light used by that view.
- The report compares the conditioned model and each per-view baseline on the same
  held-out light set for that view. The R2 quality gate is independently reported as
  conditioned PSNR within 1 dB of the baseline PSNR for every view.
- The report measures one conditioned checkpoint versus the sum of N baseline model
  files, and measures all-view light-edit latency on the same device list using the
  existing `relight_multiview` timing convention (warmup and device synchronization).
- No cloud services or CUDA-only work are required. Optional Mitsuba export remains
  an explicit prerequisite for the full Cornell-box experiment; unit and synthetic
  smoke tests must run without it.

## Architecture

### Camera-conditioned model

`nrp/torch_backend/model.py` gains an opt-in `camera_conditioned` configuration
flag. When enabled, `TorchNRP.forward` accepts a fourth `(N, 3)` `view_dir` tensor
and concatenates it with the spatial encoding, seven G-buffer auxiliary features,
and light-shape parameters before the MLP. A single `(3,)` direction is accepted
and broadcast for convenience. Unconditioned models continue to use their current
three-argument call signature and serialized configuration defaults.

R2 uses `spatial_encoding: "world3d"` so the shared network is anchored in the
scene rather than multiplexing unrelated pixel-coordinate grids. The multi-view
trainer computes one finite global world bound across all caches and stores it in
the checkpoint. The existing 2D and R1 `world3d`/`world_triplane` paths remain
unchanged when `camera_conditioned` is false.

### Multi-view data and training

Add a focused `nrp/torch_backend/conditioned_multiview.py` module containing:

- manifest records with resolved cache paths and validated camera metadata;
- camera-direction and global-world-bound helpers;
- a multi-view image pool that renders the same sampled light shapes through every
  cache and stores targets per view;
- fixed per-view validation-set construction and disjointness checks;
- one-network training over randomly sampled `(view, pool-light, pixel)` rows;
- per-view evaluation helpers that work for both conditioned and baseline models.

The training entry point returns a JSON-serializable report containing the model
configuration, view names, shared training-light vectors, validation-light vectors,
per-view metrics, timing, and checkpoint/model paths. It writes one checkpoint/model
and does not create one conditioned model per view.

### Inference and experiment runner

Extend `nrp/torch_backend/relight_multiview.py` with a shared-model loader/render
path that precomputes each view's spatial/aux tensors and view direction once. A
light edit calls the same model for each view and never calls GATHERLIGHT or reads
segments during the timed render. Existing per-view `ViewProxy` behavior remains
valid.

Add `examples/r2_conditioned.py` to reuse the Cornell-box pose/export and per-view
baseline machinery from `examples/multiview.py`. It will:

1. export or reuse N path caches and write a camera-aware manifest;
2. train or reuse the N per-view baseline models;
3. train one conditioned model;
4. evaluate both systems on identical per-view held-out lights;
5. measure memory and all-view edit latency for the conditioned and baseline paths;
6. write `out/r2-conditioned/report.json` and exit nonzero only for failed local
   checks, while recording the R1 prerequisite as a separate status field.

The existing multiview manifest writer will include camera metadata so the R2
runner and the established relighter share one path/manifest convention.

## Testing

Test-first coverage will pin:

- camera-conditioned model input shape, broadcast behavior, missing/invalid direction
  errors, save/load round-trip, and unchanged unconditioned call behavior;
- manifest resolution and camera-direction normalization, including malformed and
  zero-length camera vectors;
- global world-bound computation and same-resolution validation;
- multi-view pool target/parameter shapes and deterministic per-view validation
  disjointness;
- one-network training on two tiny hand-authored caches, including a report with
  per-view metrics and one model artifact;
- shared-model relighting parity with direct model calls and no cache use during
  the timed proxy render;
- report gate logic for pass and fail cases.

Targeted unit tests run first, followed by the complete unittest suite and Ruff.
The full Cornell-box experiment is run when the optional Mitsuba dependency and
runtime budget are available; otherwise the exact command and missing evidence
remain recorded rather than being inferred from the smoke test.

## Evidence and documentation

The experiment report is the authoritative machine-readable artifact. Update
`docs/representation-track.md` to distinguish R2 implementation/pilot evidence from
promotion status, and add a measured R2 section to `docs/performance.md` with the
exact command, device context, per-view deltas, memory comparison, latency table,
and the R1 prerequisite caveat. Do not overwrite the existing R1 negative or claim
that R2 proves novel-view generalization.
