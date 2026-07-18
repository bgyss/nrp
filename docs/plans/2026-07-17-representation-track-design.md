# Representation Track (Phase 7) — Design

**Date:** 2026-07-17. **Status:** approved design, pre-implementation.
**Prerequisites:** scale track S1–S6 closed (`docs/scale-track.md`); CUDA
stretch rungs S7/S8 remain parked and nothing here depends on them.

## Thesis

Six phases proved the SAMPLEPATHS/GATHERLIGHT/proxy decoupling — but always
per-view, per-light-type, fixed-camera. This track asks the next falsifiable
question: **can one world-anchored network represent the rendered data of a
scene across cameras**, turning the NRP from a screen-space light proxy into
the first rung of a general representation for rendered data?

The architectural move: replace the 2D multiresolution hashgrid over *pixel
coordinates* with a **3D multiresolution hashgrid over the first-hit world
position** — which every existing cache already stores in its G-buffer aux, so
no re-export is needed — and add view direction as a conditioning input. The
model input becomes:

```
hashgrid3d(world_pos) ⊕ view_dir ⊕ aux(albedo, depth, normal) ⊕ light params
```

One network then naturally serves every camera because spatial features live
in the scene, not the screen, and novel-view interpolation between trained
views becomes a meaningful, falsifiable gate.

### Rejected alternatives

- **Per-view embedding (B):** keep the pixel hashgrid, add a learned per-view
  latent. Low risk and folds N models into one, but pixel coordinates mean
  different geometry per view, so the table must multiplex all views, and
  novel views are only reachable by ungrounded embedding interpolation.
  Likely to "work" while teaching little. Not a pre-planned fallback: if R1
  fails its gate, the deliverable is the characterized negative, not a
  retreat to B.
- **Ray-space / light-field encoding (C):** encode the camera ray itself
  (Plücker or origin⊕direction) as the spatial input. Discards the locality
  the hashgrid exploits and would need far more views than laptop-scale
  exports afford.

## Crossover: perception-for-videogen

The `codex/perception-goals` branch of `~/src/perception-for-videogen`
specifies a closed loop — intent → scene program → preview render →
perception compiler (Scene4D IR) → executable verifier → repair →
**lightweight neural rendering behind a preservation gate** (Goal 06). Goal
06's appearance renderer consumes a verified base render plus diagnostic
buffers (depth, normals, instance boundaries, motion) and must provably
preserve structure.

A world-anchored NRP is a natural fit for the "cheap preview + diagnostic
buffers" side of that contract: it can emit depth/normals/world-position and
per-light radiance for any queried camera, differentiably, with no cache
access at edit time. This track carries exactly **one bridge rung (R6)** —
a concrete cross-repo artifact — and otherwise stays NRP-internal.

## The ladder

Ordering: R1 → R2 → R3 strictly (each de-risks the next); R4 after R2; R5
after R4; R6 any time after R2. If R1 misses its parity gate, the track
pivots to characterizing why — an honest negative on world-anchoring is
itself the deliverable — rather than proceeding on a broken foundation.

### R1. World-space encoding at parity

Implement a 3D multiresolution hash encoding (generalizing
`nrp/torch_backend/encoding.py`: per-level dense/hashed tables, trilinear
interpolation, geometric growth), selectable via a validated config flag.
Train the standard single-view toy and kitchen configs with
hashgrid3d(world_pos) in place of hashgrid2d(px).

- **Verify:** encoding unit tests (interpolation correctness at cell corners,
  dense/hashed threshold, gradient flow); config validation tested; existing
  2D path untouched and its tests green.
- **Gate:** held-out PSNR within **0.5 dB** of the committed 2D-pixel-hashgrid
  baselines at matched parameter budget and iterations, per scene.
- **Measure:** params, iters/s (CPU + MPS), held-out PSNR vs baseline, in
  `out/r1-worldgrid/` and `docs/performance.md`.

### R2. One network, N cameras

Train a single camera-conditioned proxy on the existing multiview cornell-box
caches (roadmap item 7's `views.json` machinery), with view direction as
input; compare against the committed per-view baseline models.

- **Verify:** loader accepts N (cache, camera) pairs into one training run;
  per-view validation sets disjoint from training lights, same convention as
  `relight_multiview`.
- **Gate:** per-view held-out PSNR within **1 dB** of the per-view baselines.
- **Measure:** memory of 1 conditioned net vs N proxies (MB); one-light-edit
  latency across all views vs `relight_multiview`'s N-model path. In
  `out/r2-conditioned/` and `docs/performance.md`.

### R3. Novel-view interpolation

Export a camera arc (~8–12 views) of the cornell box; train on alternating
views; evaluate at the held-out intermediate cameras against freshly exported
ground-truth caches (GATHERLIGHT references).

**Design seam (stated honestly):** the model takes first-hit aux as input,
and a novel camera has no cache. Novel-view aux comes from a cheap
primary-ray G-buffer pass (toy tracer / exporter primary hits only). The
report must separate error attributable to the proxy from error in this aux
source.

- **Verify:** held-out cameras never appear in training; the primary-ray aux
  pass parity-tested against cache aux at a trained view.
- **Gate:** quantified interpolation PSNR with a degradation-vs-view-distance
  curve; extrapolation beyond the arc characterized (expected negative — an
  honest statement of where the representation ends).
- **Measure:** PSNR/SSIM/FLIP per held-out view; curve in
  `out/r3-novelview/` and `docs/performance.md`.

### R4. Real scene, real scale

Multi-view kitchen exports at 512² (S3's parallel shard writes make this
affordable), camera-conditioned training through the S1-accelerated streamed
path.

- **Verify:** caches pass `PathCache.validate()`; bounded-residency guarantee
  holds (peak reported); streamed-vs-in-memory equivalence convention from S1
  respected.
- **Gate:** meets the established **18 dB** held-out envelope per view, or an
  honest negative decomposed per view/region.
- **Measure:** export wall-clock, cache GB, train wall-clock, per-view PSNR —
  extending the T1/S2 scaling tables. In `out/r4-realscene/` and
  `docs/performance.md`.

### R5. Camera in the WebGPU runtime

Extend the runtime export format and WGSL generator so the browser evaluates
the world-anchored proxy with camera as a live control — the first movable
camera in the runtime. Per-view aux/G-buffer textures are precomputed for a
set of cameras or generated by a primary-ray pass; the report states which.

- **Verify:** T4-style parity gate between torch and WGSL at each tested
  camera; existing 512²/1024² regression gates still pass.
- **Measure:** camera-move latency p95 vs the light-slider baselines (T4/H4
  conventions), 512² and 1024². In `out/r5-webgpu-camera/` and
  `docs/performance.md`.

### R6. Bridge rung — diagnostic buffers for the perception loop

The conditioned proxy emits its diagnostic buffers — depth, normals, world
position, per-light radiance — for any queried camera, in a documented,
versioned format aligned with the Scene4D `evidence` fields of
perception-for-videogen's Goal 06 appearance-renderer input contract.

- **Verify:** emitted buffers validated against cache G-buffer ground truth
  at trained views; format documented with a schema version; round-trip
  loader tested.
- **Deliverable:** an interface note in **both** repos mapping NRP onto Goal
  06's contract, stating plainly what NRP provides (differentiable per-light
  radiance + geometry buffers per camera, static scenes) and what it does
  **not** (no instance IDs, no motion/flow, no dynamic scenes beyond the
  H5-characterized residual path).
- **Measure:** buffer-emission latency per camera; report in `out/r6-bridge/`
  and `docs/performance.md`.

## Shared conventions (same as all prior phases)

- Single Apple Silicon laptop: CPU/MPS/WebGPU only; CUDA stays parked and
  nothing here depends on it.
- All new/changed code passes `mise run test` and `mise run lint`; new
  features get unit tests that skip cleanly when optional dependencies are
  absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context; `mise run pipeline-audit`
  verifies referenced paths exist.
- Honest negative results are deliverables.
- These rungs go **beyond** the paper; `docs/paper-mapping.md` flags them as
  extensions explicitly, as the E-track did.
- At implementation start: write `docs/representation-track.md` (rung/status
  doc in the S-track style) and add row 7 to `docs/tracks.md`.

## Named risks

1. **3D hashgrid collisions at toy scale** — 3D tables hash earlier than 2D
   at equal budget; R1's matched-budget gate surfaces this before any
   multi-camera claim.
2. **View-dependent transport** — gathered radiance at a first hit is
   view-independent only for the direct-ish Lambertian component; later-bounce
   paths differ per camera. The scenes are Lambertian, which makes R3
   tractable, and the design says so rather than implying generality over
   glossy transport.
3. **Multi-view export cost at 512²** — mitigated by S3's shard-write and
   exporter speedups; R4 records the actual wall-clock either way.
4. **Aux for novel views** — the primary-ray G-buffer seam in R3/R5 is a
   stated dependency, parity-tested, with its error contribution reported
   separately.
