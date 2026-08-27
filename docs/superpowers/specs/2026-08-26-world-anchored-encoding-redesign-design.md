# World-Anchored Encoding Redesign — Design

**Date:** 2026-08-26. **Status:** approved design, pre-implementation.
**Supersedes the experimental design of:** `docs/plans/2026-07-27-r1-next-experiments.md`
(rungs R1A–R1E). **Amends:** `docs/plans/2026-07-17-representation-track-design.md`
rung R1.

## Why this exists

Representation-track rung R1 has produced three campaigns of unstable, mostly
negative results (`out/r1-worldgrid/`, `out/r1-followup/`, `out/r1a/`,
`out/r1-promotion/`). The response so far has been to tune the world-anchored
encoding harder — more capacity, tri-plane allocation, initialization policies,
more seeds. This document stops that and re-states the problem, because
measurement shows the encoding was never the binding constraint.

## Diagnosis

The full test suite is green on `main` (347 tests, OK, 4 skipped). The failures
are gate failures, not defects.

Measured per-level occupancy on the real 128² Country Kitchen cache
(`out/kitchen/path_cache.npz`), for the two configs the R1 gate compares:

`pixel2d` — `examples/kitchen_torch.json`, `table_size_log2=14`:

| level | res | dense | slots | distinct verts | collision |
|---:|---:|---|---:|---:|---:|
| 0–6 | 4…78 | yes | 25…6241 | = slots | 0.0% |
| 7 | 128 | no | 16384 | 16641 | 36.4% |

`world3d` — `examples/r1_kitchen_world3d.json`, `table_size_log2=12`:

| level | res | dense | slots | distinct verts | collision |
|---:|---:|---|---:|---:|---:|
| 0 | 7 | yes | 512 | 407 | 0.0% |
| 1 | 10 | yes | 1331 | 870 | 0.0% |
| 2 | 16 | no | 4096 | 2153 | 20.4% |
| 3 | 24 | no | 4096 | 4795 | 41.6% |
| 4 | 36 | no | 4096 | 10428 | 64.0% |
| 5 | 55 | no | 4096 | 22687 | 82.0% |
| 6 | 84 | no | 4096 | 44393 | 90.8% |
| 7 | 128 | no | 4096 | 78080 | 94.8% (max 34 verts/slot) |

The 78,080 figure was measured against the code as it stood, whose cell clamp was
`clip(0, res)`. `nrp/torch_backend/occupancy.py` recomputes it as **78,084** under the
`clip(0, res - 1)` cell rule now used by `_floor_cell`; the four-vertex delta is exactly
the boundary points that clamp bound affects. Both numbers are correct for their code
state, and the argument is unchanged: 4,096 slots against ~78k distinct vertices.

Three findings, none of which is a coding bug:

1. **"Matched parameter budget" is the wrong control variable.** Matching ~106k
   parameters forced `world3d` to a 4096-slot table while `pixel2d` keeps 16384.
   Because 3D touches eight corners over a non-axis-aligned surface, its finest
   level queries 78,080 distinct vertices against 2D's 16,641. Slots per distinct
   queried vertex: **0.98 for 2D, 0.052 for 3D — a ~19× handicap awarded by the
   gate's own matching rule.** This explains why the +88%-capacity diagnostic did
   not close the gap: roughly 2× against a ~19× deficit.

2. **The 2D control is a memorizer, not a baseline.** At `res=128` on a 128²
   image, level 7 is exactly one vertex per pixel, and levels 0–6 are fully dense
   with zero collisions. `pixel2d` is effectively a free per-pixel embedding
   table. R1 asked a world-anchored encoder to match a per-pixel lookup table at
   single-view reconstruction — the one task where that table is optimal, and the
   one where it cannot generalize to any other camera. The gate measured the
   capability world-anchoring deliberately trades away.

3. **The finest-resolution schedule was copied from 2D unexamined.**
   `finest_resolution=128` means "one vertex per pixel" in 2D but "~3 cm voxels
   over a 4.2 × 3.2 × 5.2 m kitchen" in 3D. The two quantities share a number and
   nothing else.

### Genuine but minor code defects

Reported honestly, with measured impact:

- `nrp/torch_backend/encoding.py:60` — `(ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) &
  (self.table_size - 1)`. `&` binds tighter than `^`, so the mask applies only to
  the y term. Verified **numerically identical to the reference formula in every
  committed config**, because `_PRIMES[0] == 1` and `ix <= res < table_size`, so
  masking commutes with the XOR. It did not cause the negative. It is latent and
  breaks as soon as `res >= table_size`.
- `nrp/torch_backend/encoding.py:67` and `:158` — `floor(pos).clamp(0, res)`
  should clamp to `res - 1`; `x0 == res` yields a degenerate cell.
  **Correction (2026-08-26, during implementation):** this was overstated. The
  change is a verified **no-op for every in-range input**, not a defect. At
  `xy = 1.0` the old code gives `pos0 = res, frac = 0`, selecting the corner
  entry; the new code gives `pos0 = res - 1, frac = 1.0`, whose interpolation
  weights collapse onto that same entry. Measured max absolute difference over
  in-range points: 0.0. The clamp is retained as an invariant guarantee
  (`frac ∈ [0, 1]`, non-degenerate cell) that arm B's sparse lookup relies on,
  and its tests assert that invariant rather than a behavioural change.
- `nrp/torch_backend/streamed_train.py:373` — `_pixel_tensors` is called
  unconditionally, so the S1 streamed path has no world-position support at all.
  Rung R4 is structurally blocked on this.

## What changes

The gate is re-specified around generalization, and the encoding is redesigned to
remove collision pressure as an uncontrolled variable rather than tune it.

## Architecture

### `nrp/torch_backend/occupancy.py` (new)

One job: given a `PathCache` and a level-resolution schedule, return the exact set
of queried grid vertices per level plus a capacity report (distinct vertices,
slots, collision fraction, maximum slot load). It is the shared substrate for arms
A and B and the source of the diagnostic tables above, promoted from a throwaway
script to a tested artifact. It knows about caches; encoders do not.

### Encoder seam

Every spatial encoder exposes exactly three things:

- `output_dim: int`
- `forward(coords) -> (N, D)`
- `capacity_report() -> dict`

Two encoders need extra construction input, declared explicitly rather than
smuggled in through the cache:

- `needs_occupancy: bool` — `train.py` builds occupancy from the cache and passes
  it to the constructor, so the encoder never imports `PathCache`.
- `needs_normals: bool` — arm C only; the model passes the normal slice of `aux`
  when the flag is set.

`TorchNRP` builds encoders through a registry keyed by `model.spatial_encoding`,
replacing the current if/elif chain. Adding an arm becomes one registry entry.

### Arm B (primary) — `SparseVoxelEncoding`, new `nrp/torch_backend/sparse_encoding.py`

Per level: a sorted `int64` key array of occupied vertex codes and a feature table
sized exactly to occupancy. Lookup is `searchsorted` plus an exact-match test. No
hash, therefore no collisions, by construction rather than by tuning.

Keys register as buffers so `state_dict` and `--resume` handle them with no
special-casing. The key bytes are counted in the reported memory budget; that cost
is real and is stated, not hidden.

A vertex outside the occupied set — which only a novel camera produces —
contributes zero features at that level and is carried by coarser levels. The
fallback is differentiable. The per-camera out-of-occupancy fraction is reported
(see G5).

**Rationale:** R1A measured a ±1.74 dB seed spread. Collision pressure is the
largest uncontrolled contributor to it. This arm removes the variable rather than
averaging over it.

### Arm C — `NormalAwareTriPlane`, in `encoding.py`

Each point reads only the plane whose axis is most aligned with its surface
normal, chosen by a hard argmax over `|n · axis|`. One plane read per point
instead of three, so one-third the capacity pressure, and plane choice follows
geometry rather than the world frame. Gradients flow through the features, not
through the discrete selection.

**Rationale:** this is the direct fix for the −3.651 dB rotation failure in the
corrected R1C matrix, which is what killed the previous tri-plane candidate.

### Arm A (control) — occupancy-allocated `HashEncoding3D`

No new class. An `allocation: "occupancy"` option sizes each level's table from
the measured distinct-vertex count and drops levels the budget cannot serve,
instead of crushing them to 5% capacity. Arm A keeps the comparison honest about
capacity; it is not expected to fix the seed instability on its own.

### Defect fixes

Landed first and in a separate commit so they cannot confound any arm: the
precedence bug, the boundary clamp in both grids, and world-position support in
`streamed_train.py`.

## Experiment protocol

**Scene and cameras.** Toy box at 64², twelve cameras on an arc: eight training,
four held out at interpolated positions. Held-out caches are exported but used
only for evaluation references and aux — never for training.

**Prerequisite.** `nrp/toy_tracer.py:59` renders pinhole rays looking down `+z`
with only a `--camera-pos` origin override; `examples/multiview.py`'s arc
machinery is Mitsuba-only. `_camera_rays` gains a look-at target so the toy tracer
can produce a rotating arc. A translation-only arc would still break the
pixel↔world correspondence but would be a weak test of camera generalization.
Rung R3 needs the look-at basis regardless.

**Scope honesty.** Taking aux from the held-out cache isolates the representation
question. The "where does aux come from at a novel camera" seam remains R3's
problem, and the report states this rather than quietly benefiting from it.

## Gates

**G1 — promotion gate, held-out camera.** Two conditions, both binding on every
seed at every held-out camera:

1. **Comparative margin ≥ 1.0 dB** over the only thing a screen-space proxy can do
   at a novel camera: reuse the nearest trained view's `pixel2d` proxy. The
   threshold is taken from R2's existing per-view tolerance in the approved ladder
   rather than newly invented.
2. **Absolute floor of 15 dB PSNR.**

**Why the floor exists (added 2026-08-26, during implementation).** The G1 baseline
is deliberately weak: a `pixel2d` proxy indexes its hashgrid by pixel coordinate, so
at a different camera its spatial features encode the wrong view's geometry. Beating
it is a low bar, and a comparative-only pass would show that world anchoring beats a
strawman, not that it is usable. The floor makes a pass mean "usable at a novel
camera".

**The 15 dB figure is invented for this campaign and is not an established track
convention** — unlike the 1.0 dB margin. It was chosen against measured evidence:
toy-scale quality at a *trained* view is 19.17 dB (`out/toy-torch`) and 19.98–22.16 dB
(`out/r1-worldgrid`, 48²). The track's usual 18 dB envelope sits within 1–4 dB of
that, so a held-out camera could fail it structurally, producing a negative that
described the threshold rather than the representation. 15 dB leaves room for
genuine novel-view degradation while still rejecting a broken arm. The report must
state the figure's provenance rather than implying an established convention.

**G2 — capacity honesty (reported, not gated).** Single trained-view PSNR against
`pixel2d`, reporting effective capacity (slots per distinct queried vertex,
matched within ±5%) and total bytes including arm B's key arrays. Explicitly not a
promotion gate. The report records the memorizer argument so that this reversal
from the old 0.5 dB gate is auditable rather than appearing as a moved goalpost.

**G3 — stability.** Five seeds. **Per-seed pass required; no averaging** — the
existing convention, preserved. Mean, standard deviation, and paired bootstrap CI
are reported as context only. Arm B additionally *asserts* 0.0% collision on
trained views rather than merely reporting it.

**G4 — frame robustness.** Three world rotations about the up axis. The worst
orientation must still pass G1 on every seed. This is binding from the start
rather than bolted on at promotion time, because it is what killed the previous
candidate.

**G5 — fallback decomposition (arm B, mandatory).** Per held-out camera, report
the out-of-occupancy query fraction and split G1 error into in-occupancy and
out-of-occupancy pixels. Not a gate — a required decomposition, so that a good G1
cannot hide behind a favourable fallback.

**Stop condition.** If no arm passes G1 across all seeds × held-out cameras ×
orientations, the track closes as a characterized negative accompanied by the G5
decomposition. No further tuning rounds.

The prior R1 evidence is not discarded. The report reinterprets it: the ambient-3D
negative is now explained by the measured ~19× allocation handicap.

> **2026-08-27 note:** retracted. The Kitchen 128² fair-allocation
> re-measurement shows `world3d`'s allocation-fair mean (−0.355 dB) reproduces
> the original handicapped mean (−0.356 dB) almost exactly, and the 5.5×-slot
> `world_sparse` arm performs worst — so the allocation handicap is not the
> mechanism behind the negative. See
> `docs/performance.md#r1-fair-allocation-parity-re-measurement-toy-64-and-kitchen-128`.

## Verification

**Unit tests (new).** Sparse index exactness — every occupied key resolves
uniquely, with zero collisions asserted rather than assumed; interpolation
correctness at cell corners for all encoders, where output equals the table entry;
gradient flow through every arm; unseen-key fallback returns zeros and remains
differentiable; normal-aware plane selection picks the expected plane for
axis-aligned normals; `capacity_report()` matches a brute-force recount.

**Regression tests for the defect fixes.** The precedence fix is asserted against
the reference formula including a `res >= table_size` case that currently
diverges — today's committed configs coincidentally agree, so a naive test would
pass on the bug. The clamp fix is asserted at coordinates exactly 1.0.

**Integration.** `streamed_train` world-position support with an
in-memory-vs-streamed parity test per the S1 convention; existing `pixel2d` tests
stay green; a checkpoint back-compatibility test loads a committed 2D model
unchanged.

Optional-dependency tests skip via the existing `@unittest.skipUnless`
convention. All measured claims land in `out/r1-encoding-redesign/report.json` and
`docs/performance.md` with hardware context; `mise run pipeline-audit` verifies
the referenced paths exist.

## Named risks

1. **Arm B's occupancy set may not generalize across cameras.** Held-out cameras
   see the same surfaces, so occupancy should transfer, but G5 exists precisely to
   measure this rather than assume it.
2. **Key-array memory may dominate at real scale.** Toy 64² will not surface it;
   G2's byte accounting is what carries the honest number forward to R4.
3. **The look-at prerequisite touches a producer used by many committed tests.**
   The default camera path must remain bit-identical; existing toy-cache tests are
   the regression guard.
4. **Arm C's hard argmax is discontinuous across normal boundaries.** Expected to
   show as seams at surface-orientation discontinuities; G1 per-camera error maps
   should reveal it if it matters.
