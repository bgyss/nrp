# Tracks — The Milestone Progression

This project advances in phases. Each phase is a self-contained document of goal
prompts (or, for phase 3, a decision document); each links its evidence into
`performance.md`. Read them in order to see how the project got here.

| # | phase | document | status |
|---|---|---|---|
| 1 | Replication | [roadmap.md](roadmap.md) | ✅ Complete — 10/10 items, paper-scale training at 35.2 dB held-out PSNR |
| 2 | Extensions | [extensions.md](extensions.md) | ✅ Complete — E1–E9 measured, including the settled E2 negative result and the real-Chrome WebGPU runtime |
| 3 | Verdict | [pipeline-feasibility.md](pipeline-feasibility.md) | ✅ Complete — E10 decision document: "partly viable" for all three targets, blockers named |
| 4 | Production track | [production-track.md](production-track.md) | ✅ Complete — all 10 rungs measured (T1–T4, G1–G2, F1–F2, V1–V2); three closed as honest negatives/partials |
| 5 | Hardening track | [hardening-track.md](hardening-track.md) | ✅ Complete — all 7 rungs measured (H1–H7); several land as honest negatives/partials, per this program's convention |
| 6 | Scale & speed track | [scale-track.md](scale-track.md) | ✅ Complete locally — S1–S6 measured; CUDA stretch S7/S8 parked |
| 7 | Representation track | [representation-track.md](representation-track.md) | ⛔ R1 not promoted (redesigned held-out-camera gate) — no arm passes on all 5 seeds; closed as a characterized negative with a G5 decomposition; R2 pilot remains an honest negative. Fair-allocation single-view parity re-measured on two scenes with opposite outcomes: all 3 world arms pass 5/5 on toy 64², none passes on Kitchen 128²; the earlier "19× allocation handicap explains the Kitchen negative" claim is retracted (2026-08-29) The held-out-camera campaign that closed R1 has been re-read at 96 held-out lights from its committed checkpoints (no retraining): the negative survives and strengthens -- no arm passes G1/G3/G4, the failing-row count rises from 83 to 94 of 180, and every arm's mean delta shrinks; the campaign's published per-row and per-rotation deltas are retracted as measurements while its conclusion stands |

## How the phases connect

1. **Replication** (`roadmap.md`) reimplemented Sancho et al. (EGSR 2026) and
   took the toy pipeline to paper scale, with every claim traced to a committed
   report.
2. **Extensions** (`extensions.md`) stress-tested the north-star question — is
   the SAMPLEPATHS/GATHERLIGHT/proxy decoupling a building block for a
   real-time neural rendering pipeline? — across animation, dynamic geometry,
   out-of-core scale, engine runtimes, inverse art direction, production
   controls, and quality tiers.
3. **Verdict** (`pipeline-feasibility.md`) is the E10 decision document: per
   target audience (games / animated film / feature VFX), what's measured,
   what blocks, and what a production team would still have to build.
4. **Production track** (`production-track.md`) attacked the verdict's named
   blockers as a balanced ladder — each rung pairs a scale/robustness proof
   with a performance target — ending in one summit demo per target audience.
   All 10 rungs are measured; the honest negatives (G1's remaining gap, V1's
   additivity fail, F2's storage cost) and the undiagnosed quad zero-collapse
   became the next phase's work items.
5. **Hardening track** (`hardening-track.md`) fixed what the production track
   surfaced — root-caused the QuadLight zero-collapse (H1), retrained the
   V1/V2 rig post-fix (H2, honest partial: additivity still misses preview
   tier), swept textured-quad quality levers (H3, honest negative — a
   conditioning-scheme problem, not budget/capacity), ported rig compositing
   to the proven WebGPU runtime (H4, honest partial: fast but not yet
   real-time for an 8-light session), re-traced a real scene for dynamic
   geometry and found G1's toy-scale fix does not transfer at real scale (H5,
   honest negative), flipped F2's storage negative (H6), and re-issued the
   feasibility verdict against the full T1–V2+H1–H6 evidence base (H7).
   Motivating audit: `status/2026-07-11.md`.
6. **Scale & speed track** (`scale-track.md`) is the planned next phase: it
   attacks the throughput ceilings the H7 verdict left standing (Python-bound
   streaming, MPS training at only 1.6× CPU, nothing above 512² or beyond one
   real scene) and carries the NVIDIA-ecosystem CUDA port as an explicit
   stretch goal on rented cloud GPUs, with the provider comparison and cost
   estimates in `plans/2026-07-16-scale-track-research.md`.
7. **Representation track** (`representation-track.md`) tests whether replacing the
   per-view 2D pixel hashgrid with a 3D first-hit world-position hashgrid is a viable
   foundation for multi-camera proxies. R1 passes on the toy box but the direct 3D
   grid passes only 1/3 controlled Country Kitchen seeds. R1A's five-seed crossed
   matrix finds one strict candidate pass — target-scale tri-plane at 5/5 — while
   direct 3D under both policies and framework-default tri-plane fail at least one
   seed. After correcting the gather/denoiser execution confound, the full R1C
   matrix still fails multiple orientation/normalization cells and R1E's
   independent Bedroom scene fails seed 4 (−3.083 dB). Rather than keep tuning past
   that stop point, the R1 redesign re-diagnosed the negative as a ~19× allocation
   handicap built into the parameter-matching rule and re-specified promotion around
   held-out-camera generalization instead of single-view parity. Three redesigned
   arms (collision-free sparse voxel, normal-aware tri-plane, occupancy-allocated
   3D hash) all clear a positive mean delta against `pixel2d` but none passes on
   all 5 seeds; the sparse arm is the strongest and the only rotation-robust one.
   R1 therefore remains unpromoted, now closed as a characterized negative with a
   mandatory fallback decomposition; the R2 pilot is separately recorded as an
   honest quality negative, so R3–R6 remain unpromoted. A follow-up fair-allocation
   parity re-measurement (each arm sized by its own occupancy rather than matched
   to `pixel2d`'s parameter count, under the original unchanged −0.5 dB per-seed
   gate) finds all three world arms pass 5/5 seeds on the toy scene but none
   passes on Country Kitchen — and retracts the earlier claim that the ~19×
   parameter-matching handicap explained the Kitchen negative, since removing that
   handicap reproduces the same negative almost exactly.

   The held-out-camera campaign itself was re-read on 2026-08-29 against 96 held-out
   lights (12 independent draws of its own 8-light evaluation configuration), from
   the same 165 committed checkpoints and with no retraining. It reproduces the
   committed run exactly at the original light count, and at 96 lights the negative
   survives and strengthens: no arm passes G1, G3 or G4, the number of failing rows
   rises from 83 to 94 of 180, and every arm's mean delta over the `pixel2d`
   baseline shrinks by 0.32–0.47 dB. The campaign's published per-row, per-rotation
   and per-arm mean deltas are retracted as measurements of the arms (the original
   per-row estimator's ~1.33 dB standard error exceeded the 1.0 dB margin it was
   testing); the closure they supported is unchanged. See
   `docs/performance.md#encoding-redesign-campaign-re-read-at-96-held-out-lights-2026-08-29`.

Evidence conventions for all phases: every measured claim lands in a JSON
report under `out/` and in [performance.md](performance.md) with hardware
context; honest negative results are deliverables.
