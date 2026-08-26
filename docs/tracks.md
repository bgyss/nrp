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
| 7 | Representation track | [representation-track.md](representation-track.md) | ⛔ R1 promotion pending R1C/R1E — target-scale tri-plane passes 5/5 Kitchen seeds; R2 pilot remains an honest negative; R3–R6 unpromoted |

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
   seed. R1C coordinate robustness and R1E independent-scene confirmation remain
   required before R1 promotion; the R2 pilot is separately recorded as an honest
   quality negative, so R3–R6 remain unpromoted.

Evidence conventions for all phases: every measured claim lands in a JSON
report under `out/` and in [performance.md](performance.md) with hardware
context; honest negative results are deliverables.
