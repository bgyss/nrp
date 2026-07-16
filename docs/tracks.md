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

Evidence conventions for all phases: every measured claim lands in a JSON
report under `out/` and in [performance.md](performance.md) with hardware
context; honest negative results are deliverables.
