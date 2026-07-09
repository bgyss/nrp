# Tracks — The Milestone Progression

This project advances in phases. Each phase is a self-contained document of goal
prompts (or, for phase 3, a decision document); each links its evidence into
`performance.md`. Read them in order to see how the project got here.

| # | phase | document | status |
|---|---|---|---|
| 1 | Replication | [roadmap.md](roadmap.md) | ✅ Complete — 10/10 items, paper-scale training at 35.2 dB held-out PSNR |
| 2 | Extensions | [extensions.md](extensions.md) | ✅ Complete — E1–E9 measured, including the settled E2 negative result and the real-Chrome WebGPU runtime |
| 3 | Verdict | [pipeline-feasibility.md](pipeline-feasibility.md) | ✅ Complete — E10 decision document: "partly viable" for all three targets, blockers named |
| 4 | Production track | [production-track.md](production-track.md) | 🚧 In progress — trunk T1–T4 + three summit-demo branches attacking the named blockers |

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
4. **Production track** (`production-track.md`) attacks the verdict's named
   blockers as a balanced ladder — each rung pairs a scale/robustness proof
   with a performance target — ending in one summit demo per target audience.

Evidence conventions for all phases: every measured claim lands in a JSON
report under `out/` and in [performance.md](performance.md) with hardware
context; honest negative results are deliverables.
