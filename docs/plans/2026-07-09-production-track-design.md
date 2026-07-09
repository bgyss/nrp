# Design: The Production Track, Docs Reorganization, and Social Drafts

Date: 2026-07-09
Status: draft for review
Scope: everything after E10 — a new research track (`docs/production-track.md`),
a docs reorganization that makes the milestone progression legible, and three
social-post drafts under `docs/social/`.

## Problem

The replication roadmap (10/10) and the extension program (E1–E9, measured; E10,
verdict) are complete. `docs/pipeline-feasibility.md` renders "partly viable" for
all three targets (games, animated film, feature VFX) and names the blockers:

1. **Games** — dynamic geometry / frozen transport. E2's TorchNRP fine-tune
   *failed* its recovery target even with replay regularization (the one settled
   negative result).
2. **Animated film** — scale proof is cornell-box-only; production-shot
   complexity unproven.
3. **Feature VFX** — production light rigs and layered controls unproven beyond
   toy compositing.

Meanwhile the runtime side is strong (real WebGPU in real Chrome, 30/60 fps at
128–512²). The bottleneck has moved from "can it run fast" to "can it survive
real content and real change." The project needs a next track that attacks the
named blockers without letting either the robustness axis or the performance
axis get ahead of the evidence — and docs/social artifacts that make the
progression visible.

## Decisions Made (with the user, 2026-07-09)

- **Center of gravity:** balanced ladder — each milestone pairs one
  scale/robustness proof with one performance target.
- **Summit:** all three target audiences get a summit demo, mapping 1:1 onto the
  E10 verdict table.
- **Structure:** trunk-and-branches. A shared trunk of rungs every summit needs,
  then three short independent branch sequences. No duplicated evidence; a
  failed branch does not stall the others.

## Shared Conventions (inherited from roadmap/extensions)

- Every rung is a self-contained goal prompt (usable via `/goal <prompt>` or as
  a session brief) with baked-in **verification** and **measurement**.
- All new/changed code passes `mise run test` and `mise run lint`; new features
  get unit tests that skip cleanly when optional dependencies are absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context; never quote the paper's numbers
  as ours. `mise run pipeline-audit` (or a successor) verifies referenced
  report paths exist.
- Honest negative results are deliverables.
- Single Apple Silicon laptop; no CUDA assumed.
- Update `docs/paper-mapping.md` for any row a change affects; commit in the
  repo's usual message style.

## The Production Track (`docs/production-track.md`)

### Trunk — shared rungs (T1 unblocks all; T2–T4 parallelizable after T1)

**T1 — Real-scene ingestion at scale.**
Finish the vectorized (drjit wavefront) Mitsuba exporter under whichever JIT
variant is available (`llvm_ad_rgb` or `metal_ad_rgb`), scalar path kept as
fallback, CLI unchanged. Export at least one real academic scene
(kitchen/bedroom-class from the official Mitsuba scene repository; checked into
`examples/scenes/` only as a download script — never vendor assets) at ≥512²
and ≥64 spp, and train the torch proxy on it end to end.
*Verify:* existing exporter tests pass against the vectorized path; fixed-seed
scalar-vs-vectorized equivalence on the 8×8 cornell box (mean GATHERLIGHT
radiance within 2%); end-to-end training report for the new scene.
*Measure:* exporter throughput (segments/s) scalar vs vectorized at 48×48 and
128×128, target ≥20×; training time and held-out PSNR/SSIM/FLIP on the new
scene. All in `docs/performance.md`.

**T2 — Streaming and memory discipline at real-scene scale.**
Re-prove E5's sharded-cache / streamed-target-table / streamed-TorchNRP
machinery on the T1 scene, with the fp16 + shared-exponent packed cache format.
*Verify:* streamed-vs-monolithic training parity within 0.1 dB held-out PSNR on
the T1 scene; cache-size-vs-quality curve committed.
*Measure:* peak RSS during export, training, and inference, against an explicit
budget (≤8 GB); shard-streaming throughput. All in `docs/performance.md`.

**T3 — Perceptual quality gates.**
Promote SSIM/FLIP from ablation tooling to pass/fail gates: a reusable
`nrp.quality.gate` CLI generalizing the E9 trust-verdict ladder, invokable by
any report script, with named thresholds per tier (preview/draft/final).
*Verify:* unit tests for gate logic; at least two existing reports re-emitted
through the gate; a deliberately degraded render fails the gate.
*Measure:* gate evaluation overhead <5% of render time at 512².

**T4 — Runtime baseline lock.**
Re-run the browser WebGPU bench (`webgpu/bench_browser.mjs`) on the exported
T1-scene proxy and freeze it as a regression baseline checkable by a mise task.
*Verify:* parity vs PyTorch reference within existing tolerance; baseline JSON
committed; a mise task fails when frame time regresses beyond threshold.
*Measure:* frame-time **histogram** (mean and p95, not just mean) at 128/256/512²;
floor: 30 fps at 512² sustained, p95-verified.

### Games branch — Summit: interactive real-scene demo

**G1 — Dynamic geometry, second attempt (partitioned residual retraining).**
Changed hypothesis vs E2: instead of fine-tuning the whole proxy, use E2's
swept-volume invalidation to mark affected cache shards, keep the base proxy
frozen, and train a *residual* proxy over only the invalidated region,
composited at inference. Evaluate against E2's exact failed recovery target so
the comparison is apples-to-apples. Honest failure acceptable — but it must be
a different failure mode than E2's, and the report must say which.
*Verify:* recovery-target comparison table (E2 fine-tune vs G1 residual) from
one script; unit tests for shard invalidation + residual compositing.
*Measure:* invalidate-and-recover wall-clock vs full retrain; quality at
matched budget. Toy scale acceptable; T1 scene if feasible.

**G2 — Summit demo: live relight + controls in the browser.**
T1 scene in the WebGPU viewer with animated lights (E1 harness) plus ≥2 E8
production controls, sustained 30 fps at 512². If G1 succeeded, include one
moving object.
*Verify:* T3 gate at preview tier per frame; scripted interaction trace
committed alongside a screen recording artifact.
*Measure:* frame-time histogram under interaction (T4 methodology), p95 ≤33 ms.

### Film branch — Summit: shot-relight demo

**F1 — Shot harness with temporal stability.**
≥120-frame keyframed-light shot on the T1 scene through the E9 quality-tier
ladder with a per-frame trust verdict. Add a temporal metric: frame-to-frame
FLIP delta (flicker is the failure mode PSNR can't see).
*Verify:* per-frame verdict JSON; a deliberately flickering baseline (e.g.
per-frame independent noise) fails the temporal check while the proxy sequence
passes or the report explains why not.
*Measure:* per-frame render time at each tier; temporal FLIP-delta
distribution; all in `docs/performance.md`.

**F2 — Summit demo: final-tier shot.**
The F1 shot rendered at final tier with residual-identity frames, encoded as a
committed MP4 plus per-frame report.
*Verify:* every frame passes the T3 final-tier gate or is flagged with cause;
residual identity holds per frame.
*Measure:* end-to-end shot wall-clock (cache reuse amortization vs re-render);
storage: proxy+residuals vs raw frames.

### VFX branch — Summit: rig-relight demo

**V1 — Production light rig.**
≥8 lights including textured/area emitters on the T1 scene; per-light layered
proxies (scaling E1/roadmap compositing), with solo/mute per light and
verified additivity.
*Verify:* sum-of-layers vs full-rig render within tolerance (report the
tolerance); unit tests for rig serialization and per-light compositing.
*Measure:* per-light proxy size vs monolithic; compositing overhead per added
light at 512².

**V2 — Summit demo: art-direction loop.**
E7's image-target optimization driving the full V1 rig — inverse-recover
per-light intensities/colors toward a hand-authored target, then interactive
per-light grading in the viewer or headless slider loop.
*Verify:* optimization converges to the target within a stated image metric;
the recovered rig is exported and re-loadable.
*Measure:* optimizer iterations and wall-clock to convergence; interactive
grading latency per adjustment.

### Status table (lives at the top of `docs/production-track.md`)

| rung | status | evidence |
|---|---|---|
| T1–T4, G1–G2, F1–F2, V1–V2 | not started | — |

## Docs Reorganization

Principle: a reader landing in `docs/` sees a milestone progression, not a pile
of parallel documents. **No file moves or renames** (committed reports and
history cross-reference existing paths); `docs/performance.md` stays the
separate evidence ledger.

1. **New `docs/tracks.md`** (~40 lines): the spine. Four phases with one-line
   status each and links: (1) Replication — `roadmap.md`, complete 10/10;
   (2) Extensions — `extensions.md`, complete E1–E9 measured; (3) Verdict —
   `pipeline-feasibility.md` (E10); (4) Production track —
   `production-track.md`, in progress.
2. **`docs/README.md`** — reordered around the spine; `tracks.md` first;
   reference material (`quickstart`, `architecture`, `paper-mapping`,
   `performance`) grouped under a separate "Reference" heading.
3. **`roadmap.md` / `extensions.md`** — one-line banner at top (phase number,
   completion status, pointer to `tracks.md`). No content rewrites; they are
   historical records cited by reports.
4. **`pipeline-feasibility.md`** — new short closing section "What happens
   next" linking each named blocker to the rung that attacks it: frozen
   transport → G1; cornell-box-only scale → T1; production light rigs → V1;
   shot complexity → F1/F2. This explicitly closes E10's loop.
5. **`status/` convention** — future status reports include a "ladder
   position" line (e.g. "trunk: T1 in progress; branches: not started").

## Social Drafts (`docs/social/`, drafts only — user publishes manually)

All numbers quoted must exist in `docs/performance.md` or a committed report —
same evidence discipline as the docs. Source material: E1–E9 results, the E10
verdict, and the production track as the forward hook.

1. **`2026-07-09-linkedin.md`** — voice-matched to the 2026-07-02 post
   (measured, numbers-forward, negative-results-as-a-feature). Arc: closed a
   9-extension program answering "is this decoupling a building block for
   real-time neural rendering?" Highlights: real WebGPU in real Chrome at
   30/60 fps; 512² streamed-training proof; the E2 fine-tune failure stated
   plainly; the three-verdict table; production-track announcement.
2. **`2026-07-09-reddit.md`** — r/GraphicsProgramming-targeted long-form:
   "reimplemented Sancho et al. EGSR 2026, then stress-tested whether it
   generalizes." Leads with methodology and the negative results (E2 failure,
   the WebGPU crash bisection to a third-party binding defect resolved by
   running the identical shader in real Chrome). Includes repo layout and the
   claim→JSON-report traceability story.
3. **`2026-07-09-x.md`** — 5–7 post thread skeleton: hook (north-star
   question); one result + one number per post; the failure post in the
   middle; summit-demos teaser last. Each ≤280 chars, written so images/clips
   attach later.

## Implementation Order

1. `docs/production-track.md` (the ladder above, full goal-prompt form).
2. Docs reorg (tracks.md, README reorder, banners, E10 closing section).
3. Social drafts.
4. Single commit series on a branch, repo's usual message style; no rung
   implementation work in this pass — this is a docs/planning deliverable.

## Explicitly Out of Scope

- Implementing any rung (T1 etc.) — the track is the deliverable here.
- Publishing any social post.
- Moving/renaming existing docs; merging `performance.md` into narrative docs.
- Merging or touching other branches/worktrees.

## Risks

- **G1 fails like E2 did.** Contained by design: the branch structure keeps
  film/VFX summits independent, and the rung's own success criterion admits a
  documented negative result.
- **T1 scene too heavy for the laptop.** Mitigation is built into T2 (memory
  budget rung); if even streamed export exceeds budget, drop spp/resolution
  and record the ceiling honestly — that itself is a track finding.
- **Numbers drift between social drafts and reports.** Constraint stated in
  each draft's header: quote only committed numbers; verify against
  `docs/performance.md` before publishing.
