# Extensions — Is NRP a Building Block for a Real-Time Neural Rendering Pipeline?

> **Phase 2 of 4 — complete (E1–E9 measured; E10 verdict in
> [pipeline-feasibility.md](pipeline-feasibility.md)).** This is a historical
> record cited by committed reports; see [tracks.md](tracks.md) for the full
> progression.

This is a research roadmap written as goal prompts, in the same format as
`docs/roadmap.md` (each item is self-contained, designed for `/goal <prompt>` or as a
session brief, with baked-in **verification** and **measurement**). Where the original
roadmap replicated the paper, these items go *beyond* it, following the directions the
authors themselves flag in §7 (Limitations and Future Work) and §6.2/6.3 (art-directed
and generative edits), plus the obvious production questions the paper stops short of:
animation, engine integration, and final-frame quality.

**The north-star question every item feeds:** can the SAMPLEPATHS/GATHERLIGHT/proxy
decoupling serve as the core of a future real-time neural rendering pipeline that
delivers high-quality imagery at interactive speeds — for games, animated film, and
feature VFX — rather than remaining a relighting tool for static frames?

Shared conventions (same as `docs/roadmap.md`):

- All new/changed code passes `mise run test` and `mise run lint`; new features get
  unit tests that skip cleanly when optional dependencies are absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context; never quote the paper's numbers as ours.
- Honest negative results are deliverables. If an extension shows the decoupling
  *breaks down* (e.g. temporal caching costs more than it saves), the report says so
  explicitly — the point is to determine viability, not to confirm it.
- Update `docs/paper-mapping.md` for any row a change affects; commit in the repo's
  usual message style.

---

## E1. Animated lights and camera: temporal NRPs

> The paper's cache is fixed-camera, static-scene. Take the first step toward
> animation along the axis the decoupling already supports: *animated lights over a
> static scene* are free (one cache, evaluate the proxy per frame), so build the
> harness that proves it — a `nrp.torch_backend.animate` CLI that takes a keyframed
> light path (JSON: times + light params, linear/spline interpolation) and renders an
> image sequence from one resident proxy with zero cache access. Then attack the
> harder axis: *animated camera* via a time-conditioned cache — trace K camera
> keyframes (reuse the multi-view exporter machinery from roadmap item 7), train a
> single proxy conditioned on a normalized time/camera input alongside the hashgrid,
> and interpolate to unseen intermediate frames.
> **Verify:** unit tests for keyframe interpolation; per-frame animated-light output
> matches an independently evaluated single-frame relight bitwise (same code path);
> the time-conditioned proxy's PSNR on *held-out intermediate* camera frames is
> within 3 dB of its PSNR on training keyframes (otherwise report the gap as the
> finding); frame sequences written by a committed command.
> **Measure:** per-frame proxy inference latency vs frame count (should be flat —
> the interactive-speed claim); temporal stability via mean per-pixel frame-to-frame
> delta under a smoothly moving light, proxy vs GATHERLIGHT reference (flicker is
> the classic neural-rendering failure mode; quantify it, don't eyeball it);
> time-conditioned cache size vs K.

## E2. Dynamic geometry: cache invalidation and incremental retracing

> The paper assumes light transport between objects is frozen after SAMPLEPATHS
> (§7). Test how far that assumption bends before it breaks: animate the toy scene's
> sphere along a path, and implement *incremental* cache updates — retrace only
> pixels whose primary visibility changed (conservative screen-space bound from the
> G-buffer + the sphere's swept volume), splicing fresh paths into the cached set.
> Compare three regimes per frame: (a) full retrace, (b) incremental retrace +
> proxy fine-tune (a few hundred iterations warm-started from the previous frame's
> weights), (c) stale cache (no update — the honest baseline showing the artifact).
> **Verify:** the invalidation mask provably covers all changed pixels (test: pixels
> outside the mask have identical GATHERLIGHT output before/after the move, within
> MC noise); spliced caches pass `validate()`; regime (b) recovers to within 1 dB of
> regime (a)'s held-out PSNR on a 10-frame sequence.
> **Measure:** per-frame wall-clock for (a)/(b)/(c); fraction of segments retraced
> per frame vs object speed; the PSNR-vs-cost frontier across the three regimes.
> This is the single most important feasibility datapoint for games: write an
> explicit paragraph in `docs/performance.md` answering "what fraction of a frame
> budget does keeping the cache alive cost?"

## E3. Light-aware path sampling (the paper's own top fix)

> §7 identifies the key quality limitation: material-based sampling under-samples
> occluded regions (the lamp-shade failure, Fig. 14), because no NEE runs during
> data generation. Implement a light-aware sampling mode in the toy tracer: accept a
> user-declared *light placement region* (bbox or sphere where lights might later
> go) and blend material sampling with guiding toward that region (e.g. a fraction
> of bounces sampled toward the region, with proper MIS-style throughput weights so
> the cache stays unbiased). Reproduce the failure first: place a light inside a
> small open-topped box in the scene, show the standard cache's proxy fails there,
> then show the guided cache fixes it.
> **Verify:** guided caches pass all existing GATHERLIGHT consistency checks vs the
> emissive-inline reference (`nrp/compare_reference.py`) — guiding must not bias the
> estimator; a committed A/B script trains proxies on standard vs guided caches of
> equal segment budget; the guided proxy's PSNR for lights inside the occluded
> region improves by a reported margin (target ≥ 3 dB) without regressing open-region
> lights by more than 0.5 dB.
> **Measure:** segment-count density inside the declared region, standard vs guided;
> the A/B PSNR table split by light position (open vs occluded); cache size delta.

## E4. Richer light models: textured and environment lighting

> §7 leaves "textured lights or area lights of arbitrary shapes" as future work —
> and production lighting lives on textured area lights, IES profiles, and
> environment maps. Extend the light vocabulary two steps: (1) `TexturedQuadLight` —
> a quad with a low-resolution emission texture (start 8×8 RGB), GATHERLIGHT
> integrating texture lookups at segment-overlap points, the proxy conditioned on a
> compact texture embedding (flattened downsample or small learned encoder);
> (2) `EnvironmentLight` — a low-order spherical-harmonic (degree ≤ 2, 9 coeffs/channel)
> environment term gathered over escaped segments (t_max = ∞ paths). Keep both
> differentiable end to end so §5.3 inverse optimization can recover texture pixels
> and SH coefficients from image targets.
> **Verify:** GATHERLIGHT for a constant-texture quad matches the existing QuadLight
> bitwise-tolerantly (reduction test); SH gather validated against an analytic
> constant-radiance environment on the toy scene; inverse recovery test — optimize an
> 8×8 texture (or SH vector) from a rendered target to < 10% relative error; all
> existing light tests pass unchanged.
> **Measure:** proxy held-out PSNR vs parameter count as texture resolution grows
> (2×2 → 4×4 → 8×8) — this quantifies §7's warning that parameter count hurts
> convergence, which is the scaling law that decides whether production light rigs
> fit in one proxy; training time per configuration.

## E5. Out-of-core training and tiled inference at production resolution

> §7's engineering limitation: all path data is pre-loaded, capping resolution and
> sample count. Break it: (1) a sharded cache format (`PathCache.save_sharded`,
> pixel-tile shards on disk, memory-mapped or lazily loaded) with a training loader
> that streams shards on a schedule (visit every tile per epoch-equivalent, bounded
> resident set); (2) tiled full-frame inference so a 1024² frame renders through the
> proxy without materializing all intermediate activations. Demonstrate an export +
> train + relight at ≥ 512², ≥ 128 spp on a Mitsuba scene — a configuration that
> would not fit the naive loader at a stated memory budget.
> **Verify:** sharded round-trip equals the monolithic cache (identical GATHERLIGHT
> output); a streamed training run at toy scale matches an in-memory run's held-out
> PSNR within 0.3 dB at equal iterations/seed (streaming must not silently change
> the data distribution); tiled inference is bitwise-identical to untiled on
> overlapping test sizes.
> **Measure:** peak resident memory streamed vs in-memory (report the ratio and the
> enforced budget); training throughput cost of streaming (iterations/s); the 512²
> end-to-end report (cache GB, train wall-clock, held-out PSNR, full-frame proxy
> inference ms). This is the "does it scale to film frames" datapoint.

## E6. Engine-shaped runtime: exported proxy + real-time inference loop

> If NRPs are ever a pipeline core, the proxy must run inside an engine frame
> budget, decoupled from PyTorch. Build the smallest honest version: export a
> trained `TorchNRP` (hashgrid tables + MLP weights) to a self-contained artifact
> (`torch.export` / ONNX, or a documented raw-tensor format), and write a
> minimal interactive viewer — a Python loop is acceptable, but inference must go
> through the exported artifact, not the training module — with live light-parameter
> sliders (drag the light, watch the frame update). Target Metal on this machine
> (MPS via the exported graph); structure the exporter so a WebGPU/engine port is a
> backend, not a rewrite. Document the integration contract in
> `docs/engine-integration.md`: exactly what an engine must supply (G-buffer aux,
> pixel coords, light params) and what it gets back, i.e. the NRP-as-render-feature
> API.
> **Verify:** exported-artifact inference matches the PyTorch module to rtol 1e-4
> over a grid of lights (committed parity test); the viewer runs from one command
> (`mise run viewer`) and a session recording or frame dump is committed as
> evidence; export covers both sphere and quad models.
> **Measure:** full-frame exported-inference latency at 128², 256², 512² on CPU and
> MPS (extend `bench.py`), reported as fps; end-to-end slider-to-photon latency in
> the viewer; artifact size in MB (the paper's compactness argument, now as a
> shippable asset). State explicitly in `docs/performance.md` whether 30/60 fps at
> each resolution is met on this hardware and what the gap is.

## E7. The generative loop: image-space direction to physical lights, interactively

> §6.2/6.3 are the paper's real pipeline pitch: scribbles and generative-model
> outputs become *physically plausible, parameterized* lighting via inverse
> optimization through the NRP. Build the full loop as a product-shaped demo:
> (1) a scribble workflow — `optimize_lights` already has objective/protect masks,
> so add a CLI mode taking a sparse painted-target image + mask (committed fixtures;
> creating them in any paint tool is fine) and recovering lights that realize the
> scribble; (2) a generative-target workflow — take an edited/stylized target image
> of the toy scene (a committed fixture; optionally produced by any image editor or
> generative model, provenance documented), optimize N lights against it, and
> re-render via GATHERLIGHT to show the *physically consistent* version of the
> generative suggestion side by side. Report proxy-space vs re-rendered error
> separately, as `optimize_lights` already mandates.
> **Verify:** scribble recovery on a synthesized fixture (target painted from a
> known light config) recovers that config's re-rendered image to > 25 dB PSNR in
> masked regions with protected regions unchanged (< 0.5 dB drift); the generative-
> target run converges from ≥ 3 restarts to a stated objective value; both demos run
> end to end from committed commands with outputs under `out/generative/`.
> **Measure:** wall-clock per optimization at pixel fractions {1.0, 0.25, 0.05} —
> the interactivity claim for live art-direction sessions is a latency claim;
> objective-vs-iteration curves; the gap between the raw generative target and its
> physically-realized re-render (this gap *is* the finding: it quantifies how much
> physical grounding costs relative to unconstrained generation).

## E8. Post-hoc production light controls (breaking the frozen-transport limit)

> §7: non-physical controls lighters rely on — per-object light exclusivity
> (linking), custom attenuation — "cannot be changed after the NRP was trained."
> Test the obvious workaround at cache level: the cache knows each segment's
> geometry, so implement *gather-time* controls — per-layer light linking (reuse the
> layer-ownership machinery from roadmap item 8: exclude a light's contribution to
> segments whose first hit is on an excluded layer) and custom attenuation curves
> (replace inverse-square with an artist curve at gather time). Then answer the real
> question: can a proxy be *conditioned* on these controls (e.g. a per-layer light
> mask as extra input) so they stay live at proxy speed, or do they force gather-time
> fallback? Train a conditioned toy proxy and find out.
> **Verify:** gather-time linking matches the sum-of-layers algebra exactly (light
> excluded from layer A ⇒ full gather equals layer-B gather for that light; unit
> test); attenuation-curve gather has a closed-form fixture test; the conditioned
> proxy's PSNR with linking active vs inactive both within 1.5 dB of their
> respective GATHERLIGHT references — or report that conditioning fails and
> gather-time is the only route.
> **Measure:** conditioned-proxy quality vs the two-proxy (per-layer) alternative at
> equal total parameter budget; edit latency for a linking toggle: proxy-conditioned
> vs gather-time vs retrain. The write-up should state which production controls
> survive the decoupling and which genuinely require the cache.

## E9. Final-frame quality: the proxy as a preview tier with a converged escape hatch

> Film/VFX needs a path from interactive preview to final frame with *no visual
> lie*: what you approved interactively must be what the farm renders. Formalize
> NRP's place in that ladder and measure the gaps between tiers: (1) proxy
> prediction (interactive), (2) GATHERLIGHT from the cache at export spp (seconds),
> (3) GATHERLIGHT from a fresh high-spp cache (the converged reference — since
> gather is unbiased, spp is the only knob), (4) proxy + cached residual: precompute
> `GATHERLIGHT − proxy` for the *approved* light config so the displayed approval
> frame is exact, and measure how the residual's validity decays as the light moves
> away from the approval point (the "how far can you drag before re-baking"
> radius). Add a `--quality preview|draft|final` flag to the relight CLIs that
> selects the tier and always annotates output metadata with the tier used.
> **Verify:** tier-4 output at the approval config equals tier-2 bitwise (it's an
> identity by construction — test it); metadata annotation tested; a committed
> script produces the full tier-comparison report on toy + Mitsuba scenes.
> **Measure:** PSNR/SSIM/FLIP of tiers 1, 2, 4 against tier 3, and wall-clock per
> tier — the quality-vs-latency ladder in one table; residual-validity decay curve
> (PSNR vs light-parameter distance from the approval point). The deliverable is a
> defensible answer to "can a lighting supervisor trust the interactive image?" —
> stated with numbers in `docs/performance.md`.

## E10. Verdict: the pipeline feasibility report

> Synthesize E1–E9 (and the original roadmap's measurements) into a single decision
> document, `docs/pipeline-feasibility.md`, answering the north-star question for
> three concrete targets: **games** (16 ms budget, dynamic everything), **animated
> film** (interactive preview + final-frame guarantee, mostly static sets with
> animated characters/lights), **feature VFX** (per-shot bakes acceptable, layered
> compositing workflows, art-direction loops). For each target: which pipeline
> stages NRP replaces (cite measured latency/quality from `out/` reports — no
> hand-waving, and clearly label any number extrapolated from toy scale as such),
> which limitations are engineering (fused kernels, out-of-core — cite E5/E6 data)
> vs structural (frozen transport, per-light-type proxies — cite E2/E8 data), and a
> costed gap list of what a production team would still have to build. End with an
> explicit per-target verdict: *viable core / viable component / not the right
> primitive*, each justified by specific measurements.
> **Verify:** every quantitative claim links to a JSON report under `out/` or a
> `docs/performance.md` row; a claims-audit script (committed) checks that every
> `out/` path referenced in the document exists; the document distinguishes
> measured (this hardware, toy/Mitsuba scale) from extrapolated (paper's numbers,
> production scale) in every section.
> **Measure:** this item produces no new benchmarks; its rigor *is* the deliverable.
> Include one summary table: target × (interactivity met?, quality tier reached,
> hardest structural blocker, estimated engineering distance).

---

Suggested order: **E3 → E6 → E7** first — they need nothing new from each other and
directly test the three pillars (quality where it matters, speed in engine-shaped
form, the generative workflow that makes the pipeline interesting). **E1 → E2** next
(animation, easy axis then hard axis; E2 is the highest-risk/highest-information
item — schedule it early enough that a negative result can shape the rest). **E5**
whenever scale becomes the bottleneck; **E4, E8** expand expressiveness and lean on
roadmap item 8's layer machinery; **E9** once E6 exists (the ladder needs the fast
tier). **E10 is last by definition** — it has no content until the others have
numbers.
