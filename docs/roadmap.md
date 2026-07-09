# Roadmap — Ten Improvements, Written as Goal Prompts

> **Phase 1 of 4 — complete (10/10).** This is a historical record cited by
> committed reports; see [tracks.md](tracks.md) for the full progression.

Each item below is a self-contained goal prompt (designed for `/goal <prompt>` or as a
session brief). Every prompt bakes in **verification** (tests + evidence that must
exist before the goal counts as met) and **performance testing** (numbers that must be
measured and recorded). Conventions all prompts share:

- All new/changed code passes `mise run test` and `mise run lint`; new features get
  unit tests that skip cleanly when optional dependencies are absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context; never quote the paper's numbers as ours.
- Update the paper-coverage table (README + `docs/paper-mapping.md`) for any row a
  change affects, and commit with the repo's usual message style.

---

## 1. Vectorized Mitsuba exporter + a real academic scene

> Replace the scalar Python loop in `nrp/mitsuba_exporter.py` with a wavefront
> (drjit-vectorized) tracing loop that runs under whichever JIT variant is available
> (`llvm_ad_rgb` or `metal_ad_rgb`), keeping the scalar path as fallback and the CLI
> unchanged. Then export at least one real academic scene (e.g. a Mitsuba
> kitchen/bedroom scene from the official scene repository, checked into
> `examples/scenes/` only as a download script — never vendor assets) at ≥ 128×128 and
> ≥ 64 spp, and train the torch proxy on it.
> **Verify:** identical-schema caches (existing exporter tests pass against the
> vectorized path); a fixed-seed equivalence test showing scalar and vectorized
> exports of the 8×8 cornell box produce statistically compatible GATHERLIGHT images
> (mean radiance within 2%); end-to-end training report exists for the new scene.
> **Measure:** exporter throughput (segments/s) scalar vs vectorized at 48×48 and
> 128×128 (target: ≥ 20× vectorized speedup); training time and held-out PSNR/SMAPE
> on the new scene; record all in `docs/performance.md`.

## 2. Volumetric path export and volume GATHERLIGHT

> Extend the path-cache schema and GATHERLIGHT to participating media (§3.1
> "Volume rendering"): record scattering vertices from free-flight sampling in a
> homogeneous medium (add a homogeneous-volume option to the toy tracer, or export a
> Mitsuba scene with a `homogeneous` medium), so virtual lights inside the medium
> illuminate it. Keep schema backward compatibility (surface-only caches load
> unchanged; add a schema version field).
> **Verify:** unit tests for free-flight segment recording (segment lengths
> distributed per transmittance for a known σ_t, chi-square or KS sanity check);
> GATHERLIGHT with a light inside the medium produces the analytically expected
> single-scatter falloff on a slab fixture within 5%; all existing tests pass
> (schema compatibility).
> **Measure:** cache growth (segments and MB) vs surface-only for the same scene/spp;
> gather time vs surface-only; train a proxy on the volumetric scene and report
> held-out PSNR/SMAPE vs the surface-only baseline.

## 3. Fused/GPU GATHERLIGHT and GPU training

> Make reconstruction and training fast: port `gather_throughput[_quad]` to a batched
> torch implementation that runs on MPS/CUDA (all segments tested in one kernel-sized
> op, mirroring the paper's fused Triton gather), use it for pool builds, and make
> `torch_backend/train.py` fully device-resident (model, pool, batches) with
> `device: mps` working end to end. Add a `--gather-backend numpy|torch` flag to keep
> the numpy reference authoritative.
> **Verify:** torch gather matches numpy gather bitwise-tolerantly (allclose,
> rtol 1e-5) on the toy and Mitsuba caches for 50 random sphere and quad lights;
> training on MPS reaches within 0.5 dB of the CPU run's held-out PSNR at equal
> iterations and seed.
> **Measure:** gather ms/image numpy-CPU vs torch-CPU vs torch-MPS at 48², 128², 256²
> (analogue of paper Table 1's reconstruction row); total train wall-clock CPU vs MPS
> for the toy and Mitsuba configs; extend `bench.py` output and
> `docs/performance.md` accordingly.

## 4. Quad-light and multi-light inverse optimization (Table 3 replication)

> Extend `torch_backend/optimize_lights.py` to quad-light models (reparameterize
> center via logit bounds, width/height via logit of size bounds, normal via an
> unconstrained 3-vector normalized in `quad_params`) and to joint multi-light
> recovery. Then replicate the *structure* of the paper's Table 3: optimize N ∈
> {1, 3, 5} lights at pixel fractions α ∈ {1.0, 0.25, 0.05, 0.01} with ≥ 5 runs per
> cell on the Mitsuba cornell box, reporting re-rendered (GATHERLIGHT) PSNR and
> wall-clock per cell.
> **Verify:** unit tests for the quad reparameterization round-trip and its gradient
> flow (normal normalization included); a 1-light quad recovery test reaching center
> error < 0.05 on a trained toy quad model; the Table-3 grid script is committed
> (`examples/` or a CLI subcommand) and reruns from one command.
> **Measure:** the full grid (PSNR + seconds per cell) into
> `out/inverse-grid/report.json` and a table in `docs/performance.md`; state
> explicitly how the trend compares qualitatively to the paper's Table 3 (small α
> should barely hurt quality while cutting time ~linearly).

## 5. Compressed cache layout (§4.2: fp16 + rgb9e5)

> Implement the paper's packed cache: geometry in fp16, throughput in rgb9e5
> (shared-exponent 32-bit HDR format — implement encode/decode in numpy with tests),
> as an alternative `.npz` layout behind `PathCache.save(path, compressed=True)` /
> auto-detected load. Quantify the quality cost.
> **Verify:** rgb9e5 encode/decode round-trip property test (relative error ≤ 2⁻⁹
> mantissa bound for representable range, correct handling of zeros/denorm-range
> values); full-cache round-trip keeps GATHERLIGHT images within 0.5 dB of the
> float64 cache on toy + Mitsuba scenes; loaders stay backward compatible.
> **Measure:** cache size (MB) float64 vs packed for both scenes (expect ~4–6×);
> gather time delta (decode cost); train a proxy from the packed cache and show
> held-out PSNR within 0.3 dB of the float64-cache run.

## 6. Paper-scale training and the quality ceiling

> Push training toward the paper's configuration and find this implementation's
> quality ceiling: 8×256 network, hashgrid up to `finest_resolution` = image width,
> ≥ 50k iterations, pool ≥ 128, on the Mitsuba cornell box at 128×128 / 64 spp (from
> item 1's exporter, or scalar overnight). Add cosine LR decay and checkpointing (save
> every N iters, resume flag) so long runs are safe to interrupt.
> **Verify:** resume-from-checkpoint reproduces the uninterrupted loss curve within
> noise (test on a 200-iter toy run); the long-run report exists with the full loss
> curve; no regression in the standard toy configs.
> **Measure:** held-out PSNR/SMAPE vs the 3k-iteration baseline (quantify the gain
> from scale alone); wall-clock and iterations/s (CPU and MPS); PSNR-vs-iteration
> curve at checkpoints {3k, 10k, 25k, 50k} recorded in `docs/performance.md` —
> the analogue of the paper's convergence discussion.

## 7. Multi-view NRPs (§6.1)

> Support several cameras of one scene: export N ≥ 3 views of the Mitsuba cornell box
> (the exporter already takes any scene XML — add a `--sensor-index` or per-view
> camera override), train one proxy per view, and add a
> `nrp.torch_backend.relight_multiview` CLI that applies a single light edit across
> all views simultaneously, writing one image per view.
> **Verify:** per-view training reports all exceed 20 dB held-out PSNR; a
> cross-view consistency test — for one light configuration, each view's proxy
> prediction vs its own GATHERLIGHT reference within the per-view validation range
> (no view catastrophically worse: max spread < 4 dB); CLI smoke test in the suite.
> **Measure:** total multi-view edit latency (all N views, one light change) on CPU
> and MPS vs N — should be N× single-view inference with no cache access; memory of N
> resident proxies (the paper's compactness argument, stated in MB).

## 8. Per-layer compositing NRPs (§6.1, Fig. 11)

> Implement layer-separated relighting: teach the toy tracer (or the Mitsuba exporter
> via shape groups) to export *two* caches for the same camera — e.g. foreground
> sphere only vs background box only, where each layer's paths still intersect the
> full scene geometry but only the layer's first-hit pixels are owned. Train a proxy
> per layer and add a compositing CLI that relights one layer while holding the other
> layer's image fixed.
> **Verify:** the two layers' GATHERLIGHT images sum to the full-scene GATHERLIGHT
> image within MC noise (allclose on means, PSNR > 30 dB between sum and full render)
> — this is the linearity property compositing relies on; unit tests for layer
> ownership masks; a composited edit demo image is produced by a committed command.
> **Measure:** per-layer training time and held-out PSNR; composite edit latency vs
> full-scene relight; record whether layer proxies match the full-scene proxy's
> quality on their own pixels (within 1 dB).

## 9. Image-based baseline (Fig. 6 replication)

> Build the paper's key comparison: train the same torch architecture on a dataset of
> R pre-rendered images (GATHERLIGHT under R random lights, denoised — the
> "image-based" regime, no on-the-fly light resampling) for R ∈ {64, 256, 1024},
> against the path-based pool training at matched *total sample budget*, on the
> Mitsuba cornell box. This validates (or falsifies, honestly) the paper's central
> data-efficiency claim at toy scale.
> **Verify:** both regimes share the identical model/optimizer/seed and a common
> held-out validation set of ≥ 24 fresh lights (assert the validation lights are
> disjoint from every training configuration); the experiment script is committed and
> reruns end to end with one command; results JSON includes per-light PSNR
> distributions, not just means.
> **Measure:** held-out tonemapped PSNR per regime (the paper's Fig. 6 metric), gap
> in dB between path-based and the best image-based configuration, and total
> supervision-generation time per regime; write the comparison (including whether the
> paper's ≥ 2.8 dB advantage reproduces qualitatively) into `docs/performance.md`.

## 10. Ablation suite + SSIM/FLIP metrics (Table 2 / Fig. 7 replication)

> Add SSIM and FLIP to `nrp/metrics.py` (numpy implementations with cited formulas;
> unit-test SSIM = 1 on identical images, monotone degradation under noise, and FLIP
> against a couple of hand-checked fixtures). Then run the paper's component ablation
> on the Mitsuba cornell box: {None, Aux, Aux+Den, Aux+Enc, Aux+Enc+Den} × one spp
> sweep {8, 16, 32} — each cell trained with identical budget and seeds, reported on
> the common held-out light set with all four metrics (SMAPE, PSNR, SSIM, FLIP).
> **Verify:** metric unit tests as above; an ablation runner script committed under
> `examples/` that produces `out/ablation/report.json` deterministically from one
> command; every cell's config is embedded in the report for reproducibility.
> **Measure:** the full ablation table in `docs/performance.md` with a paragraph
> comparing *directions* against the paper's Table 2 (aux features should dominate;
> encoding alone may hurt on noisy targets; encoding + denoising should win) and the
> spp trend against Fig. 7 — flag any divergence explicitly rather than smoothing
> over it.

---

Suggested order: 1 → 3 → 6 first (they compound: better data, faster iteration, then
scale), 4/5 independent, 9/10 once 1+6 exist (they are science on top of the
machinery), 2/7/8 expand scope.
