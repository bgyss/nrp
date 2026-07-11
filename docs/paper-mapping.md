# Paper Mapping — Reimplementation Notes

Reference: Sergio Sancho, Alexander Rath, Marco Manzi, Pascal Chang, Amit H. Bermano,
Derek Nowrouzezahrai, Markus Gross, Marios Papas. *Neural Render Proxies for
Interactive and Differentiable Lighting.* Computer Graphics Forum 45(4), EGSR 2026.

This document walks the paper section by section and states, for each mechanism: where
it lives in this repo, whether it is faithful, substituted, or absent, and why. The
one-table summary lives in the top-level README; this is the detailed version.

## §3.1 Decoupled rendering

**SAMPLEPATHS / GATHERLIGHT split — faithful.** Virtual lights are pure emitters that
never alter path construction, so tracing (scene-dependent) and emission
(light-dependent) commute. `PathCache` stores exactly what the paper's Figure 3b
pseudocode accumulates: per-segment `(T_j, x_j → x_{j+1})` with the throughput *before*
the segment, plus escape directions (`t_max = inf`). `gather_light` implements
GATHERLIGHT verbatim: every segment crossing the light accumulates
`throughput · L_e`, normalized by the pixel's path count.

**Importance sampling constraints — faithful.** Both producers use BSDF sampling with
no next-event estimation (NEE would bake light knowledge into the paths). The Mitsuba
exporter applies throughput-based Russian roulette as the paper describes; the toy
tracer uses a fixed bounce count. Extension E3 adds an optional toy-tracer sampler
that mixes cosine BSDF sampling with a cone toward a declared spherical light-placement
region, using mixture-pdf throughput weights. This is an experiment around the
paper's undersampled-region limitation, not a replacement for the paper-faithful
default path-data pass.

**Consistency validation (ours, not the paper's):** `nrp/compare_reference.py`
re-traces the toy scene with an independent seed and evaluates the emissive light
inline — the interleaved Figure 3a algorithm — and compares against GATHERLIGHT over
the cache. Both are Monte Carlo estimates of the same integral over independent path
sets: agreement (28.97 dB, 0.03% mean radiance at 24 vs 64 spp) validates the
decoupling itself.

**Volumes — implemented (toy scale).** The paper notes GATHERLIGHT extends unchanged
to free-flight-sampled media, and this repo now demonstrates exactly that: the toy
tracer optionally fills the box with a homogeneous medium (`--medium-sigma-t`,
isotropic phase, single-scattering-albedo throughput factor at each scatter vertex),
recording segments that end at sampled scatter distances. GATHERLIGHT gained *no*
volume code — P(segment reaches d) = exp(−σ_t·d) makes transmittance implicit — and a
slab-fixture test confirms the gathered falloff of a light inside the medium matches
analytic transmittance within 5% (`tests/test_volume.py`). Cache schema v2 carries
optional medium metadata; v1 surface caches load unchanged. The Mitsuba exporter
remains surface-only.

**Known shared limitation (paper §7, ours too):** occlusion is whatever the cached
segments encode — a light radius grown past a blocker is not re-checked; undersampled
regions (e.g. inside a lamp shade, paper Fig. 14) predict poorly.

## §3.2 Neural render proxy

**Per-light-type decomposition (Eq. 1–3) — faithful.** One network per light type
(`light_type` = `"sphere"` | `"quad"`), emission E(v) factored out (networks learn the
pre-emission contribution; unit-emission lights during training), final image =
emission-weighted sum over lights (`gather_lights`, `torch_backend/relight.py`).

**Sphere lights (4 params) and quad lights (8 params) — faithful.** Quad tangent
frame is derived deterministically from the normal so (center, normal, w, h) fully
determine the light. Extension work now adds reference-GATHERLIGHT support for
`TexturedQuadLight` and degree-2 `EnvironmentLight` (`nrp/lights.py`,
`nrp/gather_light.py`), including constant-texture and constant-environment reduction
tests plus closed-form/reference inverse recovery paths (`nrp/environment_fit.py`,
`nrp/texture_fit.py`). `out/textured-quad-fit/report.json` now includes both a linear
held-out texture-proxy scaling baseline and a compact learned texture-embedding torch
proxy baseline; the main `TorchNRP` train/relight path also accepts a fixed-size
`textured_quad` parameter vector for a small 2×2 texture smoke. The paper-faithful
production-trained light types remain sphere and quad.

## §4 Implementation

**PyTorch — faithful; tiny-cuda-nn and Triton — substituted.** The hash encoding is
plain PyTorch (`torch_backend/encoding.py`). GATHERLIGHT has two implementations:
the authoritative numpy reference and a batched torch mirror
(`torch_backend/gather.py`) that tests all segments against a light in one
weight-and-scatter op on CPU/MPS/CUDA — the paper's fused Triton gather at torch-op
granularity, parity-tested against numpy (rtol 1e-5, 50 sphere + 50 quad lights on
toy and Mitsuba caches). Training is device-resident end to end (`device: mps`,
`gather_backend: torch`). Architecture matches; absolute throughput does not
(documented everywhere numbers appear).

**§4.1 path-data pass over academic scenes — implemented.** The paper records paths
inside the renderer; here `nrp/mitsuba_exporter.py` drives Mitsuba 3 from Python. The
default drjit wavefront loop (`llvm_ad_rgb`/`metal_ad_rgb`) is 39–59× the scalar
fallback's throughput (`docs/performance.md`), making real gallery scenes practical:
the Country Kitchen exports at 128×128 / 64 spp in ~4 s and trains to 25.2 dB held-out
PSNR (`examples/kitchen_torch.json`). Scene assets are downloaded on demand
(`examples/scenes/download_scene.py`), never vendored.

**§4.2 memory layout (fp16 geometry, rgb9e5 throughput) — implemented (opt-in).**
`PathCache.save(path, compressed=True)` and
`PathCache.save_sharded(directory, packed=True)` write the paper's packed layout: segment
geometry and G-buffer aux in fp16, per-segment throughput as shared-exponent rgb9e5
words (`nrp/rgb9e5.py`, the `EXT_texture_shared_exponent` conversion rules,
round-trip property-tested to the 2⁻⁹ mantissa bound). `load` auto-detects the
layout and hands back float64 arrays, so gather/training are layout-agnostic;
escape segments survive because fp16 represents inf exactly. Sizes, decode cost,
and the (negligible) quality delta are measured in `docs/performance.md`; float64
stays the monolithic default because toy caches are megabytes, not the paper's
gigabytes. T2 uses packed shards on the 52.3M-segment Country Kitchen cache and
measures 3.32× smaller storage with ≥44.90 dB GATHERLIGHT fidelity.

**§4.3 network inputs — faithful.** Beyond light parameters, exactly the paper's nine
extra inputs: pixel coordinates px (2D, encoded) and F_px = albedo (3) + depth (1) +
normal (3). Pixel coordinates go through a 2D multiresolution hashgrid [MESK22]
(dense-when-fits, hashed otherwise, bilinear interpolation, geometric level growth);
other inputs are raw, as the paper found encodings for them unhelpful.

*Deviation:* softplus output head (the paper does not specify a head; softplus keeps
contributions positive and smooth).

*numpy-backend deviation (by design):* sinusoidal encoding instead of a hashgrid, plus
derived geometric features (center − position, distance) that a compact MLP needs
without a spatial encoding — kept because that backend optimizes for readability and
finite-difference-checkable autodiff, not paper fidelity.

## §4.4 Training

**Light sampling — faithful.** Per-configuration light positions are sampled uniformly
on recorded path segments (implicit importance sampling toward contributing regions);
the visible-bbox fallback for gigantic scenes is `"sampling": "bbox"`. Sphere radius /
quad extent uniform within configured bounds; quad normals uniform on the sphere.

**Denoised target pool — faithful mechanism, scaled-down defaults.** The denoiser
forces one light per full image, so diversity comes from a pool of denoised
GATHERLIGHT images with periodic replacement. Paper: pool 300, replace 2 every 5
iterations; toy configs default to pool 64 (configurable — at 48×48 a 300-image pool
would exceed the number of meaningfully distinct configurations).

**Denoiser — faithful when the `oidn` extra is installed** (RT filter, HDR, albedo +
normal guides — OIDN's own auxiliary interface; the paper's [Áfr26]). Fallback: an
aux-guided joint bilateral filter (same guidance signal, weaker prior). Configured per
run: `"denoise": {"method": "oidn" | "bilateral"}`.

**Loss (Eq. 4) — exact.** Relative MSE with stop-gradient of the *prediction* in the
denominator and ε = 0.01: `((pred − target)² / (pred.detach()² + 0.01)).mean()`. The
stop-gradient property is unit-tested: the backward pass matches
`2(pred − target)/(pred² + ε)` to 6 decimal places.

**Training scale — paper-scale run available.** Paper: 8×256 network, 100k
iterations, ~1 h on an RTX 5090 at 680². The default toy configs stay small (4×128,
3k iterations, ~1 min CPU at 48²), but `examples/mitsuba_cornell_128_torch.json`
runs the paper's architecture — 8×256, hashgrid `finest_resolution` = image width,
pool 128 — for 50k iterations on the Mitsuba cornell box at 128×128 / 64 spp, with
cosine LR decay and full-state checkpointing (`checkpoint: {"every": N}` +
`--resume`; resume is bit-exact on CPU and unit-tested). The measured
PSNR-vs-iteration convergence curve is in `docs/performance.md`.

## §5 Evaluation

**Metrics:** PSNR, SMAPE, SSIM (Wang et al. 2004), and LDR-FLIP (Andersson et al.
2020) are all implemented in numpy (`nrp/metrics.py`). The FLIP implementation is
verified against NVIDIA's official `flip-evaluator` (uniform fixtures agree to
<1e-4; a random noisy pair to 0.07%, the residual being border padding — this
implementation uses edge-replicate padding instead of zero-fill, documented in the
module docstring). SSIM/FLIP are display-referred: HDR radiance goes through
`tonemap_srgb` (Reinhard + sRGB encode) first.

**§5.1 image-based baseline comparison (Fig. 6) — replicated in structure; the
paper's conclusion does not transfer to toy scale.** `examples/image_based_baseline.py`
trains identical model/optimizer/seed under a rolling path-data pool vs fixed sets of
R ∈ {64, 256, 1024} denoised GATHERLIGHT images on the cornell box at 128²/64 spp,
scored by tonemapped PSNR on a common 24-light held-out set asserted disjoint from
every regime's supervision lights. At matched supervision budget the fixed
1024-image set *wins* by 1.8 dB (the paper reports path-based ≥ 2.8 dB ahead).
The divergence is analyzed rather than smoothed over in `docs/performance.md`:
here both regimes share the same cheap cached-path supervision, so the comparison
isolates only the sampling schedule, and the 64-slot rolling pool's light
diversity — not data efficiency — is the binding constraint (pool 256 recovers
+1.0 dB of the gap).

**§5.2 ablations (Table 2, Fig. 7) — replicated in structure at toy scale.**
`examples/ablation.py` (`mise run ablation`) trains the five component sets
{None, Aux, Aux+Den, Aux+Enc, Aux+Enc+Den} × spp {8, 16, 32} on the Mitsuba cornell
box with identical budget and seeds — the two model switches are
`model.use_aux` / `model.use_encoding` (`torch_backend/model.py`), denoising is the
existing pool flag. Every cell is scored on one common held-out light set against a
separate high-spp reference cache, with all four paper metrics. The report
(`out/ablation/report.json`) embeds each cell's full config; direction-by-direction
comparison against the paper's Table 2 and Fig. 7 is in `docs/performance.md`.

## §5.3 Light optimization — faithful

- Eq. 5: predicted image = Σᵢ Eᵢ · N_type(px, shapeᵢ), accumulated sequentially per
  light (`--n-lights`) — sphere and quad models, joint multi-light recovery.
- Eq. 6: MSE on Reinhard-tonemapped values T(I) = I/(1+I).
- Reparameterization: bounded quantities (center; sphere radius; quad width/height)
  through the logit of their bounded domains (recovered by sigmoid — bounds can never
  be violated), color through inverse softplus from ℝ₊, and the quad normal as an
  unconstrained 3-vector normalized inside `quad_params` (gradients flow through the
  normalization and are provably tangential — unit-tested). Adam in unconstrained
  space; paper defaults lr 0.05, 500 iterations.
- Mini-batch SGD (Table 3): `--pixel-fraction α` evaluates a random pixel subset of
  size ⌊α·H·W⌋ per iteration, drawn without replacement.
  `examples/inverse_grid.py` replicates the Table-3 structure (N ∈ {1,3,5} joint
  lights × α ∈ {1.0, 0.25, 0.05, 0.01}, 5 runs/cell) on the Mitsuba cornell box.
- Every result is re-rendered through reference GATHERLIGHT; proxy-space and
  physically-gathered errors are reported separately, and restarts are *ranked* by
  the physical re-render — an addition to the paper, motivated by an observed failure
  mode where proxy loss prefers image matches at parameter-far configurations
  (quad recovery is proxy-fidelity-bound; see docs/performance.md).

*Evidence the machinery matters:* on the identical hidden-light recovery task, the
torch §5.3 implementation reaches center error 0.013 where the numpy backend's naive
clipped-Adam optimizer stalls at 0.44. Quad recovery (`--quad-check`) reaches center
error < 0.05 with a 96-spp/10k-iteration proxy.

*Not replicated:* the Mitsuba-3 and ZeroGrads baseline comparisons (Fig. 9), and the
50-run statistical protocol.

## §6 Applications

- **§6.1 interactive relighting — implemented** (both backends; benchmarked in
  `docs/performance.md`).
- **§6.1 multi-view NRPs — implemented**: the Mitsuba exporter takes
  `--sensor-index` or a per-view camera override (`--cam-origin` / `--cam-target` /
  `--cam-up` / `--fov`); `examples/multiview.py` (`mise run multiview`) exports
  3 cornell-box views, trains one proxy per view, and verifies cross-view
  consistency; `nrp.torch_backend.relight_multiview` applies one light edit across
  all resident view proxies with no path-cache access (latency scales linearly in N;
  numbers in `docs/performance.md`).
- **Extension E1 animated camera — implemented**: the toy tracer accepts a
  camera-origin override; `examples/time_conditioned_camera.py` (`mise run
  time-camera`) traces K camera keyframe caches and evaluates image-space
  interpolation on held-out intermediate cameras (baseline). `examples/
  time_conditioned_proxy.py` (`mise run time-conditioned-proxy`) goes further: a
  single `TorchNRP` conditioned on light params plus a normalized time scalar,
  trained jointly on the K keyframes, reaches held-out intermediate-camera PSNR
  within 2.06 dB of its training-keyframe PSNR (criterion: within 3 dB — met) at
  K=3, a small camera range.
- **§6.1 per-layer compositing NRPs (Fig. 11) — implemented** (toy tracer):
  `nrp.toy_tracer --layer sphere|box` records only paths whose first hit is on the
  layer's geometry (full scene still traced, so the two layer caches partition the
  full cache's segments and their GATHERLIGHT images sum to the full-scene image
  exactly — the linearity property compositing relies on, unit-tested);
  `layer_ownership_mask` gives per-layer pixel ownership; `examples/layers.py`
  (`mise run layers`) trains one proxy per layer plus a full-scene control and
  writes the composited-edit demo; `nrp.torch_backend.composite` is the CLI that
  relights one layer while holding the other layer's image fixed. Numbers in
  `docs/performance.md`.
- **§6.2 art-directed edits — implemented**: `examples/make_art_target.py` builds a
  painted objective + emphasis mask + protected region; `--mask` / `--protect` /
  `--protect-base` / `--protect-lambda` implement weighted objectives and
  keep-this-region constraints.
- **§6.3 generative targets — out of scope** as a dependency (no image model is
  bundled), but any generated image dropped in as `--target file.npy` exercises the
  same path the paper uses.
- **Extension E9 quality tiers — implemented:** `nrp.torch_backend.relight`
  exposes `--quality preview|draft|final`, output metadata sidecars, and cached
  residual correction for approved light configs; `out/quality/report.json` includes
  toy-scale PSNR/SSIM/FLIP tier metrics plus a toy residual-validity trust verdict.
  `examples/quality_tiers_production.py` reproduces this at 512x512 on real Mitsuba
  cornell-box caches (32spp export, 128spp converged reference) with a genuinely
  trained (not intentionally-untrained) streamed proxy, reaching the same
  qualitative trust verdict. This is pipeline plumbing around the paper's
  proxy/GATHERLIGHT split, not a paper mechanism. The F1/F2 production-track
  rungs extend the ladder to shot level on the T1 kitchen:
  `nrp.torch_backend.shot` adds a per-frame T3 trust verdict plus a
  frame-to-frame FLIP temporal-stability check (`out/f1-shot/report.json`),
  and `examples/f2_final_shot.py` renders the shot at final tier with
  fp16-stored residual-identity frames and a committed MP4
  (`out/f2-shot/report.json`).
- **Extension E8 production controls — toy-scale conditioned proxies implemented:**
  `gather_light_controlled` can exclude first-hit-owned pixels for light linking and
  apply a linear-distance artist attenuation curve to sphere-light gathers. This
  demonstrates that some non-physical controls can be evaluated from the cache. A
  tiny binary table proxy keeps one linking toggle live at proxy speed, and a learned
  image proxy keeps fixed-family linear and quadratic attenuation controls live for
  held-out settings. A soft mask-basis proxy also predicts a held-out mask to
  floating-point accuracy. Fully free-form production controls remain limited by the
  chosen control parameterization and training coverage, not by the cache/proxy API.
  Production-track rung G2 additionally keeps two pixel-level controls (layer-mask
  linking, first-hit artist attenuation) live *in the browser* against the real
  T1-scene proxy, gated per frame against identically-controlled GATHERLIGHT.
- **Extension E6 exported runtime — implemented:** `nrp.torch_backend.engine_runtime`
  exports sphere and quad `TorchNRP` models to TorchScript artifacts and runs parity-
  tested inference through `torch.jit.load`; `mise run viewer` writes headless slider
  frame dumps and a CPU/MPS 128/256/512 full-frame inference sweep (real MPS
  timings, not the earlier "unavailable" placeholder). `examples/export_js_viewer.py`
  (`mise run js-viewer`) closes the GUI-slider gap with a self-contained HTML/JS page
  (real interactive light-position sliders, 1e-7 parity vs PyTorch, verified under
  Node). `webgpu/bench_browser.mjs` (`mise run webgpu-bench`) closes the WebGPU
  criterion: a real WGSL compute shader running the actual exported proxy inside
  real Chrome (via Playwright), 2.4e-7 parity vs PyTorch, 30/60 fps cleared at
  128/256/512². An earlier native-binding-only attempt (`webgpu/bench.mjs`, no
  browser) reproducibly crashed on real trained-model weights — bisected to a
  defect in that specific binding (`webgpu/README.md`), resolved by running the
  identical shader in a production WebGPU implementation instead. The remaining
  gap — both browser backends sidestepped the hashgrid encoding via a
  `use_encoding=False` ablation — closed with production-track rung T4:
  `webgpu/bench_t4.mjs` (`mise run t4-bench`) ports `HashEncoding2D` to WGSL and
  runs the actual T1 kitchen proxy (hashgrid + 4×128 MLP) in real Chrome at 1.2e-6
  parity, 30 fps p95-verified at 512², locked as a regression baseline
  (`mise run t4-check`, `out/t4-runtime/baseline.json`).
- **Extension E7 image-space target loop — mostly implemented:** `mise run
  generative-loop` creates a synthesized scribble fixture and a stylized target,
  pretrains the proxy on random lights before inversion (closing the "untrained
  proxy" gap), exercises objective/protect masks through `optimize_lights`, and
  reports proxy-space plus physical GATHERLIGHT errors. `out/generative/provenance.json`
  records the deterministic fixture recipes and SHA-256 hashes for the generated
  toy targets. The toy stylized target is explicitly reported as not exactly
  realizable by one sphere light; a true hand-authored or external generative image
  fixture remains open (requires an external asset this environment cannot produce
  unprompted).

## §7 Limitations — shared

All of the paper's stated limitations apply here too: fixed transport after caching
(no post-hoc attenuation/exclusivity edits), undersampled-region artifacts,
parameter-count-driven difficulty for complex light types, and in-memory path data.
Extension and production-track work chips at these limits with light-aware toy-tracer
sampling for declared placement regions, dynamic-geometry cache splicing
(`nrp.dynamic_geometry`, primary-visibility plus multi-bounce swept-volume
invalidation) with frozen-base shard-partitioned residual retraining
(`nrp.torch_backend.residual_dynamic`, rung G1 — E2's 1 dB recovery target still
unmet, but with out-of-region fidelity structurally preserved), a
packed tile-sharded caches, streamed TorchNRP pool training, and tiled proxy inference
(`PathCache.save_sharded(..., packed=True)`, `nrp.torch_backend.streamed_train`,
`nrp.torch_backend.relight --tile-pixels`). T2 measures this path on the 512² Country
Kitchen scene at 0.80 GiB training peak RSS, versus 8.45 GiB monolithic. The E3
open-top-box occluder remains a toy lampshade-style fixture, and dynamic geometry
remains toy-scale (the Mitsuba exporter has no scene-edit retrace path). This
implementation adds its own: no fused streamed
gather kernel, plus Monte Carlo noise floors at low spp that dominate SMAPE on
near-zero pixels.

## Known deviations summary

Documented substitutions, not silent approximations — each is discussed in context
above; this is the flat list:

- **CPU (or MPS) PyTorch, no tiny-cuda-nn, no Triton kernel** — architecture is
  faithful, absolute performance is not.
- **OIDN is optional** — used when the `oidn` extra is installed
  (`"denoise": {"method": "oidn"}`); the dependency-free default is the aux-guided
  joint bilateral filter (same guidance signal, weaker prior).
- **Softplus output head** — the paper doesn't specify its head; softplus keeps
  contributions positive.
- **The Mitsuba exporter records paths from Python, not inside the renderer** — the
  default drjit wavefront loop (`llvm_ad_rgb`/`metal_ad_rgb`, auto-detected) advances
  all paths one bounce per kernel launch and pulls per-bounce results to numpy; a pure
  scalar loop (`--mode scalar`, no JIT requirements) remains the fallback and
  semantics reference. No volumes are exported (surface interactions only).
- **numpy backend diverges further by design** (sinusoidal encoding,
  target-normalized loss, extra derived geometric inputs) — it is the readable
  finite-difference-checked reference, not the paper replica; the torch backend is
  the paper replica.
- **FLIP uses edge-replicate convolution padding** instead of the reference
  implementation's zero-fill, so constant images are preserved at borders (matters at
  this repo's tiny resolutions); agreement with NVIDIA's `flip-evaluator` is <1e-4 on
  uniform fixtures and ~0.1% on natural pairs.
