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
tracer uses a fixed bounce count.

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
determine the light. Textured / arbitrary-shape area lights: future work in the paper
too.

## §4 Implementation

**PyTorch — faithful; tiny-cuda-nn and Triton — substituted.** The hash encoding is
plain PyTorch (`torch_backend/encoding.py`), GATHERLIGHT is numpy. Architecture
matches; absolute throughput does not (documented everywhere numbers appear).

**§4.1 path-data pass over academic scenes — implemented.** The paper records paths
inside the renderer; here `nrp/mitsuba_exporter.py` drives Mitsuba 3 from Python. The
default drjit wavefront loop (`llvm_ad_rgb`/`metal_ad_rgb`) is 39–59× the scalar
fallback's throughput (`docs/performance.md`), making real gallery scenes practical:
the Country Kitchen exports at 128×128 / 64 spp in ~4 s and trains to 25.2 dB held-out
PSNR (`examples/kitchen_torch.json`). Scene assets are downloaded on demand
(`examples/scenes/download_scene.py`), never vendored.

**§4.2 memory layout (fp16 geometry, rgb9e5 throughput) — not implemented.** Toy
caches are megabytes; the compressed layout matters at the paper's tens-of-gigabytes
scale. Roadmap item.

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

**Training scale — scaled down.** Paper: 8×256 network, 100k iterations, ~1 h on an
RTX 5090 at 680². Here: 4×128 + hashgrid, 3k iterations, ~1 min CPU at 48². The paper's
architecture (8×256) is a config change (`hidden_layers: 8, hidden_width: 256`), not a
code change.

## §5 Evaluation

**Metrics:** PSNR and SMAPE are implemented (`nrp/metrics.py`). SSIM and FLIP (paper
Tables 1–2) are not; adding them is a roadmap item under the ablation suite.

**§5.1 image-based baseline comparison (Fig. 6) and §5.2 ablations (Table 2, Figs.
7–8) — not replicated.** Both are experiment suites over the existing machinery;
roadmap items 9–10.

## §5.3 Light optimization — faithful

- Eq. 5: predicted image = Σᵢ Eᵢ · N_sphere(px, ℓᵢ, rᵢ), accumulated sequentially per
  light (`--n-lights`).
- Eq. 6: MSE on Reinhard-tonemapped values T(I) = I/(1+I).
- Reparameterization: center/radius through the logit of their bounded domains
  (recovered by sigmoid — bounds can never be violated), color through inverse
  softplus from ℝ₊. Adam in unconstrained space; paper defaults lr 0.05, 500
  iterations.
- Mini-batch SGD (Table 3): `--pixel-fraction α` evaluates a random pixel subset of
  size ⌊α·H·W⌋ per iteration, drawn without replacement.
- Every result is re-rendered through reference GATHERLIGHT; proxy-space and
  physically-gathered errors are reported separately.

*Evidence the machinery matters:* on the identical hidden-light recovery task, the
torch §5.3 implementation reaches center error 0.013 where the numpy backend's naive
clipped-Adam optimizer stalls at 0.44.

*Not replicated:* the Mitsuba-3 and ZeroGrads baseline comparisons (Fig. 9), and the
50-run statistical protocol.

## §6 Applications

- **§6.1 interactive relighting — implemented** (both backends; benchmarked in
  `docs/performance.md`). Multi-view and per-layer compositing NRPs — not implemented
  (roadmap items).
- **§6.2 art-directed edits — implemented**: `examples/make_art_target.py` builds a
  painted objective + emphasis mask + protected region; `--mask` / `--protect` /
  `--protect-base` / `--protect-lambda` implement weighted objectives and
  keep-this-region constraints.
- **§6.3 generative targets — out of scope** as a dependency (no image model is
  bundled), but any generated image dropped in as `--target file.npy` exercises the
  same path the paper uses.

## §7 Limitations — shared

All of the paper's stated limitations apply here too: fixed transport after caching
(no post-hoc attenuation/exclusivity edits), undersampled-region artifacts,
parameter-count-driven difficulty for complex light types, and in-memory path data.
This implementation adds its own: toy scale, no fused kernels, and Monte Carlo noise
floors at 16–24 spp that dominate SMAPE on near-zero pixels.
