# Performance — Methodology and Measured Results

## Methodology

- **Hardware:** one Apple Silicon laptop (macOS), single process. Devices: `cpu` and
  `mps` (Apple GPU via PyTorch); no CUDA hardware was available.
- **Timing:** `time.perf_counter()` around measured regions; GPU timings use 5 warmup
  frames then 30 measured frames with `torch.mps.synchronize()` /
  `torch.cuda.synchronize()` before starting and after finishing the clock
  (`nrp/torch_backend/bench.py`).
- **Quality:** PSNR (dB) and SMAPE against held-out light configurations that were
  never used in training (fresh random draws). Torch runs report quality against both
  the raw GATHERLIGHT estimate (physically grounded) and the denoised one (what the
  network was supervised with). The ablation suite additionally reports SSIM and
  LDR-FLIP (`nrp/metrics.py`, numpy implementations; FLIP verified against NVIDIA's
  `flip-evaluator`) on Reinhard-tonemapped sRGB images.
- **Reproduction:** `mise run train` / `train-torch` / `export-mitsuba` /
  `train-mitsuba` / `bench`; every run writes a JSON report next to its model.
- **Comparability warning:** the paper's numbers are RTX 5090 numbers on 680²–960×402
  production/academic scenes with tiny-cuda-nn and a fused Triton gather. Nothing
  below is comparable to the paper's absolute figures; scaling *shapes* are.

## Scenes

| scene | producer | resolution | spp | bounces | segments | cache size |
|---|---|---|---|---|---|---|
| toy box | `nrp/toy_tracer.py` | 48×48 | 24 | 3 | 165,888 | 7.1 MB |
| Mitsuba cornell box | `nrp/mitsuba_exporter.py` | 48×48 | 16 | 4 (RR) | 103,680 | 3.9 MB |
| Country Kitchen (gallery) | `nrp/mitsuba_exporter.py` (wavefront) | 128×128 | 64 | 4 (RR) | 3,266,937 | 128.7 MB |

Export cost: toy trace ~seconds; Mitsuba cornell box **1.6 s** scalar; the kitchen at
128×128 / 64 spp exports in **4.0 s** with the wavefront loop (would be ~55 s scalar
at the measured scalar throughput). The kitchen scene is downloaded on demand by
`examples/scenes/download_scene.py kitchen` (assets are never committed).

## Exporter throughput: scalar vs drjit wavefront (roadmap item 1)

`mise run bench-export` → `out/export-bench.json`. Cornell box, 16 spp, 4 bounces
(RR), best of 3 runs after a warmup export that pays the one-time Metal JIT kernel
compilation (~3 s, reported separately in the JSON). Variant: `metal_ad_rgb`
(`llvm_ad_rgb` is preferred when a system libLLVM exists; this machine has none).

| resolution | scalar (seg/s) | wavefront (seg/s) | speedup |
|---|---|---|---|
| 48×48 | 62,433 (1.66 s) | 2,447,889 (0.042 s) | **39.3×** |
| 128×128 | 59,342 (12.4 s) | 3,515,860 (0.21 s) | **59.3×** |

Both exceed the roadmap's ≥ 20× target. Scalar throughput is resolution-independent
(pure Python per-path cost); the wavefront loop keeps gaining with wavefront size as
kernel-launch overhead amortizes. A fixed-seed equivalence test (8×8 cornell box,
64 spp) keeps the two loops statistically compatible: GATHERLIGHT mean radiance
agreed within 0.15% on the run recorded here (test bound: 2%).

## Volumetric export (roadmap item 2, §3.1 "Volume rendering")

`uv run python examples/volume_report.py --out out/volume-report.json`. Toy box at
48×48 / 24 spp / 3 bounces, homogeneous medium σ_t = 2.0, single-scattering albedo
0.8, isotropic phase; surface-only trace with identical settings as baseline.
GATHERLIGHT gained no volume code: free-flight sampling makes transmittance implicit
in segment lengths (validated analytically — slab-fixture falloff of a light inside
the medium matches exp(−σ_t·d) within 5%, `tests/test_volume.py`).

| | surface | volume | ratio |
|---|---|---|---|
| segments | 165,888 | 165,888 | 1.00× |
| cache size | 7.1 MB | 7.6 MB | 1.07× |
| gather ms/image (20 lights) | 3.8 | 3.6 | 0.93× |
| torch proxy held-out PSNR | 19.17 dB | 18.84 dB | −0.33 dB |
| torch proxy held-out SMAPE | 0.92 | 0.83 | |

- **Segment count does not grow** in this tracer: the event budget is fixed
  (3 bounces), so scatter vertices *replace* surface events rather than adding
  segments. Cache bytes grow ~7% anyway — shorter, more varied segments compress
  worse. A tracer with RR/unbounded depth would grow segment count instead.
- Gather is marginally *faster* on the volume cache (shorter segments → fewer
  light overlaps to accumulate).
- Proxy quality on the volumetric scene (same architecture, budget, and seed:
  `examples/toy_sphere_volume_torch.json`) lands within 0.33 dB of the surface
  baseline — the proxy absorbs in-medium lighting without special handling. SMAPE
  improves because the glowing medium lifts pixels off zero, where SMAPE is noisiest.

## Decoupling consistency (validates §3.1, backend-independent)

GATHERLIGHT over the toy 24-spp cache vs an independent 64-spp re-trace with inline
emission (same scene, different seeds — independent path sets):

- **PSNR 28.97 dB; mean radiance 2.6994 vs 2.6986 (0.03%).**
- SMAPE 0.72, dominated by near-zero pixels where both MC estimates are noisy — this
  is variance, not bias (the means agree).

## Training and held-out quality

| run | params | train time | held-out PSNR | SMAPE | notes |
|---|---|---|---|---|---|
| numpy, toy box | 21,635 | 219 s | 19.97 dB | 0.91 | sinusoidal PE, raw targets |
| torch, toy box | 62,923 | 60 s (+0.7 s pool) | 19.17 dB (18.86 vs denoised) | 0.92 | hashgrid, bilateral-denoised pool |
| torch, Mitsuba box + OIDN | 62,923 | 48 s (+0.4 s pool) | **25.87 dB** (26.48 vs denoised) | 0.94 | paper-exact §4.1+§4.4 pipeline |
| torch, kitchen 128² + OIDN | 106,085 | 126 s (+4.8 s pool) | **25.24 dB** (23.02 vs denoised) | 0.99 | first real academic scene; wavefront-exported cache |

Readings:

- The torch backend matches numpy quality on the toy scene in **3.6× less wall
  clock**, at 3k iterations vs 150 epochs over a precomputed dataset.
- The best quality comes from the paper-exact pipeline (Mitsuba data + OIDN):
  +6.7 dB over the same architecture on the toy scene. Scene conditioning and
  denoised supervision both contribute; separating them is roadmap item 10.
- The kitchen run (`examples/kitchen_torch.json`, report in
  `out/kitchen-torch/torch_train_report.json`) holds 25.24 dB on a real interior
  scene at 7× the pixel count and 31× the segment count of the cornell runs — 12 ms/frame (84 Hz) full-frame CPU inference at 128², model
  430 KB vs the 129 MB cache it compresses. PSNR vs *denoised* is lower than vs
  raw here (23.02 dB): at 64 spp the raw GATHERLIGHT target is already clean, so
  OIDN's residual smoothing costs more than the noise it removes.
- SMAPE ≈ 0.9 everywhere: near-zero-contribution pixels dominate this metric at
  16–24 spp. Trust PSNR for aggregate quality at these sample counts.
- Model sizes: 175 KB (numpy .npz) / 257 KB (torch .pt) — the compression story of
  the paper (7 MB cache → 0.26 MB proxy even at toy scale).

## GATHERLIGHT: numpy vs batched torch (roadmap item 3)

`uv run python -m nrp.torch_backend.bench --model out/toy-torch/model.pt --gather-caches
out/toy/path_cache.npz out/mitsuba/path_cache_128.npz out/mitsuba/path_cache_256.npz
--out out/gather-bench.json` — ms per gathered image, mean over 20 random sphere
lights (3 warmup), `torch.mps.synchronize()` around the clock. torch-CPU runs fp64
(bit-compatible with numpy); MPS is fp32.

| cache | segments | numpy-CPU | torch-CPU | torch-MPS | MPS speedup |
|---|---|---|---|---|---|
| toy 48² | 165,888 | 3.3 ms | 2.6 ms | 1.6 ms | 2.0× |
| cornell 128² | 736,396 | 14.7 ms | 8.4 ms | 2.8 ms | 5.2× |
| cornell 256² | 2,945,876 | 58.6 ms | 29.5 ms | 8.1 ms | 7.2× |

The analogue of the paper's Table 1 reconstruction row: batched device gathering
scales with segment count much more gently than the numpy loop (kernel-launch
overhead dominates MPS below ~1M segments), which is what makes on-the-fly pool
replacement affordable at higher resolutions. Parity with numpy is unit-tested
(rtol 1e-5, 50 sphere + 50 quad lights, toy and Mitsuba caches;
`tests/test_torch_gather.py`).

## Device-resident training: CPU vs MPS (roadmap item 3)

Full training runs (3k iterations, identical seed, `gather_backend: torch`) on both
devices; reports in `out/bench-train/`. Held-out PSNR parity is the correctness
criterion (requirement: within 0.5 dB at equal iterations and seed).

| config | device | train wall-clock | held-out PSNR | Δ vs cpu |
|---|---|---|---|---|
| toy 48² | cpu | 43.5 s | 19.17 dB | — |
| toy 48² | mps | 50.6 s | 19.59 dB | **+0.42 dB** |
| Mitsuba cornell 48² | cpu | 38.6 s | 25.90 dB | — |
| Mitsuba cornell 48² | mps | 49.8 s | 25.65 dB | **−0.25 dB** |

- **Quality parity holds** (both configs well inside 0.5 dB; the difference is fp32
  arithmetic ordering, not a training defect).
- **MPS is ~15–30% slower end-to-end at this scale** — an honest negative: a
  62,923-param MLP at batch 4096 under-fills the GPU, so per-iteration launch
  overhead outweighs compute. The pieces that *do* scale are already GPU-favored
  (gather 5–7× at 128²–256² above; full-frame inference 4–5× at ≥256², below), so
  device residency pays off exactly where the paper operates — larger models,
  resolutions, and batches (roadmap item 6).
- Pool builds via the torch gather: 0.5 s (toy) / 0.3 s (Mitsuba) on CPU — at 48²
  the numpy and torch pool costs are equal within noise; the gather table above is
  where the gap opens.

## Inference latency (full frame, 62,923-param sphere model)

| device | 48² | 128² | 256² | 512² | 1024² |
|---|---|---|---|---|---|
| cpu | 5.9 ms / 170 Hz | 14.1 ms / 71 Hz | 47.7 ms / 21 Hz | 110.6 ms / 9.0 Hz | 354 ms / 2.8 Hz |
| mps | 2.1 ms / 481 Hz | 2.8 ms / 359 Hz | 8.5 ms / 117 Hz | **26.8 ms / 37 Hz** | 116 ms / 8.6 Hz |

- **MPS sustains the paper's ~30–60 Hz interactive band up to 512×512.** CPU holds it
  only to ~256².
- Scaling is linear in pixel count once the device is saturated (CPU: throughout;
  MPS: above ~256²) and independent of scene complexity — the cache is never touched
  at inference, which is the paper's central scaling claim.
- MPS speedup over CPU: 2.8× (48²) to 5.6× (256²) to 4.1× (512²).
- Extrapolation: at 1920×1080 (2.07 M px), expect ~230 ms (4.3 Hz) on MPS for this
  model. The paper's production rates need CUDA + tiny-cuda-nn (fp16 tensor cores) —
  roadmap item 3.

Numpy-backend comparison at 48²: 1.6 ms/frame — *faster* than torch (10.7 ms) at this
tiny size because the torch graph overhead dominates; torch wins as resolution grows.

## Inverse optimization (hidden-light recovery, toy scene, unit box)

Task: recover center (0.6, 0.7, 0.5), radius 0.12, rgb (10, 9, 8) from its rendered
image. Torch: paper §5.3 formulation, 500 steps, pixel fraction 0.25, best of 4
restarts. numpy: naive clipped Adam, 200 steps, best of 2 restarts.

| backend | center err | radius err | rgb err | re-rendered PSNR vs target |
|---|---|---|---|---|
| torch (§5.3) | **0.013** | **0.020** | 1.21 (~7–10% low, uniform) | 17.0 dB |
| numpy (naive) | 0.44 | 0.16 | 9.2 | 17.2 dB |

The 30× geometry improvement isolates the value of the paper's reparameterization +
tonemapped loss (same proxy quality class, same scene, same restart budget order).
Note the naive optimizer reaches similar *image-space* PSNR while being far off in
*parameter* space — image similarity alone does not certify parameter recovery, which
is why reports carry both.

At pixel fraction 0.25, each optimization step evaluates 576 of 2,304 pixels; the
full 500-step, 4-restart run completes in well under a minute on CPU at 48².

## Inverse-optimization grid (roadmap item 4, Table 3 structure)

`mise run inverse-grid` → `out/inverse-grid/report.json`. Mitsuba cornell box (48²,
25.9 dB sphere proxy), N hidden sphere lights jointly recovered from one rendered
target, 500 Adam steps, 5 runs per cell (fresh hidden lights + init each run);
re-rendered (reference-GATHERLIGHT) PSNR vs the target.

| N | metric | α=1.0 | α=0.25 | α=0.05 | α=0.01 |
|---|---|---|---|---|---|
| 1 | PSNR (dB) | 15.8 ± 8.8 | 15.8 ± 8.8 | 15.8 ± 8.8 | 15.8 ± 8.6 |
| 1 | s/run | 3.5 | 1.4 | 0.7 | 0.7 |
| 3 | PSNR (dB) | 17.8 ± 1.6 | 17.7 ± 1.8 | 17.8 ± 1.4 | 17.5 ± 1.5 |
| 3 | s/run | 9.7 | 4.0 | 2.0 | 1.8 |
| 5 | PSNR (dB) | 16.1 ± 2.1 | 16.1 ± 2.2 | 16.1 ± 2.1 | 16.1 ± 2.2 |
| 5 | s/run | 16.9 | 7.5 | 4.2 | 3.7 |

- **The paper's Table-3 trend reproduces qualitatively**: recovered quality is flat
  in α — dropping from all pixels to 1% of them costs at most ~0.3 dB in any row —
  while wall-clock falls ~5× from α=1.0 to α=0.05. Below that the speedup saturates
  (per-step fixed costs and the final full-frame re-render dominate at 48²), where
  the paper's GPU-scale runs keep gaining; the shape, not the constants, matches.
- N=1's large σ (8.8 dB) is bimodal single-light recovery: most runs succeed, an
  occasional hidden light lands where the proxy is weakest and the run stalls in a
  wrong basin — the same failure mode the restarts mechanism exists for (grid runs
  use a single start per run by design, to measure the raw cell cost).

## Quad-light inverse recovery (roadmap item 4)

Quad recovery is **proxy-fidelity-bound**, established by explicit diagnostics
rather than assumed: with a 24-spp/3k-iteration toy quad proxy (~20 dB), the proxy
loss at the *true* parameters (0.056) is 2× the loss at the wrong optima the
optimizer finds (0.023–0.028), and an optimization started *at* the truth walks
0.14 away — the optimizer is fine, the proxy's image-match optimum sits at wrong
parameters. Three changes fix it (`examples/inverse_grid.py --quad-check`,
`out/inverse-grid/quad_check.json`):

1. a higher-fidelity proxy (96-spp cache, 10k iterations, 2¹⁴ hash table);
2. restart ranking by the **physical GATHERLIGHT re-render** instead of proxy loss
   (proxy loss prefers parameter-far image matches; now also used by the CLI);
3. a well-conditioned fixture: near-surface dim quad (Reinhard sensitivity falls as
   1/(1+I)², so bright targets carry less positional gradient).

Result: **center error 0.039 < 0.05** (best of 12 restarts), re-rendered
18.8 dB vs the target — deterministic from one command.

## Paper-scale training (roadmap item 6, the quality ceiling)

`examples/mitsuba_cornell_128_torch.json` runs the paper's training configuration:
8×256 network + 2¹⁴-entry hashgrid with `finest_resolution` = image width
(521,061 parameters), pool 128, batch 4096, cosine LR 0.005 → 5×10⁻⁵, 50k
iterations, OIDN targets, torch gather, on the Mitsuba cornell box at 128×128 /
64 spp (2,945,475 segments, 103.6 MB float64 cache — wavefront export takes 2.6 s).
Long-run safety: `checkpoint: {"every": 1000}` writes full training state (model,
optimizer, scheduler, both RNGs, pool) and `--resume` continues the identical
trajectory — bit-exact on CPU, asserted by `tests/test_checkpoint_resume.py`.

PSNR-vs-iteration on the fixed 12-light held-out set (dedicated RNG, evaluated at
every checkpoint; PSNR vs raw GATHERLIGHT references):

| iteration | 1k | 3k | 10k | 25k | 50k |
|---|---|---|---|---|---|
| held-out PSNR (dB) | 23.61 | 26.79 | 29.05 | 31.58 | **35.19** |
| cumulative train s (MPS) | 25 | 85 | 275 | 666 | 1344 |

- **Gain from scale alone: +8.7 dB.** The 3k-iteration baseline config (4×128,
  2¹² table, pool 64, constant LR) on the *same* cache reaches 26.49 dB in 68 s;
  the paper-scale run ends at 35.19 dB (37.24 dB vs denoised), SMAPE 0.662 vs
  0.794. The big model's own 3k checkpoint (26.79 dB) barely beats the small
  baseline — capacity only pays off with iterations.
- **Convergence shape:** ~2–3.5 dB per 2.5× iterations with no plateau at 50k —
  the quality ceiling is not reached at this budget, consistent with the paper
  training to 100k. SMAPE flattens near 0.66 long before PSNR stops improving
  (SMAPE is dominated by near-zero pixels where the raw MC reference is itself
  noisy).
- **Throughput:** MPS 37.2 iters/s → 1344 s (22.4 min) wall-clock for 50k (pool
  build 3.4 s; checkpoint eval/save excluded from the training clock). CPU: 23.2
  iters/s → 2153 s (35.9 min) for the identical run, ending at 34.58 dB — within
  0.61 dB of MPS (same seeds; device arithmetic differs), MPS 1.6× faster
  wall-clock at this batch size. The 3k *baseline* config runs at 44.2 iters/s on
  MPS (68 s). Inference at 128²: 2.1 ms/frame (~480 Hz) on MPS.

## Image-based baseline vs path-based pool (roadmap item 9, Fig. 6)

`examples/image_based_baseline.py` (one command, `out/image-based/report.json`)
trains the identical model/optimizer/seed under two supervision regimes on the
Mitsuba cornell box at 128×128 / 64 spp, 3000 iterations each: **path-based**
(§4.4-style rolling pool, images continually replaced with fresh lights) vs
**image-based** (a fixed dataset of R denoised GATHERLIGHT images, no
resampling — implemented as the same pool trainer with `replace_count: 0`, so the
code path is shared). All regimes are scored on a common 24-fresh-light held-out
set (dedicated RNG, asserted disjoint from every regime's recorded supervision
lights) with tonemapped PSNR (Reinhard, peak 1 — Fig. 6's metric); per-light
distributions are in the report JSON.

| regime | supervision images | supervision s | tonemapped PSNR (dB) | linear PSNR |
|---|---|---|---|---|
| path-based (pool 64, 2/5) | 1,264 | 36.1 | 27.55 ± 4.86 | 24.47 |
| path pool 256, 2/5 | 1,456 | 43.0 | 28.57 ± 5.68 | 25.46 |
| image-based R=64 | 64 | 1.9 | 23.57 ± 4.73 | 20.24 |
| image-based R=256 | 256 | 7.1 | 27.22 ± 5.80 | 24.44 |
| image-based R=1024 | 1,024 | 25.9 | **29.38 ± 5.26** | 26.00 |

**The paper's ≥ 2.8 dB path-based advantage does *not* reproduce at this scale —
the sign flips.** At matched supervision budget (1,264 pool images vs a fixed
1,024), the fixed dataset wins by **1.83 dB** tonemapped. Honest reading of why,
rather than smoothing over it:

- **Our "image-based" images are not the paper's.** In the paper the image-based
  baseline pays full render cost per image, and path-based training resamples
  lights *per pixel* directly from path data. Here both regimes draw from the same
  cached-path GATHERLIGHT (25.9 s for 1,024 denoised 128² images), so the
  image-based regime inherits exactly the cheap supervision that is path-based
  training's actual advantage — the comparison isolates the *sampling schedule*,
  not the paper's full data-cost story.
- **The rolling pool bottlenecks light diversity.** A batch sees at most 64
  concurrent lights vs 1,024 in the fixed dataset; raising the pool to 256
  recovers +1.0 dB of the 1.83 dB gap, supporting this mechanism (the paper uses
  pool 300 *and* per-pixel resampling at production scale).
- The regimes' trend is still the paper's *within* the image-based family:
  R=64 → 256 → 1024 climbs 23.6 → 27.2 → 29.4 dB with supervision time scaling
  linearly — data diversity is decisive; only the claim that a rolling pool beats
  an equally *large and equally cheap* fixed set fails to transfer to toy scale.

## Component ablation × spp sweep (roadmap item 10, Table 2 / Fig. 7)

`mise run ablation` → `out/ablation/report.json` (deterministic on CPU: metric
fields are bit-identical across runs; every cell's full config is embedded in the
report). Mitsuba cornell box at 64×64, five component sets × spp ∈ {8, 16, 32},
every cell the identical budget and seeds: 3,000 iterations, batch 4096, pool 64
(replace 2 every 5), lr 5×10⁻³, 4×128 MLP; hashgrid 2¹²/8-level encoding and OIDN
target denoising toggled per variant (`none` = raw pixel coords + light params
only). All 15 cells are scored on one common 16-light held-out set (dedicated RNG)
against a separate 128-spp reference cache (independent seed, 1.47 M segments), so
the spp axis measures training-cache quality against a fixed clean target. SMAPE
and PSNR are on linear radiance; SSIM and FLIP on Reinhard-tonemapped sRGB
(`nrp/metrics.py`; FLIP verified against NVIDIA's `flip-evaluator`).

| spp | variant | PSNR (dB) | SSIM ↑ | FLIP ↓ | SMAPE ↓ | train s |
|---|---|---|---|---|---|---|
| 8 | none | 23.92 ± 4.20 | 0.433 | 0.165 | 0.737 | 17.8 |
| 8 | aux | 24.08 ± 5.38 | 0.537 | 0.148 | 0.609 | 17.5 |
| 8 | aux+den | 24.14 ± 4.06 | 0.541 | 0.135 | 0.626 | 21.1 |
| 8 | aux+enc | 24.86 ± 4.82 | 0.514 | 0.142 | 0.635 | 36.5 |
| 8 | aux+enc+den | **26.43 ± 5.48** | **0.544** | **0.131** | 0.630 | 39.0 |
| 16 | none | 25.84 ± 5.69 | 0.455 | 0.158 | 0.723 | 20.3 |
| 16 | aux | **26.32 ± 3.83** | **0.553** | **0.132** | **0.591** | 20.5 |
| 16 | aux+den | 24.96 ± 4.07 | 0.547 | 0.141 | 0.623 | 22.7 |
| 16 | aux+enc | 24.22 ± 3.76 | 0.534 | 0.153 | 0.690 | 38.2 |
| 16 | aux+enc+den | 25.85 ± 5.13 | 0.548 | 0.138 | 0.608 | 41.2 |
| 32 | none | 23.80 ± 4.84 | 0.433 | 0.177 | 0.735 | 25.5 |
| 32 | aux | **25.47 ± 4.51** | **0.575** | **0.124** | **0.591** | 25.8 |
| 32 | aux+den | 25.15 ± 5.01 | 0.561 | 0.138 | 0.622 | 28.1 |
| 32 | aux+enc | 24.48 ± 5.04 | 0.557 | 0.144 | 0.620 | 43.8 |
| 32 | aux+enc+den | 24.67 ± 4.60 | 0.548 | 0.135 | 0.629 | 46.1 |

Statistical context first: per-light PSNR σ is 4–6 dB over 16 lights (standard
error of the mean ≈ 1.1–1.4 dB), so single-cell PSNR gaps below ~1.5 dB are
within noise; the perceptual metrics are much tighter (FLIP σ ≈ 0.05) and the
readings below lean on directions that hold at *every* spp, not single cells.

Directions vs the paper's Table 2:

- **Aux features dominate — matches the paper.** `none` → `aux` improves SSIM by
  +0.10–0.14, FLIP by 0.017–0.053, and SMAPE by ~0.13 at every spp, at zero extra
  training cost. Notably `none` looks almost competitive on PSNR alone (23.8–25.8
  dB) — it renders a plausible low-frequency lighting wash — and only the
  structural/perceptual metrics unmask it, which is exactly why the paper reports
  SSIM/FLIP and why this item added them.
- **Encoding alone on raw targets hurts — matches the paper's noisy-target
  caveat.** `aux` → `aux+enc` degrades PSNR/SSIM/FLIP at 16 and 32 spp (−2.1 and
  −1.0 dB, FLIP +0.021/+0.020) and trades metrics at 8 spp: the hashgrid's extra
  per-pixel capacity fits Monte-Carlo noise in the raw GATHERLIGHT targets. It
  also costs ~1.8× wall-clock at this scale.
- **Encoding + denoising wins at 8 spp — matches; the win does not persist at
  higher spp — divergence, flagged.** At 8 spp `aux+enc+den` is best in all four
  metrics (+2.35 dB, best SSIM/FLIP), the paper's headline combination behaving
  as advertised where targets are noisiest. But at 16–32 spp plain `aux` edges
  out the full stack (within PSNR noise, though consistent across SSIM/FLIP at 32
  spp). Two local causes rather than one paper contradiction: (a) OIDN's residual
  smoothing costs more than the noise it removes once targets are fairly clean —
  the same effect measured on the 64-spp kitchen run above — and (b) at 3k
  iterations / 64² the hashgrid's capacity advantage hasn't paid off yet (the
  paper-scale section shows encoding capacity needs iterations: the 8×256/50k run
  gains +8.7 dB). The paper trains 100k iterations on production scenes at 512
  spp with a much stronger denoiser-to-noise ratio.
- **spp trend vs Fig. 7 — matches on perceptual metrics for the raw-target
  variants, inverts for the denoised ones.** For `aux`, SSIM climbs monotonically
  (0.537 → 0.553 → 0.575) and FLIP falls monotonically (0.148 → 0.132 → 0.124)
  with spp, the Fig. 7 shape. PSNR is not monotone anywhere (see the noise floor
  above). For `aux+enc+den` the trend *inverts* on PSNR (26.43 → 25.85 → 24.67):
  its low-spp advantage is precisely the denoiser's, so cleaning the targets
  erodes it — flagged as the expected complement of divergence (b), not smoothed
  over.

## Multi-view NRPs (roadmap item 7, §6.1)

Three views of the Mitsuba cornell box (48×48 @ 16 spp, 4 bounces, wavefront
exporter with the per-view camera override), cameras on a ±20° arc at the default
distance 3.9 with ±0.3 height offsets, all aimed at the box center. One proxy per
view, each trained exactly like the single-view Mitsuba config (62,923 params, 3k
iterations, pool 64, OIDN targets) with per-view seeds 0/1/2. Produced by
`mise run multiview` (report: `out/multiview/report.json`; view manifest:
`out/multiview/views.json`).

| view | camera origin | segments | train (s) | held-out PSNR (dB) | per-light range (dB) | SMAPE |
|---|---|---|---|---|---|---|
| view0 | (−1.33, 0.00, 3.66) | 93,011 | 35.9 | 25.41 | [15.7, 37.8] | 0.699 |
| view1 | (0.00, 0.30, 3.90) | 102,686 | 34.2 | 24.82 | [14.8, 32.4] | 0.952 |
| view2 | (1.33, −0.30, 3.66) | 92,503 | 33.9 | 21.20 | [14.4, 26.0] | 0.909 |

All three views exceed the 20 dB requirement. Cross-view consistency for one fixed
light (sphere at (0.2, 0.35, −0.1), r = 0.25): each view's proxy vs its *own*
GATHERLIGHT reference gives 27.95 / 24.71 / 24.19 dB — every view inside its
validation range, max spread **3.76 dB** (< 4 dB required, but with little margin:
view2, the most oblique camera, is consistently the weakest view).

**Edit latency** (one light change re-rendered in all N views,
`nrp.torch_backend.relight_multiview`, 20 synchronized edits, 3 warmup). Each view's
pixel inputs are precomputed at load; an edit performs N full-frame network forwards
and touches no path-cache data:

| N views | CPU ms/edit | MPS ms/edit |
|---|---|---|
| 1 | 4.02 | 7.04 |
| 2 | 7.97 | 8.30 |
| 3 | 12.30 | 12.67 |

CPU latency is N × single-view inference to within 2% (4.02 → 12.30 ms ≈ 3.06×), the
expected no-cache-access scaling; MPS is dispatch-overhead-dominated at 48² (its
per-view increment, ~2.8 ms, is below its 1-view latency), so it only catches up to
CPU at this tiny frame size and would win at paper-scale resolutions (see the
inference table above).

**Memory (the compactness argument):** 251,692 bytes of fp32 parameters per resident
view proxy — **0.76 MB total for all 3 views**, vs 3 path caches of 3.3–3.7 MB
`.npz` each (10.5 MB total) that are *not* needed at edit time.

## Per-layer compositing NRPs (roadmap item 8, §6.1 / Fig. 11)

The toy scene split into first-hit layers — foreground sphere vs background box
(`nrp.toy_tracer --layer`) — at 48×48 / 24 spp / 3 bounces, seed 1. Layer paths
still bounce off the full scene; a layer records only the paths whose *first hit*
is on its geometry, keeping the full-spp estimator denominator. Produced by
`mise run layers` (report: `out/layers/report.json`; composited demo image:
`out/layers/composite_demo.npy` — box layer held fixed under a warm light, sphere
layer relit by its proxy under a cool one).

**Linearity (the property compositing relies on):** the two layer caches partition
the full cache's 165,888 segments exactly (43,701 sphere + 122,187 box), so over 12
validation lights the sum of the layer GATHERLIGHT images reproduces the full-scene
image to `np.allclose` — min PSNR 338 dB, i.e. float round-off, comfortably past the
30 dB requirement.

**Per-layer proxies** (identical configs: 62,923 params, 3k iterations, pool 64,
bilateral denoise, seed 0):

| proxy | segments | train (s) | held-out PSNR (dB) | SMAPE |
|---|---|---|---|---|
| full scene | 165,888 | 39.5 | 21.78 | 0.844 |
| sphere layer | 43,701 | 36.1 | 24.57 | 0.346 |
| box layer | 122,187 | 38.3 | 24.81 | 0.692 |

**Owned-pixel quality:** on each layer's own pixels (proxy vs its own GATHERLIGHT
reference, shared peak), the layer proxies don't just match the full-scene proxy —
they beat it: sphere-owned pixels (609 px) 22.15 dB vs 17.47 dB (**+4.68 dB**),
box-owned pixels (1,695 px) 21.01 dB vs 17.48 dB (**+3.53 dB**). The roadmap asked
whether layer proxies stay within 1 dB of the full proxy; at this budget the result
is well outside that band *in the layers' favor* — each layer's radiance is a simpler
function (one object's first-hit transport instead of a mixture), so specialization
wins at equal capacity and iterations.

**Composite edit latency:** relighting the sphere layer and adding the fixed box
image costs **3.95 ms** vs **4.01 ms** for a full-scene proxy relight (CPU, 48×48,
30 frames) — compositing is latency-neutral at equal network size; its value is the
artistic control (hold one layer's lighting fixed) plus the per-layer quality above.

## Packed cache layout (roadmap item 5, §4.2: fp16 + rgb9e5)

`PathCache.save(path, compressed=True)` writes segment geometry and G-buffer aux as
fp16 and per-segment throughput as shared-exponent rgb9e5 (`nrp/rgb9e5.py`); `load`
auto-detects the layout and returns float64 arrays either way. Measured by
`uv run python examples/pack_bench.py --train` (report: `out/pack/report.json`),
20 random sphere lights per scene, load times best-of-3:

| scene | segments | float64 | packed | ratio | load (f64→packed) | gather ms/img (f64→packed) | gather PSNR packed vs f64, min |
|---|---|---|---|---|---|---|---|
| toy box 48² | 165,888 | 6.74 MB | 1.62 MB | **4.2×** | 37 → 17 ms | 3.20 → 3.17 | **43.6 dB** |
| Mitsuba cornell box 48² | 103,680 | 3.70 MB | 1.16 MB | **3.2×** | 22 → 12 ms | 1.96 → 1.94 | **53.9 dB** |

- **Size:** 3.2–4.2× on-disk (both layouts go through `np.savez_compressed`, so the
  ratio is measured after zlib — raw array bytes shrink ~4.3×; the toy cache packs
  better because its float64 mantissas are incompressible MC noise, while the
  Mitsuba cache's zlib already exploited structure). The lower end of the roadmap's
  4–6× expectation; honest as measured.
- **Decode cost:** packing moves all decode work to `load` — and packed *load* is
  ~2× **faster** (half the bytes to decompress dominates the fp16/rgb9e5 decode).
  Gather time is unchanged (identical float64 arrays after load), so there is no
  per-image decode tax, unlike the paper's in-kernel decode.
- **Fidelity:** packed-cache GATHERLIGHT images are ≥ 43 dB (toy) / ≥ 54 dB
  (Mitsuba) faithful to the float64 cache's over 20 random lights — far inside the
  0.5 dB budget for any downstream metric.
- **End-to-end training cost:** toy torch config trained from float64 vs packed
  cache over seeds {0, 1, 2} (identical config, same seeds): held-out PSNR
  21.69 ± 2.12 dB (float64) vs 21.63 ± 2.56 dB (packed) — **mean delta −0.06 dB**,
  within the 0.3 dB target. Per-seed deltas (−0.66/+0.07/+0.40 dB) are trajectory
  noise: the caches' tiny numerical differences perturb the SGD path chaotically,
  so single-seed comparisons measure seed sensitivity (σ ≈ 2.1–2.6 dB across
  seeds), not packing damage — hence the mean-over-seeds claim, per the repo's
  statistical-testing convention.

## Denoising

OIDN (RT, HDR, albedo+normal guides) on a flat-2.0 HDR signal with σ=0.5 noise:
MSE 0.248 → 0.0054 (**46×**), mean 1.98 (HDR preserved). The bilateral fallback on
the same fixture achieves ~2× (it is a much weaker prior — expected). Pool build cost
with OIDN at 48×48: 0.35 s for 64 images (~5 ms/image).

## Animated-light temporal NRP harness (extensions E1, static-camera slice)

`nrp.torch_backend.animate` renders a keyframed light sequence from one resident
torch proxy. During frame rendering the path cache is used only for pixel coordinates
and G-buffer auxiliary tensors; no GATHERLIGHT segment traversal occurs. The committed
demo command is `mise run animate`, using `examples/animated_lights.json`,
`out/toy-torch/model.pt`, and `out/toy/path_cache.npz`. Outputs are
`out/animate/frame_*.npy` plus `out/animate/report.json`.

Measured on the standard 48×48 toy torch proxy (62,923 params, 3k iterations, CPU):

| frame count | proxy ms/frame |
|---:|---:|
| 1 | 4.46 |
| 8 | 4.72 |
| 24 | 4.79 |
| 48 | 4.76 |

The near-flat per-frame latency is the expected static-scene/animated-light result:
one cache-backed proxy can be evaluated frame by frame without retracing or gathering.
The 24-frame output sequence took 0.116 s total (**4.81 ms/frame**, 208 fps equivalent)
at 48×48.

Temporal stability was measured as mean absolute per-pixel frame-to-frame delta under
the smooth light path. The proxy delta was **0.0698** vs **0.0665** for the
GATHERLIGHT reference sequence, a **1.05×** ratio. The reference sequence is computed
only for this metric pass (`--measure-reference`); it is not part of the animation
runtime path. The animated-camera/time-conditioned half of E1 is not implemented yet,
so no held-out intermediate-camera PSNR or time-conditioned cache-size-vs-K result is
claimed here.

## Out-of-core foundation (extensions E5, sharded cache + streamed-target slice)

`PathCache.save_sharded` writes a tile-sharded cache directory with a manifest and one
compressed `.npz` per pixel tile; `PathCache.load_sharded` reconstructs the monolithic
cache while restoring original segment order. `nrp.torch_backend.relight --tile-pixels`
and `relight_tiled` render proxy outputs in bounded pixel chunks. The out-of-core
report also streams shard files directly to build a fixed-light supervised target
table, without reconstructing the full segment arrays.

Measured by `mise run out-of-core` (report: `out/out-of-core/report.json`) on a
24×24 / 8 spp toy cache with 9,216 segments:

| check | result |
|---|---:|
| monolithic cache size | 381,005 bytes |
| sharded cache directory size | 423,777 bytes |
| save sharded | 0.026 s |
| load sharded | 0.013 s |
| sharded vs monolithic GATHERLIGHT max diff | 0.0 |
| streamed target max diff vs monolithic | 3.33e-16 |
| streamed target PSNR vs monolithic | 334.67 dB |
| streamed peak loaded segments | 1,024 (11.1% of cache) |
| streamed target pass | 0.0076 s |
| tiled vs untiled proxy max diff | 0.0 |

The sharded layout is larger at this tiny scale because each tile pays `.npz` and
manifest overhead; that is expected and is not a production-scale memory claim. This
slice now proves exact cache reconstruction, streamed target construction at toy
scale, and chunked inference, but it does **not** yet satisfy E5's full completion
criteria: the actual torch optimizer still trains from monolithic caches, there is no
peak RSS process measurement, and there is no 512² / 128 spp Mitsuba end-to-end run.

## Dynamic geometry cache invalidation (extensions E2, one-bounce slice)

`nrp.dynamic_geometry` identifies pixels whose first-hit G-buffer changes after the
toy sphere moves and splices fresh segments for only those pixels into the old cache.
This is the primary-visibility slice of E2: it proves cache-level invalidation and
segment replacement for one-bounce paths, not secondary-bounce transport updates or
proxy fine-tuning.

Measured by `mise run dynamic-geometry` (report:
`out/dynamic-geometry/report.json`) on a 32×32 / 8 spp / 10-frame toy sequence with
the sphere moving +0.16 along x:

| check | result |
|---|---:|
| mean full retrace | 4.26 ms/frame |
| mean splice pass | 0.77 ms/frame |
| mean invalid pixels | 29.7% |
| mean retraced segment fraction | 29.7% |
| full retrace share of 16 ms frame | 26.6% |
| splice-only share of 16 ms frame | 4.8% |
| outside invalidation mask max diff | 0.0 |
| incremental splice vs full retrace | exact (`inf` PSNR, 0 max abs) |
| stale cache PSNR at final frame | 10.76 dB |

For this constrained one-bounce setup, keeping the cache alive costs a small fraction
of a 16 ms game frame for the splice operation itself, and the stale-cache baseline
falls quickly as the sphere moves. The important caveat is structural: once secondary
bounces are enabled, unchanged first-hit pixels can still see changed indirect
transport. The full E2 criterion therefore remains open until multi-bounce
invalidation and warm-started proxy fine-tuning are implemented and measured.

## Light-aware path sampling (extensions E3, spherical guide-region slice)

`trace_path_cache` now accepts a spherical light-placement region and a guide
probability. At eligible surface bounces, the sampler mixes the existing
cosine-weighted Lambertian direction with a uniform solid-angle cone toward the
declared region, using `brdf * cos / mixture_pdf` throughput weights. If the guide
cone is not fully above the surface hemisphere, that ray falls back to cosine
sampling.

Measured by `mise run light-aware-sampling` (report:
`out/light-aware-sampling/report.json`) on a 32×32 / 12 spp / 3-bounce toy cache with
guide probability 0.5:

| check | standard | guided |
|---|---:|---:|
| segments | 36,864 | 36,864 |
| region-hit fraction | 5.18% | 35.03% |
| trace time | 65.28 ms | 28.04 ms |
| PSNR vs independent guided reference | 28.68 dB | 32.98 dB |
| SMAPE vs independent guided reference | 1.183 | 0.270 |

The equal-segment cache puts **6.76×** more segments through the declared placement
region and improves the direct GATHERLIGHT estimate for a light in that region by
**4.29 dB** in this toy setup. Cache size is unchanged because the sampler changes
directions, not path count. This satisfies the low-level E3 sampling-density and
unbiasedness-consistency slice, but not the full E3 criterion by itself: it does not
train proxies or reproduce the occluder/lamp-shade failure.

`mise run light-aware-proxy-ab` adds a toy standard-vs-guided proxy A/B on the same
placement region (report: `out/light-aware-proxy-ab/report.json`). Both runs use
20×20 / 8 spp caches, identical segment budgets, the same 2,743-parameter sphere
proxy, and 350 CPU training iterations:

| check | standard | guided |
|---|---:|---:|
| region-hit fraction | 4.98% | 34.57% |
| mean held-out validation PSNR | 13.68 dB | 14.83 dB |
| fixed in-region light PSNR | 10.24 dB | 11.88 dB |
| fixed open-region light PSNR | 4.70 dB | 14.46 dB |
| train time | 0.41 s | 0.40 s |

The guided proxy improves the fixed in-region light by **1.64 dB** and the open-region
fixture by **9.77 dB** at equal cache size. This is positive evidence that the guided
cache can help proxy training at toy scale, but it still misses E3's target
improvement of at least 3 dB for the occluded-region reproduction. A larger or more
targeted occluder fixture is still required before E3 can be marked complete.

## Quality-tier relight ladder (extensions E9, CLI plumbing slice)

`nrp.torch_backend.relight` now accepts `--quality preview|draft|final` and writes a
metadata sidecar next to every `.npy` output. `preview` is the proxy output, `draft`
is GATHERLIGHT from the current cache, and `final` is GATHERLIGHT from `--final-cache`
when supplied. `--residual-light` applies the E9 cached residual
`GATHERLIGHT(approved) - proxy(approved)` to preview output.

Measured by `mise run quality-tiers` (report: `out/quality/report.json`) on a
16×16 toy cache, using an 8-spp draft cache and a 32-spp final cache:

| tier | source | ms | PSNR vs final |
|---|---|---:|---:|
| preview | proxy | 0.98 | -15.23 dB |
| draft | cached GATHERLIGHT | 0.13 | 10.81 dB |
| final | high-spp GATHERLIGHT | 0.69 | inf |

The proxy in this report is intentionally untrained; the measurement is for the
quality-tier plumbing, sidecar metadata, and residual identity rather than visual
quality. At the approved light config, proxy plus cached residual matches cached
GATHERLIGHT to max absolute error **5.6e-17**. Moving the light by dx =
{0.05, 0.10, 0.20} drops residual-corrected PSNR vs cached GATHERLIGHT to
{24.66, 23.72, 20.38} dB, which is the expected residual-validity decay. This does
**not** yet satisfy E9's full final-frame study: no SSIM/FLIP tier table, no fresh
high-spp production-scale cache, and no supervisor-trust verdict.

## Gather-time production controls (extensions E8, cache fallback slice)

`gather_light_controlled` adds post-hoc controls at GATHERLIGHT time: per-pixel
first-hit exclusion for light linking, and a simple linear-distance artist
attenuation curve for sphere lights. This is the cache fallback path E8 asks us to
test; proxy-conditioned live controls are not implemented yet.

Measured by `mise run production-controls` (report:
`out/production-controls/report.json`) on a 32×32 / 12 spp toy cache with 24,576
segments:

| check | result |
|---|---:|
| full gather vs sphere+box layers max diff | 0.0 |
| exclude-sphere gather vs box-layer gather max diff | 0.0 |
| exclude-sphere gather vs box-layer PSNR | inf |
| full gather | 0.63 ms |
| linked gather | 0.91 ms |
| attenuated gather | 1.15 ms |
| attenuated/default mean-radiance ratio | 0.918 |

The linking algebra is exact for the toy layer partition: excluding the sphere-owned
pixels from the full cache leaves the box-layer contribution for that light. The
attenuation curve used here is `max(0, 1 - 0.1 * distance(origin, light_center))`,
validated by a closed-form unit fixture. The result is intentionally a negative
boundary for proxy speed: these controls survive the decoupling at gather time, but
they still require cache access until a conditioned proxy is trained and validated.

## Exported engine-shaped runtime (extensions E6, TorchScript slice)

`nrp.torch_backend.engine_runtime` exports a `TorchNRP` to a TorchScript artifact plus
JSON metadata and renders by loading that artifact through `torch.jit.load`, without
loading the training checkpoint in the frame loop. `docs/engine-integration.md`
documents the current input/output contract.

Measured by `mise run viewer` (report: `out/engine-runtime/report.json`) on a 32×32
toy cache and a small sphere-light proxy:

| check | result |
|---|---:|
| artifact size | 38,763 bytes |
| module vs exported runtime max diff | 0.0 |
| exported runtime allclose rtol 1e-4 | true |
| exported runtime latency | 0.43 ms/frame |
| exported runtime rate | 2,326 fps |
| headless slider loop mean | 1.14 ms/frame |
| headless slider loop max | 7.26 ms/frame |

This proves the exported-artifact inference path and records a frame-dump "viewer"
loop under `out/engine-runtime/viewer_frames/`. It does **not** yet satisfy the full
E6 benchmark matrix: no 128²/256²/512² sweep, no MPS measurement, and no GUI slider.

## Environment-light inverse recovery (extensions E4, SH slice)

`EnvironmentLight` uses degree-2 spherical harmonics over escaped segments
(`seg_tmax = inf`). Because reference GATHERLIGHT is linear in the 9 RGB coefficient
triples, `nrp.environment_fit` builds an explicit design matrix and solves the
inverse problem with least squares.

Measured by `mise run environment-fit` (report: `out/environment-fit/report.json`) on
a deterministic 48-pixel escaped-ray fixture:

| check | result |
|---|---:|
| equations / unknowns | 144 / 27 |
| least-squares rank | 27 |
| relative coefficient error | 1.26e-15 |
| reconstruction max abs error | 1.67e-15 |
| reconstruction PSNR | 305.22 dB |

This satisfies the E4 inverse-recovery criterion for the SH environment slice
(< 10% relative error) and gives future torch inverse/proxy work a closed-form
reference. It does **not** yet satisfy the full richer-light study: textured quad
inverse recovery, proxy conditioning on richer light parameters, and held-out proxy
PSNR vs texture resolution {2×2, 4×4, 8×8} are still open.

## Image-space target to physical lights (extensions E7, toy demo slice)

`mise run generative-loop` runs two toy-scale inverse-lighting workflows through the
existing `optimize_lights` mask/protect machinery and writes
`out/generative/report.json` plus target, mask, and realized-GATHERLIGHT images.

The synthesized scribble fixture is initialized at the known hidden light and exists
to verify the mask/protect accounting path:

| scribble check | result |
|---|---:|
| masked-region PSNR | 155.01 dB |
| protected-region MSE vs base | 1.5e-17 |
| passes E7 thresholds | true |
| wall-clock | 740 ms |

The stylized/generative target is a deliberately non-physical edit optimized from
three restarts at pixel fraction 0.25 for 20 steps each:

| restart | proxy loss first | proxy loss last | GATHERLIGHT PSNR vs target |
|---:|---:|---:|---:|
| 0 | 1.323 | 0.710 | 14.13 dB |
| 1 | 1.750 | 1.123 | 13.28 dB |
| 2 | 1.799 | 1.376 | 12.19 dB |

Best physical re-render PSNR is **14.13 dB** and the protected-region MSE is
**0.0051**. The raw stylized image cannot be exactly realized by one physical sphere
light at this budget; that physical-realization gap is the useful finding for this
slice. This does **not** yet satisfy the full E7 product demo: no trained high-quality
proxy, no committed hand-authored/generative image provenance, and no latency sweep
over pixel fractions {1.0, 0.25, 0.05}.
