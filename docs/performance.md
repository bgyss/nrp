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
| Country Kitchen — T1 scene | `nrp/mitsuba_exporter.py` (wavefront) | 512×512 | 64 | 4 (RR) | 52,264,226 | 2,092.6 MB |

Export cost: toy trace ~seconds; Mitsuba cornell box **1.6 s** scalar; the kitchen at
128×128 / 64 spp exports in **4.0 s** with the wavefront loop (would be ~55 s scalar
at the measured scalar throughput), and at 512×512 / 64 spp (the T1 scene of
`docs/production-track.md`) in **27.2 s** for 52.3M segments — a ~15-minute job at
scalar throughput. The kitchen scene is downloaded on demand by
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
| torch, kitchen 512² + OIDN (T1) | 409,189 | 854 s (+44 s pool) | **21.01 dB** (20.56 vs denoised), SSIM 0.30, ꟻLIP 0.146 | 0.88 | T1 scene end to end; 118.6 ms/frame (8.4 Hz) CPU inference at 512² |

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
- The T1 run (`examples/kitchen_512_torch.json`, report in
  `out/kitchen-512-torch/torch_train_report.json`) trains the same architecture
  (wider hashgrid, 409k params, 1.6 MB) end to end on the 52.3M-segment 512² cache:
  21.01 dB / SSIM 0.30 / ꟻLIP 0.146 held out, 854 s train wall-clock on CPU. One
  validation light (a small emitter placed nearly inside geometry) scores −5.3 dB
  and drags the means; the median light is ~24.1 dB. Per-light metrics are in the
  report. SSIM/FLIP are newly recorded here for T3's perceptual gates.
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
runtime path.

The animated-camera half now has a toy baseline in `mise run time-camera`
(`out/time-camera/report.json`). The toy tracer accepts a camera-origin override and
the script traces **K=3** camera keyframe caches on a ±0.04 x-axis camera move at
20×20, 8 spp, 2 bounces. Those keyframe caches contain 6,400 segments each and occupy
**798,180 bytes total** (**266,060 bytes/keyframe**). A simple image-space linear
interpolator between keyframe GATHERLIGHT frames reaches **27.30 dB** at held-out
t=0.25 and **25.98 dB** at held-out t=0.75, for **26.64 dB mean** versus directly
traced intermediate-camera GATHERLIGHT.

This is deliberately not claimed as the completed E1 time-conditioned proxy: it
measures camera-cache K scaling and establishes a held-out interpolation baseline.

### Time-conditioned TorchNRP proxy (closes E1's held-out criterion)

`examples/time_conditioned_proxy.py` (`mise run time-conditioned-proxy`, report:
`out/time-camera/proxy_report.json`) trains one `TorchNRP` model on the same K=3
camera keyframes, with the light-parameter input extended by one scalar (normalized
time) so a single set of weights covers all keyframes rather than interpolating
their images post hoc. Each keyframe's own G-buffer aux (camera-pose-dependent)
supervises training; held-out intermediate times are evaluated against a *freshly
traced* cache at that exact camera pose (real ground truth, not an interpolation of
existing frames).

On the same 20×20/8spp/2-bounce/±0.04 setup, 800 iterations, seed 0:

| check | result |
|---|---:|
| mean training-keyframe PSNR | 28.74 dB |
| mean held-out intermediate PSNR (t=0.25, 0.75) | 26.68 dB |
| held-out − train gap | **2.06 dB** |
| within 3 dB criterion | **met** |
| mean full-frame inference latency | 0.77 ms/frame |

This closes E1's outstanding criterion at toy scale: a single neural proxy
conditioned on time generalizes to unseen intermediate camera poses within the 3 dB
target. The caveat is scale — K=3 keyframes and a ±0.04 camera range are a small
interpolation problem; whether the gap holds at larger K or camera-motion range is
untested.

## Out-of-core foundation (extensions E5, sharded cache + streamed-optimizer slice)

`PathCache.save_sharded` writes a tile-sharded cache directory with a manifest and one
compressed `.npz` per pixel tile; `PathCache.load_sharded` reconstructs the monolithic
cache while restoring original segment order. `nrp.torch_backend.relight --tile-pixels`
and `relight_tiled` render proxy outputs in bounded pixel chunks. The out-of-core
report also streams shard files directly to build a fixed-light supervised target
table and to train a tiny per-pixel image proxy optimizer without reconstructing the
full segment arrays.

Measured by `mise run out-of-core` (report: `out/out-of-core/report.json`) on a
24×24 / 8 spp toy cache with 9,216 segments:

| check | result |
|---|---:|
| monolithic cache size | 381,005 bytes |
| sharded cache directory size | 423,777 bytes |
| save sharded | 0.023 s |
| load sharded | 0.010 s |
| sharded vs monolithic GATHERLIGHT max diff | 0.0 |
| streamed target max diff vs monolithic | 3.33e-16 |
| streamed target PSNR vs monolithic | 334.67 dB |
| streamed peak loaded segments | 1,024 (11.1% of cache) |
| streamed peak loaded segment bytes | 90,112 (11.1% of resident segment bytes) |
| estimated resident segment-memory ratio | 9.0× |
| process RSS before/after stream | 209,829,888 / 209,829,888 bytes |
| streamed target pass | 0.0063 s |
| streamed optimizer loss | 8.11e-3 → 1.24e-7 |
| streamed optimizer max diff vs monolithic optimizer | 3.33e-16 |
| streamed optimizer PSNR vs monolithic optimizer | 333.14 dB |
| tiled vs untiled proxy max diff | 0.0 |

The sharded layout is larger at this tiny scale because each tile pays `.npz` and
manifest overhead; that is expected and is not a production-scale memory claim. This
slice now proves exact cache reconstruction, streamed target construction at toy
scale, a streamed optimizer loop with the same result as its monolithic counterpart,
and chunked inference.

### Streamed TorchNRP pool training (`nrp.torch_backend.streamed_train`)

The remaining E5 optimizer gap — "the actual TorchNRP optimizer still trains from
monolithic caches" — is now closed at toy scale. `StreamedImagePool` renders each
pool slot's GATHERLIGHT target (sphere lights) by visiting shard tiles one at a time
instead of holding a `TorchPathCache`'s full segment arrays resident; only the
per-pixel G-buffer (albedo/depth/normal, small) stays resident. `train_streamed`
reuses the exact same rng draw order (numpy `default_rng` for light sampling, a
`torch.Generator` seeded identically for pixel/pool-index sampling) as the in-memory
training loop, so loss curves are directly comparable iteration-for-iteration rather
than just statistically close.

Measured by `mise run streamed-torchnrp` (`examples/streamed_torchnrp_train.py`,
report: `out/out-of-core/streamed_torchnrp_report.json`) on a 20×20 / 8 spp toy cache
(6,400 segments, 6×6 tiles), 200 iterations, seed 0:

| check | monolithic | streamed |
|---|---:|---:|
| loss (first → last) | 0.7002 → 0.7636 | 0.7002 → 0.7636 (bitwise) |
| held-out PSNR vs raw GATHERLIGHT | 9.12 dB | 9.12 dB |
| resident/peak segment bytes | 563,200 (full cache) | 50,688 |

- PSNR gap: 0.00 dB (criterion: within 0.3 dB) — the two loss curves are bit-identical
  at this toy scale because both paths compute the same targets from the same
  segments in the same order.
- Resident segment-memory ratio: 11.1× lower peak segment bytes for the streamed
  path.

This closes E5's TorchNRP-optimizer criterion at toy scale.

### 512×512 / 128 spp Mitsuba end-to-end report (closes E5's last criterion)

`examples/mitsuba_512_streamed.py` (`mise run export-mitsuba-512` then
`mise run mitsuba-512-streamed`, report: `out/out-of-core/mitsuba_512_report.json`)
exports a 512×512/128spp/4-bounce Mitsuba cornell-box cache, shards it, trains the
same streamed TorchNRP sphere-light proxy at this resolution, and measures tiled
full-frame inference — the production-scale "does it scale to film frames" datapoint
E5 asks for:

| check | result |
|---|---:|
| segments | 94,237,448 |
| monolithic cache size | 3.35 GB |
| monolithic resident segment bytes | 8.29 GB |
| sharded cache size (128×128 tiles) | 3.50 GB |
| save-sharded wall-clock | 306.3 s |
| streamed pool build | 382.2 s |
| streamed train (150 iters) | 621.7 s |
| streamed total | 1,004.9 s (~16.7 min) |
| peak segment bytes loaded (streamed) | 574.1 MB |
| resident segment-memory ratio | **14.4×** lower than monolithic |
| loss (first → last) | 4.15 → 0.0014 |
| held-out PSNR vs raw GATHERLIGHT | **32.57 dB** |
| tiled full-frame inference | 118.3 ms (16,384-pixel tiles) |

This is a genuine configuration that would not fit the naive in-memory loader at a
reasonable memory budget: the monolithic cache needs 8.29 GB of resident segment
arrays, while the streamed path never holds more than 574 MB (14.4×  less) at any
point during pool building or training, at the cost of much more wall-clock time
(shard I/O and repeated `np.add.at` gather passes over ~5.9M-segment shards dominate
— this is a scalar-numpy-loop implementation, not a fused kernel). The held-out PSNR
(32.57 dB) is well above the toy-scale runs (~9–14 dB) because 128 spp at 512×512
supplies far more segments per pixel. This closes E5's last open criterion; the
caveat is that streaming cost is currently I/O- and Python-loop-bound rather than
compute-bound, which is an engineering distance (a batched/vectorized streaming
gather, analogous to `TorchPathCache`, would close it), not a structural blocker.

### T2 real-scene packed streaming (Country Kitchen, 512×512 / 64 spp)

`mise run t2` is the complete reproducible workflow: it measures full and reduced
exports, then `mise run t2-streaming` re-proves the E5 machinery on the T1 Country Kitchen cache
with 64×64 tiles and the paper's packed fp16 geometry + shared-exponent rgb9e5
throughput. Report: `out/t2-streaming/report.json`. The run used CPU, seed 0, an
eight-image raw target pool, 300 iterations, and an explicit **8 GiB peak-RSS
budget**. Each phase runs in a fresh process so its `ru_maxrss` is attributable.
Measurements are from an Apple M1 Max (10 CPU cores, 64 GiB RAM, macOS 26.6).
The exporter writes its own configuration/timing/RSS/hardware JSON via `--report`;
the two source reports are `out/t2-streaming/export_512x512_64spp.json` and
`out/t2-streaming/export_128x128_64spp.json`.

| check | monolithic | packed streamed |
|---|---:|---:|
| on-disk cache | 2.09 GB | 630.20 MB (3.32× smaller) |
| training peak RSS | **9.07 GB / 8.45 GiB (over budget)** | **0.85 GB / 0.80 GiB** |
| pool build | 8.37 s | 37.18 s |
| total train phase | 35.62 s | 100.18 s |
| held-out PSNR vs packed GATHERLIGHT | 21.873 dB | 21.873 dB |
| PSNR gap (criterion ≤ 0.1 dB) | — | **0.000 dB, pass** |
| 512² tiled inference | 235.5 ms | 224.3 ms |
| inference peak RSS | — | 1.19 GB / 1.11 GiB |

The actual 512²/64-spp wavefront SAMPLEPATHS export took 182.4 s end to end
(20.4 s trace plus serialization) and peaked at **10.33 GB / 9.62 GiB**, over budget.
Following T2's required fallback, a 128²/64-spp scalar export establishes a passing
ceiling at 3.26 GB / 3.04 GiB peak RSS (63.0 s end to end; 52.5 s trace). The
intermediate 512²/48-spp measurement was not available, so the bracket is reported
rather than inferred. Packed-shard conversion of the full 512² T1 cache took 115.0 s
and peaked at 8.33 GB / 7.76 GiB, under the
8 GiB budget. The largest decoded shard is 79.1 MB. Streamed supervision processed
segments at **11.24 million segments/s**. Across six fixed fresh lights, packed vs
float64 GATHERLIGHT fidelity is 59.72 dB mean and **44.90 dB minimum**. This is the
committed cache-size/quality curve: 2.09 GB float64 is the reference point; the
630.20 MB packed point retains ≥44.90 dB image fidelity.

The full monolithic comparison was run (rather than silently subsampled) and exceeded
the budget by 0.45 GiB. The packed streamed training and inference phases remain far
inside it, so T2's production path succeeds while documenting the monolithic ceiling.
The cost is a 4.4× slower pool build from scalar shard visits; this is the next
optimization target. T2 therefore records an export ceiling rather than claiming the
full T1 export fits; packed streaming keeps the subsequent full-resolution phases
within budget.

## Dynamic geometry cache invalidation (extensions E2, one-bounce slice)

`nrp.dynamic_geometry` identifies pixels whose first-hit G-buffer changes after the
toy sphere moves and splices fresh segments for only those pixels into the old cache.
This is the primary-visibility slice of E2: it proves cache-level invalidation and
segment replacement for one-bounce paths, not secondary-bounce transport updates or
TorchNRP weight fine-tuning. It also includes a deliberately narrow image-space
warm-start proxy to measure whether incremental cache targets can repair stale
pixels quickly.

Measured by `mise run dynamic-geometry` (report:
`out/dynamic-geometry/report.json`) on a 32×32 / 8 spp / 10-frame toy sequence with
the sphere moving +0.16 along x:

| check | result |
|---|---:|
| mean full retrace | 4.45 ms/frame |
| mean splice pass | 0.71 ms/frame |
| mean image-proxy fine-tune | 0.30 ms/frame |
| mean invalid pixels | 29.7% |
| mean retraced segment fraction | 29.7% |
| full retrace share of 16 ms frame | 27.8% |
| splice-only share of 16 ms frame | 4.4% |
| splice + image-proxy share of 16 ms frame | 6.3% |
| outside invalidation mask max diff | 0.0 |
| incremental splice vs full retrace | exact (`inf` PSNR, 0 max abs) |
| stale cache PSNR at final frame | 10.76 dB |
| image-proxy PSNR after fine-tune vs full | 65.86 dB min / 69.69 dB mean |

For this constrained one-bounce setup, keeping the cache alive costs a small fraction
of a 16 ms game frame for splice plus a tiny warm-start repair baseline, and the
stale-cache baseline falls quickly as the sphere moves. The proxy measurement is not
the paper proxy: it stores one RGB value per pixel and updates only invalidated
pixels, so it is evidence about incremental target availability rather than neural
transport generalization. The important caveat is structural: once secondary bounces
are enabled, unchanged first-hit pixels can still see changed indirect transport.
Multi-bounce invalidation is now implemented and measured (below); warm-started
TorchNRP weight fine-tuning has also been implemented and measured, with a negative
result.

### Multi-bounce invalidation: swept-volume masking (closes E2's structural gap)

`nrp.dynamic_geometry.swept_volume_invalidation_mask` generalizes primary-visibility
invalidation to indirect bounces: it flags any pixel with *any* cached segment, at
*any* bounce depth, whose ray could pass through the moving object's swept volume
(a conservative bounding sphere — object radius plus half the travel distance,
centered at the midpoint of the before/after positions — proven in
`test_swept_bounding_sphere_contains_endpoints_and_object_extent` to contain the
true capsule-shaped swept region). Primary-visibility invalidation only inspects each
path's *first* segment, so it misses pixels whose indirect illumination changed
because the moving object altered transport between two other, still-visible
surfaces without ever occluding either of their camera rays.

`examples/dynamic_geometry.py::multi_bounce_invalidation_comparison` reproduces this
failure and shows the fix, on a 32×32/8spp/2-bounce before/after pair (sphere moves
+0.12 along x, `mise run dynamic-geometry`, field `multi_bounce_invalidation` in
`out/dynamic-geometry/report.json`):

| mask | invalid pixels | max abs diff vs full retrace | PSNR vs full retrace |
|---|---:|---:|---:|
| primary-visibility only | 362 | 0.366 | 33.43 dB |
| primary + swept-volume | 896 | 0.0 | ∞ (exact) |

Primary-only invalidation misses 534 pixels whose indirect bounces changed (a real,
measured failure — not a hypothetical), producing visible error (33.43 dB, not
exact) after splicing. Adding the swept-volume mask catches those pixels and
recovers an exact match to a full retrace, same as the one-bounce case. Test
coverage: `tests/test_dynamic_geometry.py::test_swept_volume_mask_flags_pixels_with_any_bounce_depth_segment_in_region`
and `::test_multi_bounce_spliced_cache_matches_full_retrace_with_swept_mask` (the
latter is the 2-bounce analogue of the existing exact one-bounce splice test). The
cost is over-invalidation: the swept-volume mask is conservative (any segment
*potentially* affected is invalidated, not only segments *actually* re-occluded), so
it invalidates more pixels than the true minimal set — a correctness-first choice
appropriate for a cache-invalidation scheme, not a performance-first one.

### TorchNRP weight fine-tune regimes (honest negative result)

`TorchNRPWarmStartProxy` (`examples/dynamic_geometry.py`) wraps a real `TorchNRP`
model (pixel xy + G-buffer aux → RGB, single fixed light) and runs the three regimes
E2 specifies: (a) full retrace — full retrain (300 Adam iterations) on every frame's
fully-retraced cache, warm-started from the previous frame's weights; (b) incremental
— warm-started fine-tune (same 300 iterations) restricted to only the invalidated
pixels, using the spliced cache's target; (c) stale — the frozen weights from before
that frame's update, no training at all. All three are measured against the fully
retraced frame's GATHERLIGHT image.

Measured by `mise run dynamic-geometry` on the same 32×32 / 8 spp / 10-frame sequence:

| regime | mean PSNR vs full | mean ms/frame |
|---|---:|---:|
| (a) full retrace + full retrain | 44.88 dB | 207.1 |
| (b) incremental splice + masked-pixel fine-tune | 25.20 dB | 94.3 |
| (b2) incremental + self-distillation replay | 33.76 dB | 166.9 |
| (c) stale (no update) | 22.13 dB | — |

- Gap (a) − (b): **19.7 dB**, far outside the ≥ criterion of "within 1 dB." E2's
  explicit ask is an honest result either way — this is the negative one.
- Interpretation: fine-tuning only on the invalidated pixels' loss lets the shared
  MLP weights drift away from a good fit on the *unchanged* pixels (there is no
  regularizer holding the rest of the image steady), so regime (b) is barely better
  than doing nothing (regime (c), 22.13 dB) despite costing real per-frame compute.

### Replay-regularized retry (regime b2): partial fix, not sufficient

`TorchNRPWarmStartProxy.fine_tune_with_replay` tests the follow-up question directly:
does mixing a sample of unchanged-pixel targets into the incremental fine-tune batch
close the gap? Since a real incremental-update pipeline has no extra ground truth for
unchanged pixels, replay uses **self-distillation** — the model's own predictions on
unmasked pixels, captured before that frame's update — as the regularization target,
mixed 1:1 with the real invalidated-pixel supervision every batch.

- Gap (a) − (b2): **11.1 dB**, down from 19.7 dB for plain incremental fine-tuning —
  a real, substantial improvement (`replay_closes_the_gap: true` in the report,
  meaning the gap shrank by more than 0.5 dB), but still **far outside the 1 dB
  criterion** (`regime_b2_within_1db_of_a: false`).
- Cost: replay roughly doubles fine-tune wall-clock (166.9 ms vs 94.3 ms/frame) since
  every step now runs two forward/backward passes (invalidated + replay batches).
- Conclusion: self-distillation replay is a real, worthwhile mitigation (8.5 dB
  recovered) but not a fix. The remaining gap is consistent with genuine capacity/
  optimization limits of segment-local fine-tuning on a shared small MLP, not simply
  "forgetting" that a naive regularizer can patch. A production system would likely
  need either a larger reserved-capacity architecture (e.g. per-region adapter
  weights) or accept that dynamic geometry forces periodic full retrains rather than
  purely incremental updates.
- Test coverage: `tests/test_dynamic_geometry.py::test_torchnrp_warm_start_fine_tune_only_touches_masked_pixels`,
  `::test_torchnrp_warm_start_no_op_on_empty_mask`,
  `::test_torchnrp_fine_tune_with_replay_runs_and_produces_finite_loss`,
  `::test_torchnrp_fine_tune_with_replay_no_op_on_empty_mask`.

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

`mise run light-aware-proxy-ab` adds a toy standard-vs-guided proxy A/B with a small
geometric open-top-box occluder around the declared light-placement region (report:
`out/light-aware-proxy-ab/report.json`). Both runs use 20×20 / 8 spp caches,
identical segment budgets, the same 2,743-parameter sphere proxy, and 350 CPU
training iterations:

| check | standard | guided |
|---|---:|---:|
| region-hit fraction | 0.47% | 15.73% |
| mean held-out validation PSNR | 9.71 dB | 9.59 dB |
| fixed in-region light PSNR | -6.72 dB | 3.38 dB |
| fixed open-region light PSNR | 7.35 dB | 16.44 dB |
| train time | 0.44 s | 0.51 s |
| in-region target met | false | true |
| open-region regression within 0.5 dB | false | true |

The guided proxy improves the fixed in-region light by **10.10 dB** and the
open-region fixture by **9.09 dB** at equal cache size. This satisfies E3's proxy A/B
margin target on a deterministic open-top-box occluder fixture: standard sampling puts
only 0.47% of segments through the declared region, while guided sampling puts 15.73%
there. The remaining caveat is scale and realism, not the existence of the failure
reproduction: the fixture is a toy geometric box, not a full lampshade asset or
production lighting scene.

## Quality-tier relight ladder (extensions E9, CLI plumbing slice)

`nrp.torch_backend.relight` now accepts `--quality preview|draft|final` and writes a
metadata sidecar next to every `.npy` output. `preview` is the proxy output, `draft`
is GATHERLIGHT from the current cache, and `final` is GATHERLIGHT from `--final-cache`
when supplied. `--residual-light` applies the E9 cached residual
`GATHERLIGHT(approved) - proxy(approved)` to preview output.

Measured by `mise run quality-tiers` (report: `out/quality/report.json`) on a
16×16 toy cache, using an 8-spp draft cache and a 32-spp final cache:

SSIM and FLIP are computed after the repo's documented display preprocessing:
Reinhard tonemap plus sRGB encoding.

| tier | source | ms | PSNR vs final | SSIM vs final | FLIP vs final |
|---|---|---:|---:|---:|---:|
| preview | proxy | 27.55 | -14.11 dB | 0.023 | 0.954 |
| draft | cached GATHERLIGHT | 0.16 | 10.81 dB | 0.190 | 0.166 |
| final | high-spp GATHERLIGHT | 0.39 | inf | 1.000 | 0.000 |

The proxy in this report is intentionally untrained; the measurement is for the
quality-tier plumbing, sidecar metadata, and residual identity rather than visual
quality. At the approved light config, proxy plus cached residual matches cached
GATHERLIGHT to max absolute error **5.6e-17**. Moving the light by dx =
{0.05, 0.10, 0.20} drops residual-corrected PSNR vs cached GATHERLIGHT to
{24.67, 23.73, 20.40} dB, which is the expected residual-validity decay.

The report now includes a toy-scale supervisor-trust verdict with a 1e-12 residual
identity tolerance and a 25 dB residual-validity threshold. The approved frame is
exact, but the first measured move (dx=0.05) falls below 25 dB, so the verdict is:
**trust the approved frame only; re-bake residual after any measured light move**.
### Production-scale trust verdict (closes E9's last criterion)

`examples/quality_tiers_production.py` (`mise run quality-tiers-production`, report:
`out/quality/production_report.json`) reuses the E5 512×512 Mitsuba cornell-box
caches: a 32spp "export" cache (tiers 1/2/4) and the 128spp cache from the E5 report
(tier 3, converged reference). The proxy is trained via the streamed pipeline
(`nrp.torch_backend.streamed_train`) on the 32spp cache's shards — a real trained
proxy, not the intentionally-untrained one in the toy report above.

| tier | source | ms | PSNR vs final (128spp) | SSIM vs final | FLIP vs final |
|---|---|---:|---:|---:|---:|
| preview | proxy | 86.9 | 33.76 dB | 0.9944 | 0.0426 |
| draft | cached GATHERLIGHT (32spp) | 2,021.7 | 35.72 dB | 0.9965 | 0.0087 |
| final | GATHERLIGHT (128spp, fresh) | 9,589.7 | inf | 1.000 | 0.000 |

Unlike the toy report, this proxy is trained and predicts within 2 dB of the export-spp
cache itself, both close to the converged 128spp reference — 32spp cornell-box noise
is low enough that the gap between draft and final is small at this scene's
complexity. At the approved light config, proxy plus cached residual matches cached
GATHERLIGHT (32spp) to max absolute error **0.0** (exact, same identity as the toy
report). Moving the light by dx = {5, 10, 20} world units (this scene's units, not
normalized) drops residual-corrected PSNR vs cached GATHERLIGHT to
**{-13.67, -18.46, -25.68} dB** — a much sharper decay than the toy report's
{24.67, 23.73, 20.40} dB, because the moved-light re-renders here use the same
proxy without re-training and the sphere radius (8 units) is large relative to the
tested dx range, so small moves already change occlusion substantially.

The production-scale supervisor-trust verdict (1e-12 identity tolerance, 25 dB
validity threshold): **trust the approved frame only; re-bake residual after any
measured light move** — the same qualitative verdict as the toy report, now backed
by a real trained proxy and a genuinely converged high-spp reference. This closes
E9's last open criterion. The caveat: this is one scene (cornell-box) and one light
family (sphere); the decay radius is scene- and light-scale-dependent (compare the
toy report's normalized-unit dx values to this report's world-unit dx values), so
the *qualitative* verdict (don't trust residual correction beyond the approval point)
generalizes more confidently than the *specific* decay numbers.

## Production light controls (extensions E8)

`gather_light_controlled` adds post-hoc controls at GATHERLIGHT time: per-pixel
first-hit exclusion for light linking, and a simple linear-distance artist
attenuation curve for sphere lights. This is the cache fallback path E8 asks us to
test, plus proxy-conditioned toy paths for controls whose output is linear in the
chosen control parameterization.

Measured by `mise run production-controls` (report:
`out/production-controls/report.json`) on a 32×32 / 12 spp toy cache with 24,576
segments:

| check | result |
|---|---:|
| full gather vs sphere+box layers max diff | 0.0 |
| exclude-sphere gather vs box-layer gather max diff | 0.0 |
| exclude-sphere gather vs box-layer PSNR | inf |
| full gather | 0.63 ms |
| linked gather | 0.79 ms |
| attenuated gather | 0.85 ms |
| attenuated/default mean-radiance ratio | 0.918 |

The linking algebra is exact for the toy layer partition: excluding the sphere-owned
pixels from the full cache leaves the box-layer contribution for that light. The
attenuation curve used here is `max(0, 1 - 0.1 * distance(origin, light_center))`,
validated by a closed-form unit fixture.

The report also includes a tiny binary table proxy conditioned on the linking toggle:
it stores the inactive and active linked images, so it has the same 6,144 parameters
as a two-proxy table at this resolution. The active and inactive predictions match
their GATHERLIGHT references exactly (`inf` PSNR, 0 max abs). Active-toggle prediction
takes **0.010 ms**, an **81×** speedup over linked gather in this toy run. This proves
that a precomputed binary control can stay live at proxy speed.

The same report now includes a learned least-squares image proxy conditioned on the
continuous attenuation controls `(intercept, slope)` for the fixed
`linear_distance` curve family. Trained on four control settings, it predicts the
held-out setting `(1.1, -0.1)` to **333.65 dB** PSNR versus GATHERLIGHT with
**1.11e-16** max absolute error. Prediction takes **0.015 ms** versus **0.98 ms**
for held-out attenuated gather, a **64×** speedup. This proves that one fixed
continuous control family can be conditioned and kept live without cache traversal.

The E8 report now also measures broader conditioned-control fixtures:

| conditioned control | held-out error | edit/inference latency | comparison |
|---|---:|---:|---:|
| 4-basis soft link mask | 331.25 dB PSNR, 5.55e-17 max abs | 0.036 ms | exact vs image-space mask application |
| quadratic distance attenuation | 323.22 dB PSNR, 1.94e-16 max abs | 0.010 ms | 158× faster than polynomial gather |

These results answer E8 at toy scale: binary linking, soft mask-basis controls, and
linear/quadratic attenuation curves can stay live through a conditioned image proxy
when the proxy receives the relevant control weights as inputs. The caveat is now
parameterization scale, not correctness: a fully free-form per-pixel mask or arbitrary
artist curve would require enough input dimensions and training coverage to span that
control space.

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
| exported runtime latency | 0.46 ms/frame |
| exported runtime rate | 2,153 fps |
| headless slider loop mean | 1.19 ms/frame |
| headless slider loop max | 6.72 ms/frame |

The same report includes an exported-runtime device sweep using valid synthetic
G-buffer caches to isolate full-frame inference cost from path tracing. This
machine's PyTorch build now has a working MPS backend (`torch.backends.mps.is_built()
== True`), so the MPS row is a real measurement, not the earlier "unavailable"
placeholder:

| device | resolution | available? | ms/frame | fps | 30 fps? | 60 fps? |
|---|---:|---|---:|---:|---|---|
| CPU | 128×128 | yes | 4.74 | 210.9 | yes | yes |
| CPU | 256×256 | yes | 14.67 | 68.1 | yes | yes |
| CPU | 512×512 | yes | 33.40 | 29.9 | no | no |
| MPS | 128×128 | yes | 106.10 | 9.4 | no | no |
| MPS | 256×256 | yes | 28.61 | 35.0 | yes | no |
| MPS | 512×512 | yes | 38.15 | 26.2 | no | no |

MPS is *slower* than CPU at 128×128 (fixed per-dispatch overhead dominates at this
tiny model/resolution) and only pulls ahead of CPU at 512×512, where it still misses
30 fps. Neither backend hits 30 fps at 512×512 for this exported artifact on this
hardware — state that plainly rather than rounding up. This proves the
exported-artifact inference path and records a frame-dump "viewer" loop under
`out/engine-runtime/viewer_frames/`.

### A real GUI slider, and a step toward the WebGPU/engine-backend criterion

`examples/export_js_viewer.py` (`mise run js-viewer`, output:
`out/engine-runtime/js_viewer/`) closes the "no GUI slider" gap and takes a concrete
step toward "structure the exporter so a WebGPU/engine port is a backend, not a
rewrite": it trains a small proxy with `use_encoding=False` (so the forward pass is
plain `Linear -> ReLU -> ... -> Linear -> softplus`, no hashgrid), extracts the raw
weight matrices, and writes a **self-contained HTML page** (`viewer.html`, no
external requests, no build step) that re-implements that exact forward pass in
~40 lines of vanilla JavaScript. Four range-input sliders (light center x/y/z,
radius) drive real interactive re-rendering: every slider move recomputes the full
image through the JS forward pass and redraws a canvas — a genuine GUI, unlike the
headless frame-dump loop above (this environment has no display or GUI toolkit
available for a native app, so a browser page is the honest choice here).

Parity between the JS and PyTorch forward passes (the E6 "matches the PyTorch
module" criterion, now for a non-TorchScript backend) is verified two ways: the
page itself runs a JS-side comparison against an embedded Python-computed reference
image at load time, and `tests/test_export_js_viewer.py::test_js_forward_pass_matches_python_reference`
runs the same JS logic under Node (skips cleanly if `node` is absent) and asserts
max abs diff < 1e-4. Measured max abs diff on this machine: **1.0e-7**.

### A real WebGPU compute-shader backend (closes E6's last criterion)

`webgpu/bench_browser.mjs` (`mise run webgpu-bench`, report:
`out/engine-runtime/webgpu_browser_report.json`) is a real WGSL compute shader
replicating the exact forward pass, executed inside real Google Chrome (via
Playwright) against the **actual exported proxy's weights** — not a browser demo
with synthetic data, the real trained model:

| check | result |
|---|---:|
| parity vs. PyTorch reference (max abs diff) | 2.4e-7 |
| adapter | Apple M1 Max, Metal 3 (real hardware) |
| 128×128 latency | ~1.7 ms/frame (580 fps) |
| 256×256 latency | ~2.8–4.0 ms/frame (250–350 fps) |
| 512×512 latency | ~9.4 ms/frame (107 fps) |

All three resolutions clear both 30 fps and 60 fps — a substantially better result
than the TorchScript CPU/MPS matrix above (which misses 30 fps at 512×512 on both
backends). Test coverage: `tests/test_webgpu_browser_bench.py` (end-to-end,
spawns real Chrome; skips cleanly if Playwright/Chrome aren't installed).

The path here is worth recording: an earlier attempt, `webgpu/bench.mjs`, used
native Dawn bindings (the `webgpu` npm package) to avoid needing a browser at all,
and reproducibly segfaulted when given the real exported proxy's weights — 100%
failure across 60+ trials, while a synthetic-weight smoke test of the identical
pipeline succeeded reliably (8/8). A ~150-trial bisection (full table in
`webgpu/README.md`) narrowed this to a defect in that specific native binding's
handling of real trained-model float32 data (ruled out: magnitude, distribution,
buffer size, memory layout, three independent package versions). Running the
byte-for-byte identical shader inside a production WebGPU implementation (Chrome's)
instead of the experimental Node-only binding resolved it completely — confirming
the diagnosis and delivering the actual criterion. `webgpu/bench.mjs` and
`webgpu/smoke.mjs` are kept as a documented negative result illustrating that
diagnosis; `bench_browser.mjs` is the working, measured backend.

This uses the same exported model as the JS/canvas viewer (`use_encoding=False`, so
the forward pass is portable without a hashgrid port — one remaining ablation
shared by both browser-backend demos, not new to this one). A genuine WebGPU port
of the paper's exact hashgrid-encoded architecture (`HashEncoding2D` in WGSL)
remains a separate, larger future step; what closes here is the E6 criterion as
stated — a real, executed, measured WebGPU compute-shader backend running the
actual exported proxy.

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
reference.

`mise run textured-quad-fit` adds the corresponding reference inverse-recovery check
for `TexturedQuadLight` (report: `out/textured-quad-fit/report.json`). For fixed quad
geometry and nearest-neighbor texture lookup, GATHERLIGHT is linear in the RGB texel
values, so `nrp.texture_fit` builds one design matrix per color channel and solves
least squares. The synthetic fixture gives each texel independent observations.

| texture | RGB params | rank per channel | relative texture error | reconstruction PSNR |
|---:|---:|---:|---:|---:|
| 2×2 | 12 | 4 / 4 | 2.34e-16 | 319.62 dB |
| 4×4 | 48 | 16 / 16 | 1.72e-16 | 324.19 dB |
| 8×8 | 192 | 64 / 64 | 1.96e-16 | 322.09 dB |

This satisfies textured-quad reference inverse recovery at the requested 8×8
resolution and records the parameter-count growth from 12 to 192 RGB values.

The same report now includes a linear texture-proxy scaling baseline: each texture
size is fitted from the same 48 random quad-hit observations and evaluated on a
separate 256-observation held-out cache. This isolates the parameter-count pressure
behind the paper's warning that richer light parameterizations hurt convergence.

| texture | RGB params | train obs | rank / unknowns per channel | held-out PSNR |
|---:|---:|---:|---:|---:|
| 2×2 | 12 | 48 | 4 / 4 | 319.61 dB |
| 4×4 | 48 | 48 | 16 / 16 | 320.47 dB |
| 8×8 | 192 | 48 | 32 / 64 | 10.66 dB |

The 8×8 case becomes underdetermined at equal observation budget and collapses on
held-out samples.

The report now adds a compact learned texture-embedding proxy: a torch MLP encodes
each flattened RGB texture into an 8D embedding, then predicts textured-quad
GATHERLIGHT observations from UV, segment throughput, and that embedding. Each row
uses 24 training textures, 6 held-out textures, 96 observations per training texture,
256 held-out observations, and 600 CPU optimization steps:

| texture | RGB params | proxy params | mean held-out PSNR | min held-out PSNR |
|---:|---:|---:|---:|---:|
| 2×2 | 12 | 4,187 | 20.08 dB | 19.05 dB |
| 4×4 | 48 | 5,915 | 21.19 dB | 20.07 dB |
| 8×8 | 192 | 12,827 | 22.27 dB | 17.62 dB |

This satisfies a learned texture-embedding proxy-scaling slice for E4 and shows that
the learned proxy avoids the catastrophic 8×8 collapse of the equal-observation
linear baseline on this synthetic texture bank.

The same run also verifies first-class `TexturedQuadLight` support in the main
`TorchNRP` train/relight path for a small 2×2 texture parameterization:

| check | result |
|---|---:|
| light type | `textured_quad` |
| light parameter dimension | 20 |
| model parameters | 2,005 |
| iterations | 120 |
| validation PSNR vs raw | 13.27 dB |
| held-out relight PSNR | 12.31 dB |

This completes the E4 toy-scale evidence loop: reference inverse recovery, a
texture-resolution proxy scaling table, and a first-class TorchNRP train/relight path
for textured-quad lights. The remaining caveat is quality and scale rather than API
coverage: the first-class TorchNRP smoke uses 2×2 textures, while the broader learned
scaling table remains an experiment harness rather than a production model.

## Image-space target to physical lights (extensions E7, toy demo slice)

`mise run generative-loop` runs two toy-scale inverse-lighting workflows through the
existing `optimize_lights` mask/protect machinery and writes
`out/generative/report.json` plus target, mask, realized-GATHERLIGHT images, and
`out/generative/provenance.json`.

The proxy is now pretrained on random sphere lights before being used for inversion
(`pretrain_proxy` in `examples/generative_loop.py`, 24 random lights, 800 Adam steps)
— closing the "high-quality proxy run" gap. Previously `optimize()` differentiated
through a randomly initialized network that had never seen the scene, which is not a
meaningful test of the paper's actual pipeline (train a proxy, then invert through
it). Windowed mean loss (first vs last 10% of iterations, not single-minibatch
values, per this repo's testing convention) drops from **0.330 to 0.169**.

The synthesized scribble fixture is initialized at the known hidden light and exists
to verify the mask/protect accounting path:

| scribble check | result |
|---|---:|
| masked-region PSNR | 155.01 dB |
| protected-region MSE vs base | 1.5e-17 |
| passes E7 thresholds | true |
| wall-clock | 2.3 ms |

The stylized/generative target is a deliberately non-physical edit optimized from
three restarts at pixel fraction 0.25 for 20 steps each, now through the pretrained
proxy:

| restart | proxy loss first | proxy loss last | GATHERLIGHT PSNR vs target |
|---:|---:|---:|---:|
| 0 | 1.055 | 0.238 | 13.19 dB |
| 1 | 0.382 | 0.157 | 12.43 dB |
| 2 | 3.329 | 2.343 | 11.53 dB |

Best physical re-render PSNR is **13.19 dB** and the protected-region MSE is
**0.0052** — comparable to the pre-pretraining numbers (14.13 dB), since a single
sphere light's expressiveness, not proxy quality, is the binding constraint here.
The raw stylized image cannot be exactly realized by one physical sphere light at
this budget; that physical-realization gap is the useful finding for this slice.

The same report includes the required pixel-fraction latency sweep on the stylized
target, using 10 optimization steps per fraction:

| pixel fraction | wall-clock | ms/step | final proxy loss | GATHERLIGHT PSNR |
|---:|---:|---:|---:|---:|
| 1.0 | 13.4 ms | 1.34 | 0.245 | 13.16 dB |
| 0.25 | 11.5 ms | 1.15 | 3.032 | 10.43 dB |
| 0.05 | 10.8 ms | — | — | — |

At this 14×14 toy scale, optimizer overhead dominates, so lowering the pixel fraction
does not reduce wall-clock linearly.

The report includes committed provenance for the current toy fixtures:
`out/generative/provenance.json` records the deterministic repo-local generation
method, the known-light scribble recipe, the stylized target multipliers, optimizer
settings, proxy-pretrain configuration, and SHA-256 hashes for six generated `.npy`
files. It explicitly records that no external image model, editor, or asset was used
for this slice. This closes the provenance gap and the "high-quality proxy run" gap
for the synthetic/stylized toy targets.

### A genuinely hand-authored target (closes E7's last gap)

`examples/hand_authored_target.py` (`mise run hand-authored-target`, report:
`out/generative/hand_authored_report.json`) is different in kind from the fixtures
above: those are *derived* (the scribble is GATHERLIGHT from a known light; the
"stylized" target is a deterministic filter applied to a render). This target has no
reference to any render, light, or scene geometry at all — it is an explicit,
hand-picked list of `(row, col, RGB)` strokes (a small pixel-art plus-sign with an
accent dot) typed directly into `hand_authored_strokes()`. The coordinates and
colors *are* the authored content, satisfying the extension's "creating them in any
paint tool is fine" bar — the deliverable is authored pixel content, not the
specific tool used to author it.

The same pipeline as the other E7 workflows applies: pretrain a proxy (24 random
lights, 800 steps), then optimize 2 physical sphere lights against the hand-authored
target through objective/protect masks, 3 restarts:

| check | result |
|---|---:|
| best restart GATHERLIGHT-tonemapped MSE | 0.0262 |
| best restart proxy-space PSNR vs target | 10.24 dB |
| target vs. best realized-GATHERLIGHT PSNR | 12.56 dB |
| protected-corner region MSE vs base | 0.0052 |

As with the stylized target, a hand-drawn plus-sign cannot be exactly realized by
two physical sphere lights — the same physical-realization-gap finding, now against
content with no algorithmic connection to the physics at all.
`out/generative/hand_authored_provenance.json` records the full stroke list and sets
`hand_authored: true`, `derived_from_render: false`, distinct from the other
fixtures' `hand_authored: false`. Test coverage:
`tests/test_hand_authored_target.py` (4 tests: stroke bounds, render correctness,
provenance flags, end-to-end CLI report).

This closes E7's last open criterion at toy scale.

## Perceptual quality gates (production track, rung T3)

`nrp.quality.gate` promotes SSIM/FLIP/PSNR from ablation tooling to named pass/fail
gates (preview / draft / final thresholds, repo conventions embedded in every gate
result) with a CLI that gates image pairs or re-emits existing report JSONs with
`quality_gate` verdicts attached. Two existing reports were re-emitted with
conclusions unchanged: the E9 toy quality ladder (`mise run quality-tiers`, gate now
built in — untrained preview and 8 spp draft fail their tiers honestly; the final
tier and the residual-identity approval frame pass at final tier) and the E10
ablation report (`out/ablation/report_gated.json` — every smoke-scale cell fails the
preview SSIM bar, consistent with that report's stated scale limits). A deliberately
degraded render failing the gate is unit-tested (`tests/test_quality_gate.py`,
19 tests).

Gate evaluation overhead at 512×512, measured on the T1 kitchen cache/model
(`mise run gate-overhead`, `out/quality/gate_overhead.json`, Apple M1 Max CPU,
min over 3 repeats):

| quantity | seconds |
|---|---:|
| preview render (proxy inference) | 0.120 |
| draft render (cached GATHERLIGHT, 52.3 M segments) | 1.711 |
| final-tier reference render (same-cache gather) | 1.672 |
| gate evaluation (tonemap + PSNR + SSIM + FLIP) | 0.146 |

A gate consumes a rendered pair (gated image + reference), so overhead is
gate / (gated render + reference render): **4.3%** for the E9 approval flow (draft
gated against a final reference) — under the 5% target, and conservative because
the same-cache reference understates true final-tier cost. Stated plainly rather
than rounded away: against a *single* draft render the ratio is 8.5%, and the
preview proxy render (0.120 s) is faster than the gate itself, so per-frame gating
of interactive previews needs cheaper metrics (a G2/T4 concern). Getting here
required a real optimization: `nrp.metrics` separable convolutions now use
`sliding_window_view` + matvec instead of per-tap Python sums (identical results to
≤1e-15; full gate evaluation 0.22 s → 0.15 s, SSIM/FLIP tests unchanged).

## Runtime baseline lock (production track, rung T4)

T4 locks the browser-WebGPU runtime floor on the real T1 scene, upgrading the E6
result (toy proxy, `use_encoding=False`, mean-only timing) in three ways: it runs
the **actual T1 kitchen proxy** (409 k parameters: 10-level hashgrid + 4×128 MLP,
`out/kitchen-512-torch/model.pt`) with the hashgrid encoding ported to WGSL — the
step the E6 section explicitly deferred; it feeds the real exported 512² G-buffer
rather than synthetic aux; and it reports a frame-time histogram (p95, not just
mean) under a per-frame jittered light uniform, the interactive-relight access
pattern.

Pipeline: `mise run t4-export` (`examples/export_webgpu_runtime.py`) dumps the
proxy — MLP weights, hashgrid tables, per-level metadata — plus xy/aux G-buffer and
a PyTorch reference image as flat float32 blobs (`out/t4-runtime/export/`), with a
numpy reimplementation of the exported forward pass self-checked against the torch
module at export time (max abs diff 1.6e-6; unit-tested in
`tests/test_export_webgpu_runtime.py` including hashed-vs-dense table levels).
`mise run t4-bench` (`webgpu/bench_t4.mjs`) runs the WGSL port in real Chrome via
Playwright (same harness as E6), 200 timed frames per resolution after 10 warmup
frames; timing covers uniform upload, dispatch, and `onSubmittedWorkDone` — no
per-frame readback, weights/tables/G-buffer resident.

Apple M1 Max (Metal 3), Chrome; parity vs the PyTorch reference at 512²:
**1.2e-6 max abs diff** (tolerance 2e-4):

| resolution | mean | p50 | p95 | min–max | fps (mean) | fps (p95) | 30 fps p95 | 60 fps p95 |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 128² | 2.21 ms | 2.10 ms | 2.80 ms | 1.9–5.8 ms | 452 | 357 | pass | pass |
| 256² | 5.93 ms | 5.60 ms | 7.40 ms | 5.4–9.6 ms | 169 | 135 | pass | pass |
| 512² | 20.96 ms | 20.50 ms | 23.30 ms | 19.5–28.3 ms | 48 | 43 | **pass** | fail |

The T4 floor — 30 fps at 512² sustained, p95-verified — **holds** on the real
scene (p95 23.3 ms < 33.3 ms), with ~30% headroom. 60 fps holds through 256²
but not at 512²; recorded as-is, the floor is 30.

Getting under the floor required real shader work, recorded because the failed
variants are instructive: a naive per-thread scalar loop over the storage-buffer
weights ran at 219 ms p95 at 512² (4.6 fps, ~7× over the floor); rewriting it with
vec4 dot products halved that (132 ms); workgroup-shared weight tiles alone changed
nothing (108 ms) because the true bottleneck was dynamically-indexed function-space
arrays (activations/accumulators) spilling to device memory. The shipped shader
generates fully-unrolled constant-index accumulator statements (registers), reads
each activation once per 16-vec4 output block, and streams transposed weights
through a ≤16 KiB workgroup tile: 20.9 ms mean — ~10× the naive version, all
parity-identical.

The lock: `out/t4-runtime/baseline.json` (committed) freezes per-resolution
mean/p95 plus the thresholds; `mise run t4-check` re-runs the bench and **fails**
if parity exceeds 2e-4, any resolution's p95 regresses more than 30% over the
frozen baseline, or 512² misses the 30 fps p95 floor outright. The same check runs
as `tests/test_export_webgpu_runtime.py::T4BaselineCheckTests` (spawns real
Chrome; skips cleanly without node/Playwright/export artifacts, repo convention).
Full per-frame timings live in `out/t4-runtime/report.json` (also committed).

## Dynamic geometry, second attempt (production track, rung G1)

E2's settled negative result was that segment-local TorchNRP weight fine-tuning
misses its 1 dB recovery target by 11–20 dB even with replay regularization. G1
tests a changed hypothesis on E2's *exact* fixture (32×32 / 8 spp / 10 frames,
sphere moving +0.16 along x, same light, same 300-iteration / lr 5e-3 per-frame
budget, warm-started per frame like every E2 regime): mark invalidated cache
shards with E2's swept-volume + primary-visibility masks aggregated to an 8×8
tile grid, keep the base proxy **frozen**, and train a signed-output residual
MLP (`nrp.torch_backend.residual_dynamic.ResidualNRP`, linear head — a residual
must go negative) over only the invalidated region, composited additively at
inference. Outside the region the composite equals the frozen base bitwise
(unit-proven), so E2's failure mode is structurally impossible.

Measured by `mise run g1-residual` (report: `out/g1-residual/report.json`,
committed), one script producing all five regimes; E2's regimes reproduce their
published numbers exactly:

| regime | mean PSNR vs full | gap vs (a) | within 1 dB | mean ms/frame |
|---|---:|---:|---|---:|
| (a) full retrace + full retrain | 44.88 dB | — | — | 141.8 |
| (b) E2 masked-pixel fine-tune | 25.20 dB | 19.7 dB | no | 69.8 |
| (b2) E2 fine-tune + replay | 33.76 dB | 11.1 dB | no | 121.1 |
| (c) stale (no update) | 22.13 dB | 22.8 dB | no | — |
| (d) G1 frozen base + shard residual | **37.15 dB** | **7.7 dB** | **no** | 84.5 |

**The letter of the target is still not met** — but the failure mode is
categorically different from E2's, which is what the rung asked to establish:

- Regime (b) fails by *global forgetting*: its out-of-mask PSNR averages
  22.95 dB (the fine-tune degrades pixels the geometry change never touched).
  Regime (d)'s out-of-mask PSNR averages **54.90 dB** and cannot drift by
  construction.
- Per-frame floor: (d) never drops below **25.3 dB**; (b) collapses to 10.6 dB
  (worse than doing nothing) and (a) itself oscillates down to 16.4 dB mid-
  sequence. At the per-frame **median**, (d) is 1.2 dB *better* than (a)
  (median gap −1.17 dB) and beats it outright on 6 of 10 frames.
- The 7.7 dB mean gap is dominated by the two near-static frames (dx ≤ 0.018)
  where (a) retrains on an almost-unchanged target and scores 110+ dB while
  (d) is capped at ~58 dB by the frozen base's own fit. G1's residual failure
  mode is therefore *in-region underfit against an overfit reference on
  degenerate frames*, not forgetting.
- Cost: invalidate-and-recover (mask + splice + spliced GATHERLIGHT + residual
  train) averages 84.5 ms/frame vs 141.8 ms for full retrace + retrain — 0.60×,
  at matched optimization budget.

T1-scene feasibility: not run — the Mitsuba exporter has no scene-edit/retrace
path for a moved object (it records paths for a fixed scene), so invalidation
targets cannot be produced for the kitchen; the toy fixture is kept because the
rung requires the apples-to-apples comparison against E2's numbers. Unit
coverage: `tests/test_residual_dynamic.py` (shard-grid aggregation incl.
non-divisible edges, signed head, bitwise outside-region compositing, windowed
loss decrease, save/load round-trip, report-loop smoke).

## Summit demo: live relight + controls in the browser (production track, rung G2)

The T1 kitchen proxy relit live in real Chrome (WebGPU, Apple M1 Max / metal-3):
`webgpu/demo/` is an interactive viewer — animated lights in the E1 keyframe
format, the two E8 production controls (per-layer light linking via the exported
first-hit layer mask, 10.1% of pixels; a first-hit linear-distance artist
attenuation curve) applied in-shader, an emission tint (Eq. 3), and — G1's
deliverable — a toy-scale moving-object panel compositing the frozen base proxy
with per-frame signed residuals next to the full-retrace GATHERLIGHT reference.
`mise run g2-serve` opens it; `mise run g2-demo` replays the committed
interaction trace (`webgpu/demo/trace.json`) headlessly and writes the evidence.

**The proxy had to be retrained to clear the gate.** T1's 3000-iteration model
(21.0 dB mean val PSNR) tops out at SSIM ≈ 0.78 against OIDN-denoised references
on interior lights, and its only *raw*-reference pass region is degenerate — a
light enclosing the camera, rendering a near-constant image (std 0.006). A
15k-iteration cosine-schedule run of the same architecture
(`examples/kitchen_512_torch_g2.json`, 61 min CPU, report:
`out/kitchen-512-torch-g2/torch_train_report.json`) lifts mean val PSNR to
29.4 dB and clears preview tier across the whole committed light path (dense
17-point pre-verification sweep plus control states: 32–37.5 dB / SSIM
0.881–0.906 / FLIP ≤ 0.075).

Measured by `mise run g2-demo` (report: `out/g2-demo/report.json`, committed;
recording: `out/g2-demo/recording.webm`, committed) — 481 timed frames at 512²
under the full interaction timeline (light animation + linking toggle +
attenuation + tint, plus the G1 panel's two extra dispatches and three canvas
presents per frame):

| check | result |
|---|---:|
| frame time mean / p50 / p95 / max | 26.7 / 27.2 / 30.7 / 35.4 ms |
| fps p95 (criterion: p95 ≤ 33 ms) | **32.6 fps — passes** |
| G1-panel GPU-vs-torch composite parity | 4.8e-7 max abs (all 10 frames) |
| preview-tier gate (12 sampled trace frames) | **12/12 pass** |
| gate metrics vs OIDN-denoised gather | 32.1–37.3 dB, SSIM ≥ 0.883, FLIP ≤ 0.075 |
| same frames vs raw 64-spp gather | SSIM 0.297–0.438 (noise-bound; PSNR ≥ 30.7 dB) |
| gate evaluation per frame | 0.15 s |

The per-frame gate (`mise run g2-gate`, report: `out/g2-demo/gate.json`,
committed) scores the browser's actual output buffers against GATHERLIGHT
references carrying the *identical* pixel-level control algebra — linking zeroes
the same layer-mask pixels, attenuation applies the same first-hit curve, tint
folds into emission — so the gate measures proxy fidelity, not control
approximation. The reference is the **OIDN-denoised** gather, the class the
proxy is supervised on (§4.4): the raw 64-spp gather's Monte Carlo noise bounds
SSIM at ~0.2–0.44 on these frames regardless of proxy quality (recorded per
frame as `raw_reference_metrics`), while linear-HDR PSNR — noise-robust — stays
≥ 30.7 dB against the raw reference too. Honest caveats, stated plainly: the
committed light path was *authored inside* the proxy's verified pass region
(radius ≈ 0.28–0.37 interior lights among the training distribution); arbitrary
light positions do not gate at preview tier (T1's per-light variance table
already showed this), and the linking/attenuation controls are the pixel-level
(first-hit) control family — E8's own gather-time linking algebra, but not a
per-segment attenuation. The moving object ships at G1's toy scale with the
scale labeled in the UI, since G1's result is toy-scale and the exporter has no
scene-edit retrace path for the kitchen.

Frame-time context: the earlier identical workload measured 23.1/24.2 ms
(mean/p95) on a quieter machine; the committed run carried ~5 load average of
unrelated processes. Both hold the 33 ms criterion; the T4 single-proxy bench
(no canvas, no G1 panel) remains the tighter runtime baseline. Integration
coverage: `tests/test_g2_demo.py` (real-Chrome render + G1 parity, skips
without the browser toolchain), `tests/test_export_webgpu_demo.py`,
`tests/test_g2_gate.py`.

## Shot harness with temporal stability (production track, rung F1)

A 120-frame keyframed-light shot on the T1 kitchen — the G2-verified interior
orbit (`examples/f1_shot_kitchen.json`; authored inside the proxy's verified
pass region, same caveat as G2) — through the E9 tier ladder with a per-frame
T3 trust verdict. Tier mapping on this single-cache scene: preview = proxy
inference, draft = raw cached GATHERLIGHT, final = OIDN-denoised GATHERLIGHT
(the supervision-class reference, §4.4). `mise run f1-shot` under the nix
devshell (report: `out/f1-shot/report.json`, committed; harness:
`nrp/torch_backend/shot.py`, unit tests: `tests/test_shot.py`).

Per-frame trust verdict at preview tier: **120/120 frames pass**
(31.6–37.8 dB / SSIM 0.881–0.898 / FLIP 0.065–0.081 vs the denoised
reference). Raw-reference metrics recorded per frame as in G2: vs the
un-denoised 64-spp gather the same frames score 30.4–34.5 dB but SSIM
0.298–0.356 — the MC-noise bound, not proxy quality.

Temporal stability — the metric the ladder lacked: frame-to-frame FLIP between
consecutive tonemapped frames, checked as *excess over the reference
sequence's own per-pair deltas* (the light moves; flicker is change the
reference doesn't have). Named repo convention: excess_max = 0.02 per pair.

| sequence | FLIP delta mean / p95 / max | verdict |
|---|---:|---|
| final-tier reference | 0.0155 / 0.0184 / 0.0188 | (baseline for excess) |
| proxy (preview) | 0.0150 / 0.0189 / 0.0191 | **pass** — worst excess 0.0019; mean excess −0.0004 (the proxy is slightly *smoother* than the denoised reference) |
| flickering baseline (reference + per-frame independent noise, 30.0 dB/frame) | 0.1681 / 0.1789 / 0.1816 | **fails** — worst excess 0.165; per-frame PSNR alone cannot see this |

Per-tier render time per frame at 512² (ms, mean / p95, Apple M1 Max, CPU
torch gather): preview 231 / 336, draft 722 / 854, final 806 / 933 (final =
draft gather + OIDN). Gate + temporal metrics run on top of that per frame;
the 120-frame shot completed in ~4 min wall-clock end to end.

## Summit demo: final-tier shot (production track, rung F2)

The F1 shot rendered at final tier with residual-identity frames: `mise run
f2-shot` under the nix devshell (report: `out/f2-shot/report.json`; video:
`out/f2-shot/shot.mp4`, 120 frames at 512², 24 fps, h264 crf 18 — both
committed; script: `examples/f2_final_shot.py`, unit tests:
`tests/test_f2_final_shot.py`). Per frame the pipeline stores what production
would store: the shared proxy plus an fp16 compressed residual against the
final-tier reference (OIDN-denoised GATHERLIGHT — single-cache scene, the
supervision-class reference); the *stored* reconstruction is gated at T3 final
tier per frame, so the fp16 quantization is the only error source being gated.

| check | result |
|---|---:|
| final-tier gate (40 dB / SSIM 0.98 / FLIP 0.02) | **120/120 frames pass** (≥ 105.9 dB, SSIM ≥ 0.9999999, FLIP ≤ 7.0e-5); flagged: none |
| exact residual identity (float64) | ≤ 1.1e-19 max abs, all frames |
| fp16-stored reconstruction error | ≤ 4.9e-4 max abs (tolerance 1.4e-3 = 1e-3 × max residual) |
| shot wall-clock (cache reuse) | 185.3 s total; 1.54 s/frame mean, 1.90 p95 |
| re-render-every-frame estimate | 21 887 s (120 × 182.39 s measured T2 export wall-clock) — **118× amortization** |
| storage: proxy + residuals vs raw frames | 165.7 MiB vs 142.0 MiB (**1.17× — residuals cost *more***, fp16 npz both sides) |
| MP4 | 0.22 MiB |

The storage row is an honest negative: at this proxy quality the per-frame
residual is dominated by reference noise/detail at fp16 mantissa scale and
compresses *worse* than the frames themselves, so proxy+residual storage only
wins if residuals are kept for a subset of frames (approval frames) or
quantized more aggressively — the win here is the 118× wall-clock
amortization, not bytes. Honest caveats: the final tier is the denoised 64-spp
gather, not a higher-spp ground truth (the kitchen ships one cache; the T2
report records the export ceiling); the residual-identity gate is exact *by
construction* in float64 — the per-frame check verifies the stored fp16
artifact, which is what a pipeline would actually keep; the amortization
figure is an estimate built on the measured T2 export wall-clock, not a
120-frame re-export. Hardware: Apple M1 Max, CPU torch gather + OIDN.

## Production light rig (production track, rung V1)

An 8-light rig on the T1 kitchen scene, one independent per-light
`TorchNRP` proxy per light, composited with `nrp.torch_backend.rig.LightRig`
(scaling the E1/roadmap per-layer-compositing machinery from 2 layers to N
lights): `mise run v1-rig` (report: `out/v1-rig/report.json`; harness:
`nrp/torch_backend/rig.py`; unit tests: `tests/test_rig.py`). Each light was
trained at a reduced 800-iteration budget (vs T1's 3000) to fit N=8
independent proxies in the rung's time budget.

Rig composition — 3 `SphereLight` + 3 `QuadLight` + 2 `TexturedQuadLight`
(8×8 texel emission grids):

| light | type | role |
|---|---|---|
| key | sphere | primary sphere key light |
| fill | sphere | cool fill sphere |
| rim | sphere | warm rim/kicker sphere |
| window | quad | ceiling-adjacent window panel |
| ceiling_panel | quad | overhead panel |
| practical | quad | warm practical wall fixture |
| neon_sign | textured_quad | checkerboard-textured emitter |
| tv_glow | textured_quad | smooth radial-gradient textured emitter |

Per-light training (800 iters each, val PSNR/SSIM/FLIP vs the raw
gather-lights mean for that light in isolation):

| light | type | train wall-clock (s) | val PSNR (dB) | val SSIM | val FLIP | params | model size |
|---|---|---:|---:|---:|---:|---:|---:|
| key | sphere | 219.2 | 20.59 | 0.217 | 0.205 | 409,189 | 1.57 MiB |
| fill | sphere | 228.5 | 19.39 | 0.221 | 0.232 | 409,189 | 1.57 MiB |
| rim | sphere | 202.7 | 17.04 | 0.248 | 0.180 | 409,189 | 1.57 MiB |
| window | quad | 201.3 | 20.38 | 0.238 | 0.268 | 409,701 | 1.57 MiB |
| ceiling_panel | quad | 207.6 | 20.33 | 0.264 | 0.253 | 409,701 | 1.57 MiB |
| practical | quad | 215.8 | 22.35 | 0.416 | 0.204 | 409,701 | 1.57 MiB |
| neon_sign | textured_quad | 713.4 | 12.35 | 0.264 | 0.562 | 434,277 | 1.66 MiB |
| tv_glow | textured_quad | 760.8 | 13.35 | 0.242 | 0.541 | 434,277 | 1.66 MiB |

The two textured-quad lights train ~3.4× slower per iteration than the
sphere/quad lights (the 8×8 texture adds a per-texel emission lookup to the
GATHERLIGHT target) and land at the lowest val PSNR of the rig (12.3–13.4 dB)
at this reduced 800-iteration budget — expected given the harder,
higher-frequency target and the same iteration count as the simpler light
types.

**Val PSNR does not mean genuine proxy quality for all 8 lights.** The 3
`QuadLight` proxies (`window`, `ceiling_panel`, `practical`) score 20–22 dB
val PSNR above — on par with or better than the spheres — but produce
*exactly zero* raw output on this cache (verified in the V2 section below via
`out/v2-artloop/report.json`'s `colorable_light_raw_output_magnitude`); their
PSNR is a zero-output proxy scoring well against a near-dark target, not
learned quality. The two `TexturedQuadLight` proxies (12.3–13.4 dB, lowest in
the rig) are nonzero-output and genuinely contributing, making them — not the
quads — the real low-quality drivers among this rig's per-light proxies. See
the V2 section's "Zero-gradient caveat" for the full investigation.

**Additivity gate — honest negative.** The rig's composited render
(`LightRig.render`, sum of the 8 trained per-light proxies) was checked
against `gather_lights` for the full active rig at T3's preview tier (PSNR ≥
20 dB, SSIM ≥ 0.80, FLIP ≤ 0.15). The reference image matters here: an
earlier draft of this check compared the rig sum against the **raw**
(un-denoised) GATHERLIGHT render, which bounds SSIM well below any
proxy-quality ceiling from MC noise alone (the same phenomenon documented in
`nrp/torch_backend/shot.py`'s module docstring) — raw-reference SSIM was
0.117, uninformative about proxy quality. Fixed (commit `e1af7fb`) by
denoising the reference to match what each per-light proxy was actually
supervised against (`denoise_method="oidn"`), matching the pool-training
target class of §4.4:

| reference | PSNR (dB) | SSIM | FLIP |
|---|---:|---:|---:|
| raw (un-denoised) gather_lights — pre-fix, uninformative | 25.44 | 0.117 | 0.360 |
| OIDN-denoised gather_lights — post-fix (matches training target) | **25.61** | **0.622** | **0.352** |

Post-fix verdict at preview tier: PSNR **passes** (25.6 ≥ 20 dB), SSIM and
FLIP both **fail** (0.622 < 0.80; 0.352 > 0.15) — `"fail at preview tier:
ssim, flip"`. The methodology fix mattered (SSIM moved from 0.117 to 0.622,
an order-of-magnitude-meaningful jump once MC noise stopped dominating the
metric) but the underlying result is a genuine preview-tier failure, not an
artifact of a broken evaluation.

**Correction:** an earlier draft of this paragraph attributed the SSIM/FLIP
failure to "the reduced 800-iteration-per-light training budget... those
per-light errors compound additively across all 8 lights." That causal claim
is wrong and has been withdrawn. The V2 section below (art-direction loop)
diagnoses the actual mechanism on this same rig: 3 of the 8 per-light proxies
— `window`, `ceiling_panel`, `practical`, all `QuadLight` — produce exactly
zero raw output on this cache (`colorable_light_raw_output_magnitude` in
`out/v2-artloop/report.json`, `mean: 0.0, max: 0.0` for all three), while
`rim` (a `SphereLight`) has the *worst* val PSNR of the rig at 17.04 dB and
still contributes nonzero output — ruling out iteration-count/PSNR as the
explanation for the additivity error. See the V2 section's "Zero-gradient
caveat" for the full investigation (the root cause of the `QuadLight`
zero-output pattern itself is not yet diagnosed). The additivity-relevant
quality story for V1's rig is instead the two genuinely-contributing but
lowest-scoring proxies, the `TexturedQuadLight`s `neon_sign` (12.35 dB) and
`tv_glow` (13.35 dB) — nonzero output, low val PSNR — which are the real
drivers of whatever proxy-quality-driven component the additivity error has.
Effectively, the "8-light" composite has only 5 genuinely-contributing
per-light proxies on this cache (3 spheres + 2 textured quads), not 8; the
SSIM/FLIP failure is still reported as-is, per the shared convention that
honest negatives are deliverables, not something to spin as a pass.

**Sizes: per-light rig vs monolithic baseline.** A monolithic (non-relightable)
baseline was trained on the combined 8-light image with a matched total
iteration budget (6400 iters = 8 × 800):

| | bytes | MiB |
|---|---:|---:|
| per-light rig (8 proxies, total) | 13,350,808 | 12.73 |
| monolithic baseline (1 proxy) | 553,013 | 0.53 |
| **ratio (rig / monolithic)** | | **24.1×** |

The per-light rig is ~24.1× larger in total than the single non-relightable
baseline — the cost of per-light relightability (solo/mute, independent
per-light edits) versus one proxy that only reproduces the fixed 8-light
composite.

**Compositing overhead vs active light count.** Wall-clock to render the
composited rig at 512² as active lights are added one at a time (1→8),
measured by `LightRig.render`:

| n_lights | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 141.2 | 300.0 | 327.2 | 446.1 | 555.4 | 683.2 | 797.2 | 950.0 |

Roughly linear in light count: least-squares fit ≈ 111.0 ms/light + 25.5 ms
intercept (measured directly from `out/v1-rig/report.json`'s
`compositing_overhead_ms` — do not use any other document's copy of this
number).

Hardware: macOS-26.6-arm64-arm-64bit, CPU torch (8 threads).

## Summit demo: art-direction loop (production track, rung V2)

E7's image-target optimization driving the full V1 rig: `mise run v2-artloop`
(report: `out/v2-artloop/report.json`; harness: `nrp/torch_backend/art_loop.py`;
unit tests: `tests/test_art_loop.py`). A hand-authored target changed the 6
colorable V1 lights' rgb (`optimize_colors`, 500 Adam steps, lr 0.05); the 2
`TexturedQuadLight` lights (`neon_sign`, `tv_glow`) have no color concept —
their emission is baked into a fixed texture — and were excluded from the
optimization entirely.

| light | V1 (neutral guess) rgb | art-direction target rgb | recovered rgb | genuinely recovered? |
|---|---|---|---|---|
| key | [1.0, 1.0, 1.0] | [4.5, 3.5, 2.5] | [4.500, 3.500, 2.500] | **yes** — gradient recovery |
| fill | [1.0, 1.0, 1.0] | [0.6, 0.8, 1.8] | [0.600, 0.800, 1.800] | **yes** — gradient recovery |
| rim | [1.0, 1.0, 1.0] | [0.8, 1.2, 2.5] | [0.800, 1.200, 2.500] | **yes** — gradient recovery |
| window | [1.0, 1.0, 1.0] | [2.5, 2.0, 1.2] | [1.0, 1.0, 1.0] | **no** — zero-gradient proxy, see caveat |
| ceiling_panel | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | [1.0, 1.0, 1.0] | **no** — zero-gradient proxy, see caveat |
| practical | [1.0, 1.0, 1.0] | [3.5, 2.0, 0.5] | [1.0, 1.0, 1.0] | **no** — zero-gradient proxy, see caveat |

**Zero-gradient caveat — read before citing the convergence gate below.**
`colorable_light_raw_output_magnitude` in the report shows `window`,
`ceiling_panel`, and `practical` each produce exactly zero raw proxy output on
this cache (`mean: 0.0, max: 0.0`), inherited from their V1 per-light
training. The root cause is not the reduced 800-iteration training budget:
all 6 colorable lights trained for the identical 800 iterations, and `rim`
(a `SphereLight`, one of the 3 genuinely-recovered lights below) has the
*worst* val PSNR of all 6 at 17.04 dB — worse than all 3 zero-output lights
(`window` 20.38 dB, `ceiling_panel` 20.33 dB, `practical` 22.35 dB, per
`out/v1-rig/report.json`'s `per_light_training`), which rules out iteration
count as the explanation. The pattern the data does support: all 3
zero-output lights are `QuadLight`, all 3 nonzero-output lights are
`SphereLight` — a clean type correlation whose deeper cause (e.g. something
about how `QuadLight` inputs interact with this proxy/cache) is not yet
diagnosed. With zero raw
output, `u_rgb` receives zero gradient for these three lights throughout
`optimize_colors`, so their `recovered_rgbs` above are simply the untouched
neutral initial guess `[1.0, 1.0, 1.0]` — not a genuine recovery of the
hand-authored target, even though they happen to contribute nothing to the
rendered image either way. Only `key`, `fill`, and `rim` were genuinely,
verifiably recovered via gradient descent: their `recovered_rgbs` match
`art_direction_target_rgb` to five significant figures and their raw-output
magnitudes are nonzero (mean 0.001–0.021, max 0.34–1.30). **This is 3 of 6
colorable lights recovered, not "all 8 lights" or "all 6 colorable lights" —**
do not cite this result as unqualified convergence across the rig.

**Convergence gate (image-space, all 8 lights composited).** Because the
proxy-vs-target comparison is on the *composited render*, and the 3
zero-gradient lights contribute nothing to that render regardless of their
color, the overall image-space metric still reads as a clean pass — this is
"image-space-vacuous" for those three lights specifically, not evidence they
converged:

| tier | thresholds (PSNR / SSIM / FLIP) | measured | verdict |
|---|---|---|---|
| draft | ≥ 30 dB / ≥ 0.90 / ≤ 0.08 | 154.5 dB / 0.9999999999999992 / 8.4e-7 | **pass** (no fallback to preview tier needed) |

`reload_identical: true` — the recovered rig round-trips through
`LightRig.save`/`load` bit-identically.

**Interactive grading latency.** The headless slider loop (10 adjustments,
re-rendering the composited 512² rig per adjustment) measured
`latency_ms_mean` **950.1 ms**, `latency_ms_p95` **1011.4 ms** — this is the
per-adjustment interactive-grading-latency datapoint the rung asks for; it is
well above real-time (30 fps ≈ 33 ms) at this CPU-torch, 8-light, 512² scale.

**Wall-clock to convergence.** The full `optimize_colors` run (500 Adam
steps) took **1320.7 s (~22.0 min)** wall-clock. This is slower than a plan
assumption expected (the original Task 4 brief assumed full-batch rendering
at rig scale would be fast). A code review found an avoidable inefficiency,
not fixed here (it lives in already-approved Task 4 code, out of scope for
this task): the loop gives the 2 `TexturedQuadLight` lights (`neon_sign`,
`tv_glow`) a full proxy forward pass on every one of the 500 steps even
though their output has zero gradient with respect to `u_rgb` and is
otherwise constant across the optimization — roughly 2/8 of the per-step
compute is spent on lights that could have had their contribution hoisted
out of the loop as a one-time constant. The 1320.7 s figure should be read as
a measured cost with a known, unaddressed, avoidable-inefficiency root
cause — not as an inherent lower bound on this optimization's cost.

Hardware: macOS-26.6-arm64-arm-64bit, CPU torch (8 threads) — same machine/run
as the V1 section above.

## QuadLight zero-collapse diagnosis and fix (hardening track, rung H1)

V1/V2 established that 3 of the kitchen rig's 8 proxies (`window`,
`ceiling_panel`, `practical` — all `QuadLight`) produce exactly-zero raw
output at their authored parameters (`out/v1-rig/report.json`,
`out/v2-artloop/report.json`), leaving only 5/8 rig lights genuinely
contributing and making 3/6 of V2's "recovered" colors vacuous (the untouched
neutral guess, never actually gradient-updated). The 2026-07-11 audit
(`docs/status/2026-07-11.md`) ruled out training budget (`rim`, a working
`SphereLight`, has the *worst* val PSNR of the working lights) and confirmed
the supervision target itself is nonzero at the authored params (numpy
`gather_throughput_quad` gives mean 0.0028–0.0053, 13–30% of pixels lit — the
same order as `rim`'s 0.0082/27%), narrowing the failure to the training
loop. Full report and per-hypothesis evidence: `out/h1-quad-fix/report.json`.

**Discriminating experiment matrix** (all on `out/kitchen-512/path_cache.npz`,
the rig's real architecture: hidden_width 128, 4 layers, the paper's
hashgrid encoding, lr 0.005, ε 0.01, batch 8192, pool 64):

| variable changed | collapses within budget? | what it rules in/out |
|---|---|---|
| ε 0.01 → 1.0 (150 iters) | yes | not a denominator-magnitude effect |
| global grad-norm clip 1.0 (150 iters) | yes | not a single large-gradient spike — Adam already normalizes step size, so clipping the aggregated gradient after the fact changes nothing |
| quad normal restricted to a 15° cone around the authored normal (150 iters) | yes | not "rare bright outlier from unconstrained full-sphere normal sampling" — collapses even when the pool's brightest target is capped at 0.31 (vs 1.0 unrestricted) |
| lr 0.005 → 0.0005, 10× lower (150 iters) | **no** | LR magnitude is the actual lever |
| hidden ReLU → LeakyReLU(0.01), lr unchanged (150 iters) | yes | the dead zone is the **output softplus**, not hidden-layer dying ReLUs — a saturated final activation blocks gradient regardless of what the hidden layers do |
| LR warmup (100 of 800 iters), same final lr | delayed, not prevented (mean 1.4e-21 by iter 800) | warmup only postpones crossing the dead zone over a long enough budget, it doesn't remove the underlying drift |
| **sphere `rim` at the identical unmodified config** (150 iters) | **same drift, not yet fatal** (raw_pre_min −246 vs quad's −916; pred mean decays 2.6e-6 → 1.1e-33) | the mechanism is shared by both light types — quad crosses the dead zone within budget on this cache, sphere's shallower drift usually doesn't, which is why `rim` survives as the working rig's worst performer, not why quads are uniquely broken |
| output bias initialized to `inverse_softplus(pool target median)` instead of nn.Linear's default | **no**, for both quad and sphere, over the full 800-iter budget | the fix |

**Mechanism.** `nn.Linear`'s default bias init gives `softplus(~0) ≈ 0.69` at
step 0, regardless of scene — but this cache's true pool-target median is
~0.001–0.005, roughly 130–700× dimmer. Because most pool samples are near
that dim median, the relative-MSE loss (Eq. 4, stop-gradient-of-prediction
denominator) has a persistently negative-signed gradient for most steps.
Adam's per-step displacement is ≈lr *independent of raw gradient magnitude*
(the ε and grad-clip ablations above confirm this — changing gradient scale
by 100× or capping its norm does not change the drift trajectory), so this
one-directional signal marches the pre-softplus logit toward −∞ at a roughly
constant per-iteration rate. Once it passes float32's softplus-derivative-
underflow point (`sigmoid(z) → 0` for `z ≲ −100`), the chain rule multiplies
every upstream gradient by exactly 0 and the network is permanently frozen —
this happens for `window` within the first ~50 of 800 training iterations.
The failure is not quad-specific: `rim` (`sphere`) walks the identical path,
just more slowly, which is exactly why it is the rig's lowest-PSNR *working*
light — it is deep into the same decay curve without (yet) having crossed
the underflow threshold within its 800-iteration budget.

**Fix** (`nrp/torch_backend/model.py`): `TorchNRP.init_output_scale(scale)`
zeros the output layer's weight and sets its bias to
`inverse_softplus(scale)`, so training starts near the true target scale
instead of traversing several orders of magnitude under a one-directional
gradient. `nrp/torch_backend/train.py::train` calls it once, immediately
after the pool is built (non-resume path), with
`scale = pool.targets.mean(dim=-1).median()`. Measured overhead: **0.14 s**
(one median over the full 512² × 64-slot pool) against a ~320 s per-light
train — negligible. Unit tests: `tests/test_torch_backend.py`'s
`QuadZeroCollapseTests` pin a fast (≈4 s) synthetic reproduction shaped like
the kitchen pool's target distribution — one test asserts it still collapses
without the fix (documents the bug so a future refactor that accidentally
"fixes" the repro doesn't go unnoticed), the other asserts the fix prevents
it; `ModelTests` adds direct unit coverage of `init_output_scale` and
`inverse_softplus`.

**Verification on the real cache** (`out/h1-quad-fix/`, same 800-iter budget
and architecture as `examples/v1_rig.py`; **oidn was unavailable in this
shell** — not run under `nix develop` — so both the fixed quads and freshly
retrained spheres below use `denoise.method=bilateral`, an apples-to-apples
comparison with each other but not directly with `out/v1-rig/report.json`'s
oidn numbers):

| light | type | raw output mean | raw output max | val PSNR vs raw (dB) | train s (+pool s) |
|---|---|---|---|---|---|
| window | quad | 9.10e-3 | 1.418 | 15.51 | 324.6 (+61.9) |
| ceiling_panel | quad | 4.02e-3 | 0.553 | 15.52 | 330.3 (+61.8) |
| practical | quad | 1.82e-3 | 0.100 | 11.65 | 338.4 (+62.2) |
| key | sphere | 1.19e-2 | 1.216 | 16.22 | 319.0 (+60.2) |
| fill | sphere | 4.96e-2 | 2.827 | 15.98 | 328.0 (+61.8) |
| rim | sphere | 3.33e-3 | 0.707 | 12.07 | 314.5 (+64.2) |

All 3 previously-collapsed quads now produce nonzero raw output at their
authored parameters, at the order of magnitude `gather_throughput_quad`
predicts (§ above, 0.0028–0.0053 mean), and their val PSNR (11.65–15.52 dB)
falls in the same range as the freshly retrained spheres (12.07–16.22 dB) —
the systematic quad-vs-sphere gap is gone. Note the *pre-fix* `out/v1-rig/
report.json` actually reported deceptively high val PSNR for these same
three collapsed quads (20.33–22.35 dB): the fixed held-out validation lights
are drawn from the same `sample_light()` distribution as the training pool
(mostly near-black targets), so an all-zero predictor scores well against
mostly-dark targets. Per-light val PSNR alone cannot detect this collapse —
only a direct raw-output-magnitude probe at the light's own authored params
can, which is why H1's verify criterion is framed that way rather than as a
PSNR threshold. Training wall-clock is ~100 s higher across the board here
than the original oidn-denoised V1 run; per the 0.14 s fix-overhead
measurement above, that gap is attributable to bilateral vs. oidn denoising
in this shell, not the fix — H2 should re-measure under oidn (`nix develop`)
for a clean budget comparison.
