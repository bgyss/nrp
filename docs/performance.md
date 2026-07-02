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
  network was supervised with).
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
