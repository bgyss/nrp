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

## Denoising

OIDN (RT, HDR, albedo+normal guides) on a flat-2.0 HDR signal with σ=0.5 noise:
MSE 0.248 → 0.0054 (**46×**), mean 1.98 (HDR preserved). The bilateral fallback on
the same fixture achieves ~2× (it is a much weaker prior — expected). Pool build cost
with OIDN at 48×48: 0.35 s for 64 images (~5 ms/image).
