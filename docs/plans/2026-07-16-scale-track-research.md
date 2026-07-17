# Scale & Speed Track — Research and Design (2026-07-16)

Research backing `docs/scale-track.md` (phase 6). Written after the hardening
track closed (H1–H7, see `docs/tracks.md`); the H7 verdict revision in
`docs/pipeline-feasibility.md` is the authoritative statement of what remains
open. This document collects (1) the measured performance ceilings the next
cycle should attack, (2) the larger-scene ladder, (3) a design for a CUDA
port that works in the NVIDIA ecosystem (stretch goal), and (4) cloud-provider
research and cost estimates, since no local CUDA hardware exists.

## 1. Where the pipeline is slow today (measured, from `docs/performance.md`)

| bottleneck | measured today | why it's the ceiling |
|---|---|---|
| **Streamed pool build + train at 512²/128spp** | 1,004.9 s total (382 s pool build, 622 s for only 150 iters) on the Mitsuba cornell cache | Shard streaming is scalar-numpy: repeated `np.add.at` passes over ~5.9M-segment shards. The in-memory `TorchPathCache` batched gather already exists and is 5–7× faster on MPS at 128²–256² — it has never been applied per-shard. The E5 write-up itself calls this "an engineering distance, not a structural blocker." |
| **Shard save** | 306.3 s to write the 3.35 GB 512² cache as 128×128 tiles | Serial per-tile `np.savez_compressed`; embarrassingly parallel. |
| **Training throughput** | MPS only 1.6× CPU at paper scale (37.2 vs 23.2 iters/s, 50k iters = 22.4 min); MPS *slower* than CPU at toy scale | Small MLP + batch 4096 under-fills the GPU; per-iteration launch overhead dominates. Untested levers: larger batches, fp16/AMP, `torch.compile` (works on MPS since torch 2.4, maturity varies). |
| **Full-frame inference at ≥1024²** | 116 ms (8.6 Hz) MPS at 1024²; extrapolated ~230 ms (4.3 Hz) at 1080p | The performance doc already concludes: "The paper's production rates need CUDA + tiny-cuda-nn (fp16 tensor cores)." The WebGPU runtime is faster than TorchScript-MPS (107 fps at 512² in Chrome) but has never been benchmarked at 1024². |
| **Rig quality, not speed** | H2 additivity still fails preview tier (SSIM 0.725 < 0.80, FLIP 0.257 > 0.15) — measured *before* H3's kernel conditioning landed | The rig's two textured-quad proxies were the quality floor (12–13 dB) when the gate was last run; H3 has since fixed exactly that (19.64/19.99 dB). The obvious next experiment — rerun the additivity gate with kernel-conditioned textured quads in the rig — has not been done. |

Largest scene proven end to end: Country Kitchen at 512×512 / 64 spp
(52.3M segments, `out/kitchen-512/`); largest cache streamed: 512²/128spp
cornell (94.2M segments, 3.35 GB packed, 8.29 GB resident float64).
Nothing has been attempted at 1024² or on a second gallery scene.

**Deliberately out of scope for this track:** dynamic geometry. H5 confirmed
the structural blocker at real scale (neither fine-tune nor residual regimes
come within 14 dB of the 1 dB recovery target); attacking it again needs a new
idea, not more budget. It stays on the backlog, not in this track.

## 2. Larger-scene ladder

1. **1024×1024 kitchen** — 4× the pixels of anything trained so far; packed
   cache projected ~13–14 GB at 64 spp (linear in pixels from the 512² data
   point). Only feasible through the sharded/streamed path — which is exactly
   why the streamed-gather speedup (S1) must land first: the current scalar
   streaming loop would take hours per pool build at this size.
2. **A second, geometrically different gallery scene** (e.g. bedroom or
   bathroom from the same Bitterli/Mitsuba gallery the kitchen came from,
   via `examples/scenes/download_scene.py` conventions — assets never
   vendored). Every real-scene claim in the repo currently rests on one scene;
   a second scene tests whether the H1/H3 fixes and the 18–20 dB envelope are
   kitchen-specific.
3. **Resolution/spp scaling table** — export + train cost and held-out PSNR
   vs (resolution, spp) so the verdict docs can state cost-per-quality at
   production frame sizes instead of extrapolating.

## 3. CUDA port design (stretch goal)

The torch backend is already device-generic: `TorchPathCache` and
`train.py`/`bench.py` take a `device` argument and CUDA is nominally a string
away (`device: cuda`). The port is therefore staged so each stage is
independently valuable and an honest negative at any stage still produces a
benchmark table.

**Stage A — bring-up (no new code paths).** Run the existing suite and the
existing benchmarks with `device: cuda` on a rented GPU. Expected issues are
small: fp64 tensors (CUDA prefers fp32; MPS already forced the fp32 path, so
the machinery exists), any `torch.mps.synchronize()` calls in benches need a
device-dispatched sync, and seeds/generators need `torch.Generator(device=)`
audit. Deliverable: the cpu/mps/cuda three-column version of every existing
benchmark table (gather, train, inference, inverse), plus parity tests that
skip cleanly without CUDA (`@unittest.skipUnless(torch.cuda.is_available())`).

**Stage B — CUDA-native fast path (the paper's actual regime).** Two options,
in preference order:

1. **`torch.compile` (Triton) first.** Zero new dependencies, one config
   flag, and it targets exactly our bottleneck shape (small-op launch
   overhead on a small MLP). On CUDA, `torch.compile` + fp16 autocast +
   larger batch is likely to recover most of the fused-kernel win. Also
   applies to the gather: the batched segment-overlap test is a
   reduction-heavy elementwise kernel Triton handles well.
2. **tiny-cuda-nn as the escalation.** The paper's own stack (fused MLP +
   multiresolution hashgrid, fp16 tensor cores). Installable as a torch
   extension (`pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`)
   but it compiles against the local CUDA toolkit — pin a container image
   (e.g. RunPod's pytorch CUDA images) and treat it like the mitsuba/oidn
   optional extras: a `cuda` extra, tests skip cleanly, numpy/torch reference
   stays authoritative. Risk: tcnn's hashgrid layout differs from ours
   (interleaved per-level tables, fp16 params), so parity is statistical
   (equal-quality-at-equal-budget), not bitwise — the roadmap-item-3
   convention (within 0.5 dB at equal iterations) already covers this.

Success criteria worth pinning: 50k-iteration paper-scale training in single-
digit minutes (vs 22.4 min MPS), and full-frame inference at 1080p inside the
30 Hz band (vs ~4.3 Hz extrapolated MPS) — those are the two numbers the
performance doc explicitly defers to "CUDA + tiny-cuda-nn."

**Reproducibility constraint:** the owner has no CUDA hardware, so every CUDA
claim must be reproducible from a committed script that provisions nothing
implicitly — a `cloud/` (or `examples/cuda/`) runbook: exact image, `uv sync`
extras, commands, and the report JSON copied back into `out/`. Hardware
context in `docs/performance.md` must name the rented GPU and provider.

## 4. Cloud provider research and cost estimates

Prices checked 2026-07-16 (see sources). The workload is small by cloud-GPU
standards: single GPU, ≤24 GB VRAM is ample (the 512² packed cache is 3.35 GB;
the model is <1M params), bursty interactive sessions of a few hours.

| provider | relevant offer | ~$/hr | notes |
|---|---|---|---|
| **RunPod (recommended)** | RTX 4090 24 GB secure cloud | ~$0.35–0.69 | Per-second billing, persistent network volumes, SSH + prebuilt PyTorch/CUDA images, $10 signup credit. Best friction/cost balance for iterative dev. |
| Vast.ai | RTX 4090 marketplace | ~$0.09–0.59 | Cheapest raw price; peer-to-peer, no SLA, spot can be reclaimed on 15 s notice. Fine for reruns of a scripted benchmark, annoying for interactive bring-up. |
| Lambda | A100 80 GB / H100 SXM | ~$2.06 / ~$2.99 | Premium tier with SLA; overkill for this model size. Only relevant if tcnn needs an architecture check on A100/H100. |
| AWS | g5.xlarge (A10G 24 GB) | ~$1.01 | 2–3× RunPod's 4090 for less compute; only worth it if AWS credits exist. |
| Google Colab Pro | T4/L4/A100 (quota'd) | $9.99/mo | Cheapest absolute entry, but notebook-shaped: no SSH-first workflow, sessions preempt — poor fit for the runbook-reproducibility requirement. |

**Recommendation:** RunPod RTX 4090 secure cloud as the primary target
(consumer Ada card ≈ the class of hardware the paper's interactive numbers
imply; 24 GB fits everything; tcnn compiles for it routinely), with Vast.ai as
the cheap rerun/verification option and one short A100 session only if a
tensor-core-generation comparison is wanted.

**Cost estimate (at $0.50/hr 4090 midpoint):**

| activity | GPU-hours | est. cost |
|---|---|---:|
| Stage A bring-up: env setup, test suite, parity, full bench matrix | 6–10 | $3–5 |
| Stage B: torch.compile/AMP sweep + paper-scale 50k runs | 8–12 | $4–6 |
| Stage B: tiny-cuda-nn build + equal-budget quality parity runs | 8–15 | $4–8 |
| 1024² kitchen train on CUDA (optional cross-check of S2) | 4–8 | $2–4 |
| margin for reruns / broken sessions (×1.5) | — | ~$7–12 |
| **total campaign** | **~30–50** | **≈ $20–35** |

Even worst-case (everything on Lambda A100 at $2.06/hr) the campaign stays
under ~$110. Budget risk is negligible; the real cost is session logistics,
which the runbook requirement addresses.

Sources: [RunPod pricing](https://www.runpod.io/pricing),
[Vast.ai pricing](https://vast.ai/pricing),
[GPU cloud pricing comparison 2026 (altstreet)](https://altstreet.investments/tools/gpu/gpu-price-comparison),
[Cloud GPU rental guide 2026 (promptquorum)](https://www.promptquorum.com/power-local-llm/cloud-gpu-rental-guide-2026),
[H100 rental price comparison (IntuitionLabs)](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison),
[L4 cloud pricing (getdeploying)](https://getdeploying.com/gpus/nvidia-l4),
[A10G cloud pricing (getdeploying)](https://getdeploying.com/gpus/nvidia-a10g),
[tiny-cuda-nn (NVlabs)](https://github.com/NVlabs/tiny-cuda-nn).

## 5. Track structure decision

Six core rungs + two CUDA stretch rungs, written as goal prompts in
`docs/scale-track.md`. Ordering: S1 (streamed gather speedup) unlocks S2/S3
(larger scenes); S4 (rig re-gate with H3's fix) is independent and cheap;
S5/S6 are local performance work that also de-risks the CUDA stage; S7/S8
are the stretch goals and depend only on S6's config plumbing.
