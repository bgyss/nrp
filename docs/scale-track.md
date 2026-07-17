# Scale & Speed Track — Larger Scenes, Faster Everything, and a CUDA Stretch

Phase 6. The hardening track (`docs/hardening-track.md`) closed with the H7
verdict revision (`docs/pipeline-feasibility.md`): the pipeline is proven on
one real scene at 512², the WebGPU rig is real-time, and the named remaining
gaps are quality (rig additivity) and *scale/throughput* (streaming is
Python-loop-bound, nothing has run at 1024² or on a second scene, and the
performance doc explicitly defers paper-rate training/inference to "CUDA +
tiny-cuda-nn"). This track attacks throughput and scene scale directly, and
carries the NVIDIA-ecosystem port as an explicit **stretch goal** (S7/S8) —
the owner has no CUDA hardware, so those rungs run on rented cloud GPUs under
a committed, reproducible runbook.

Design research, provider comparison, and cost estimates:
`docs/plans/2026-07-16-scale-track-research.md`. Estimated total cloud spend
for the stretch rungs: **≈ $20–35** (30–50 GPU-hours on a RunPod RTX 4090 at
~$0.35–0.69/hr; worst-case ~$110 on Lambda A100).

## Status

| rung | title | status | evidence |
|---|---|---|---|
| S1 | Vectorize the streamed gather (kill the Python-loop bound) | **done** (6.04× end-to-end, target ≥5×; 10× stretch missed) | `out/s1-streamed-gather/*.json`, `docs/performance.md#vectorized-streamed-gather-scale-track-rung-s1` |
| S2 | 1024² and a second real scene | **done** (1024²/32spp kitchen through streamed path, 23.0 dB, 158 MB peak; bedroom 512² at 21.89 dB > 18 dB bar) | `out/s2-scale/*.json`, `out/bedroom-512-torch/torch_train_report.json`, `docs/performance.md#1024-and-a-second-real-scene-scale-track-rung-s2` |
| S3 | Export/shard throughput at scale | **done** (save-sharded 5.78× float64 / 9.13× packed, target ≥3×; exporter 1.87× via bincount G-buffer scatter) | `out/s3-shard-write/*.json`, `docs/performance.md#exportshard-throughput-at-scale-scale-track-rung-s3` |
| S4 | Re-earn the rig additivity gate with H3's kernel conditioning | todo | — |
| S5 | Local training throughput: torch.compile / AMP / batch sweep | todo | — |
| S6 | WebGPU runtime at 1024² + device/precision config plumbing | todo | — |
| S7 | **(stretch)** CUDA bring-up on a rented cloud GPU | todo | — |
| S8 | **(stretch)** Paper-parity CUDA fast path (torch.compile → tiny-cuda-nn) | todo | — |

Ordering: S1 → S2/S3 (larger scenes are only affordable after the streaming
speedup); S4 independent and cheap — do it first if a quick win is wanted;
S5 → S6 → S7 → S8 (each de-risks the next). S7/S8 are stretch: the track
closes honorably at S6 if cloud work is deferred.

## Shared conventions (same as all prior phases)

- All new/changed code passes `mise run test` and `mise run lint`; new
  features get unit tests that skip cleanly when optional dependencies —
  now including CUDA itself — are absent.
- Every measured claim lands in a JSON report under `out/` **and** in
  `docs/performance.md` with hardware context; never quote the paper's
  numbers as ours. `mise run pipeline-audit` verifies referenced paths exist.
- Honest negative results are deliverables.
- Local rungs (S1–S6): single Apple Silicon laptop, no CUDA assumed. Stretch
  rungs (S7–S8): rented cloud GPU; every claim reproducible from a committed
  runbook (exact image, sync commands, run commands), report JSONs copied
  back into `out/`, and `docs/performance.md` names the GPU and provider.
- Update `docs/paper-mapping.md` for any row a change affects; commit in the
  repo's usual message style.

---

### S1. Vectorize the streamed gather (kill the Python-loop bound)

> E5's 512²/128spp streamed run works but is scalar-numpy-bound: 382 s pool
> build + 622 s for 150 iterations, dominated by repeated `np.add.at` passes
> over ~5.9M-segment shards — while the batched `TorchPathCache` gather
> (roadmap item 3) already exists and is 5–7× faster on MPS at much smaller
> segment counts. Give `StreamedImagePool` (and the streamed target builder)
> a `gather_backend: torch` path that constructs a per-shard `TorchPathCache`
> on the fly, gathers the shard's contribution in one batched op on
> cpu/mps, and accumulates into the pool slot — preserving the streamed
> path's bounded-residency guarantee (never more than one shard's segments
> resident). Parallelize shard *decompression* (the other measured cost) if
> profiling shows it still dominates after the gather is batched.
> **Verify:** streamed-torch pool targets match the streamed-numpy path
> within the existing torch-gather parity tolerance (rtol 1e-5) on the toy
> sharded cache and one real shard; peak resident segment bytes unchanged
> (assert in the report, same accounting as `out/out-of-core/`); the
> bit-exact rng-order property of `train_streamed` vs in-memory training is
> either preserved or its loss explicitly documented and the statistical
> equivalence shown instead.
> **Measure:** pool-build and total train wall-clock at 512²/128spp vs the
> committed 382 s / 1,004.9 s baselines (target: **≥ 5×** end-to-end,
> stretch 10×); per-shard gather ms numpy vs torch-cpu vs torch-mps;
> peak-resident bytes unchanged. All in `docs/performance.md`.

### S2. 1024² and a second real scene

> Nothing has ever run above 512², and every real-scene claim rests on one
> scene (the Country Kitchen). (a) Export the kitchen at **1024×1024** (spp
> chosen deliberately from a small cost/quality probe — record the tradeoff),
> shard it packed, and train a sphere-light proxy end to end through the
> S1-accelerated streamed path. (b) Download-script a **second** Mitsuba
> gallery scene with materially different geometry/materials (e.g. bedroom or
> bathroom; assets never vendored, per `examples/scenes/` convention), export
> at 512²/64spp, and run the same train + held-out evaluation as T1 did for
> the kitchen.
> **Verify:** both caches pass `PathCache.validate()`; the 1024² run
> completes within the streamed path's bounded-residency guarantee on the
> laptop (report the peak); the second scene's proxy beats 18 dB held-out
> PSNR (the low edge of the established sphere/quad envelope) or lands an
> honest negative with the failure characterized (which light types, which
> regions).
> **Measure:** the resolution/spp scaling table — export wall-clock, packed
> cache GB, shard time, pool-build + train wall-clock, held-out PSNR — for
> {512², 1024²} × kitchen and 512² × scene-2, extending the T1/T2 tables;
> tiled full-frame inference latency at 1024². All in `docs/performance.md`.

### S3. Export/shard throughput at scale

> The 512² export pipeline's fixed costs are now material: 306.3 s to write
> the sharded cache (serial per-tile `np.savez_compressed`) and a
> single-threaded wavefront→numpy conversion. Parallelize shard writing
> (process pool or threaded zlib), profile the exporter's Python-side
> conversion, and fix the top measured cost. This is what makes S2's 1024²
> exports (and any future retrace work) affordable.
> **Verify:** parallel-written sharded caches load bit-identically to
> serial ones (existing sharded round-trip tests extended); exporter
> equivalence tests still pass.
> **Measure:** save-sharded wall-clock at 512² vs the 306.3 s baseline
> (target ≥ 3×) and at 1024²; exporter segments/s before/after any
> conversion fix. All in `docs/performance.md`.

### S4. Re-earn the rig additivity gate with H3's kernel conditioning

> H2's additivity gate still fails preview tier (SSIM 0.725 < 0.80, FLIP
> 0.257 > 0.15) — but it was measured with the rig's two `TexturedQuadLight`
> proxies at their pre-H3 12–13 dB floor. H3 has since landed per-texel
> kernel conditioning that puts textured quads at 19.64/19.99 dB, inside the
> sphere/quad envelope. Rebuild the 8-light rig with kernel-conditioned
> textured-quad proxies (budget per H2's deliberate-budget convention),
> re-run the additivity gate and the V2 art-direction loop, and fix the
> known `recovery_caveats` false-negative (the fixed epsilon that reported
> 6/6 when hand-checking showed 5/6) so the automated check matches the
> hand-checked truth.
> **Verify:** 8/8 proxies nonzero; the recovery check's epsilon logic
> unit-tested against the H2 false-negative case (must report 5/6 on the H2
> artifacts, and genuine/vacuous machinery retained); rig round-trip still
> bit-identical.
> **Measure:** additivity PSNR/SSIM/FLIP vs preview-tier thresholds — pass,
> or an honest fail with the residual error decomposed per light so the next
> floor is named; colorable-light recovery count; slider-loop p95 on the H4
> WebGPU runtime with the rebuilt rig (confirm the 1.3/30.9 ms results hold
> with kernel-conditioned proxies, whose per-texel head changes the shader's
> work). All in `docs/performance.md`.

### S5. Local training throughput: torch.compile / AMP / batch sweep

> Paper-scale training is 22.4 min on MPS at only 1.6× CPU — the measured
> reason is a small MLP at batch 4096 under-filling the GPU. Sweep the three
> untested levers on the paper-scale config: batch size (4096 → 16k/64k with
> LR scaled accordingly), fp16/bf16 autocast, and `torch.compile`, each
> ablated separately then combined. An honest negative on any lever (e.g.
> torch.compile immaturity on MPS) is a deliverable — it becomes S7's
> baseline expectation on CUDA.
> **Verify:** equal-quality criterion per roadmap item 3: held-out PSNR
> within 0.5 dB of the fp32 eager baseline at equal effective sample budget;
> any train-config additions covered by tests; numpy reference untouched.
> **Measure:** iters/s and wall-clock to 35 dB (the committed 50k baseline
> quality) per lever and combined, CPU and MPS; the chosen default config
> justified. All in `docs/performance.md`.

### S6. WebGPU runtime at 1024² + device/precision config plumbing

> The Chrome WebGPU runtime is the project's fastest proven backend (107 fps
> at 512², beating TorchScript-MPS) but has never been benchmarked above
> 512², while MPS full-frame inference is 8.6 Hz at 1024². Extend the T4
> bench methodology to 1024² (and 1080p non-square if the runtime's tiling
> permits), including an H4 rig-compositing session at 1024². While in the
> plumbing: make device (`cpu|mps|cuda`) and precision first-class,
> validated config/CLI options across train/bench/relight/optimize — the
> exact seam S7 needs — with `cuda` requested-but-unavailable failing with a
> clear message.
> **Verify:** T4's parity gate at 1024²; config validation unit-tested,
> including the cuda-unavailable path; existing 512² regression gate
> (`out/t4-runtime/report.json` conventions) still passes.
> **Measure:** frame-time histogram / p95 at 1024² (30 fps p95 pass/fail
> verdict, same criterion as T4); rig slider p95 at 1024² vs H4's 512²
> numbers. All in `docs/performance.md`.

---

## Stretch goals — the NVIDIA-ecosystem port

Run on rented cloud GPUs (recommended: RunPod RTX 4090 secure cloud,
~$0.35–0.69/hr; alternatives and cost table in the research doc). Every
claim must be reproducible from the committed runbook; CUDA tests skip
cleanly on machines without CUDA, keeping the laptop suite green.

### S7. (stretch) CUDA bring-up on a rented cloud GPU

> The torch backend is nominally device-generic but `device: cuda` has never
> executed. Write `cloud/README.md` + a provisioning script (pinned
> container image, `uv sync` extras, exact commands), then on a rented 4090:
> run the full test suite, fix what breaks (expected: fp64/fp32 dtype
> policy, `torch.mps.synchronize()` → device-dispatched sync, generator
> device audit), and produce the three-device benchmark matrix.
> **Verify:** full suite green on the CUDA box; CUDA parity tests (gather
> rtol 1e-5 vs numpy; training within 0.5 dB of CPU at equal seed/iters)
> committed and skipping cleanly locally; runbook reproduces the report from
> a fresh instance (do it twice, second time from the runbook alone).
> **Measure:** cpu/mps/cuda columns for the gather table (toy/128²/256² +
> one kitchen shard), paper-scale training iters/s and wall-clock to 35 dB,
> and full-frame inference at 48²–1024²; GPU-hours and dollars actually
> spent vs the ≤ $10 estimate for this rung. All in `docs/performance.md`
> with provider + GPU named.

### S8. (stretch) Paper-parity CUDA fast path (torch.compile → tiny-cuda-nn)

> The two numbers `docs/performance.md` explicitly defers to "CUDA +
> tiny-cuda-nn": paper-scale training minutes and 1080p interactive
> inference. Attack in escalation order: (1) S5's torch.compile/AMP/batch
> levers on CUDA — Triton is mature there and may suffice; (2) only if the
> targets are still missed, add tiny-cuda-nn (fused MLP + hashgrid, fp16
> tensor cores) as an optional `cuda` extra behind the existing model
> interface, with the pure-torch implementation remaining authoritative.
> tcnn parity is statistical (equal quality at equal budget), not bitwise —
> its hashgrid layout and fp16 params differ by construction.
> **Verify:** equal-budget held-out PSNR within 0.5 dB of the pure-torch
> CUDA baseline for whichever fast path ships; tcnn (if used) is
> import-guarded exactly like mitsuba/oidn, suite green without it; runbook
> extended and re-verified from fresh instance.
> **Measure:** 50k paper-scale wall-clock (target: **≤ 5 min**, vs 22.4 min
> MPS) and full-frame 1920×1080 inference latency (target: **≥ 30 Hz**, vs
> ~4.3 Hz extrapolated MPS); honest statement of which target each
> escalation stage hit or missed; total stretch-goal spend vs the ≈ $20–35
> campaign estimate. All in `docs/performance.md`.
