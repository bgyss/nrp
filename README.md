# nrp — Neural Render Proxies, a sample reimplementation

A small, CPU-runnable reimplementation of Sancho et al., *"Neural Render Proxies for
Interactive and Differentiable Lighting"* (Computer Graphics Forum 45(4), EGSR 2026),
written as an educational sample, not a production renderer. Every piece runs end to
end on a laptop in minutes, with no GPU required.

The paper's core idea: path tracing decouples into a **light-agnostic path pass**
(SAMPLEPATHS — expensive, needs the scene, done once) and an **emission pass**
(GATHERLIGHT — cheap, needs only the light parameters and cached path segments). A
compact per-light-type MLP then compresses the path cache into a differentiable proxy:
forward evaluation gives interactive relighting, backpropagation gives gradient-based
inverse lighting from image-space objectives.

Two backends share one path-cache/GATHERLIGHT vocabulary:

- **`nrp/` (numpy)** — the dependency-light reference: hand-rolled,
  finite-difference-checked autodiff, sinusoidal positional encoding. Good for reading.
- **`nrp/torch_backend/` (PyTorch)** — the paper's architecture (§4): 2D
  multiresolution hash encoding [MESK22], the paper's exact input set and loss
  (relative MSE, Eq. 4), pool-of-denoised-images training (§4.4), and the paper's
  inverse-optimization formulation (§5.3).

## Documentation

Full docs live in [docs/](docs/):

| doc | what's there |
|---|---|
| [tracks.md](docs/tracks.md) | **start here** — the five-phase milestone progression and how the docs below connect |
| [quickstart.md](docs/quickstart.md) | every CLI, copy-pasteable, from `uv sync` to inverse lighting |
| [architecture.md](docs/architecture.md) | pipeline diagram, path-cache schema, module-by-module notes, training-config reference |
| [paper-mapping.md](docs/paper-mapping.md) | section-by-section paper coverage, faithful vs. substituted vs. out of scope, and the known-deviations list |
| [performance.md](docs/performance.md) | benchmark methodology and every measured result, with hardware context |
| [roadmap.md](docs/roadmap.md) | phase 1 (complete): the ten-item replication roadmap, written as ready-to-run goal prompts |
| [extensions.md](docs/extensions.md) | phase 2 (complete): the E1–E9 program stress-testing the decoupling as a real-time-pipeline building block |
| [pipeline-feasibility.md](docs/pipeline-feasibility.md) | phase 3 (complete): the E10 verdict per target audience (games / film / VFX), every claim traced to a report |
| [production-track.md](docs/production-track.md) | phase 4 (complete): the trunk-and-branches ladder attacking the verdict's blockers, ending in three summit demos |
| [hardening-track.md](docs/hardening-track.md) | phase 5 (complete): fixed what phase 4 surfaced, re-earned the caveated claims, refreshed the verdict |
| [status/](docs/status/) | dated status reports, including the [2026-07-11 full audit](docs/status/2026-07-11.md) |

## Toolchain (nix + mise + uv)

- **nix** — reproducible tool source of truth: `nix develop` gives python 3.12, uv,
  ruff (direnv users: `.envrc` loads it automatically).
- **mise** — non-nix alternative and task runner: `mise trust && mise install`, then
  `mise run test|lint|fmt|train|train-torch|smoke`.
- **uv** — Python project/venv manager: `uv sync` installs numpy + torch (the only
  required runtime dependencies) and the dev group; `uv run <cmd>` runs inside the
  venv. Optional extras: `uv sync --extra mitsuba --extra oidn` (or `mise run
  sync-all`) adds the Mitsuba 3 exporter and the OIDN denoiser. On macOS the `oidn`
  wheel needs `libtbb.12.dylib`; the nix devshell provides it (oneTBB +
  `DYLD_FALLBACK_LIBRARY_PATH`), so run through direnv or `nix develop` — no
  Homebrew TBB required.

## Quickstart

```sh
uv sync

# numpy reference backend: trace the toy scene and train (~4 min CPU)
uv run python -m nrp.train --config examples/toy_sphere.json

# torch backend (paper architecture): reuses the same path cache
uv run python -m nrp.torch_backend.train --config examples/toy_sphere_torch.json

# tests / lint
uv run python -m unittest discover -s tests    # or: mise run test
uv run ruff check .                            # or: mise run lint
```

Mitsuba scenes, paper-scale training, real gallery scenes, benchmarking, relighting,
and inverse lighting are all in **[docs/quickstart.md](docs/quickstart.md)**.

## Status

The project has progressed through four completed phases beyond the initial
implementation — replication, extensions, a feasibility verdict, and a
production track ending in one summit demo per target audience — and is now in
a fifth (hardening). [docs/tracks.md](docs/tracks.md) is the spine; the
[2026-07-11 audit](docs/status/2026-07-11.md) is the current full picture.
Every number below is measured on a single Apple Silicon laptop (no CUDA) and
traces to a committed report — none are quoted from the paper (see
[docs/performance.md](docs/performance.md)).

**Phase 1 — Replication (complete).** All ten roadmap items: decoupling
validated against an independently re-traced reference (29.3 dB / 0.03% mean
radiance); paper-scale training to **35.19 dB** held-out PSNR on the Mitsuba
cornell box; packed fp16/rgb9e5 caches (3.2–4.2× smaller); multi-view and
per-layer compositing proxies. Plus an honest negative: the paper's
data-efficiency claim (Fig. 6) does not reproduce at toy scale.

**Phase 2 — Extensions (complete).** E1–E9 stress-tested the decoupling
against production questions: animated lights, dynamic geometry, out-of-core
scale (streamed training at 512×512/128 spp), engine runtimes (the exported
proxy as a WebGPU compute shader in real Chrome, 2.4e-7 parity vs PyTorch),
inverse art direction, production controls, and quality tiers. Two negative
results were the most useful deliverables: segment-local proxy fine-tuning
after a geometry change misses its recovery target by 11–20 dB (a settled
structural finding), and a native WebGPU binding's segfault was bisected to
the binding itself, not the shader.

**Phase 3 — Verdict (complete).** [pipeline-feasibility.md](docs/pipeline-feasibility.md)
grades games / animated film / feature VFX each "partly viable" and names the
blocker per audience.

**Phase 4 — Production track (complete).** Ten rungs attacking those blockers,
ending in three summit demos: **games** — the real 409k-param kitchen proxy
(hashgrid in WGSL) relit live in real Chrome with production controls, p95
30.7 ms at 512² under interaction, 12/12 frames passing the perceptual gate;
**film** — a 120-frame shot at final tier via proxy + fp16 residual identity,
118× wall-clock amortization vs re-rendering (honest negative: residual
storage costs 1.17× the raw frames); **VFX** — an 8-light rig with per-light
proxies, solo/mute, and a gradient-based art-direction loop. Three rungs
closed as honest negatives or partials, including the track's biggest open
finding: three `QuadLight` proxies trained to exactly-zero output, making the
rig's 154.5 dB inverse-convergence score half vacuous (only 3 of 6 colorable
lights genuinely recovered).

**Phase 5 — Hardening track (complete).**
[hardening-track.md](docs/hardening-track.md) root-caused the quad
zero-collapse (nn.Linear's default output-head init vs. this cache's dim
true target scale — fixed by re-initing near the pool's own target scale),
then retrained the full 8-light rig post-fix: all 8 proxies now produce real
output, but the additivity gate still misses preview tier and the
art-direction recovery is genuinely 5/6 colorable lights, not the
automatically-reported 6/6 (a false negative in the report's own vacuity
check, caught by hand). The two `TexturedQuadLight` proxies cannot be
brought into the rig's quality envelope by more iterations or more model
capacity — a conditioning-input-scheme problem, not a budget one. Rig
compositing now runs on the proven WebGPU runtime (clean GPU-vs-CPU parity,
5× per-light speedup over CPU, but not yet real-time for an 8-light
session). Real-scene dynamic geometry (a genuinely re-traced, edited Mitsuba
scene, not a toy fixture) shows G1's toy-scale residual-training fix does
**not** transfer to real scale — neither tested regime meets the 1 dB
recovery target. F2's storage negative is flipped (approval-frame gating
beats raw storage at zero quality cost; int8 quantization is a documented
floor). The feasibility verdict is re-issued against the full T1–V2+H1–H6
evidence base in [pipeline-feasibility.md](docs/pipeline-feasibility.md)'s
2026-07-16 revision (original verdict kept intact as history).

## Known deviations from the paper

Documented substitutions, not silent approximations — full list with rationale in
[docs/paper-mapping.md](docs/paper-mapping.md#known-deviations-summary): CPU/MPS
PyTorch only (no tiny-cuda-nn/Triton), OIDN optional (bilateral filter is the
dependency-free default), a softplus output head, Python-side path recording in the
Mitsuba exporter, and a numpy backend that diverges further by design for
readability.

## Layout

```
nrp/                 numpy reference backend + shared vocabulary (cache, lights, gather)
nrp/mitsuba_exporter.py  Mitsuba 3 scene -> path cache (optional extra; wavefront + scalar loops)
nrp/export_bench.py  exporter throughput benchmark (scalar vs wavefront)
nrp/torch_backend/   paper-architecture backend (hashgrid, pool training, inverse, bench,
                     streaming, shot/rig/art-loop harnesses, residual dynamic geometry)
nrp/quality/         perceptual pass/fail gates (preview/draft/final tiers, PSNR/SSIM/FLIP)
webgpu/              WGSL runtime: shader generator, browser bench, interactive demo viewer,
                     and the documented native-binding negative result
examples/            training configs, demo/report scripts per rung, art-directed target builder
examples/scenes/     gallery-scene download script (assets never committed)
tests/               unit tests (geometry, gather, hashgrid, loss gradients, reparam, exporter,
                     OIDN, quality gates, rig/shot/art-loop, WebGPU export, smokes)
docs/                the five-phase track docs, architecture, paper mapping, performance
                     ledger, dated status reports, social drafts
out/                 committed JSON reports + demo artifacts every measured claim cites
flake.nix / mise.toml / .envrc   toolchain
```

## License

MIT — see [LICENSE](LICENSE).
