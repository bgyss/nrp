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
| [quickstart.md](docs/quickstart.md) | every CLI, copy-pasteable, from `uv sync` to inverse lighting |
| [architecture.md](docs/architecture.md) | pipeline diagram, path-cache schema, module-by-module notes, training-config reference |
| [paper-mapping.md](docs/paper-mapping.md) | section-by-section paper coverage, faithful vs. substituted vs. out of scope, and the known-deviations list |
| [performance.md](docs/performance.md) | benchmark methodology and every measured result |
| [roadmap.md](docs/roadmap.md) | the ten-item improvement roadmap, written as ready-to-run goal prompts |
| [status/](docs/status/) | dated status reports tracking roadmap progress |

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

All ten roadmap items are complete — see [docs/status/2026-07-02.md](docs/status/2026-07-02.md)
for the full before/after picture and [docs/paper-mapping.md](docs/paper-mapping.md)
for section-by-section coverage. Headline numbers (toy scene, laptop CPU, none
quoted from the paper — see [docs/performance.md](docs/performance.md) for
methodology and every measured result):

- **Decoupling validated:** GATHERLIGHT over the cache agrees with an independently
  re-traced reference to PSNR 29.3 dB / 0.03% mean radiance.
- **Paper-scale quality ceiling:** 8×256 hashgrid MLP, 50k iterations →
  **35.19 dB** held-out PSNR on the Mitsuba cornell box (up from 25.87 dB at 3k
  iterations).
- **GPU path is real:** batched torch GATHERLIGHT is 5–7× faster than numpy on MPS
  at ≥128²; inference sustains the paper's ~30–60 Hz interactive range up to 512×512
  on this laptop's GPU.
- **Compactness, demonstrated three ways:** packed fp16/rgb9e5 caches (3.2–4.2×
  smaller), multi-view proxies (0.76 MB for 3 views vs 10.5 MB of caches), and
  per-layer compositing proxies (latency-neutral vs. a full-scene relight).
- **One honest negative result:** the paper's data-efficiency claim (Fig. 6) does
  *not* reproduce at toy scale — a fixed image dataset beats the rolling training
  pool by 1.8 dB at matched budget, traced to pool-size-limited light diversity
  rather than smoothed over.

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
nrp/torch_backend/   paper-architecture backend (hashgrid, pool training, inverse, bench)
examples/            training configs + art-directed target builder
examples/scenes/     gallery-scene download script (assets never committed)
tests/               unit tests (geometry, gather, hashgrid, loss gradients, reparam, exporter, OIDN, smokes)
docs/                architecture, paper mapping, performance, status reports, roadmap
flake.nix / mise.toml / .envrc   toolchain
```

## License

MIT — see [LICENSE](LICENSE).
