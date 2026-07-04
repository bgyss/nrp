---
name: nrp-path-cache-and-gather
description: Use when changing NRP path-cache schema, toy or Mitsuba path producers, light definitions, numpy GATHERLIGHT, torch GATHERLIGHT parity, packed cache serialization, volume segment semantics, or tests that depend on PathCache and gather_light behavior.
---

# NRP Path Cache and Gather

## Overview

Preserve the SAMPLEPATHS / GATHERLIGHT contract across both backends. Treat
`nrp/gather_light.py` as the reference implementation and keep every producer,
serialization path, and torch mirror compatible with `nrp/path_cache.py`.

## First Files

Read only the files relevant to the change:

- `nrp/path_cache.py` for schema, validation, `.npz`, JSON, and packed-cache behavior.
- `nrp/gather_light.py` and `nrp/lights.py` for authoritative light intersection and accumulation.
- `nrp/torch_backend/gather.py` when the change affects device-side gather or pool builds.
- `nrp/toy_tracer.py` and `nrp/mitsuba_exporter.py` when producers or segment semantics change.
- `tests/test_path_cache.py`, `tests/test_gather_light.py`, `tests/test_quad_lights.py`,
  `tests/test_torch_gather.py`, `tests/test_volume.py`, and exporter tests as applicable.
- `docs/architecture.md`, `docs/paper-mapping.md`, and `docs/performance.md` only when behavior,
  evidence, or paper coverage changes.

## Invariants

- Keep `PathCache` the single shared artifact for numpy, torch, toy, and Mitsuba workflows.
- Store pre-emission path throughput per segment; emission `rgb` remains a linear scale factor.
- Keep lights virtual and non-blocking. Occlusion comes only from cached segment endpoints.
- Keep `gather_light` / `gather_lights` linear in emission and additive across lights.
- Keep v1 surface caches loadable when adding schema fields. New metadata must be optional unless
  a migration path exists.
- Keep `seg_dir` unit length, `seg_tmax` positive or `inf`, `seg_pixel` in range, and aux buffers
  shaped `(height, width, ...)`.
- Treat torch gather as a mirror of numpy gather. It may be lower precision on MPS/CUDA, but parity
  should be tested against numpy with an explicit tolerance.

## Workflow

1. Start from a tiny hand-authored cache test when possible. It is faster and clearer than a traced
   stochastic fixture.
2. Update `PathCache.validate()` before relying on new fields elsewhere.
3. Update serialization in both directions: `.npz` save/load and JSON `to_dict` / `from_dict`
   when tests or tiny fixtures need the field.
4. Update producers last, after the cache contract and gather behavior are pinned by tests.
5. If changing sphere or quad intersection semantics, test miss, hit, tangent or near-edge,
   finite `tmax`, escape `inf`, intensity scaling, and multi-hit accumulation cases.
6. If changing torch gather, add or update numpy-vs-torch parity over both sphere and quad lights.
7. If changing volume behavior, verify that GATHERLIGHT itself stays light-agnostic; transmittance
   should come from segment sampling unless the design explicitly changes.

## Verification

Run the narrowest relevant checks first:

```sh
uv run python -m unittest tests.test_path_cache
uv run python -m unittest tests.test_gather_light
uv run python -m unittest tests.test_quad_lights
uv run python -m unittest tests.test_torch_gather
```

For producer, packed-cache, volume, or optional-dependency changes, add the matching targeted tests:

```sh
uv run python -m unittest tests.test_volume
uv run python -m unittest tests.test_packed_cache
uv run python -m unittest tests.test_exporter_denoise_bench
```

Finish with:

```sh
uv run python -m unittest discover -s tests
uv run ruff check .
```

When tests are stochastic, compare aggregate means, PSNR, SMAPE, or allclose tolerances. Do not
assert on a single minibatch or one noisy Monte Carlo sample.
