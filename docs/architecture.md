# Architecture

## The pipeline

```
                    SAMPLEPATHS (once, needs scene, no lights)
  nrp/toy_tracer.py ──────────────┐
  nrp/mitsuba_exporter.py ────────┤──► PathCache (.npz)
                                  │      segments + throughputs + G-buffer aux
                                  ▼
                    GATHERLIGHT (cheap, needs only light params)
  nrp/gather_light.py: per-pixel emission accumulation over cached segments
                                  │
              ┌───────────────────┼──────────────────────┐
              ▼                   ▼                      ▼
   reference images       training targets         re-render check for
   for any light          (optionally denoised)    inverse results
                                  │
                                  ▼
                    NEURAL RENDER PROXY (per light type)
  numpy: nrp/model.py + nrp/train.py       torch: nrp/torch_backend/{model,train}.py
                                  │
              ┌───────────────────┴──────────────────────┐
              ▼                                          ▼
   forward relighting (Eq. 3)                inverse optimization (§5.3)
   nrp/relight.py                            nrp/optimize_lights.py
   nrp/torch_backend/relight.py              nrp/torch_backend/optimize_lights.py
```

## The path-cache schema (`nrp/path_cache.py`)

The central artifact. For a fixed camera and static scene (S = total segments):

| field | shape | meaning |
|---|---|---|
| `n_paths` | (H·W,) | paths traced per pixel (0 = undersampled, reported not interpolated) |
| `seg_pixel` | (S,) | row-major pixel index of each segment |
| `seg_origin` | (S, 3) | segment start point |
| `seg_dir` | (S, 3) | unit direction |
| `seg_tmax` | (S,) | segment length; `inf` marks an escape direction |
| `seg_throughput` | (S, 3) | path throughput accumulated **before** this segment (first segment of every path is 1) |
| `albedo/depth/normal/position` | (H, W, …) | first-hit G-buffer aux features |

Serializations: compressed `.npz` (tracer exports) and a JSON dict form for tiny
hand-authored caches in tests. `validate()` enforces shapes, index ranges, positive
t_max, and unit directions.

## Lights (`nrp/lights.py`)

Virtual pure emitters — they never block or scatter cached paths, so one cache serves
every light configuration. `SphereLight` (center, radius; 4 shape params) and
`QuadLight` (center, normal, width, height; 8 params — the tangent frame is derived
deterministically from the normal). Vectorized segment-overlap tests
(`segment_hits_sphere`, `segment_hits_quad`) drive GATHERLIGHT. `light_from_dict`
dispatches JSON specs; emission `rgb` is the E(v) factor of Eq. 1 and scales linearly.

## GATHERLIGHT (`nrp/gather_light.py`)

`gather_throughput[_quad]` returns the per-pixel **pre-emission** contribution — the
quantity the proxies learn. `gather_light` scales by rgb; `gather_lights` sums a list
(linearity of transport, Eq. 1). CPU/numpy; a fused GPU kernel is future work.

## Producers

- **`nrp/toy_tracer.py`** — dependency-free educational tracer: hard-coded
  Cornell-style unit box + diffuse sphere, Lambertian, cosine-weighted sampling,
  fixed bounce count. Also renders the *independent* emissive-inline reference used
  by `nrp/compare_reference.py` for the decoupling consistency check.
- **`nrp/mitsuba_exporter.py`** (extra: `mitsuba`) — drives Mitsuba 3's scalar
  variant from Python over any scene XML (or `builtin:cornell-box`): BSDF sampling,
  no NEE, throughput Russian roulette after bounce 2 (`--no-russian-roulette` for
  deterministic counts). Emitters in the scene are ignored (light-agnostic pass).

## numpy backend (`nrp/`)

The readable reference: `model.py` is an MLP with hand-rolled autodiff
(finite-difference-checked in tests), sinusoidal positional encoding, and extra
derived geometric inputs (first-hit→light offset + distance) that the compact MLP
needs without a spatial encoding. `train.py` trains against precomputed GATHERLIGHT
targets for uniformly sampled lights. `optimize_lights.py` is a plain clipped-Adam
optimizer — deliberately naive; the torch backend shows how much the paper's §5.3
machinery improves on it.

## torch backend (`nrp/torch_backend/`) — the paper replica

- **`encoding.py`** — 2D multiresolution hash encoding [MESK22]: per-level dense or
  hashed feature tables, bilinear interpolation, geometric resolution growth.
- **`model.py`** — `TorchNRP`: hashgrid(px) ⊕ aux(7 = albedo+depth+normal) ⊕ light
  shape params (4 sphere / 8 quad) → MLP → softplus. `relative_mse_loss` is Eq. 4
  exactly (stop-gradient prediction in the denominator, ε = 0.01); its gradient is
  unit-tested against the closed form.
- **`sampling.py`** — §4.4 light-position strategies: uniform-on-recorded-segments
  (implicit importance sampling) or visible-bbox fallback.
- **`denoise.py`** — `denoise_image` dispatch: `"oidn"` (paper's denoiser; RT filter,
  HDR, albedo+normal guides; extra: `oidn`) or `"bilateral"` (dependency-free
  aux-guided joint bilateral).
- **`train.py`** — the §4.4 pool scheme: `pool.size` denoised GATHERLIGHT images,
  every training pixel samples its target uniformly from the pool,
  `pool.replace_count` images replaced every `pool.replace_every` iterations.
- **`relight.py`** — Eq. 3 forward relighting CLI (single lights or lists), bench mode.
- **`optimize_lights.py`** — §5.3: Eq. 5 multi-light sum, Reinhard-tonemapped MSE
  (Eq. 6), logit/inverse-softplus reparameterization, pixel-fraction mini-batch SGD,
  restarts, objective/protect masks, and a mandatory GATHERLIGHT re-render of the
  result so proxy-space and physical errors are reported separately.
- **`bench.py`** — cross-device (cpu/mps/cuda) full-frame inference benchmark with
  warmup and proper synchronization.

## Training-config reference (torch backend)

```jsonc
{
  "cache": "path.npz",          // resolved relative to the config file
  "out_dir": "dir",             // outputs: model.pt, torch_train_report.json
  "trace": {...},               // optional: toy-tracer params to create the cache if missing
  "light_type": "sphere",       // or "quad"
  "light_bounds": {             // shape-parameter sampling ranges
    "radius_min": 0.05, "radius_max": 0.25          // sphere
    // "size_min": ..., "size_max": ...             // quad width/height
  },
  "sampling": "segments",       // light positions: "segments" | "bbox"
  "pool": { "size": 64, "replace_every": 5, "replace_count": 2 },
  "denoise": { "enabled": true, "method": "bilateral" },  // or "oidn"; extra keys → bilateral kwargs
  "iters": 3000, "batch_pixels": 4096, "lr": 0.005,
  "model": {
    "hidden_width": 128, "hidden_layers": 4,
    "encoding": { "levels": 8, "features_per_level": 2, "table_size_log2": 12,
                  "base_resolution": 4, "finest_resolution": 48 }
  },
  "n_val_lights": 12, "seed": 0, "device": "cpu"   // "mps"/"cuda" honored if available
}
```

The numpy config (`examples/toy_sphere.json`) differs: epoch-based
(`epochs`/`batch_size`/`n_train_lights`), `hidden` as a list, and `light_bounds`
includes `center_min`/`center_max` (uniform box sampling instead of segment sampling).

## Testing conventions

Tests import modules directly via a `sys.path` shim (the repo is not an installed
package; `[tool.uv] package = false`). Optional-dependency tests skip cleanly:
`@unittest.skipUnless(HAVE_MITSUBA, ...)`, `@unittest.skipUnless(oidn_available(), ...)`.
Statistical assertions compare windowed means, never single minibatch losses.
