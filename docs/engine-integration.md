# Engine Integration Contract

This document describes the exported-runtime slice of the NRP-as-render-feature API.
It is intentionally small: an engine supplies per-pixel attributes and light
parameters, runs an exported proxy artifact, and receives RGB radiance.

## Artifact

`nrp.torch_backend.engine_runtime export` writes:

- `runtime.pt`: TorchScript graph containing hashgrid tables and MLP weights.
- `runtime.pt.json`: metadata with `format`, `light_type`, model config, parameter
  count, and artifact size.

The runtime path loads `runtime.pt` through `torch.jit.load`; it does not load the
training checkpoint or instantiate `TorchNRP`.

## Inputs

For each frame or tile, the engine supplies tensors at one common pixel grain:

| tensor | shape | meaning |
|---|---:|---|
| `pixel_xy` | `(N, 2)` | normalized pixel coordinates in `[0, 1]` |
| `aux` | `(N, 7)` | albedo RGB, depth, normal XYZ |
| `light_params` | `(N, 4)` or `(N, 8)` | sphere or quad shape parameters, broadcast per pixel |

Sphere parameters are `center.xyz, radius`. Quad parameters are
`center.xyz, normal.xyz, width, height`; the normal is normalized by the training-side
parameterization before export.

Emission RGB is multiplied outside the artifact, matching the paper's decomposition:
the proxy predicts pre-emission transport, and the frame accumulates
`proxy(pixel, light_shape) * emission_rgb` for each light.

## Output

The artifact returns:

| tensor | shape | meaning |
|---|---:|---|
| `contribution` | `(N, 3)` | pre-emission RGB transport contribution |

The caller reshapes or composites the result into the render target. For multi-light
edits, call the artifact once per light and sum the emission-scaled outputs.

## Current Limits

- The committed runtime backend is TorchScript on CPU by default. MPS can be selected
  where the local PyTorch build supports it, but the current report only claims CPU
  measurements.
- The exported artifact covers sphere and quad TorchNRP models. Textured and
  environment lights currently have reference GATHERLIGHT support only.
- The headless viewer loop in `mise run viewer` simulates slider positions and writes
  frame dumps; it is not a GUI integration.
