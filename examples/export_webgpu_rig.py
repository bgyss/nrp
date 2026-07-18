"""H4: export an N-light `LightRig` to the WGSL rig-compositor's blob format.

Generalizes `examples/export_webgpu_demo.py::export_kitchen` (sphere-light-only,
one proxy) to any rig -- sphere/quad/textured_quad, N lights -- for
`webgpu/bench_h4.mjs`'s per-light "write first active, add the rest" dispatch
sequence. `pixels.bin` (xy + G-buffer aux) is shared across every light (same
cache); each light gets its own `mlp_<name>.bin` / `tables_<name>.bin` (only if its
proxy uses the hashgrid encoding) plus a manifest entry with its own
`mlp_dims`/`encoding`/`light_param_dim` (these can differ per light -- textured_quad
proxies have a much larger `light_param_dim` than sphere/quad, and this exporter
does not assume they match). Each light's numpy-replica self-check reuses
`export_webgpu_runtime.numpy_forward`, same tolerance as the G2 exporter.

Usage:
  uv run python examples/export_webgpu_rig.py \
      --rig out/h2-rig/rig.json --models-dir out/h2-rig/models \
      --cache out/kitchen-512/path_cache.npz --out-dir out/h4-rig-export
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.export_webgpu_runtime import (  # noqa: E402
    export_encoding,
    export_mlp,
    load_gbuffer,
    numpy_forward,
    pixel_arrays,
)
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.rig import LightRig  # noqa: E402
from nrp.torch_backend.train import light_param_vector  # noqa: E402

SELF_CHECK_TOLERANCE = 1e-3


def export_light(
    model: TorchNRP, name: str, light, xy: np.ndarray, aux: np.ndarray, out_dir: Path
) -> dict:
    n_px = xy.shape[0]
    light_params = light_param_vector(light).astype(np.float32)
    mlp_flat, mlp_dims = export_mlp(model)
    if model.encoding is not None:
        tables_flat, level_meta = export_encoding(model)
        features_per_level = model.encoding.features_per_level
        table_size = model.encoding.table_size
    else:
        tables_flat, level_meta = None, None
        features_per_level, table_size = 0, 0

    with torch.no_grad():
        reference = (
            model(
                torch.as_tensor(xy),
                torch.as_tensor(aux),
                torch.as_tensor(light_params).expand(n_px, -1),
            )
            .numpy()
            .astype(np.float32)
        )
    replica = numpy_forward(
        xy,
        aux,
        light_params,
        mlp_flat,
        mlp_dims,
        tables_flat,
        level_meta,
        features_per_level,
        table_size,
        texture_kernel=model.texture_kernel,
    ).astype(np.float32)
    self_check = float(np.max(np.abs(replica - reference)))

    mlp_flat.tofile(out_dir / f"mlp_{name}.bin")
    if tables_flat is not None:
        tables_flat.tofile(out_dir / f"tables_{name}.bin")

    rgb = getattr(light, "rgb", None)
    return {
        "name": name,
        "light_type": model.light_type,
        "mlp_dims": mlp_dims,
        "aux_dim": 7,
        "light_param_dim": model.light_param_dim,
        "texture_kernel": model.texture_kernel,
        "encoding": None
        if level_meta is None
        else {
            "features_per_level": features_per_level,
            "table_size": table_size,
            "hash_prime": 2654435761,
            "levels": level_meta,
        },
        "light_params": light_params.tolist(),
        "rgb": None if rgb is None else np.asarray(rgb, dtype=np.float64).tolist(),
        "parameter_count": model.parameter_count,
        "files": {
            "mlp": f"mlp_{name}.bin",
            "tables": None if tables_flat is None else f"tables_{name}.bin",
        },
        "numpy_replica_vs_torch_max_abs_diff": self_check,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rig", required=True, help="rig.json (v1_rig.py/LightRig.save format)")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.rig) as f:
        rig_dict = json.load(f)
    models_manifest = rig_dict.get("models")
    if models_manifest:
        models = {
            name: TorchNRP.load(str(Path(args.models_dir) / Path(rel).name))
            for name, rel in models_manifest.items()
        }
    else:
        models = {
            rl["name"]: TorchNRP.load(str(Path(args.models_dir) / f"{rl['name']}.pt"))
            for rl in rig_dict["lights"]
        }
    rig = LightRig.from_dict(rig_dict, models)

    gbuf = load_gbuffer(args.cache)
    xy, aux = pixel_arrays(gbuf)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pixels = np.concatenate([xy, aux], axis=1).astype(np.float32)
    pixels.tofile(out_dir / "pixels.bin")

    lights_manifest = []
    for rl in rig.lights:
        model = rig.models[rl.name]
        model.eval()
        lights_manifest.append(export_light(model, rl.name, rl.light, xy, aux, out_dir))

    # CPU composite reference (LightRig.render's full-rig sum, Eq. 3 over Eq. 1's
    # active-light sum) for the browser compositor's GPU-vs-CPU parity check --
    # same convention as G2's G1-panel composite parity.
    composite_reference = rig.render(PathCache.load(args.cache)).astype(np.float32)
    composite_reference.tofile(out_dir / "composite_reference.bin")

    manifest = {
        "rung": "H4",
        "rig": str(args.rig),
        "cache": str(args.cache),
        "resolution": [gbuf["width"], gbuf["height"]],
        "files": {"pixels": "pixels.bin", "composite_reference": "composite_reference.bin"},
        "lights": lights_manifest,
        "worst_self_check": max(m["numpy_replica_vs_torch_max_abs_diff"] for m in lights_manifest),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if manifest["worst_self_check"] > SELF_CHECK_TOLERANCE:
        raise SystemExit(
            f"numpy replica vs torch disagrees by {manifest['worst_self_check']:.3e} "
            f"(tolerance {SELF_CHECK_TOLERANCE:.3e}) -- refusing to ship {manifest_path}"
        )
    print(f"wrote {manifest_path} ({len(lights_manifest)} lights)")


if __name__ == "__main__":
    main()
