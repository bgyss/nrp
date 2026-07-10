"""G2 (docs/production-track.md): export the browser demo's WebGPU blobs.

Two modes, sharing the T4 exporter's blob format (webgpu/shader_gen.mjs consumes
both):

- default (kitchen): the T4 export (hashgrid proxy + G-buffer + reference) plus the
  demo's control inputs — first-hit positions (``positions.bin``) and the per-pixel
  light-linking layer mask (``link_mask.bin``: 1.0 where the first hit falls inside
  ``--link-box``, the demo's "layer"; linking a light off that layer zeroes exactly
  those pixels, E8's pixel-level gather-time linking algebra).
- ``--g1``: converts ``out/g1-residual/`` artifacts (frozen base + per-frame
  residual proxies + spliced G-buffers + region masks + full-retrace references)
  into per-frame blobs for the demo's toy-scale moving-object panel.

Both modes self-check a numpy replica of the exported format against the torch
modules and refuse to write a manifest that disagrees beyond 1e-3.
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
    export_mlp,
    load_gbuffer,
    numpy_forward,
    pixel_arrays,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.residual_dynamic import ResidualNRP  # noqa: E402

SELF_CHECK_TOLERANCE = 1e-3


def build_link_mask(position: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """1.0 where the first-hit position lies inside the axis-aligned box, else 0.0."""
    p = position.reshape(-1, 3)
    inside = np.all((p >= np.asarray(box_min)) & (p <= np.asarray(box_max)), axis=1)
    return inside.astype(np.float32)


def export_kitchen(args: argparse.Namespace) -> dict:
    model = TorchNRP.load(args.model)
    model.eval()
    if model.light_type != "sphere":
        raise SystemExit("the demo exporter supports sphere-light proxies only")
    gbuf = load_gbuffer(args.cache)
    xy, aux = pixel_arrays(gbuf)
    n_px = xy.shape[0]

    box_min = np.array(args.link_box[:3], dtype=np.float64)
    box_max = np.array(args.link_box[3:], dtype=np.float64)
    link_mask = build_link_mask(gbuf["position"], box_min, box_max)
    positions = gbuf["position"].reshape(-1, 3).astype(np.float32)

    mean_pos = gbuf["position"].reshape(-1, 3).mean(axis=0)
    center = np.array([mean_pos[0], mean_pos[1] + 0.5, mean_pos[2]], dtype=np.float64)
    light_params = np.array([*center, args.light_radius], dtype=np.float32)

    mlp_flat, mlp_dims = export_mlp(model)
    if model.encoding is not None:
        from examples.export_webgpu_runtime import export_encoding

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
    ).astype(np.float32)
    self_check = float(np.max(np.abs(replica - reference)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp_flat.tofile(out_dir / "mlp.bin")
    if tables_flat is not None:
        tables_flat.tofile(out_dir / "tables.bin")
    pixels = np.concatenate([xy, aux], axis=1).astype(np.float32)
    pixels.tofile(out_dir / "pixels.bin")
    reference.tofile(out_dir / "reference.bin")
    positions.tofile(out_dir / "positions.bin")
    link_mask.tofile(out_dir / "link_mask.bin")

    manifest = {
        "rung": "G2",
        "model": str(args.model),
        "cache": str(args.cache),
        "light_type": model.light_type,
        "resolution": [gbuf["width"], gbuf["height"]],
        "mlp_dims": mlp_dims,
        "aux_dim": 7,
        "light_param_dim": model.light_param_dim,
        "encoding": None
        if level_meta is None
        else {
            "features_per_level": features_per_level,
            "table_size": table_size,
            "hash_prime": 2654435761,
            "levels": level_meta,
        },
        "default_light": {"center": center.tolist(), "radius": args.light_radius},
        "parameter_count": model.parameter_count,
        "demo": {
            "link_box": {"min": box_min.tolist(), "max": box_max.tolist()},
            "link_pixel_fraction": float(link_mask.mean()),
            "attenuation": {"type": "first_hit_linear_distance", "k_default": 0.0},
        },
        "files": {
            "mlp.bin": int(mlp_flat.size),
            "tables.bin": 0 if tables_flat is None else int(tables_flat.size),
            "pixels.bin": int(pixels.size),
            "reference.bin": int(reference.size),
            "positions.bin": int(positions.size),
            "link_mask.bin": int(link_mask.size),
        },
        "numpy_replica_vs_torch_max_abs_diff": self_check,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_g1_gbuffer(path: Path) -> dict:
    with np.load(path) as z:
        return {
            "width": int(z["width"]),
            "height": int(z["height"]),
            "albedo": np.asarray(z["albedo"], dtype=np.float64),
            "depth": np.asarray(z["depth"], dtype=np.float64),
            "normal": np.asarray(z["normal"], dtype=np.float64),
            "position": np.asarray(z["position"], dtype=np.float64),
        }


def export_g1_frame(
    base: TorchNRP,
    residual: ResidualNRP,
    gbuf: dict,
    region_mask: np.ndarray,
    light_params: np.ndarray,
) -> dict:
    """Flat blobs + numpy self-check for one G1 frame (base + masked residual)."""
    xy, aux = pixel_arrays(gbuf)
    n_px = xy.shape[0]
    base_flat, base_dims = export_mlp(base)
    res_flat, res_dims = export_mlp(residual)
    with torch.no_grad():
        params = torch.as_tensor(light_params).expand(n_px, -1)
        torch_base = base(torch.as_tensor(xy), torch.as_tensor(aux), params).numpy()
        torch_res = residual(torch.as_tensor(xy), torch.as_tensor(aux), params).numpy()
    mask_flat = region_mask.reshape(-1).astype(np.float32)
    torch_composite = torch_base + torch_res * mask_flat[:, None]

    def replica(flat, dims, activation):
        return numpy_forward(
            xy, aux, light_params, flat, dims, None, None, 0, 0, output_activation=activation
        )

    np_composite = (
        replica(base_flat, base_dims, "softplus")
        + replica(res_flat, res_dims, "linear") * mask_flat[:, None]
    )
    self_check = float(np.max(np.abs(np_composite - torch_composite)))
    pixels = np.concatenate([xy, aux], axis=1).astype(np.float32)
    return {
        "base_flat": base_flat,
        "base_dims": base_dims,
        "residual_flat": res_flat,
        "residual_dims": res_dims,
        "pixels": pixels,
        "mask": mask_flat,
        "self_check_max_abs_diff": self_check,
    }


def export_g1(args: argparse.Namespace) -> dict:
    src = Path(args.g1_dir)
    report = json.loads((src / "report.json").read_text())
    fixture = report["fixture"]
    n_frames = int(fixture["frames"])
    light = fixture["light"]
    light_params = np.array([*light["center"], light["radius"]], dtype=np.float32)

    base = TorchNRP.load(str(src / "models" / "base.pt"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames_meta = []
    worst_check = 0.0
    base_dims = None
    residual_dims = None
    for frame in range(n_frames):
        residual = ResidualNRP.load(str(src / "models" / f"residual_frame_{frame:04d}.pt"))
        gbuf = load_g1_gbuffer(src / "frames" / f"gbuffer_frame_{frame:04d}.npz")
        region_mask = np.load(src / "frames" / f"region_mask_{frame:04d}.npy")
        target = np.load(src / "frames" / f"target_frame_{frame:04d}.npy").astype(np.float32)
        blobs = export_g1_frame(base, residual, gbuf, region_mask, light_params)
        worst_check = max(worst_check, blobs["self_check_max_abs_diff"])
        if frame == 0:
            blobs["base_flat"].tofile(out_dir / "base_mlp.bin")
            base_dims = blobs["base_dims"]
            residual_dims = blobs["residual_dims"]
        blobs["residual_flat"].tofile(out_dir / f"residual_mlp_{frame:04d}.bin")
        blobs["pixels"].tofile(out_dir / f"pixels_{frame:04d}.bin")
        blobs["mask"].tofile(out_dir / f"region_mask_{frame:04d}.bin")
        target.tofile(out_dir / f"reference_{frame:04d}.bin")
        frames_meta.append(
            {
                "frame": frame,
                "sphere_center": report["frames_detail"][frame]["sphere_center"],
                "self_check_max_abs_diff": blobs["self_check_max_abs_diff"],
            }
        )

    manifest = {
        "rung": "G2 (G1 moving-object panel)",
        "source": str(src),
        "resolution": [int(fixture["resolution"][0]), int(fixture["resolution"][1])],
        "frames": n_frames,
        "light": light,
        "aux_dim": 7,
        "light_param_dim": 4,
        "encoding": None,
        "base": {"mlp_dims": base_dims, "output_activation": "softplus"},
        "residual": {"mlp_dims": residual_dims, "output_activation": "linear"},
        "g1_recovery_target_met": report["recovery_target_met_by_regime_d"],
        "frames_detail": frames_meta,
        "numpy_replica_vs_torch_max_abs_diff": worst_check,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--g1", action="store_true", help="export the G1 moving-object panel")
    parser.add_argument("--model", default="out/kitchen-512-torch/model.pt")
    parser.add_argument("--cache", default="out/kitchen-512/path_cache.npz")
    parser.add_argument("--g1-dir", default="out/g1-residual")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--light-radius", type=float, default=0.3)
    parser.add_argument(
        "--link-box",
        type=float,
        nargs=6,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        default=[-0.5, 0.0, 0.0, 1.5, 1.2, 2.0],
        help="first-hit position box defining the linking layer (kitchen mode)",
    )
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = "out/g2-demo/export-g1" if args.g1 else "out/g2-demo/export"

    manifest = (export_g1 if args.g1 else export_kitchen)(args)
    check = manifest["numpy_replica_vs_torch_max_abs_diff"]
    summary = {
        "out_dir": args.out_dir,
        "resolution": manifest["resolution"],
        "numpy_replica_vs_torch_max_abs_diff": check,
    }
    if not args.g1:
        summary["link_pixel_fraction"] = manifest["demo"]["link_pixel_fraction"]
    print(json.dumps(summary, indent=2))
    if check > SELF_CHECK_TOLERANCE:
        raise SystemExit(f"exported-format self-check failed: max abs diff {check}")


if __name__ == "__main__":
    main()
