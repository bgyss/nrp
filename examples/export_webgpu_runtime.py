"""T4 (docs/production-track.md): export a trained TorchNRP — hashgrid encoding
included — plus the real scene's G-buffer as flat float32 blobs a WebGPU backend can
consume directly.

Unlike `export_js_viewer.py` (which sidesteps the encoding by training a
`use_encoding=False` toy proxy), this exports the *actual* T1-scene proxy: the
hashgrid tables are dumped alongside the MLP weights and the WGSL shader in
`webgpu/bench_t4.mjs` reimplements the level-by-level lookup. Outputs, in --out-dir:

  - manifest.json  — model architecture, encoding metadata (per-level resolution,
    dense/hashed flag, table offset/size), default light, blob byte counts, and the
    exporter's own numpy-vs-torch forward self-check result.
  - mlp.bin        — float32; per layer: row-major weight matrix then bias vector.
  - tables.bin     — float32; hashgrid tables concatenated level by level.
  - pixels.bin     — float32; per pixel (row-major): xy (2) + aux (7: albedo,
    depth, normal), the same inputs `pixel_tensors` feeds the torch model.
  - reference.bin  — float32; (H*W*3) PyTorch forward output for the default light,
    the parity target the WebGPU backend must match.

Only the aux G-buffer arrays are read from the cache .npz (lazily — the T1 cache's
segment arrays are ~2 GiB and irrelevant to inference).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.model import TorchNRP  # noqa: E402


def load_gbuffer(cache_path: str) -> dict:
    """Lazily read width/height + aux arrays from a path-cache .npz (v1 or v2 layout;
    aux buffers are stored unpacked in both)."""
    with np.load(cache_path) as z:
        return {
            "width": int(z["width"]),
            "height": int(z["height"]),
            "albedo": np.asarray(z["albedo"], dtype=np.float64),
            "depth": np.asarray(z["depth"], dtype=np.float64),
            "normal": np.asarray(z["normal"], dtype=np.float64),
            "position": np.asarray(z["position"], dtype=np.float64),
        }


def pixel_arrays(gbuf: dict) -> tuple[np.ndarray, np.ndarray]:
    """Same convention as nrp.torch_backend.train.pixel_tensors, as float32 numpy."""
    h, w = gbuf["height"], gbuf["width"]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    aux = np.concatenate(
        [
            gbuf["albedo"].reshape(-1, 3),
            gbuf["depth"].reshape(-1, 1),
            gbuf["normal"].reshape(-1, 3),
        ],
        axis=1,
    )
    return xy.astype(np.float32), aux.astype(np.float32)


def export_mlp(model: TorchNRP) -> tuple[np.ndarray, list[int]]:
    """Flatten Linear layers in forward order: weight rows then bias, per layer."""
    chunks: list[np.ndarray] = []
    dims: list[int] = []
    for module in model.mlp:
        if isinstance(module, torch.nn.Linear):
            if not dims:
                dims.append(module.in_features)
            dims.append(module.out_features)
            chunks.append(module.weight.detach().numpy().astype(np.float32).reshape(-1))
            chunks.append(module.bias.detach().numpy().astype(np.float32))
    return np.concatenate(chunks), dims


def export_encoding(model: TorchNRP) -> tuple[np.ndarray, list[dict]]:
    """Concatenate hashgrid tables; per-level metadata carries everything the WGSL
    lookup needs (resolution, dense flag, offset in floats, entry count)."""
    enc = model.encoding
    levels: list[dict] = []
    chunks: list[np.ndarray] = []
    offset = 0
    for level, res in enumerate(enc.resolutions):
        table = enc.tables[level].detach().numpy().astype(np.float32)
        levels.append(
            {
                "resolution": res,
                "dense": bool(enc._dense[level]),
                "offset_floats": offset,
                "entries": table.shape[0],
            }
        )
        chunks.append(table.reshape(-1))
        offset += table.size
    return np.concatenate(chunks), levels


def numpy_forward(
    xy: np.ndarray,
    aux: np.ndarray,
    light_params: np.ndarray,
    mlp_flat: np.ndarray,
    mlp_dims: list[int],
    tables_flat: np.ndarray | None,
    level_meta: list[dict] | None,
    features_per_level: int,
    table_size: int,
) -> np.ndarray:
    """Reimplement the exported forward pass from the flat blobs alone (float32).

    This is the format's semantic contract: the WGSL shader in bench_t4.mjs mirrors
    this function, and the exporter self-checks it against the torch module.
    """
    if tables_flat is not None:
        feats = []
        for meta in level_meta:
            res = meta["resolution"]
            table = tables_flat[
                meta["offset_floats"] : meta["offset_floats"] + meta["entries"] * features_per_level
            ].reshape(meta["entries"], features_per_level)
            pos = xy.astype(np.float32) * np.float32(res)
            pos0 = np.clip(np.floor(pos).astype(np.int64), 0, res)
            frac = pos - pos0
            x0, y0 = pos0[:, 0], pos0[:, 1]
            x1 = np.minimum(x0 + 1, res)
            y1 = np.minimum(y0 + 1, res)
            if meta["dense"]:

                def idx(ix, iy, res=res, entries=meta["entries"]):
                    return (iy * (res + 1) + ix) % entries

            else:

                def idx(ix, iy, entries=meta["entries"]):
                    return (ix ^ ((iy * 2654435761) & (table_size - 1))) % entries

            wx = frac[:, 0:1]
            wy = frac[:, 1:2]
            feats.append(
                table[idx(x0, y0)] * (1 - wx) * (1 - wy)
                + table[idx(x1, y0)] * wx * (1 - wy)
                + table[idx(x0, y1)] * (1 - wx) * wy
                + table[idx(x1, y1)] * wx * wy
            )
        px = np.concatenate(feats, axis=1)
    else:
        px = xy.astype(np.float32)
    h = np.concatenate(
        [px, aux, np.broadcast_to(light_params, (xy.shape[0], light_params.size))], axis=1
    ).astype(np.float32)
    off = 0
    n_layers = len(mlp_dims) - 1
    for layer in range(n_layers):
        in_dim, out_dim = mlp_dims[layer], mlp_dims[layer + 1]
        w = mlp_flat[off : off + in_dim * out_dim].reshape(out_dim, in_dim)
        off += in_dim * out_dim
        b = mlp_flat[off : off + out_dim]
        off += out_dim
        h = h @ w.T + b
        if layer < n_layers - 1:
            h = np.maximum(h, 0.0)
    return np.log1p(np.exp(np.minimum(h, 30.0))) + np.maximum(h - 30.0, 0.0)  # stable softplus


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="out/kitchen-512-torch/model.pt")
    parser.add_argument("--cache", default="out/kitchen-512/path_cache.npz")
    parser.add_argument("--out-dir", default="out/t4-runtime/export")
    parser.add_argument(
        "--light-radius", type=float, default=0.3, help="default sphere light radius"
    )
    args = parser.parse_args()

    model = TorchNRP.load(args.model)
    model.eval()
    if model.light_type != "sphere":
        raise SystemExit("T4 exporter currently supports sphere-light proxies only")

    gbuf = load_gbuffer(args.cache)
    xy, aux = pixel_arrays(gbuf)
    n_px = xy.shape[0]

    # Default light: hover above the scene's mean first-hit position — arbitrary but
    # scene-plausible (kitchen coordinates are metric, not [0,1]); recorded in the
    # manifest so the parity check is fully reproducible.
    mean_pos = gbuf["position"].reshape(-1, 3).mean(axis=0)
    center = np.array([mean_pos[0], mean_pos[1] + 0.5, mean_pos[2]], dtype=np.float64)
    light_params = np.array([*center, args.light_radius], dtype=np.float32)

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
    ).astype(np.float32)
    self_check_max_abs_diff = float(np.max(np.abs(replica - reference)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mlp_flat.tofile(out_dir / "mlp.bin")
    if tables_flat is not None:
        tables_flat.tofile(out_dir / "tables.bin")
    pixels = np.concatenate([xy, aux], axis=1).astype(np.float32)
    pixels.tofile(out_dir / "pixels.bin")
    reference.tofile(out_dir / "reference.bin")

    manifest = {
        "rung": "T4",
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
        "files": {
            "mlp.bin": int(mlp_flat.size),
            "tables.bin": 0 if tables_flat is None else int(tables_flat.size),
            "pixels.bin": int(pixels.size),
            "reference.bin": int(reference.size),
        },
        "numpy_replica_vs_torch_max_abs_diff": self_check_max_abs_diff,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "resolution",
                    "mlp_dims",
                    "parameter_count",
                    "numpy_replica_vs_torch_max_abs_diff",
                    "default_light",
                )
            },
            indent=2,
        )
    )
    if self_check_max_abs_diff > 1e-3:
        raise SystemExit(
            f"exported-format self-check failed: max abs diff {self_check_max_abs_diff}"
        )


if __name__ == "__main__":
    main()
