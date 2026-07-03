"""Component ablation + spp sweep (roadmap item 10, paper Table 2 / Fig. 7).

Trains the paper's five architecture/supervision variants — None (raw pixel coords,
no aux, raw targets), Aux, Aux+Den, Aux+Enc, Aux+Enc+Den — on the Mitsuba cornell
box, once per SAMPLEPATHS spp in {8, 16, 32}. Every cell shares the identical model
budget, optimizer, iteration count, batch schedule, and seeds; the *only* differences
are the two model input switches (`use_aux`, `use_encoding`) and whether pool targets
are denoised. *Every* cell — across variants and across spp — is scored on one common
held-out light set (fresh lights from a dedicated RNG, never seen in training)
against GATHERLIGHT references from a separate high-spp reference cache
(`--ref-spp`, default 128), so the spp axis measures training-cache quality against
a fixed clean target rather than against each cache's own noise. Metrics are the
paper's four:
SMAPE + PSNR on linear radiance, SSIM + FLIP on Reinhard-tonemapped sRGB
(`nrp.metrics.tonemap_srgb`; FLIP is display-referred by definition).

Deterministic from one command (fixed seeds, CPU default; timing fields are the only
run-to-run variation) and every cell's full training config is embedded in the report:

  uv run python examples/ablation.py --out out/ablation/report.json

Needs the `mitsuba` extra (and `oidn` for the paper's denoiser; falls back to the
bilateral filter with a note in the report). `--producer toy` swaps in the
dependency-free toy tracer for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import torch  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.metrics import flip, psnr, smape, ssim, tonemap_srgb  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.denoise import oidn_available  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.train import light_param_vector, pixel_tensors, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# The paper's Table-2 component sets: aux G-buffer features, hashgrid encoding of the
# pixel coordinates, denoised supervision targets.
VARIANTS: dict[str, dict[str, bool]] = {
    "none": {"aux": False, "enc": False, "den": False},
    "aux": {"aux": True, "enc": False, "den": False},
    "aux_den": {"aux": True, "enc": False, "den": True},
    "aux_enc": {"aux": True, "enc": True, "den": False},
    "aux_enc_den": {"aux": True, "enc": True, "den": True},
}


def export_cache(path: Path, producer: str, width: int, height: int, spp: int, seed: int) -> None:
    if producer == "toy":
        from nrp.toy_tracer import trace_path_cache

        cache = trace_path_cache(width, height, spp, 3, seed=seed)
    else:
        from nrp.mitsuba_exporter import (
            _import_mitsuba,
            _load_mitsuba,
            _load_scene,
            export_path_cache,
            export_path_cache_wavefront,
            pick_jit_variant,
        )

        mode = "wavefront" if pick_jit_variant(_import_mitsuba()) else "scalar"
        mi = _load_mitsuba(mode)
        scene = _load_scene(mi, "builtin:cornell-box", width, height)
        export = export_path_cache_wavefront if mode == "wavefront" else export_path_cache
        cache = export(scene, mi, width, height, spp, 4, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache.save(str(path))


def cell_config(
    args, cache_path: Path, out_dir: Path, switches: dict[str, bool], denoise_method: str
) -> dict:
    """One cell's full training config — identical budget and seeds across cells; only
    the model switches and target denoising differ."""
    return {
        "cache": str(cache_path),
        "out_dir": str(out_dir),
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.5},
        "sampling": "segments",
        "pool": {"size": args.pool_size, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": switches["den"], "method": denoise_method},
        "iters": args.iters,
        "batch_pixels": args.batch_pixels,
        "lr": 0.005,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "use_aux": switches["aux"],
            "use_encoding": switches["enc"],
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 12,
                "base_resolution": 4,
                "finest_resolution": args.width,
            },
        },
        "n_val_lights": 4,  # train.py's internal val set; scoring uses the common set
        "seed": args.seed,
        "device": args.device,
        "gather_backend": "numpy",
    }


def build_val_set(ref_cache: PathCache, seed: int, n_val: int) -> list[dict]:
    """Common held-out light set for every cell: dedicated RNG stream (never used in
    training), lights sampled on and references gathered from the high-spp reference
    cache with the authoritative numpy gather."""
    rng = np.random.default_rng([seed, 0xAB1A])
    val = []
    for _ in range(n_val):
        light = sample_light(ref_cache, rng, "sphere", {"radius_min": 0.1, "radius_max": 0.5})
        val.append(
            {
                "light": light.to_dict(),
                "params": light_param_vector(light),
                "raw": gather_light(ref_cache, light),
            }
        )
    return val


def score(model_path: Path, cache: PathCache, val_set: list[dict], device) -> dict:
    """All four paper metrics per held-out light: SMAPE/PSNR on linear radiance,
    SSIM/FLIP on tonemapped sRGB."""
    model = TorchNRP.load(str(model_path)).to(device)
    model.eval()
    xy, aux = pixel_tensors(cache, device)
    n_px = xy.shape[0]
    per_light: dict[str, list[float]] = {"psnr_db": [], "smape": [], "ssim": [], "flip": []}
    with torch.no_grad():
        for entry in val_set:
            params = torch.as_tensor(entry["params"], dtype=torch.float32, device=device)
            pred = model(xy, aux, params.expand(n_px, -1)).cpu().numpy().astype(np.float64)
            pred = pred.reshape(cache.height, cache.width, 3)
            ref = entry["raw"]
            per_light["psnr_db"].append(psnr(pred, ref))
            per_light["smape"].append(smape(pred, ref))
            per_light["ssim"].append(ssim(tonemap_srgb(pred), tonemap_srgb(ref), data_range=1.0))
            per_light["flip"].append(flip(tonemap_srgb(pred), tonemap_srgb(ref)))
    out = {f"{k}_per_light": v for k, v in per_light.items()}
    for k, v in per_light.items():
        out[f"{k}_mean"] = float(np.mean(v))
        out[f"{k}_std"] = float(np.std(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "out/ablation/report.json"))
    parser.add_argument("--producer", choices=["mitsuba", "toy"], default="mitsuba")
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--spp", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--batch-pixels", type=int, default=4096)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument(
        "--ref-spp",
        type=int,
        default=128,
        help="spp of the separate reference cache all cells are scored against",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu", help="cpu keeps the report bit-deterministic")
    parser.add_argument(
        "--denoise-method",
        choices=["auto", "oidn", "bilateral"],
        default="auto",
        help="auto: OIDN (the paper's denoiser) when available, else bilateral",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    out_root = out_path.parent
    out_root.mkdir(parents=True, exist_ok=True)

    denoise_method = args.denoise_method
    if denoise_method == "auto":
        denoise_method = "oidn" if oidn_available() else "bilateral"
        print(f"denoiser: {denoise_method}")

    device = torch.device(args.device)

    # Reference cache: higher spp, independent seed — the common ground truth every
    # cell is scored against.
    ref_path = out_root / f"cache_{args.producer}_{args.width}x{args.height}_ref{args.ref_spp}.npz"
    if not ref_path.exists():
        t0 = time.perf_counter()
        export_cache(ref_path, args.producer, args.width, args.height, args.ref_spp, args.seed + 1)
        print(f"exported {ref_path.name} in {time.perf_counter() - t0:.1f}s")
    ref_cache = PathCache.load(str(ref_path))
    val_set = build_val_set(ref_cache, args.seed, args.n_val)

    cells: dict[str, dict] = {}
    for spp in args.spp:
        cache_path = out_root / f"cache_{args.producer}_{args.width}x{args.height}_spp{spp}.npz"
        if not cache_path.exists():
            t0 = time.perf_counter()
            export_cache(cache_path, args.producer, args.width, args.height, spp, args.seed)
            print(f"exported {cache_path.name} in {time.perf_counter() - t0:.1f}s")
        cache = PathCache.load(str(cache_path))

        cells[str(spp)] = {}
        for name in args.variants:
            cfg = cell_config(
                args, cache_path, out_root / f"spp{spp}_{name}", VARIANTS[name], denoise_method
            )
            print(f"--- spp {spp}, variant {name}")
            report = train(cfg)
            scores = score(out_root / f"spp{spp}_{name}" / "model.pt", cache, val_set, device)
            cells[str(spp)][name] = {
                "config": cfg,
                "components": VARIANTS[name],
                "parameter_count": report["parameter_count"],
                "train_seconds": report["train_seconds"],
                "pool_build_seconds": report["pool_build_seconds"],
                **scores,
            }
            print(
                f"spp {spp} {name}: PSNR {scores['psnr_db_mean']:.2f} dB, "
                f"SSIM {scores['ssim_mean']:.4f}, FLIP {scores['flip_mean']:.4f}, "
                f"SMAPE {scores['smape_mean']:.3f}"
            )

    summary = {
        "producer": args.producer,
        "scene": "builtin:cornell-box" if args.producer == "mitsuba" else "toy box",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "variants": args.variants,
        "iters": args.iters,
        "seed": args.seed,
        "device": args.device,
        "denoise_method": denoise_method,
        "n_val_lights": args.n_val,
        "ref_spp": args.ref_spp,
        "ref_cache_segments": ref_cache.segment_count,
        "val_lights": [v["light"] for v in val_set],
        "metric_conventions": {
            "psnr_db": "linear radiance, peak = per-light reference max",
            "smape": "linear radiance, eps 1e-3",
            "ssim": "Reinhard-tonemapped sRGB, data_range 1.0",
            "flip": "LDR-FLIP on Reinhard-tonemapped sRGB, 67.02 ppd",
        },
        "cells": cells,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
