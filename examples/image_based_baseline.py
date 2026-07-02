"""Image-based baseline vs path-based pool training (roadmap item 9, paper Fig. 6).

The paper's central data-efficiency claim: training on path data (a live pool whose
images are continually replaced with fresh light configurations) beats training on a
fixed dataset of R pre-rendered images. This script replicates that comparison at toy
scale on the Mitsuba cornell box. The image-based regime *is* the pool trainer with
replacement disabled (`replace_count: 0`), so all regimes share the identical model,
optimizer, seed, batch schedule, and code path — the only difference is whether the
supervision set is fixed (R ∈ {64, 256, 1024} images) or refreshed (pool 64 + 2
images every 5 iterations = 64 + 2·iters/5 total).

All regimes are scored on a common held-out set of ≥ 24 fresh lights (dedicated RNG),
asserted disjoint from every regime's recorded supervision lights, with per-light
tonemapped PSNR (Reinhard I/(1+I), peak 1 — the paper's Fig. 6 metric) against raw
GATHERLIGHT references from the authoritative numpy gather. One command:

  uv run python examples/image_based_baseline.py --device mps --out out/image-based/report.json
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
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.train import light_param_vector, pixel_tensors, train  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def base_config(cache: str, out_root: str, device: str, iters: int, seed: int) -> dict:
    return {
        "cache": cache,
        "out_dir": out_root,  # per-regime out_dir set later
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.5},
        "sampling": "segments",
        "denoise": {"enabled": True, "method": "oidn"},
        "iters": iters,
        "batch_pixels": 4096,
        "lr": 0.005,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 12,
                "base_resolution": 4,
                "finest_resolution": 128,
            },
        },
        "n_val_lights": 12,
        "seed": seed,
        "device": device,
        "gather_backend": "torch",
        "record_supervision_lights": True,
    }


def tonemap(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + x)


def eval_common_val(model_path: str, cache: PathCache, val_set: list[dict], device) -> dict:
    model = TorchNRP.load(model_path).to(device)
    model.eval()
    xy, aux = pixel_tensors(cache, device)
    n_px = xy.shape[0]
    tm_psnrs, lin_psnrs = [], []
    with torch.no_grad():
        for entry in val_set:
            params = torch.as_tensor(entry["params"], dtype=torch.float32, device=device).expand(
                n_px, -1
            )
            pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
            tm_psnrs.append(psnr(tonemap(pred), tonemap(entry["raw"]), peak=1.0))
            lin_psnrs.append(psnr(pred, entry["raw"]))
    return {
        "psnr_tonemapped_per_light": tm_psnrs,
        "psnr_tonemapped_mean": float(np.mean(tm_psnrs)),
        "psnr_tonemapped_std": float(np.std(tm_psnrs)),
        "psnr_linear_per_light": lin_psnrs,
        "psnr_linear_mean": float(np.mean(lin_psnrs)),
    }


def assert_val_disjoint(val_set: list[dict], train_params: list[list[float]], regime: str):
    train_arr = np.asarray(train_params)
    for entry in val_set:
        gap = np.abs(train_arr - np.asarray(entry["params"])).max(axis=1).min()
        if gap <= 1e-9:
            raise AssertionError(f"validation light appears in {regime} training set")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=str(ROOT / "out/mitsuba/path_cache_128_64spp.npz"))
    parser.add_argument("--out", default=str(ROOT / "out/image-based/report.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-val", type=int, default=24)
    parser.add_argument("--image-counts", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument(
        "--path-pool-sizes",
        type=int,
        nargs="+",
        default=[64, 256],
        help="pool sizes for the path-based regime; the first is the headline "
        "'path_based' cell (§4.4 default-style pool), extras probe pool-diversity "
        "sensitivity",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    out_root = out_path.parent
    out_root.mkdir(parents=True, exist_ok=True)

    cache = PathCache.load(args.cache)
    cfg0 = base_config(args.cache, str(out_root), args.device, args.iters, args.seed)

    # Common held-out validation set: fresh lights from a dedicated RNG stream,
    # raw GATHERLIGHT references from the authoritative numpy gather.
    val_rng = np.random.default_rng([args.seed, 1009])
    val_set = []
    t0 = time.perf_counter()
    for _ in range(args.n_val):
        light = sample_light(cache, val_rng, cfg0["light_type"], cfg0["light_bounds"], "segments")
        val_set.append(
            {
                "light": light.to_dict() if hasattr(light, "to_dict") else None,
                "params": light_param_vector(light),
                "raw": gather_light(cache, light).reshape(-1, 3),
            }
        )
    print(f"built {args.n_val}-light validation set in {time.perf_counter() - t0:.1f}s")

    regimes: dict[str, dict] = {}
    for i, p in enumerate(args.path_pool_sizes):
        name = "path_based" if i == 0 else f"path_pool{p}"
        regimes[name] = {"pool": {"size": p, "replace_every": 5, "replace_count": 2}}
    for r in args.image_counts:
        regimes[f"image_{r}"] = {"pool": {"size": r, "replace_every": 5, "replace_count": 0}}

    device = torch.device(args.device)
    results: dict = {}
    for name, override in regimes.items():
        cfg = json.loads(json.dumps(cfg0))  # deep copy
        cfg.update(override)
        cfg["out_dir"] = str(out_root / name)
        print(f"--- regime {name}: pool {cfg['pool']}")
        report = train(cfg)
        assert_val_disjoint(val_set, report["supervision_light_params"], name)
        scores = eval_common_val(str(out_root / name / "model.pt"), cache, val_set, device)
        results[name] = {
            "pool": cfg["pool"],
            "supervision_images": report["supervision_images"],
            "supervision_seconds": report["supervision_seconds"],
            "train_seconds": report["train_seconds"],
            **scores,
        }
        print(
            f"{name}: {report['supervision_images']} supervision images "
            f"({report['supervision_seconds']:.1f}s), tonemapped PSNR "
            f"{scores['psnr_tonemapped_mean']:.2f} ± {scores['psnr_tonemapped_std']:.2f} dB"
        )

    best_image = max(
        (r for r in results if r.startswith("image_")),
        key=lambda r: results[r]["psnr_tonemapped_mean"],
    )
    gap = (
        results["path_based"]["psnr_tonemapped_mean"] - results[best_image]["psnr_tonemapped_mean"]
    )
    summary = {
        "cache": args.cache,
        "resolution": [cache.width, cache.height],
        "iters": args.iters,
        "seed": args.seed,
        "n_val_lights": args.n_val,
        "val_lights": [v["light"] for v in val_set],
        "best_image_based": best_image,
        "path_minus_best_image_db": gap,
        "regimes": results,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"path-based beats best image-based ({best_image}) by {gap:+.2f} dB tonemapped; "
        f"wrote {out_path}"
    )


if __name__ == "__main__":
    main()
