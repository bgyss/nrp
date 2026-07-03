"""E3 standard-vs-guided proxy A/B on a toy spherical placement region."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.light_aware_sampling import region_density  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.torch_backend.train import train  # noqa: E402


def make_cfg(root: Path, name: str, region: dict, guided: bool, args) -> dict:
    return {
        "cache": str(root / name / "path_cache.npz"),
        "out_dir": str(root / name),
        "trace": {
            "width": args.width,
            "height": args.height,
            "spp": args.spp,
            "bounces": args.bounces,
            "seed": 61,
            **(
                {"light_region": region, "guide_probability": args.guide_probability}
                if guided
                else {}
            ),
        },
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.06, "radius_max": 0.18},
        "sampling": "segments",
        "pool": {"size": 12, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": False},
        "iters": args.iters,
        "batch_pixels": min(512, args.width * args.height),
        "lr": 0.006,
        "model": {
            "hidden_width": 32,
            "hidden_layers": 2,
            "encoding": {
                "levels": 4,
                "features_per_level": 2,
                "table_size_log2": 8,
                "base_resolution": 4,
                "finest_resolution": args.width,
            },
        },
        "n_val_lights": 4,
        "seed": 5,
        "device": "cpu",
    }


def eval_fixed_lights(model_path: Path, cache: PathCache, lights: dict[str, SphereLight]) -> dict:
    model = TorchNRP.load(str(model_path))
    out = {}
    for name, light in lights.items():
        pred = relight(model, cache, [light])
        ref = gather_light(cache, light)
        out[name] = {
            "psnr_db_vs_own_cache_gather": psnr(pred, ref),
            "reference_mean": float(ref.mean()),
            "prediction_mean": float(pred.mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/light-aware-proxy-ab/report.json")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--iters", type=int, default=350)
    parser.add_argument("--guide-probability", type=float, default=0.5)
    args = parser.parse_args()

    out_path = Path(args.out)
    root = out_path.parent
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    region = {"type": "sphere", "center": [0.45, 0.75, 0.45], "radius": 0.12}
    configs = {
        "standard": make_cfg(root, "standard", region, False, args),
        "guided": make_cfg(root, "guided", region, True, args),
    }
    reports = {}
    for name, cfg in configs.items():
        reports[name] = train(cfg)

    caches = {name: PathCache.load(cfg["cache"]) for name, cfg in configs.items()}
    lights = {
        "inside_region": SphereLight(center=region["center"], radius=region["radius"]),
        "open_region": SphereLight(center=[0.75, 0.75, 0.35], radius=0.12),
    }
    fixed_eval = {
        name: eval_fixed_lights(Path(configs[name]["out_dir"]) / "model.pt", caches[name], lights)
        for name in configs
    }
    density = {name: region_density(cache, region) for name, cache in caches.items()}
    inside_gain_db = (
        fixed_eval["guided"]["inside_region"]["psnr_db_vs_own_cache_gather"]
        - fixed_eval["standard"]["inside_region"]["psnr_db_vs_own_cache_gather"]
    )
    open_regression_db = (
        fixed_eval["guided"]["open_region"]["psnr_db_vs_own_cache_gather"]
        - fixed_eval["standard"]["open_region"]["psnr_db_vs_own_cache_gather"]
    )
    report = {
        "extension": "E3",
        "scope": "toy standard-vs-guided proxy A/B for a spherical placement region",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "iters": args.iters,
        "guide_probability": args.guide_probability,
        "light_region": region,
        "density": density,
        "train": {
            name: {
                "train_seconds": reports[name]["train_seconds"],
                "pool_build_seconds": reports[name]["pool_build_seconds"],
                "val_psnr_db_vs_raw_mean": reports[name]["val_psnr_db_vs_raw_mean"],
                "iters_per_second": reports[name]["iters_per_second"],
            }
            for name in configs
        },
        "fixed_lights": fixed_eval,
        "inside_region_psnr_gain_db_guided_minus_standard": inside_gain_db,
        "open_region_psnr_delta_db_guided_minus_standard": open_regression_db,
        "cache_size_delta_segments": (
            caches["guided"].segment_count - caches["standard"].segment_count
        ),
        "limitations": [
            "This is a toy placement-region A/B, not the full occluder/lamp-shade reproduction.",
            "The criterion target is reported directly; this small CPU run is not "
            "tuned to guarantee +3 dB.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
