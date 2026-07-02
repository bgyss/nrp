"""Table-3-style inverse-optimization grid + quad recovery check (roadmap item 4).

Replicates the *structure* of the paper's Table 3 on the Mitsuba cornell box: jointly
recover N ∈ {1, 3, 5} hidden sphere lights at pixel fractions α ∈ {1.0, 0.25, 0.05,
0.01}, ≥ 5 runs per cell (fresh hidden lights and init per run), reporting re-rendered
GATHERLIGHT PSNR against the target and wall-clock per run. One command, deterministic:

  uv run python examples/inverse_grid.py --model out/mitsuba-torch/model.pt \
      --cache out/mitsuba/path_cache.npz --out out/inverse-grid/report.json

`--quad-check` instead trains a full-quality toy quad proxy (96 spp, 10k iterations)
and verifies 1-light quad recovery (center error < 0.05, best of 12 restarts ranked by
the physical GATHERLIGHT re-render) — the roadmap's quad verification, kept out of the
unit suite because of the ~4 min training cost:

  uv run python examples/inverse_grid.py --quad-check --out out/inverse-grid/quad_check.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_lights  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.optimize_lights import (  # noqa: E402
    DEFAULT_BOUNDS,
    optimize,
    random_init,
)
from nrp.torch_backend.sampling import sample_light  # noqa: E402

N_LIGHTS = [1, 3, 5]
PIXEL_FRACTIONS = [1.0, 0.25, 0.05, 0.01]
RUNS_PER_CELL = 5
STEPS = 500
LR = 0.05

# The Mitsuba cornell box spans roughly [-1,1]^3 — the toy-scene DEFAULT_BOUNDS
# would confine lights to a corner of it.
CORNELL_BOUNDS = {
    "center_min": [-0.9, -0.9, -0.9],
    "center_max": [0.9, 0.9, 0.9],
    "radius_min": 0.05,
    "radius_max": 0.5,
    "size_min": 0.05,
    "size_max": 0.6,
}


def run_grid(model_path: str, cache_path: str, out_path: str) -> None:
    model = TorchNRP.load(model_path)
    cache = PathCache.load(cache_path)
    bounds = dict(CORNELL_BOUNDS)
    bounds["radius_min"] = 0.1  # match the training bounds of the cornell config
    cells = []
    for n_lights in N_LIGHTS:
        for alpha in PIXEL_FRACTIONS:
            runs = []
            for run in range(RUNS_PER_CELL):
                seed = 1000 * n_lights + run
                rng = np.random.default_rng(seed)
                true_lights = [
                    sample_light(
                        cache,
                        rng,
                        "sphere",
                        {"radius_min": bounds["radius_min"], "radius_max": bounds["radius_max"]},
                        "segments",
                    )
                    for _ in range(n_lights)
                ]
                for light in true_lights:
                    light.rgb = 2.0 + 8.0 * rng.random(3)
                target = gather_lights(cache, true_lights)
                init = random_init(rng, "sphere", bounds, n_lights)
                t0 = time.perf_counter()
                rep = optimize(
                    model,
                    cache,
                    target,
                    init,
                    bounds,
                    steps=STEPS,
                    lr=LR,
                    pixel_fraction=alpha,
                    seed=seed,
                )
                seconds = time.perf_counter() - t0
                runs.append(
                    {
                        "seed": seed,
                        "seconds": seconds,
                        "gather_psnr_db": rep["gather_vs_target_psnr_db"],
                        "proxy_psnr_db": rep["proxy_vs_target_psnr_db"],
                    }
                )
            cell = {
                "n_lights": n_lights,
                "pixel_fraction": alpha,
                "steps": STEPS,
                "runs": runs,
                "gather_psnr_db_mean": float(np.mean([r["gather_psnr_db"] for r in runs])),
                "gather_psnr_db_std": float(np.std([r["gather_psnr_db"] for r in runs])),
                "seconds_mean": float(np.mean([r["seconds"] for r in runs])),
            }
            cells.append(cell)
            print(
                f"N={n_lights} alpha={alpha:<5}: {cell['gather_psnr_db_mean']:.2f} "
                f"± {cell['gather_psnr_db_std']:.2f} dB, {cell['seconds_mean']:.1f} s/run"
            )
    report = {
        "model": model_path,
        "cache": cache_path,
        "bounds": bounds,
        "runs_per_cell": RUNS_PER_CELL,
        "cells": cells,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")


def run_quad_check(out_path: str) -> None:
    """Train a toy quad proxy and verify 1-light recovery (center error < 0.05).

    The recipe is the product of an explicit conditioning study (docs/performance.md):
    quad recovery is proxy-fidelity-bound, so the check uses a 96-spp cache and 10k
    iterations (a 24-spp/3k-iteration proxy plateaus at ~0.17 center error), a
    near-surface dim fixture (Reinhard sensitivity falls as 1/(1+I)^2), 12 restarts,
    and physical GATHERLIGHT re-ranking of the restarts.
    """
    import tempfile

    from nrp.lights import QuadLight
    from nrp.torch_backend.train import train
    from nrp.toy_tracer import trace_path_cache

    tmp = tempfile.mkdtemp()
    cache_path = os.path.join(tmp, "cache.npz")
    cache = trace_path_cache(48, 48, spp=96, max_bounces=3, seed=1)
    cache.save(cache_path)
    cfg = {
        "cache": cache_path,
        "out_dir": os.path.join(tmp, "out"),
        "light_type": "quad",
        "light_bounds": {"size_min": 0.08, "size_max": 0.4},
        "sampling": "segments",
        "pool": {"size": 96, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "radius": 2},
        "iters": 10000,
        "batch_pixels": 4096,
        "lr": 0.005,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 14,
                "base_resolution": 4,
                "finest_resolution": 48,
            },
        },
        "n_val_lights": 6,
        "seed": 0,
        "device": "cpu",
        "gather_backend": "torch",
    }
    train_report = train(cfg)
    model = TorchNRP.load(os.path.join(tmp, "out", "model.pt"))

    true = QuadLight(
        center=[0.5, 0.85, 0.5],
        normal=[0.0, -1.0, 0.0],
        width=0.3,
        height=0.3,
        rgb=[1.5, 1.5, 1.5],
    )
    target = gather_lights(cache, [true])
    best = None
    for restart in range(12):
        rng = np.random.default_rng(restart)
        init = random_init(rng, "quad", DEFAULT_BOUNDS, 1)
        rep = optimize(
            model,
            cache,
            target,
            init,
            DEFAULT_BOUNDS,
            steps=STEPS,
            lr=LR,
            pixel_fraction=0.25,
            seed=restart,
        )
        # Rank restarts by the physical re-render (see optimize_lights.main).
        if best is None or rep["gather_tonemapped_mse"] < best["gather_tonemapped_mse"]:
            best = rep
    opt = best["optimized_lights"][0]
    center_error = float(np.linalg.norm(np.array(opt["center"]) - true.center))
    result = {
        "proxy_val_psnr_db": train_report["val_psnr_db_vs_raw_mean"],
        "true_light": true.to_dict(),
        "optimized_light": opt,
        "center_error": center_error,
        "gather_psnr_db": best["gather_vs_target_psnr_db"],
        "passed": center_error < 0.05,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "optimized_light"}, indent=2))
    if not result["passed"]:
        raise SystemExit(f"quad recovery FAILED: center error {center_error:.4f} >= 0.05")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", help="trained sphere model .pt (grid mode)")
    parser.add_argument("--cache", help="path cache .npz (grid mode)")
    parser.add_argument("--quad-check", action="store_true", help="run the quad recovery check")
    parser.add_argument("--out", required=True, help="output JSON report")
    args = parser.parse_args()
    if args.quad_check:
        run_quad_check(args.out)
    else:
        if not (args.model and args.cache):
            parser.error("grid mode needs --model and --cache")
        run_grid(args.model, args.cache, args.out)


if __name__ == "__main__":
    main()
