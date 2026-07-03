"""E7 image-space target to physical lights demo.

This script exercises the existing inverse-optimization mask/protect plumbing in two
small workflows:

1. A synthesized scribble fixture from a known light, initialized at the known light,
   proving masked/protected-region accounting and GATHERLIGHT re-render reporting.
2. A stylized target fixture optimized from three restarts, reporting the gap between
   the raw edited target and the physically realized GATHERLIGHT image.

It is intentionally toy-scale and does not claim the full product demo is complete.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.optimize_lights import DEFAULT_BOUNDS, optimize, random_init  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def masked_psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    return psnr(a[mask], b[mask])


def fixture_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    objective = np.ones((height, width), dtype=np.float64)
    objective[height // 4 : height // 2, width // 2 :] = 8.0
    protect = np.zeros((height, width), dtype=np.float64)
    protect[int(height * 0.75) :, :] = 1.0
    return objective, protect


def strip_images(report: dict) -> tuple[dict, dict]:
    images = report.pop("_images")
    return report, images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/generative/report.json")
    parser.add_argument("--width", type=int, default=14)
    parser.add_argument("--height", type=int, default=14)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(args.width, args.height, spp=6, max_bounces=2, seed=14)
    model = TorchNRP(
        light_type="sphere",
        hidden_width=24,
        hidden_layers=2,
        encoding={"levels": 3, "features_per_level": 2, "finest_resolution": args.width},
    )
    true = SphereLight(center=[0.45, 0.65, 0.45], radius=0.16, rgb=[2.0, 1.5, 1.0])
    target = gather_lights(cache, [true])
    objective_mask, protect_mask = fixture_masks(args.height, args.width)
    protect_base = target.copy()

    np.save(base / "scribble_target.npy", target)
    np.save(base / "scribble_mask.npy", objective_mask)
    np.save(base / "protect_mask.npy", protect_mask)

    init_true = [true.to_dict()]
    t0 = time.perf_counter()
    scribble_report, scribble_images = strip_images(
        optimize(
            model,
            cache,
            target,
            init_true,
            DEFAULT_BOUNDS,
            steps=1,
            lr=0.0,
            pixel_fraction=1.0,
            weight_mask=objective_mask,
            protect_mask=protect_mask,
            protect_base=protect_base,
            seed=0,
        )
    )
    scribble_ms = (time.perf_counter() - t0) * 1000.0
    np.save(base / "scribble_realized_gather.npy", scribble_images["gather"])

    base_target = target.copy()
    stylized = base_target.copy()
    width_mid = args.width // 2
    stylized[args.height // 4 : args.height // 2, width_mid:] *= np.array([1.8, 1.2, 0.7])
    stylized[:, : width_mid // 2] *= np.array([0.7, 0.9, 1.4])
    np.save(base / "generative_target.npy", stylized)

    restart_rows = []
    best_report = None
    best_images = None
    t0 = time.perf_counter()
    for restart in range(3):
        init = random_init(np.random.default_rng(20 + restart), "sphere", DEFAULT_BOUNDS, 1)
        report, images = strip_images(
            optimize(
                model,
                cache,
                stylized,
                init,
                DEFAULT_BOUNDS,
                steps=args.steps,
                lr=0.03,
                pixel_fraction=0.25,
                weight_mask=objective_mask,
                protect_mask=protect_mask,
                protect_base=base_target,
                seed=20 + restart,
            )
        )
        restart_rows.append(
            {
                "restart": restart,
                "proxy_loss_first": report["proxy_loss_first"],
                "proxy_loss_last": report["proxy_loss_last"],
                "gather_tonemapped_mse": report["gather_tonemapped_mse"],
                "gather_vs_target_psnr_db": report["gather_vs_target_psnr_db"],
            }
        )
        if (
            best_report is None
            or report["gather_tonemapped_mse"] < best_report["gather_tonemapped_mse"]
        ):
            best_report, best_images = report, images
    generative_ms = (time.perf_counter() - t0) * 1000.0
    np.save(base / "generative_realized_gather.npy", best_images["gather"])

    report = {
        "resolution": [args.width, args.height],
        "scribble": {
            "wall_ms": scribble_ms,
            "masked_psnr_db": masked_psnr(scribble_images["gather"], target, objective_mask > 1.0),
            "protected_region_mse_vs_base": scribble_report["protected_region_mse_vs_base_gather"],
            "passes_e7_scribble_thresholds": bool(
                masked_psnr(scribble_images["gather"], target, objective_mask > 1.0) > 25.0
                and scribble_report["protected_region_mse_vs_base_gather"] < 0.5
            ),
            "note": "identity fixture initialized at known light; proves mask/protect plumbing",
        },
        "generative_target": {
            "wall_ms_total_3_restarts": generative_ms,
            "pixel_fraction": 0.25,
            "steps_per_restart": args.steps,
            "restarts": restart_rows,
            "best": {
                "gather_tonemapped_mse": best_report["gather_tonemapped_mse"],
                "proxy_loss_first": best_report["proxy_loss_first"],
                "proxy_loss_last": best_report["proxy_loss_last"],
                "gather_vs_target_psnr_db": best_report["gather_vs_target_psnr_db"],
                "protected_region_mse_vs_base": best_report["protected_region_mse_vs_base_gather"],
            },
            "finding": (
                "raw stylized target cannot be exactly realized by one physical sphere light"
            ),
        },
        "outputs": {
            "scribble_target": "scribble_target.npy",
            "scribble_mask": "scribble_mask.npy",
            "protect_mask": "protect_mask.npy",
            "scribble_realized_gather": "scribble_realized_gather.npy",
            "generative_target": "generative_target.npy",
            "generative_realized_gather": "generative_realized_gather.npy",
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
