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
import hashlib
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_provenance(base: Path, report_paths: dict, config: dict) -> dict:
    """Write deterministic provenance for E7 fixtures and realized outputs."""
    files = {
        name: {
            "path": rel_path,
            "sha256": file_sha256(base / rel_path),
        }
        for name, rel_path in report_paths.items()
    }
    provenance = {
        "scope": "E7 synthetic scribble and stylized target provenance",
        "generation": {
            "method": "deterministic repo-local numpy fixture generation",
            "external_generator": None,
            "hand_authored": False,
            "notes": [
                "scribble_target is GATHERLIGHT from a known SphereLight fixture",
                "generative_target is a deterministic stylization of that base target",
                "no external image model, editor, or asset was used in this toy slice",
            ],
        },
        "config": config,
        "files": files,
        "limitations": [
            "This provenance covers the committed synthetic/stylized toy fixtures only.",
            (
                "A high-quality proxy run and a true hand-authored or external generative "
                "image remain open."
            ),
        ],
    }
    path = base / "provenance.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def strip_images(report: dict) -> tuple[dict, dict]:
    images = report.pop("_images")
    return report, images


def run_inverse(
    model: TorchNRP,
    cache,
    target: np.ndarray,
    init: list[dict],
    steps: int,
    pixel_fraction: float,
    objective_mask: np.ndarray,
    protect_mask: np.ndarray,
    protect_base: np.ndarray,
    seed: int,
) -> tuple[dict, dict, float]:
    t0 = time.perf_counter()
    report, images = strip_images(
        optimize(
            model,
            cache,
            target,
            init,
            DEFAULT_BOUNDS,
            steps=steps,
            lr=0.03,
            pixel_fraction=pixel_fraction,
            weight_mask=objective_mask,
            protect_mask=protect_mask,
            protect_base=protect_base,
            seed=seed,
        )
    )
    return report, images, (time.perf_counter() - t0) * 1000.0


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
        report, images, _ = run_inverse(
            model,
            cache,
            stylized,
            init,
            args.steps,
            0.25,
            objective_mask,
            protect_mask,
            base_target,
            20 + restart,
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

    latency_sweep = []
    for fraction in (1.0, 0.25, 0.05):
        init = random_init(
            np.random.default_rng([70, int(fraction * 100)]),
            "sphere",
            DEFAULT_BOUNDS,
            1,
        )
        sweep_report, _sweep_images, sweep_ms = run_inverse(
            model,
            cache,
            stylized,
            init,
            max(5, args.steps // 2),
            fraction,
            objective_mask,
            protect_mask,
            base_target,
            70 + int(fraction * 100),
        )
        latency_sweep.append(
            {
                "pixel_fraction": fraction,
                "steps": max(5, args.steps // 2),
                "wall_ms": sweep_ms,
                "ms_per_step": sweep_ms / max(5, args.steps // 2),
                "proxy_loss_first": sweep_report["proxy_loss_first"],
                "proxy_loss_last": sweep_report["proxy_loss_last"],
                "gather_tonemapped_mse": sweep_report["gather_tonemapped_mse"],
                "gather_vs_target_psnr_db": sweep_report["gather_vs_target_psnr_db"],
                "proxy_loss_curve": sweep_report["proxy_loss_curve"],
            }
        )

    outputs = {
        "scribble_target": "scribble_target.npy",
        "scribble_mask": "scribble_mask.npy",
        "protect_mask": "protect_mask.npy",
        "scribble_realized_gather": "scribble_realized_gather.npy",
        "generative_target": "generative_target.npy",
        "generative_realized_gather": "generative_realized_gather.npy",
    }
    provenance = write_provenance(
        base,
        outputs,
        {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "cache": {"spp": 6, "max_bounces": 2, "seed": 14},
            "true_light": true.to_dict(),
            "stylization": {
                "warm_objective_region_multiplier": [1.8, 1.2, 0.7],
                "cool_left_region_multiplier": [0.7, 0.9, 1.4],
            },
            "optimizer": {
                "restarts": 3,
                "pixel_fraction": 0.25,
                "steps_per_restart": args.steps,
            },
        },
    )

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
        "latency_sweep": latency_sweep,
        "outputs": {**outputs, "provenance": "provenance.json"},
        "provenance": {
            "path": "provenance.json",
            "file_count": len(provenance["files"]),
            "external_generator": provenance["generation"]["external_generator"],
            "method": provenance["generation"]["method"],
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
