"""R2: one camera-conditioned NRP over the existing Cornell-box view harness.

The runner reuses ``examples.multiview`` for camera poses, Mitsuba export, and the
per-view baseline configuration. It then trains one world-anchored conditioned
model, evaluates both systems on identical per-view held-out lights, and records
the quality, memory, and all-view light-edit latency evidence in one JSON report.

The R1 prerequisite is recorded separately and remains unpromoted. A passing local
R2 pilot therefore does not claim that the representation track has been promoted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.multiview import (  # noqa: E402
    CONSISTENCY_LIGHT,
    export_view,
    train_view_config,
    view_poses,
)
from nrp.lights import light_from_dict  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.conditioned_multiview import (  # noqa: E402
    build_validation_sets,
    load_camera_manifest,
    train_conditioned,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight_multiview import (  # noqa: E402
    conditioned_edit_latency_ms,
    edit_latency_ms,
    load_conditioned_views,
    load_views,
)
from nrp.torch_backend.train import evaluate, model_tensors  # noqa: E402


def quality_gate(rows: list[dict], tolerance_db: float = 1.0) -> dict:
    """Apply R2's per-view no-more-than-``tolerance_db`` quality loss gate."""
    if tolerance_db < 0.0:
        raise ValueError("tolerance_db must be non-negative")
    per_view = []
    for row in rows:
        baseline = float(row["baseline_psnr_db"])
        conditioned = float(row["conditioned_psnr_db"])
        delta = round(conditioned - baseline, 6)
        per_view.append(
            {
                "view": row["view"],
                "baseline_psnr_db": baseline,
                "conditioned_psnr_db": conditioned,
                "delta_db": delta,
                "tolerance_db": tolerance_db,
                "within_tolerance": delta >= -tolerance_db,
            }
        )
    deltas = [row["delta_db"] for row in per_view]
    return {
        "tolerance_db": tolerance_db,
        "per_view": per_view,
        "worst_delta_db": min(deltas) if deltas else None,
        "passed": bool(per_view) and all(row["within_tolerance"] for row in per_view),
    }


def camera_pairs_present(poses: list[dict]) -> bool:
    """Check the pose records that are used to write the manifest camera pairs."""
    return bool(poses) and all("origin" in pose and "target" in pose for pose in poses)


def write_camera_manifest(path: Path, poses: list[dict]) -> None:
    """Write the established view manifest plus the physical camera pair for R2."""
    entries = []
    for pose in poses:
        entries.append(
            {
                "name": pose["name"],
                "model": f"{pose['name']}/model.pt",
                "cache": f"{pose['name']}/path_cache.npz",
                "camera": {
                    "origin": pose["origin"],
                    "target": pose["target"],
                },
            }
        )
    path.write_text(json.dumps(entries, indent=2))


def _baseline_rows(
    manifest_path: Path,
    baseline_paths: list[Path],
    cfg: dict,
    conditioned_report: dict,
) -> list[dict]:
    """Evaluate every per-view baseline on the conditioned run's validation lights."""
    views = load_camera_manifest(manifest_path)
    caches = [PathCache.load(str(view.cache_path)) for view in views]
    validation_sets = build_validation_sets(caches, cfg, seed=int(cfg.get("seed", 0)))
    rows = []
    device = "cpu"
    for view, cache, model_path, entries, conditioned_row in zip(
        views,
        caches,
        baseline_paths,
        validation_sets,
        conditioned_report["views"],
        strict=True,
    ):
        model = TorchNRP.load(str(model_path)).to(device).eval()
        spatial, aux = model_tensors(cache, model, device)
        metrics = evaluate(
            model,
            entries,
            spatial,
            aux,
            device,
            hw=(cache.height, cache.width),
        )
        rows.append(
            {
                "view": view.name,
                "baseline_psnr_db": float(
                    sum(metric["psnr_db_vs_raw"] for metric in metrics) / len(metrics)
                ),
                "conditioned_psnr_db": float(conditioned_row["val_psnr_db_vs_raw_mean"]),
                "validation_count": len(entries),
                "validation_light_params": [entry["params"].tolist() for entry in entries],
            }
        )
    return rows


def _latency_report(manifest_path: Path, conditioned_model: Path, devices: list[str]) -> dict:
    lights = [light_from_dict(CONSISTENCY_LIGHT)]
    conditioned = {}
    per_view = {}
    for device in devices:
        conditioned_model_obj, conditioned_views = load_conditioned_views(
            str(manifest_path), str(conditioned_model), device=device
        )
        conditioned[device] = [
            {
                "n_views": count,
                "ms_per_edit": conditioned_edit_latency_ms(
                    conditioned_model_obj, conditioned_views[:count], lights, frames=20, warmup=3
                ),
            }
            for count in range(1, len(conditioned_views) + 1)
        ]
        baseline_views = load_views(str(manifest_path), device=device)
        per_view[device] = [
            {
                "n_views": count,
                "ms_per_edit": edit_latency_ms(
                    baseline_views[:count], lights, frames=20, warmup=3
                ),
            }
            for count in range(1, len(baseline_views) + 1)
        ]
    return {"conditioned": conditioned, "per_view_baselines": per_view}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/r2-conditioned/report.json")
    parser.add_argument("--manifest", help="camera-aware view manifest to create or reuse")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--denoise", choices=["oidn", "bilateral"], default="bilateral"
    )
    parser.add_argument("--devices", nargs="+", default=None)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-conditioned", action="store_true")
    args = parser.parse_args()
    if args.n_views < 2:
        raise SystemExit("--n-views must be >= 2 for R2")

    report_path = Path(args.out).resolve()
    base = report_path.parent
    base.mkdir(parents=True, exist_ok=True)
    poses = view_poses(args.n_views)
    for index, pose in enumerate(poses):
        pose["seed"] = args.seed + index
    manifest_path = Path(args.manifest).resolve() if args.manifest else base / "views.json"

    for pose in poses:
        cache_path = base / pose["name"] / "path_cache.npz"
        if args.skip_export and cache_path.exists():
            continue
        export_view(pose, str(cache_path), args.width, args.height, args.spp, args.bounces)
    write_camera_manifest(manifest_path, poses)

    baseline_paths = []
    baseline_configs = []
    for pose in poses:
        out_dir = base / pose["name"]
        cfg = train_view_config(pose, str(out_dir), str(out_dir / "path_cache.npz"), args)
        cfg["seed"] = pose["seed"]
        baseline_configs.append(cfg)
        model_path = out_dir / "model.pt"
        report_file = out_dir / "torch_train_report.json"
        if not (
            args.skip_baseline
            and model_path.exists()
            and report_file.exists()
        ):
            from nrp.torch_backend.train import train as train_torch

            train_torch(cfg)
        baseline_paths.append(model_path)

    conditioned_cfg = {
        "manifest": str(manifest_path),
        "out_dir": str(base / "conditioned"),
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.5},
        "sampling": "segments",
        "pool": {"size": 64, "replace_every": 5, "replace_count": 2},
        "denoise": (
            {"enabled": True, "method": "oidn"}
            if args.denoise == "oidn"
            else {"enabled": True, "method": "bilateral", "radius": 2}
        ),
        "iters": args.iters,
        "batch_pixels": 4096,
        "lr": 0.005,
        "model": {
            "camera_conditioned": True,
            "spatial_encoding": "world3d",
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 12,
                "base_resolution": 4,
                "finest_resolution": args.width,
            },
        },
        "n_val_lights": 12,
        "seed": args.seed,
        "device": "cpu",
    }
    conditioned_dir = base / "conditioned"
    conditioned_model = conditioned_dir / "model.pt"
    conditioned_train_report_path = conditioned_dir / "conditioned_train_report.json"
    if (
        args.skip_conditioned
        and conditioned_train_report_path.exists()
        and conditioned_model.exists()
    ):
        conditioned_report = json.loads(conditioned_train_report_path.read_text())
    else:
        conditioned_report = train_conditioned(conditioned_cfg)

    rows = _baseline_rows(manifest_path, baseline_paths, conditioned_cfg, conditioned_report)
    gate = quality_gate(rows)
    baseline_bytes = [os.path.getsize(path) for path in baseline_paths]
    devices = args.devices
    if devices is None:
        from nrp.torch_backend.bench import available_devices

        devices = available_devices()
    latency = _latency_report(manifest_path, conditioned_model, devices)
    report = {
        "track": "representation",
        "rung": "R2",
        "status": "implemented_pilot_unpromoted_r1_prerequisite",
        "scene": "builtin:cornell-box",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "view_count": args.n_views,
        "devices": devices,
        "denoise": args.denoise,
        "seed": args.seed,
        "manifest": manifest_path.name,
        "views": rows,
        "quality_gate": gate,
        "memory_mb": {
            "conditioned_model": os.path.getsize(conditioned_model) / 1e6,
            "per_view_baselines_total": sum(baseline_bytes) / 1e6,
            "per_view_baselines": [value / 1e6 for value in baseline_bytes],
        },
        "latency_ms_per_edit": latency,
        "checks": {
            "manifest_camera_pairs": camera_pairs_present(poses),
            "validation_disjoint": bool(
                all(conditioned_report["validation_disjoint_by_view"])
            ),
            "one_conditioned_model": conditioned_report["view_count"] == args.n_views,
            "per_view_quality_gate": gate["passed"],
        },
        "r1_prerequisite": {
            "promoted": False,
            "status": "blocked",
            "reason": (
                "R1 world-anchoring promotion gate remains unmet; R2 pilot evidence is "
                "recorded but does not promote the representation track."
            ),
        },
        "conditioned_train_report": "conditioned/conditioned_train_report.json",
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    failed = [name for name, passed in report["checks"].items() if not passed]
    if failed:
        raise SystemExit(f"R2 checks failed: {failed}")


if __name__ == "__main__":
    main()
