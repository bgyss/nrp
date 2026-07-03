"""Multi-view NRP experiment (roadmap item 7, paper §6.1).

Exports N >= 3 camera views of the Mitsuba cornell box (per-view camera override in
`nrp.mitsuba_exporter`), trains one torch proxy per view, then verifies and measures
the paper's multi-view claims:

- **Verify:** every view's held-out validation PSNR exceeds 20 dB; for one fixed light
  configuration, each view's proxy-vs-own-GATHERLIGHT PSNR is no worse than 2 dB below
  that view's validation range and the max spread across views is < 4 dB (no view
  catastrophically worse); a `views.json` manifest is written for the
  `nrp.torch_backend.relight_multiview` CLI.
- **Measure:** total edit latency (one light change, all N views re-rendered) vs N on
  every requested device — should scale as N x single-view inference with no cache
  access — and the resident memory of the N proxies in MB (the compactness argument).

Requires the mitsuba extra. Results land in out/multiview/report.json; the script
exits nonzero if any verification check fails.

Usage:
  uv run python examples/multiview.py --out out/multiview/report.json
  uv run python examples/multiview.py --skip-export --skip-train   # re-measure only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import light_from_dict  # noqa: E402
from nrp.torch_backend.denoise import oidn_available  # noqa: E402
from nrp.torch_backend.relight_multiview import (  # noqa: E402
    cross_view_consistency,
    edit_latency_ms,
    load_views,
)
from nrp.torch_backend.train import train as train_torch  # noqa: E402

#: The fixed light configuration used for the cross-view consistency check. Chosen
#: inside the box, off-center, so every view sees a nontrivial illumination pattern.
CONSISTENCY_LIGHT = {
    "type": "sphere",
    "center": [0.2, 0.35, -0.1],
    "radius": 0.25,
    "rgb": [1.0, 1.0, 1.0],
}


def view_poses(n: int) -> list[dict]:
    """N distinct cameras outside the cornell box's open front, all aimed inside.

    Cameras sit on an arc of +-20 degrees around the default front camera (distance
    3.9, matching `mi.cornell_box()`), with a small alternating height offset so no
    two poses are related by a pure y-rotation.
    """
    poses = []
    for i in range(n):
        theta = math.radians(-20.0 + 40.0 * i / (n - 1)) if n > 1 else 0.0
        y = (0.0, 0.3, -0.3)[i % 3]
        poses.append(
            {
                "name": f"view{i}",
                "origin": [3.9 * math.sin(theta), y, 3.9 * math.cos(theta)],
                "target": [0.0, 0.0, 0.0],
            }
        )
    return poses


def export_view(pose: dict, out_path: str, width: int, height: int, spp: int, bounces: int):
    from nrp.mitsuba_exporter import (
        _import_mitsuba,
        _load_mitsuba,
        _load_scene,
        build_sensor,
        export_path_cache,
        export_path_cache_wavefront,
        pick_jit_variant,
    )

    mode = "wavefront" if pick_jit_variant(_import_mitsuba()) else "scalar"
    mi = _load_mitsuba(mode)
    scene = _load_scene(mi, "builtin:cornell-box", width, height)
    sensor = build_sensor(mi, width, height, pose["origin"], pose["target"])
    export = export_path_cache_wavefront if mode == "wavefront" else export_path_cache
    cache = export(scene, mi, width, height, spp, bounces, seed=0, sensor=sensor)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cache.save(out_path)
    return cache


def train_view_config(pose: dict, out_dir: str, cache_path: str, args) -> dict:
    denoise = (
        {"enabled": True, "method": "oidn"}
        if args.denoise == "oidn"
        else {"enabled": True, "method": "bilateral", "radius": 2}
    )
    return {
        "cache": cache_path,
        "out_dir": out_dir,
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.5},
        "sampling": "segments",
        "pool": {"size": 64, "replace_every": 5, "replace_count": 2},
        "denoise": denoise,
        "iters": args.iters,
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
                "finest_resolution": args.width,
            },
        },
        "n_val_lights": 12,
        # Distinct seeds so the views train on independent light sets; the
        # cross-view consistency light is fixed and shared.
        "seed": pose["seed"],
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/multiview/report.json")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument(
        "--denoise",
        choices=["oidn", "bilateral"],
        default="oidn" if oidn_available() else "bilateral",
    )
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="devices for the latency measurement (default: cpu + mps/cuda if available)",
    )
    parser.add_argument("--skip-export", action="store_true", help="reuse existing caches")
    parser.add_argument("--skip-train", action="store_true", help="reuse existing models")
    args = parser.parse_args()
    if args.n_views < 3:
        raise SystemExit("--n-views must be >= 3 (roadmap item 7)")

    base = Path(args.out).resolve().parent
    poses = view_poses(args.n_views)
    for i, pose in enumerate(poses):
        pose["seed"] = i

    # 1. Export one path cache per camera pose.
    for pose in poses:
        cache_path = base / pose["name"] / "path_cache.npz"
        if args.skip_export and cache_path.exists():
            print(f"[{pose['name']}] reusing {cache_path}")
            continue
        t0 = time.perf_counter()
        cache = export_view(pose, str(cache_path), args.width, args.height, args.spp, args.bounces)
        print(
            f"[{pose['name']}] exported {cache.segment_count} segments from "
            f"origin {pose['origin']} in {time.perf_counter() - t0:.1f}s"
        )

    # 2. Train one proxy per view.
    view_rows = []
    for pose in poses:
        out_dir = base / pose["name"]
        cfg = train_view_config(pose, str(out_dir), str(out_dir / "path_cache.npz"), args)
        report_path = out_dir / "torch_train_report.json"
        if args.skip_train and (out_dir / "model.pt").exists() and report_path.exists():
            print(f"[{pose['name']}] reusing {out_dir / 'model.pt'}")
            report = json.loads(report_path.read_text())
        else:
            print(f"[{pose['name']}] training ({args.iters} iters, {args.denoise} denoise) ...")
            report = train_torch(cfg)
        per_light = [m["psnr_db_vs_raw"] for m in report["val_lights"]]
        view_rows.append(
            {
                "name": pose["name"],
                "origin": pose["origin"],
                "target": pose["target"],
                "seed": pose["seed"],
                "path_cache_segments": report["path_cache_segments"],
                "train_seconds": report["train_seconds"],
                "val_psnr_db_mean": report["val_psnr_db_vs_raw_mean"],
                "val_psnr_db_range": [min(per_light), max(per_light)],
                "val_smape_mean": report["val_smape_vs_raw_mean"],
            }
        )

    # 3. Manifest for the relight_multiview CLI (paths relative to the manifest).
    manifest_path = base / "views.json"
    manifest_path.write_text(
        json.dumps(
            [
                {
                    "name": p["name"],
                    "model": f"{p['name']}/model.pt",
                    "cache": f"{p['name']}/path_cache.npz",
                }
                for p in poses
            ],
            indent=2,
        )
    )
    print(f"wrote {manifest_path}")

    # 4. Cross-view consistency for the fixed light (proxy vs own GATHERLIGHT).
    views = load_views(str(manifest_path), device="cpu")
    lights = [light_from_dict(CONSISTENCY_LIGHT)]
    consistency = cross_view_consistency(views, lights)
    for row in consistency["per_view"]:
        print(f"consistency [{row['view']}] proxy vs GATHERLIGHT: {row['psnr_db']:.2f} dB")
    print(f"cross-view spread: {consistency['psnr_db_spread']:.2f} dB")

    # 5. Latency vs N per device, and resident-proxy memory.
    devices = args.devices
    if devices is None:
        from nrp.torch_backend.bench import available_devices

        devices = available_devices()
    latency = {}
    for device in devices:
        dev_views = load_views(str(manifest_path), device=device)
        latency[device] = [
            {
                "n_views": n,
                "ms_per_edit": edit_latency_ms(dev_views[:n], lights, frames=20, warmup=3),
            }
            for n in range(1, len(dev_views) + 1)
        ]
        for row in latency[device]:
            print(f"latency [{device}] {row['n_views']} view(s): {row['ms_per_edit']:.2f} ms/edit")
    memory_mb = sum(v.model_bytes for v in views) / 1e6
    print(f"resident memory of {len(views)} proxies: {memory_mb:.2f} MB")

    # 6. Verification checks (roadmap item 7).
    checks = {
        "all_views_exceed_20db": all(v["val_psnr_db_mean"] > 20.0 for v in view_rows),
        "consistency_within_validation_range": all(
            row["psnr_db"] >= v["val_psnr_db_range"][0] - 2.0
            for row, v in zip(consistency["per_view"], view_rows, strict=True)
        ),
        "cross_view_spread_below_4db": consistency["psnr_db_spread"] < 4.0,
    }

    report = {
        "scene": "builtin:cornell-box",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "iters": args.iters,
        "denoise": args.denoise,
        "consistency_light": CONSISTENCY_LIGHT,
        "views": view_rows,
        "consistency": consistency,
        "latency_ms_per_edit": latency,
        "resident_memory_mb": memory_mb,
        "model_bytes_per_view": [v.model_bytes for v in views],
        "checks": checks,
    }
    os.makedirs(base, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"verification checks failed: {failed}")
    print("all verification checks passed")


if __name__ == "__main__":
    main()
