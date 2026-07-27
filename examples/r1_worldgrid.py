"""R1 matched-budget pixel2d versus world3d parity experiment.

Runs the standard toy and 128x128 Country Kitchen training configurations with
identical seeds, iterations, pool settings, denoisers, and held-out lights. The
world-grid variants change only the spatial representation plus small hash-table
budget adjustments that keep total parameters within 0.5% of the 2D model.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.train import load_config, train

SCENES = {
    "toy": {
        "pixel_config": "examples/toy_sphere_torch.json",
        "world_config": "examples/r1_toy_world3d.json",
        "committed_baseline_report": "out/toy-torch/torch_train_report.json",
    },
    "kitchen": {
        "pixel_config": "examples/kitchen_torch.json",
        "world_config": "examples/r1_kitchen_world3d.json",
        "committed_baseline_report": "out/kitchen-torch/torch_train_report.json",
    },
}


def device_available(device: str) -> bool:
    if device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    return False


def cpu_brand() -> str | None:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def load_or_train(
    config_path: str,
    out_dir: Path,
    device: str,
    reuse: bool,
    denoise_method: str | None,
    repo_root: Path,
) -> dict:
    report_path = out_dir / "torch_train_report.json"
    if reuse and report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        cfg = load_config(config_path)
        cfg["out_dir"] = str(out_dir)
        cfg["device"] = device
        if denoise_method is not None:
            cfg.setdefault("denoise", {})["method"] = denoise_method
        report = train(cfg)
    cache_path = Path(report["config"]["cache"])
    try:
        report["config"]["cache"] = cache_path.relative_to(repo_root).as_posix()
    except ValueError:
        pass
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarize_pair(
    scene: str,
    device: str,
    pixel: dict,
    world: dict,
    committed_baseline: dict,
    committed_baseline_path: str,
) -> dict:
    pixel_psnr = float(pixel["val_psnr_db_vs_raw_mean"])
    world_psnr = float(world["val_psnr_db_vs_raw_mean"])
    delta = world_psnr - pixel_psnr
    pixel_params = int(pixel["parameter_count"])
    world_params = int(world["parameter_count"])
    committed_psnr = float(committed_baseline["val_psnr_db_vs_raw_mean"])
    committed_delta = world_psnr - committed_psnr
    return {
        "scene": scene,
        "device": device,
        "iterations": int(pixel["config"]["iters"]),
        "seed": int(pixel["config"].get("seed", 0)),
        "pixel2d": {
            "parameter_count": pixel_params,
            "iters_per_second": pixel["iters_per_second"],
            "held_out_psnr_db": pixel_psnr,
            "report": f"out/r1-worldgrid/{scene}/{device}/pixel2d/torch_train_report.json",
        },
        "world3d": {
            "parameter_count": world_params,
            "iters_per_second": world["iters_per_second"],
            "held_out_psnr_db": world_psnr,
            "report": f"out/r1-worldgrid/{scene}/{device}/world3d/torch_train_report.json",
        },
        "parameter_delta": world_params - pixel_params,
        "parameter_delta_percent": (world_params / pixel_params - 1.0) * 100.0,
        "same_run_control": {
            "psnr_delta_db_world3d_minus_pixel2d": delta,
            "gate_pass": delta >= -0.5,
        },
        "committed_baseline": {
            "report": committed_baseline_path,
            "parameter_count": int(committed_baseline["parameter_count"]),
            "iterations": int(committed_baseline["config"]["iters"]),
            "held_out_psnr_db": committed_psnr,
        },
        "psnr_delta_db_world3d_minus_committed_pixel2d": committed_delta,
        "gate_threshold_db": -0.5,
        "gate_pass": committed_delta >= -0.5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="out/r1-worldgrid")
    parser.add_argument(
        "--committed-baseline-root",
        default=".",
        help="repo root containing the committed baseline out/ artifacts",
    )
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=sorted(SCENES))
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=["cpu", "mps", "cuda"],
        default=["cpu", "mps"],
    )
    parser.add_argument("--reuse", action="store_true", help="reuse existing per-run JSON reports")
    parser.add_argument(
        "--denoise-method",
        choices=["bilateral", "oidn"],
        default=None,
        help="explicit matched-pair override; default preserves each standard config",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pairs = []
    skipped = []
    for scene in args.scenes:
        spec = SCENES[scene]
        baseline_path = Path(args.committed_baseline_root) / spec["committed_baseline_report"]
        if not baseline_path.exists():
            raise SystemExit(f"committed baseline report not found: {baseline_path}")
        committed_baseline = json.loads(baseline_path.read_text())
        for device in args.devices:
            existing_pair = all(
                (
                    out_root
                    / scene
                    / device
                    / spatial
                    / "torch_train_report.json"
                ).exists()
                for spatial in ("pixel2d", "world3d")
            )
            if not device_available(device) and not (args.reuse and existing_pair):
                skipped.append({"scene": scene, "device": device, "reason": "device unavailable"})
                continue
            reports = {}
            for spatial, key in (("pixel2d", "pixel_config"), ("world3d", "world_config")):
                config_path = str(root / spec[key])
                reports[spatial] = load_or_train(
                    config_path,
                    out_root / scene / device / spatial,
                    device,
                    args.reuse,
                    args.denoise_method,
                    root,
                )
            pairs.append(
                summarize_pair(
                    scene,
                    device,
                    reports["pixel2d"],
                    reports["world3d"],
                    committed_baseline,
                    spec["committed_baseline_report"],
                )
            )

    cpu_pairs = [pair for pair in pairs if pair["device"] == "cpu"]
    expected_scenes = set(args.scenes)
    gate_complete = {pair["scene"] for pair in cpu_pairs} == expected_scenes
    gate_pass = gate_complete and all(pair["gate_pass"] for pair in cpu_pairs)
    report = {
        "rung": "R1",
        "title": "World-space encoding at parity",
        "gate": {
            "definition": (
                "world3d held-out PSNR must be within 0.5 dB of the committed "
                "pixel2d baseline at matched parameters and iterations, per scene"
            ),
            "device": "cpu",
            "complete": gate_complete,
            "pass": gate_pass,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": cpu_brand(),
            "torch_version": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "cuda_available": torch.cuda.is_available(),
        },
        "denoise_override": args.denoise_method,
        "pairs": pairs,
        "skipped": skipped,
    }
    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {report_path}")
    if not gate_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
