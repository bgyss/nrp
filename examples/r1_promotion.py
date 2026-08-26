"""R1C/R1E promotion audit for the target-scale world-triplane candidate.

R1A identified one candidate that clears the unchanged per-seed gate on the Country
Kitchen cache: ``world_triplane`` with target-scale output initialization. This
runner carries that exact candidate into coordinate-robustness experiments and a
second 128² real-scene confirmation. It promotes only when R1A, every R1C row, and
every R1E seed pass independently.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.r1a_variance import ARM_MODELS  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.train import (  # noqa: E402
    build_val_set,
    evaluate,
    load_config,
    model_tensors,
    train,
)

GATE_DELTA_DB = -0.5
CANDIDATE = "world_triplane"
POLICY = "target_scale"
REQUIRED_SEEDS = frozenset(range(5))


def configure_deterministic_cpu() -> None:
    """Make paired promotion arms independent of process scheduling."""
    # OIDN's TBB worker pool otherwise produces different floating-point reductions
    # in separate processes, which can move a borderline PSNR gate by >1 dB.
    for name in ("TBB_NUM_THREADS", "OIDN_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this setting only before any inter-op work has started;
        # the intra-op setting and deterministic algorithm guard still apply.
        pass
    torch.use_deterministic_algorithms(True)


def rotation_matrix_y(degrees: float) -> np.ndarray:
    """Return a right-handed rotation about the scene's Y/up axis."""
    angle = np.deg2rad(float(degrees))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def canonical_world_transform(positions: np.ndarray) -> tuple[dict, np.ndarray]:
    """Build a rotation-stable PCA frame and return its transformed positions."""
    points = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3 or not np.isfinite(points).all():
        raise ValueError("positions must contain at least three finite points")
    origin = points.mean(axis=0)
    centered = points - origin
    _, basis = np.linalg.eigh(centered.T @ centered)
    basis = basis[:, ::-1]
    projections = centered @ basis
    skew = (projections**3).sum(axis=0)
    for index in range(3):
        if abs(float(skew[index])) > 1e-12:
            sign = 1.0 if skew[index] > 0.0 else -1.0
        else:
            pivot = int(np.argmax(np.abs(basis[:, index])))
            sign = 1.0 if basis[pivot, index] >= 0.0 else -1.0
        basis[:, index] *= sign
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    canonical = centered @ basis
    return {"origin": origin.tolist(), "basis": basis.tolist()}, canonical


def transform_cache(
    cache: PathCache, rotation: np.ndarray, translation: np.ndarray | None = None
) -> PathCache:
    """Apply a rigid coordinate change to every world-space cache field."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal")
    translation = np.zeros(3, dtype=np.float64) if translation is None else np.asarray(translation)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("translation must be a finite three-vector")
    return PathCache(
        width=cache.width,
        height=cache.height,
        n_paths=cache.n_paths.copy(),
        seg_pixel=cache.seg_pixel.copy(),
        seg_origin=cache.seg_origin @ rotation.T + translation,
        seg_dir=cache.seg_dir @ rotation.T,
        seg_tmax=cache.seg_tmax.copy(),
        seg_throughput=cache.seg_throughput.copy(),
        albedo=cache.albedo.copy(),
        position=cache.position @ rotation.T + translation,
        depth=cache.depth.copy(),
        normal=cache.normal @ rotation.T,
        medium=copy.deepcopy(cache.medium),
    )


def percentile_bounds(
    positions: np.ndarray, lower: float = 1.0, upper: float = 99.0
) -> dict:
    """Return robust componentwise bounds for first-hit positions."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(positions).all():
        raise ValueError("positions must be finite")
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("percentile bounds require 0 <= lower < upper <= 100")
    lo, hi = np.quantile(positions, [lower / 100.0, upper / 100.0], axis=0)
    if np.any(hi <= lo):
        raise ValueError("percentile bounds must span a non-zero range on every axis")
    return {"min": lo.tolist(), "max": hi.tolist()}


def out_of_bounds_fraction(positions: np.ndarray, bounds: dict) -> float:
    """Return the fraction of positions clamped by a model's world bound."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lower = np.asarray(bounds["min"], dtype=np.float64)
    upper = np.asarray(bounds["max"], dtype=np.float64)
    outside = np.any((positions < lower) | (positions > upper), axis=1)
    return float(outside.mean())


def promotion_gate(
    r1a_pass: bool,
    r1c_runs: list[dict],
    r1e_runs: list[dict],
    threshold_db: float = GATE_DELTA_DB,
    r1c_complete: bool = True,
    r1e_complete: bool = True,
) -> dict:
    """Apply the R1 promotion rule without averaging away a failing run."""
    r1c_failures = [row for row in r1c_runs if row["delta_db"] < threshold_db]
    r1e_failures = [row for row in r1e_runs if row["delta_db"] < threshold_db]
    r1c_pass = r1c_complete and bool(r1c_runs) and not r1c_failures
    r1e_pass = r1e_complete and bool(r1e_runs) and not r1e_failures
    return {
        "threshold_db": threshold_db,
        "r1a_pass": bool(r1a_pass),
        "r1c_pass": r1c_pass,
        "r1e_pass": r1e_pass,
        "r1c_run_count": len(r1c_runs),
        "r1e_run_count": len(r1e_runs),
        "r1c_complete": bool(r1c_complete),
        "r1e_complete": bool(r1e_complete),
        "r1c_failures": r1c_failures,
        "r1e_failures": r1e_failures,
        "promoted": bool(r1a_pass and r1c_pass and r1e_pass),
    }


def promotion_stop_reason(gate: dict) -> str | None:
    """Describe the first binding reason a promotion decision remains closed."""
    if gate["r1c_failures"]:
        row = gate["r1c_failures"][0]
        return (
            "R1C fails the per-seed gate: "
            f"{row['rotation_degrees']:g}°/{row['bounds_mode']} seed {row['seed']} "
            f"delta {row['delta_db']:.3f} dB < {gate['threshold_db']:.3f} dB."
        )
    if gate["r1e_failures"]:
        row = gate["r1e_failures"][0]
        return (
            "R1E fails the per-seed gate: "
            f"{row.get('scene', 'independent scene')} seed {row['seed']} "
            f"delta {row['delta_db']:.3f} dB < {gate['threshold_db']:.3f} dB."
        )
    if not gate["r1c_complete"]:
        return "R1C coverage is incomplete; no promotion decision is allowed."
    if not gate["r1e_complete"]:
        return "R1E coverage is incomplete; no promotion decision is allowed."
    if not gate["r1a_pass"]:
        return "R1A has no candidate arm passing every required seed."
    return None


def r1a_seed_rows(report: dict) -> list[dict]:
    """Convert the carried-forward R1A candidate comparison into R1C base rows."""
    rows = []
    comparison = report["comparisons"][CANDIDATE][POLICY]
    for entry in comparison["per_seed"]:
        delta = float(entry["summary"]["mean_db"])
        rows.append(
            {
                "seed": int(entry["seed"]),
                "rotation_degrees": 0.0,
                "bounds_mode": "aabb",
                "delta_db": delta,
                "gate_pass": bool(entry["gate_pass"]),
                "source": "r1a",
            }
        )
    return rows


def r1c_coverage_complete(rotations: list[float], bounds_modes: list[str]) -> bool:
    """Require the full planned three-rotation/two-normalization R1C matrix."""
    return {float(value) for value in rotations} >= {0.0, 90.0, 180.0} and {
        str(value) for value in bounds_modes
    } >= {"aabb", "percentile"}


def r1c_runs_coverage_complete(runs: list[dict]) -> bool:
    """Require every planned R1C cell to contain exactly the five fixed seeds."""
    expected_rotations = {0.0, 90.0, 180.0}
    expected_bounds = {"aabb", "percentile"}
    by_cell: dict[tuple[float, str], list[dict]] = {}
    for row in runs:
        cell = (float(row.get("rotation_degrees", 0.0)), str(row["bounds_mode"]))
        by_cell.setdefault(cell, []).append(row)
    for rotation in expected_rotations:
        for bounds_mode in expected_bounds:
            cell_rows = by_cell.get((rotation, bounds_mode), [])
            if len(cell_rows) != len(REQUIRED_SEEDS):
                return False
            if {int(row["seed"]) for row in cell_rows} != REQUIRED_SEEDS:
                return False
    return True


def r1e_seed_coverage_complete(runs: list[dict]) -> bool:
    """Require one R1E result for each of the five fixed promotion seeds."""
    return len(runs) == len(REQUIRED_SEEDS) and {
        int(row["seed"]) for row in runs
    } == REQUIRED_SEEDS


def aggregate_reports(
    r1a_path: str | Path,
    r1c_path: str | Path,
    r1e_path: str | Path,
    out_path: str | Path,
    root: str | Path | None = None,
) -> dict:
    """Combine completed R1A/R1C-slice/R1E artifacts without inventing coverage."""
    root_path = Path(root or Path.cwd()).resolve()
    r1a = json.loads(Path(r1a_path).read_text())
    r1c = json.loads(Path(r1c_path).read_text())
    r1e = json.loads(Path(r1e_path).read_text())

    def sanitize(rows: list[dict]) -> list[dict]:
        result = []
        for row in rows:
            copy_row = copy.deepcopy(row)
            if "output_dir" in copy_row:
                try:
                    copy_row["output_dir"] = Path(copy_row["output_dir"]).resolve().relative_to(
                        root_path
                    ).as_posix()
                except ValueError:
                    pass
            result.append(copy_row)
        return result

    r1c_runs = sorted(
        sanitize(r1c["r1c"]["runs"]),
        key=lambda row: (
            float(row.get("rotation_degrees", 0.0)),
            row.get("bounds_mode", ""),
            row["seed"],
        ),
    )
    r1e_runs = sorted(sanitize(r1e["r1e"]["runs"]), key=lambda row: row["seed"])
    expected_rotations = {0.0, 90.0, 180.0}
    expected_bounds = {"aabb", "percentile"}
    observed_rotations = {
        float(row.get("rotation_degrees", 0.0)) for row in r1c_runs
    }
    observed_bounds = {row.get("bounds_mode") for row in r1c_runs}
    rotations_complete = expected_rotations.issubset(observed_rotations)
    bounds_complete = expected_bounds.issubset(observed_bounds)
    coverage_complete = rotations_complete and bounds_complete and r1c_runs_coverage_complete(
        r1c_runs
    )
    r1a_pass = "world_triplane/target_scale" in r1a["gate"]["passing_world_anchored_arms"]
    gate = promotion_gate(
        r1a_pass,
        r1c_runs,
        r1e_runs,
        r1c_complete=coverage_complete,
        r1e_complete=bool(r1e["promotion"].get("r1e_complete")) and r1e_seed_coverage_complete(
            r1e_runs
        ),
    )
    report = {
        "experiment": "R1 promotion audit",
        "rung": "R1",
        "status": "promoted" if gate["promoted"] else "candidate_not_promoted",
        "candidate": {
            "representation": CANDIDATE,
            "output_bias_policy": POLICY,
            "gate_threshold_db": GATE_DELTA_DB,
            "training_denoiser": r1e.get("candidate", {}).get(
                "training_denoiser", {"method": "oidn"}
            ),
            "gather_backend": r1e.get("candidate", {}).get("gather_backend", "numpy"),
        },
        "r1a": {
            "report": "out/r1a/report.json",
            "candidate_pass": r1a_pass,
            "source_report": str(Path(r1a_path).resolve().relative_to(root_path)),
        },
        "r1c": {
            "source_command": r1c.get("command"),
            "scene": "Country Kitchen",
            "observed_rotations_degrees": sorted(observed_rotations),
            "observed_bounds_modes": sorted(observed_bounds),
            "expected_rotations_degrees": sorted(expected_rotations),
            "expected_bounds_modes": sorted(expected_bounds),
            "coverage_complete": coverage_complete,
            "runs": r1c_runs,
        },
        "r1e": {
            "source_command": r1e.get("command"),
            "scene": r1e["r1e"]["scene"],
            "runs": r1e_runs,
        },
        "promotion": gate,
        "stop_condition": promotion_stop_reason(gate),
    }
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _candidate_config(
    base: dict,
    cache_path: Path,
    out_dir: Path,
    representation: str,
    seed: int,
    bounds: dict | None,
    world_transform: dict | None = None,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["cache"] = str(cache_path)
    cfg["out_dir"] = str(out_dir)
    cfg["seed"] = int(seed)
    cfg["device"] = "cpu"
    cfg["model"].update(copy.deepcopy(ARM_MODELS[representation]))
    cfg["model"]["init_output_scale"] = True
    cfg["gather_backend"] = cfg.get("gather_backend", "numpy")
    if bounds is None:
        cfg["model"].pop("world_bounds", None)
    else:
        cfg["model"]["world_bounds"] = bounds
    if world_transform is None:
        cfg["model"].pop("world_transform", None)
    else:
        cfg["model"]["world_transform"] = world_transform
    return cfg


def _evaluate_model(model_path: Path, cache: PathCache, val_set: list[dict]) -> float:
    device = torch.device("cpu")
    model = TorchNRP.load(str(model_path)).to(device).eval()
    spatial, aux = model_tensors(cache, model, device)
    metrics = evaluate(model, val_set, spatial, aux, device)
    return float(np.mean([row["psnr_db_vs_raw"] for row in metrics]))


def _run_pair(
    base_cfg: dict,
    cache: PathCache,
    cache_path: Path,
    seed: int,
    out_dir: Path,
    bounds_mode: str,
    rotation_degrees: float,
    reuse: bool,
    coordinate_frame: str = "raw",
) -> dict:
    if coordinate_frame not in {"raw", "pca"}:
        raise ValueError(f"unknown coordinate frame {coordinate_frame!r}")
    world_transform = None
    position_for_bounds = cache.position
    if coordinate_frame == "pca":
        world_transform, position_for_bounds = canonical_world_transform(cache.position)
    bounds = None
    if bounds_mode == "percentile":
        bounds = percentile_bounds(position_for_bounds)
    elif coordinate_frame == "pca":
        points = np.asarray(position_for_bounds, dtype=np.float64).reshape(-1, 3)
        bounds = {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}
    elif bounds_mode != "aabb":
        raise ValueError(f"unknown bounds mode {bounds_mode!r}")
    control_dir = out_dir / "pixel2d"
    candidate_dir = out_dir / CANDIDATE
    control_cfg = _candidate_config(base_cfg, cache_path, control_dir, "pixel2d", seed, None)
    candidate_cfg = _candidate_config(
        base_cfg,
        cache_path,
        candidate_dir,
        CANDIDATE,
        seed,
        bounds,
        world_transform,
    )
    for cfg, directory in ((control_cfg, control_dir), (candidate_cfg, candidate_dir)):
        report_path = directory / "torch_train_report.json"
        model_path = directory / "model.pt"
        if not (reuse and report_path.exists() and model_path.exists()):
            train(cfg)
    val_cfg = copy.deepcopy(base_cfg)
    val_cfg["seed"] = int(seed)
    val_set = build_val_set(cache, val_cfg)
    control_psnr = _evaluate_model(control_dir / "model.pt", cache, val_set)
    candidate_psnr = _evaluate_model(candidate_dir / "model.pt", cache, val_set)
    return {
        "seed": int(seed),
        "rotation_degrees": float(rotation_degrees),
        "bounds_mode": bounds_mode,
        "control_psnr_db": control_psnr,
        "candidate_psnr_db": candidate_psnr,
        "delta_db": float(candidate_psnr - control_psnr),
        "gate_pass": bool(candidate_psnr - control_psnr >= GATE_DELTA_DB),
        "candidate_bounds": bounds,
        "out_of_bounds_fraction": (
            out_of_bounds_fraction(position_for_bounds, bounds) if bounds is not None else 0.0
        ),
        "output_dir": str(out_dir),
        "coordinate_frame": coordinate_frame,
    }


def _run_pair_worker(arguments: tuple) -> dict:
    """Process-pool entry point; each job owns one cache/model output directory."""
    (
        base_cfg,
        cache_path,
        seed,
        out_dir,
        bounds_mode,
        rotation_degrees,
        reuse,
        coordinate_frame,
    ) = arguments
    configure_deterministic_cpu()
    cache = PathCache.load(cache_path)
    return _run_pair(
        base_cfg,
        cache,
        Path(cache_path),
        seed,
        Path(out_dir),
        bounds_mode,
        rotation_degrees,
        reuse,
        coordinate_frame,
    )


def run_r1c(
    base_cfg: dict,
    cache: PathCache,
    root: Path,
    out_root: Path,
    seeds: list[int],
    rotations: list[float],
    bounds_modes: list[str],
    reuse: bool,
    workers: int = 1,
    coordinate_frame: str = "raw",
) -> list[dict]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if coordinate_frame not in {"raw", "pca"}:
        raise ValueError(f"unknown coordinate frame {coordinate_frame!r}")
    jobs = []
    for degrees in rotations:
        transformed = transform_cache(cache, rotation_matrix_y(degrees))
        for bounds_mode in bounds_modes:
            cache_dir = out_root / "r1c" / f"rotation_{degrees:g}" / bounds_mode
            cache_dir.mkdir(parents=True, exist_ok=True)
            if coordinate_frame == "raw" and degrees == 0.0 and bounds_mode == "aabb":
                continue
            transformed_path = out_root / "r1c" / f"rotation_{degrees:g}" / "cache.npz"
            if not (reuse and transformed_path.exists()):
                transformed.save(transformed_path)
            jobs.extend(
                (
                    base_cfg,
                    str(transformed_path),
                    seed,
                    str(cache_dir / f"seed{seed}"),
                    bounds_mode,
                    degrees,
                    reuse,
                    coordinate_frame,
                )
                for seed in seeds
            )
    if workers == 1:
        return [_run_pair_worker(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_pair_worker, job) for job in jobs]
        return [future.result() for future in as_completed(futures)]


def run_r1e(
    base_cfg: dict,
    cache: PathCache,
    root: Path,
    out_root: Path,
    scene_name: str,
    seeds: list[int],
    reuse: bool,
    workers: int = 1,
    coordinate_frame: str = "raw",
) -> list[dict]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    cache_path = root / "out" / "r1-promotion" / f"{scene_name.replace(' ', '_').lower()}_cache.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not (reuse and cache_path.exists()):
        cache.save(cache_path)
    jobs = [
        (
            base_cfg,
            str(cache_path),
            seed,
            str(out_root / "r1e" / f"seed{seed}"),
            "aabb",
            0.0,
            reuse,
            coordinate_frame,
        )
        for seed in seeds
    ]
    if workers == 1:
        runs = [_run_pair_worker(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_run_pair_worker, job) for job in jobs]
            runs = [future.result() for future in as_completed(futures)]
    for row in runs:
        row["scene"] = scene_name
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="examples/kitchen_torch.json")
    parser.add_argument("--second-cache", required=True)
    parser.add_argument("--second-scene", default="Bedroom")
    parser.add_argument("--out", default="out/r1-promotion/report.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--rotations", nargs="+", type=float, default=[0.0, 90.0, 180.0])
    parser.add_argument(
        "--bounds-modes",
        nargs="+",
        choices=["aabb", "percentile"],
        default=["aabb", "percentile"],
    )
    parser.add_argument("--denoise-method", choices=["oidn", "bilateral"], default=None)
    parser.add_argument("--gather-backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--coordinate-frame", choices=["raw", "pca"], default="raw")
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--skip-r1c", action="store_true")
    parser.add_argument("--skip-r1e", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    base_cfg = load_config(str(config_path))
    configure_deterministic_cpu()
    if args.denoise_method is not None:
        base_cfg.setdefault("denoise", {})["method"] = args.denoise_method
    base_cfg["gather_backend"] = args.gather_backend
    kitchen_cache = PathCache.load(base_cfg["cache"])
    second_cache = PathCache.load(str(Path(args.second_cache).resolve()))
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_root = out_path.parent
    out_root.mkdir(parents=True, exist_ok=True)
    r1a_path = root / "out" / "r1a" / "report.json"
    r1a = json.loads(r1a_path.read_text())
    r1a_pass = "world_triplane/target_scale" in r1a["gate"]["passing_world_anchored_arms"]
    r1c_runs = r1a_seed_rows(r1a) if args.coordinate_frame == "raw" else []
    if not args.skip_r1c:
        r1c_runs += run_r1c(
            base_cfg,
            kitchen_cache,
            root,
            out_root,
            args.seeds,
            args.rotations,
            args.bounds_modes,
            args.reuse,
            args.workers,
            args.coordinate_frame,
        )
        if args.coordinate_frame == "pca":
            r1a_rows = [
                row
                for row in r1c_runs
                if row["rotation_degrees"] == 0.0 and row["bounds_mode"] == "aabb"
            ]
            r1a_pass = bool(r1a_rows) and all(
                row["delta_db"] >= GATE_DELTA_DB for row in r1a_rows
            )
    r1e_runs = []
    if not args.skip_r1e:
        r1e_runs = run_r1e(
            base_cfg,
            second_cache,
            root,
            out_root,
            args.second_scene,
            args.seeds,
            args.reuse,
            args.workers,
            args.coordinate_frame,
        )
    gate = promotion_gate(
        r1a_pass,
        r1c_runs,
        r1e_runs,
        r1c_complete=(not args.skip_r1c) and r1c_runs_coverage_complete(r1c_runs),
        r1e_complete=(not args.skip_r1e) and r1e_seed_coverage_complete(r1e_runs),
    )
    report = {
        "experiment": "R1 promotion audit",
        "rung": "R1",
        "status": "promoted" if gate["promoted"] else "candidate_not_promoted",
        "command": "UV_CACHE_DIR=.uv-cache uv run python examples/r1_promotion.py "
        + " ".join(sys.argv[1:]),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "device": "cpu",
            "torch_num_threads": 1,
            "workers": args.workers,
            "deterministic_algorithms": True,
            "thread_environment": {
                "TBB_NUM_THREADS": "1",
                "OIDN_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
        },
        "candidate": {
            "representation": CANDIDATE,
            "output_bias_policy": POLICY,
            "gate_threshold_db": GATE_DELTA_DB,
            "training_denoiser": base_cfg.get("denoise", {}),
            "gather_backend": args.gather_backend,
            "coordinate_frame": args.coordinate_frame,
        },
        "r1a": {"report": "out/r1a/report.json", "candidate_pass": r1a_pass},
        "r1c": {
            "scene": "Country Kitchen",
            "rotations_degrees": args.rotations,
            "bounds_modes": args.bounds_modes,
            "runs": r1c_runs,
        },
        "r1e": {"scene": args.second_scene, "runs": r1e_runs},
        "promotion": gate,
        "stop_condition": promotion_stop_reason(gate),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(gate, indent=2))
    print(f"wrote {out_path}")
    if not gate["promoted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
