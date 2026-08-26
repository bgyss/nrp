"""R1A: five-seed paired variance decomposition for world-anchored encodings.

The matrix is deliberately limited to the 128² Country Kitchen cache. Each seed
gets one held-out validation-light set, and that same set is evaluated for every
representation and output-bias policy. A world-anchored arm can promote only when
its paired delta is at least -0.5 dB for every seed; mean-only parity is not enough.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0x51A
REPRESENTATIONS = ("pixel2d", "world3d", "world_triplane")
OUTPUT_BIAS_POLICIES = {
    "target_scale": {
        "init_output_scale": True,
        "description": "target-scale output bias initialized from the training pool",
    },
    "framework_default": {
        "init_output_scale": False,
        "description": "framework-default output bias",
    },
}

ARM_MODELS = {
    "pixel2d": {
        "spatial_encoding": "pixel2d",
        "encoding": {
            "levels": 8,
            "features_per_level": 2,
            "table_size_log2": 14,
            "base_resolution": 4,
            "finest_resolution": 128,
        },
    },
    "world3d": {
        "spatial_encoding": "world3d",
        "encoding": {
            "levels": 8,
            "features_per_level": 2,
            "table_size_log2": 12,
            "base_resolution": 7,
            "finest_resolution": 128,
        },
    },
    "world_triplane": {
        "spatial_encoding": "world_triplane",
        "encoding": {
            "levels": 3,
            "features_per_level": 2,
            "table_size_log2": 13,
            "base_resolution": 4,
            "finest_resolution": 128,
        },
    },
}


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


def paired_bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Return a deterministic percentile CI for already-paired observations."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one observation")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return {
        "lower_db": float(lower),
        "upper_db": float(upper),
        "confidence": 0.95,
        "n": int(values.size),
        "resamples": int(resamples),
        "seed": int(seed),
        "method": "percentile bootstrap over paired observations",
    }


def summarize_values(
    values: np.ndarray | list[float], *, resamples: int, bootstrap_seed: int
) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("cannot summarize an empty value set")
    return {
        "n": int(values.size),
        "mean_db": float(values.mean()),
        "median_db": float(np.median(values)),
        "std_db": float(values.std()),
        "min_db": float(values.min()),
        "max_db": float(values.max()),
        "paired_bootstrap_95_ci_db": paired_bootstrap_ci(
            values, resamples=resamples, seed=bootstrap_seed
        ),
    }


def pair_validation_metrics(control: list[dict], candidate: list[dict]) -> list[dict]:
    """Pair candidate/control PSNRs by validation-light index and verify identity."""
    if len(control) != len(candidate):
        raise ValueError("paired validation sets must have equal lengths")
    paired = []
    for index, (control_row, candidate_row) in enumerate(zip(control, candidate, strict=True)):
        if control_row.get("light") != candidate_row.get("light"):
            raise ValueError(f"validation light mismatch at index {index}")
        paired.append(
            {
                "light_index": index,
                "light": control_row.get("light"),
                "control_psnr_db": float(control_row["psnr_db_vs_raw"]),
                "candidate_psnr_db": float(candidate_row["psnr_db_vs_raw"]),
                "delta_db": float(
                    candidate_row["psnr_db_vs_raw"] - control_row["psnr_db_vs_raw"]
                ),
            }
        )
    return paired


def validation_light_specs(val_set: list[dict], light_type: str) -> list[dict]:
    """Serialize the frozen validation lights without storing rendered images."""
    specs = []
    for index, entry in enumerate(val_set):
        light = entry["light"]
        spec = light.to_dict()
        spec.setdefault("type", light_type)
        specs.append({"light_index": index, "light": spec})
    return specs


def validation_fingerprint(specs: list[dict]) -> str:
    payload = json.dumps(specs, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def make_arm_config(
    base: dict, representation: str, policy: str, seed: int, out_dir: Path
) -> dict:
    if representation not in ARM_MODELS:
        raise ValueError(f"unknown representation {representation!r}")
    if policy not in OUTPUT_BIAS_POLICIES:
        raise ValueError(f"unknown output-bias policy {policy!r}")
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["out_dir"] = str(out_dir)
    cfg["model"].update(copy.deepcopy(ARM_MODELS[representation]))
    cfg["model"]["init_output_scale"] = OUTPUT_BIAS_POLICIES[policy]["init_output_scale"]
    cfg["r1a_output_bias_policy"] = policy
    cfg["r1a_representation"] = representation
    cfg["record_supervision_lights"] = True
    return cfg


def _sanitize_report(report: dict, root: Path, path: Path) -> dict:
    cache_path = Path(report["config"]["cache"])
    try:
        report["config"]["cache"] = cache_path.relative_to(root).as_posix()
    except ValueError:
        pass
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_or_reuse(
    base_cfg: dict,
    representation: str,
    policy: str,
    seed: int,
    out_root: Path,
    root: Path,
    reuse: bool,
) -> tuple[TorchNRP, dict, str]:
    arm_dir = out_root / "train" / policy / representation / f"seed{seed}"
    report_path = arm_dir / "torch_train_report.json"
    model_path = arm_dir / "model.pt"
    if reuse and report_path.exists() and model_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = train(make_arm_config(base_cfg, representation, policy, seed, arm_dir))
    report = _sanitize_report(report, root, report_path)
    return (
        TorchNRP.load(str(model_path)),
        report,
        f"out/r1a/train/{policy}/{representation}/seed{seed}/torch_train_report.json",
    )


def evaluate_model(model: TorchNRP, cache: PathCache, val_set: list[dict]) -> list[dict]:
    model = model.to(torch.device("cpu")).eval()
    spatial, aux = model_tensors(cache, model, torch.device("cpu"))
    return evaluate(model, val_set, spatial, aux, torch.device("cpu"))


def build_frozen_validation_sets(
    cache: PathCache, base_cfg: dict, seeds: tuple[int, ...] | list[int]
) -> tuple[dict[int, list[dict]], dict[str, list[dict]]]:
    """Build exactly one held-out set per seed for all matrix arms to share."""
    validation_sets = {}
    specs_by_seed = {}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        val_set = build_val_set(cache, cfg)
        validation_sets[seed] = val_set
        specs = validation_light_specs(val_set, cfg["light_type"])
        specs_by_seed[str(seed)] = specs
    return validation_sets, specs_by_seed


def _relative_path(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _pair_seed_summary(
    control_metrics: list[dict],
    candidate_metrics: list[dict],
    *,
    seed: int,
    bootstrap_seed: int,
    resamples: int,
) -> dict:
    per_light = pair_validation_metrics(control_metrics, candidate_metrics)
    deltas = np.asarray([row["delta_db"] for row in per_light], dtype=np.float64)
    return {
        "seed": seed,
        "per_light_deltas": per_light,
        "summary": summarize_values(
            deltas, resamples=resamples, bootstrap_seed=bootstrap_seed
        ),
        "gate_threshold_db": GATE_DELTA_DB,
        "gate_pass": bool(deltas.mean() >= GATE_DELTA_DB),
    }


def build_comparisons(
    metrics_by_arm: dict[tuple[str, str, int], list[dict]],
    seeds: tuple[int, ...] | list[int],
    *,
    resamples: int,
    bootstrap_seed: int,
) -> dict:
    comparisons = {}
    for candidate_index, candidate in enumerate(("world3d", "world_triplane")):
        policy_rows = {}
        for policy_index, policy in enumerate(OUTPUT_BIAS_POLICIES):
            per_seed = []
            for seed in seeds:
                per_seed.append(
                    _pair_seed_summary(
                        metrics_by_arm[(policy, "pixel2d", seed)],
                        metrics_by_arm[(policy, candidate, seed)],
                        seed=seed,
                        bootstrap_seed=(
                            bootstrap_seed + candidate_index * 100 + policy_index * 10 + seed
                        ),
                        resamples=resamples,
                    )
                )
            seed_deltas = np.asarray(
                [row["summary"]["mean_db"] for row in per_seed], dtype=np.float64
            )
            passing = int(sum(row["gate_pass"] for row in per_seed))
            policy_rows[policy] = {
                "candidate": candidate,
                "output_bias_policy": policy,
                "per_seed": per_seed,
                "across_seed_summary": summarize_values(
                    seed_deltas,
                    resamples=resamples,
                    bootstrap_seed=bootstrap_seed + 1000 + candidate_index * 10 + policy_index,
                ),
                "gate": {
                    "threshold_db": GATE_DELTA_DB,
                    "passing_seed_count": passing,
                    "seed_count": len(seeds),
                    "pass": passing == len(seeds),
                    "definition": (
                        "every seed-level mean of paired held-out-light PSNR deltas "
                        "must be at least -0.5 dB"
                    ),
                },
            }
        comparisons[candidate] = policy_rows
    return comparisons


def _arm_summary(
    representation: str,
    policy: str,
    seed: int,
    report: dict,
    report_path: str,
    metrics: list[dict],
) -> dict:
    psnrs = np.asarray([row["psnr_db_vs_raw"] for row in metrics], dtype=np.float64)
    return {
        "representation": representation,
        "output_bias_policy": policy,
        "seed": seed,
        "parameter_count": int(report["parameter_count"]),
        "iters_per_second": report["iters_per_second"],
        "train_seconds": report["train_seconds"],
        "report": report_path,
        "validation": metrics,
        "validation_psnr_db_mean": float(psnrs.mean()),
        "validation_psnr_db_median": float(np.median(psnrs)),
        "validation_psnr_db_std": float(psnrs.std()),
    }


def run_matrix(
    base_cfg: dict,
    cache: PathCache,
    *,
    root: Path,
    out_root: Path,
    seeds: tuple[int, ...] | list[int],
    reuse: bool,
    resamples: int,
    bootstrap_seed: int,
) -> dict:
    validation_sets, validation_specs = build_frozen_validation_sets(cache, base_cfg, seeds)
    metrics_by_arm = {}
    training_arms = []
    for seed in seeds:
        for policy in OUTPUT_BIAS_POLICIES:
            for representation in REPRESENTATIONS:
                model, train_report, report_path = train_or_reuse(
                    base_cfg, representation, policy, seed, out_root, root, reuse
                )
                metrics = evaluate_model(model, cache, validation_sets[seed])
                metrics_by_arm[(policy, representation, seed)] = metrics
                training_arms.append(
                    _arm_summary(
                        representation, policy, seed, train_report, report_path, metrics
                    )
                )
                print(
                    f"{representation} / {policy} seed {seed}: "
                    f"{training_arms[-1]['validation_psnr_db_mean']:.2f} dB, "
                    f"{train_report['parameter_count']} params"
                )
                del model

    return {
        "validation_specs": validation_specs,
        "validation_fingerprints": {
            seed: validation_fingerprint(validation_specs[str(seed)]) for seed in seeds
        },
        "training_arms": training_arms,
        "comparisons": build_comparisons(
            metrics_by_arm,
            seeds,
            resamples=resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="examples/kitchen_torch.json")
    parser.add_argument("--out-dir", default="out/r1a")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--reuse", action="store_true", help="reuse complete arm checkpoints")
    parser.add_argument(
        "--denoise-method",
        choices=["bilateral", "oidn"],
        default=None,
        help="override the Kitchen config denoiser for every arm",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    base_cfg = load_config(str(config_path))
    if args.denoise_method is not None:
        base_cfg.setdefault("denoise", {})["method"] = args.denoise_method
    cache = PathCache.load(base_cfg["cache"])
    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates")
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")

    matrix = run_matrix(
        base_cfg,
        cache,
        root=root,
        out_root=out_root,
        seeds=seeds,
        reuse=args.reuse,
        resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    passing_world_arms = [
        f"{candidate}/{policy}"
        for candidate, policies in matrix["comparisons"].items()
        for policy, comparison in policies.items()
        if comparison["gate"]["pass"]
    ]
    training_config = {
        key: value for key, value in base_cfg.items() if key not in {"out_dir", "seed"}
    }
    training_config["cache"] = _relative_path(base_cfg["cache"], root)
    report = {
        "experiment": "R1A variance decomposition",
        "rung": "R1A",
        "status": "complete",
        "command": (
            "UV_CACHE_DIR=.uv-cache uv run python examples/r1a_variance.py "
            f"--seeds {' '.join(str(seed) for seed in seeds)}"
            + (f" --denoise-method {args.denoise_method}" if args.denoise_method else "")
        ),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": cpu_brand(),
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "scene": {
            "name": "Country Kitchen",
            "cache": _relative_path(base_cfg["cache"], root),
            "resolution": [cache.width, cache.height],
            "segments": cache.segment_count,
            "light_type": base_cfg["light_type"],
            "denoiser": base_cfg["denoise"],
        },
        "matrix": {
            "seeds": list(seeds),
            "representations": list(REPRESENTATIONS),
            "output_bias_policies": OUTPUT_BIAS_POLICIES,
            "expected_arm_count": len(seeds) * len(REPRESENTATIONS) * len(OUTPUT_BIAS_POLICIES),
            "completed_arm_count": len(matrix["training_arms"]),
            "validation_lights_per_seed": base_cfg.get("n_val_lights", 12),
            "validation_freeze": (
                "one held-out light set was generated per seed before training and "
                "the same set object was evaluated for every representation/policy arm"
            ),
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "training_config": {
            **training_config,
        },
        "validation": {
            "lights": matrix["validation_specs"],
            "fingerprints": matrix["validation_fingerprints"],
        },
        "gate": {
            "threshold_db": GATE_DELTA_DB,
            "definition": (
                "for each output-bias policy, every world3d and world_triplane seed-level "
                "mean delta versus the same-policy pixel2d control must be at least -0.5 dB"
            ),
            "passing_world_anchored_arms": passing_world_arms,
            "any_world_anchored_arm_pass": bool(passing_world_arms),
            "all_crossed_world_anchored_arms_pass": all(
                comparison[policy]["gate"]["pass"]
                for comparison in matrix["comparisons"].values()
                for policy in OUTPUT_BIAS_POLICIES
            ),
            "r2_started": False,
        },
        "training_arms": matrix["training_arms"],
        "comparisons": matrix["comparisons"],
        "conclusions": {
            "r1a_candidate_pass": bool(passing_world_arms),
            "passing_world_anchored_arms": passing_world_arms,
            "r2_authorized": False,
            "note": (
                "R1A does not promote or authorize R2 unless a world-anchored arm "
                "passes the unchanged per-seed gate"
            ),
        },
    }
    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["conclusions"], indent=2))
    print(f"wrote {report_path}")
    if not report["conclusions"]["r1a_candidate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
