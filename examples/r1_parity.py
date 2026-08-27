"""R1 parity: does a fairly-allocated world-anchored encoding match pixel2d
at a single trained view?

Rung R1 originally asked this exact question and answered no, but that measurement
was confounded: the "matched parameter budget" control handed the 3D encoder ~4,096
table slots for ~78,084 distinct queried grid vertices (~19x capacity handicap) while
`pixel2d` got 16,384 slots for 16,641 vertices. A later redesign campaign
(examples/r1_encoding_redesign.py) replaced this parity question with a held-out-
camera generalization question rather than re-answering it -- so parity under fair
allocation has never actually been tested. This is a NEW measurement, not a retry of
that failed one; a negative is a valid result here.

Design (single view, no camera arc, no rotations):
  - Arms: pixel2d (control), world_sparse, world_normal_triplane, world3d
    (allocation="occupancy").
  - Every arm shares the same base_resolution/finest_resolution ladder
    (--base-resolution, --finest-resolution; defaults 4/64). finest_resolution is
    chosen to match the render's pixel dimension (64 for the 64^2 toy render, 128 for
    the 128^2 Kitchen render), so pixel2d's natural one-vertex-per-pixel relationship
    to the image is mirrored by the world arms' finest level. Levels/table_size_log2
    differ only where the arm's own geometry requires it (world_normal_triplane reads
    one plane per point, so it gets 4 levels; world_sparse has no dense/hashed table
    to size, so it has no table_size_log2).
  - Capacity is NOT matched -- matching it is precisely what invalidated the
    original R1 measurement. Each arm's parameter count and capacity_report() are
    recorded instead.
  - 5 seeds, one held-out validation-light set per seed shared by all four arms for
    that seed (examples/r1a_variance.py's paired structure, reused not reimplemented).
  - Gate: the UNCHANGED original R1 gate. Each world arm's paired PSNR delta versus
    the same-run pixel2d control must be >= -0.5 dB for EVERY seed. Per-seed pass;
    a favourable mean never rescues a failing seed.

Usage:
  uv run python examples/r1_parity.py --seeds 0 --iters 50 --out-dir out/r1-parity-smoke
  uv run python examples/r1_parity.py --out-dir out/r1-parity
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.r1a_variance import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    build_frozen_validation_sets,
    cpu_brand,
    pair_validation_metrics,
    summarize_values,
    validation_fingerprint,
)
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.train import (  # noqa: E402
    evaluate,
    load_trained_model,
    model_tensors,
    train,
)

#: The unchanged original R1 gate: a world arm's paired delta vs. the same-run
#: pixel2d control must be at least this many dB, for every seed, to pass.
GATE_DELTA_DB = -0.5
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
CONTROL_ARM = "pixel2d"
WORLD_ARMS = ("world_sparse", "world_normal_triplane", "world3d")
ARMS = (CONTROL_ARM, *WORLD_ARMS)

#: A single trained-view cache from the completed R1-encoding-redesign campaign
#: (64x64, 16spp, 4 bounces): reused rather than re-traced, per this experiment's
#: single-view design (no camera arc, no rotations).
DEFAULT_CACHE = "out/r1-encoding-redesign/seed0/train0.npz"

#: Default resolution ladder, preserved exactly: base_resolution=4, finest_resolution=64
#: (one vertex per pixel at the finest level, matching pixel2d's natural relationship
#: to the 64^2 toy render). Callers that need a different render resolution (e.g. 128
#: for Kitchen) build a fresh ladder with `build_arm_models`.
DEFAULT_BASE_RESOLUTION = 4
DEFAULT_FINEST_RESOLUTION = 64


def build_arm_models(
    *,
    base_resolution: int = DEFAULT_BASE_RESOLUTION,
    finest_resolution: int = DEFAULT_FINEST_RESOLUTION,
) -> dict[str, dict]:
    """Build the per-arm model configs for a given base/finest resolution ladder.

    Every arm shares `base_resolution`/`finest_resolution`, including the control.
    Per-arm level counts and table sizes differ only where the arm's own geometry
    requires it -- see the module docstring -- and are NOT touched here.
    """
    return {
        "pixel2d": {
            "spatial_encoding": "pixel2d",
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 14,
                "base_resolution": base_resolution,
                "finest_resolution": finest_resolution,
            },
        },
        "world_sparse": {
            "spatial_encoding": "world_sparse",
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "base_resolution": base_resolution,
                "finest_resolution": finest_resolution,
            },
        },
        "world_normal_triplane": {
            "spatial_encoding": "world_normal_triplane",
            "encoding": {
                "levels": 4,
                "features_per_level": 2,
                "table_size_log2": 14,
                "base_resolution": base_resolution,
                "finest_resolution": finest_resolution,
            },
        },
        "world3d": {
            "spatial_encoding": "world3d",
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 14,
                "base_resolution": base_resolution,
                "finest_resolution": finest_resolution,
                "allocation": "occupancy",
            },
        },
    }


#: Module-level default ladder (base_resolution=4, finest_resolution=64), preserved
#: exactly for backward compatibility -- callers that need a different render
#: resolution should call `build_arm_models` and thread the result through
#: `make_arm_config`'s `arm_models` argument instead of mutating this.
ARM_MODELS: dict[str, dict] = build_arm_models()

#: Shared training/pool/denoise settings, matching examples/r1_toy_world3d.json.
BASE_TRAIN_CONFIG: dict = {
    "light_type": "sphere",
    "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
    "sampling": "segments",
    "pool": {"size": 64, "replace_every": 5, "replace_count": 2},
    "denoise": {"enabled": True, "method": "bilateral", "radius": 2},
    "iters": 3000,
    "batch_pixels": 4096,
    "lr": 0.005,
    "model": {"hidden_width": 128, "hidden_layers": 4},
    "n_val_lights": 12,
}


def make_arm_config(
    base: dict, arm: str, seed: int, out_dir: Path, arm_models: dict[str, dict] = ARM_MODELS
) -> dict:
    """Build one arm's training config: the shared ladder plus this arm's overrides.

    `base` is never mutated -- every nested structure that gets touched (`model`,
    `model.encoding`) is deep-copied before use. `arm_models` defaults to the
    module-level base=4/finest=64 ladder; pass the result of `build_arm_models` to
    use a different resolution ladder (e.g. for a different render resolution).
    """
    if arm not in arm_models:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(arm_models)}")
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["out_dir"] = str(out_dir)
    cfg["model"] = copy.deepcopy(cfg.get("model", {}))
    cfg["model"].update(copy.deepcopy(arm_models[arm]))
    return cfg


def arm_gate_verdict(per_seed_deltas: list[float] | np.ndarray, seeds: tuple[int, ...]) -> dict:
    """Per-seed pass/fail against the unchanged -0.5 dB gate.

    Raises rather than silently reporting a pass when there is nothing to gate:
    zero seeds, or a seed/delta count mismatch. A gate that reports a pass with no
    seeds evaluated is the exact recurring defect this experiment must not repeat.
    """
    if len(seeds) == 0:
        raise ValueError("cannot compute a gate verdict with zero seeds")
    deltas = list(per_seed_deltas)
    if len(deltas) != len(seeds):
        raise ValueError(
            f"per_seed_deltas has {len(deltas)} entries, expected one per seed ({len(seeds)})"
        )
    per_seed_pass = [bool(delta >= GATE_DELTA_DB) for delta in deltas]
    return {
        "threshold_db": GATE_DELTA_DB,
        "seed_count": len(seeds),
        "passing_seed_count": int(sum(per_seed_pass)),
        "per_seed_pass": per_seed_pass,
        "per_seed_delta_db": [float(d) for d in deltas],
        "pass": bool(all(per_seed_pass)),
        "definition": (
            "every seed's paired mean PSNR delta versus the same-run pixel2d control "
            "must be at least -0.5 dB (the unchanged original R1 gate)"
        ),
    }


def any_world_arm_passes(world_gates: dict[str, dict]) -> bool:
    """Whether at least one world-anchored arm's gate passed.

    Raises if no world arm was evaluated at all -- an empty mapping must never
    resolve to a vacuous pass (Python's `any({}) == False` already gets this right
    for the "no arm passed" case, but an empty *input* signals the experiment did
    not run, which is a different failure this function must not paper over).
    """
    if not world_gates:
        raise ValueError("no world arms were evaluated")
    return any(gate["pass"] for gate in world_gates.values())


def build_report(
    *,
    seeds: tuple[int, ...],
    training_arms: list[dict],
    world_gates: dict[str, dict],
    hardware: dict,
    extra: dict | None = None,
) -> dict:
    """Assemble the final report dict from already-computed results (no I/O)."""
    report = {
        "experiment": "R1 parity: fairly-allocated world-anchored encoding vs. pixel2d "
        "at a single trained view",
        "rung": "R1 (re-measured under fair allocation)",
        "seeds": list(seeds),
        "control_arm": CONTROL_ARM,
        "world_arms": list(WORLD_ARMS),
        "hardware": hardware,
        "training_arms": training_arms,
        "gate": world_gates,
        "any_world_arm_pass": any_world_arm_passes(world_gates),
    }
    if extra:
        report.update(extra)
    return report


def evaluate_model(model: TorchNRP, cache: PathCache, val_set: list[dict]) -> list[dict]:
    model = model.to(torch.device("cpu")).eval()
    spatial, aux = model_tensors(cache, model, torch.device("cpu"))
    return evaluate(model, val_set, spatial, aux, torch.device("cpu"))


def _relative_path(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def run_experiment(
    base_cfg: dict,
    cache: PathCache,
    *,
    root: Path,
    out_root: Path,
    seeds: tuple[int, ...],
    resamples: int,
    bootstrap_seed: int,
    arm_models: dict[str, dict] = ARM_MODELS,
) -> dict:
    validation_sets, validation_specs = build_frozen_validation_sets(cache, base_cfg, seeds)
    metrics_by_arm: dict[tuple[str, int], list[dict]] = {}
    training_arms: list[dict] = []

    for seed in seeds:
        for arm in ARMS:
            arm_dir = out_root / "train" / arm / f"seed{seed}"
            arm_cfg = make_arm_config(base_cfg, arm, seed, arm_dir, arm_models=arm_models)
            train_report = train(arm_cfg)
            model_path = arm_dir / "model.pt"
            model = load_trained_model(str(model_path), cache)
            metrics = evaluate_model(model, cache, validation_sets[seed])
            metrics_by_arm[(arm, seed)] = metrics
            capacity_report = model.encoding.capacity_report() if model.encoding else None
            psnrs = np.asarray([row["psnr_db_vs_raw"] for row in metrics], dtype=np.float64)
            training_arms.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "parameter_count": int(train_report["parameter_count"]),
                    "capacity_report": capacity_report,
                    "iters_per_second": train_report["iters_per_second"],
                    "train_seconds": train_report["train_seconds"],
                    "report": _relative_path(arm_dir / "torch_train_report.json", root),
                    "validation": metrics,
                    "validation_psnr_db_mean": float(psnrs.mean()),
                    "validation_psnr_db_std": float(psnrs.std()),
                }
            )
            print(
                f"{arm} seed {seed}: {psnrs.mean():.2f} dB, "
                f"{train_report['parameter_count']} params"
            )
            del model

    world_gates = {}
    per_arm_comparisons = {}
    for arm_index, arm in enumerate(WORLD_ARMS):
        per_seed = []
        seed_mean_deltas = []
        for seed_index, seed in enumerate(seeds):
            control = metrics_by_arm[(CONTROL_ARM, seed)]
            candidate = metrics_by_arm[(arm, seed)]
            per_light = pair_validation_metrics(control, candidate)
            deltas = np.asarray([row["delta_db"] for row in per_light], dtype=np.float64)
            per_seed.append(
                {
                    "seed": seed,
                    "per_light_deltas": per_light,
                    "summary": summarize_values(
                        deltas,
                        resamples=resamples,
                        bootstrap_seed=bootstrap_seed + arm_index * 100 + seed_index,
                    ),
                }
            )
            seed_mean_deltas.append(float(deltas.mean()))
        gate = arm_gate_verdict(seed_mean_deltas, seeds)
        gate["across_seed_summary"] = summarize_values(
            np.asarray(seed_mean_deltas, dtype=np.float64),
            resamples=resamples,
            bootstrap_seed=bootstrap_seed + 1000 + arm_index,
        )
        world_gates[arm] = gate
        per_arm_comparisons[arm] = per_seed

    return {
        "validation_specs": validation_specs,
        "validation_fingerprints": {
            str(seed): validation_fingerprint(validation_specs[str(seed)]) for seed in seeds
        },
        "training_arms": training_arms,
        "world_gates": world_gates,
        "comparisons": per_arm_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", default="out/r1-parity")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--iters", type=int, default=BASE_TRAIN_CONFIG["iters"])
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--finest-resolution",
        type=int,
        default=DEFAULT_FINEST_RESOLUTION,
        help="Finest hashgrid resolution for every arm; match the render's pixel "
        "dimension (default 64, for the 64^2 toy render).",
    )
    parser.add_argument(
        "--base-resolution",
        type=int,
        default=DEFAULT_BASE_RESOLUTION,
        help="Base (coarsest) hashgrid resolution for every arm (default 4).",
    )
    parser.add_argument(
        "--denoise-method",
        default=BASE_TRAIN_CONFIG["denoise"]["method"],
        choices=["bilateral", "oidn"],
        help="Pool/validation-target denoiser (default: bilateral).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if not cache_path.exists():
        raise SystemExit(f"cache not found: {cache_path}")
    cache = PathCache.load(str(cache_path))

    base_cfg = copy.deepcopy(BASE_TRAIN_CONFIG)
    base_cfg["cache"] = str(cache_path)
    base_cfg["iters"] = args.iters
    base_cfg["denoise"] = copy.deepcopy(base_cfg["denoise"])
    base_cfg["denoise"]["method"] = args.denoise_method
    if args.denoise_method == "oidn":
        base_cfg["denoise"].pop("radius", None)

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates")
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")
    if args.finest_resolution < 1:
        raise SystemExit("--finest-resolution must be positive")
    if args.base_resolution < 1:
        raise SystemExit("--base-resolution must be positive")

    arm_models = build_arm_models(
        base_resolution=args.base_resolution, finest_resolution=args.finest_resolution
    )

    result = run_experiment(
        base_cfg,
        cache,
        root=root,
        out_root=out_root,
        seeds=seeds,
        resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        arm_models=arm_models,
    )

    report = build_report(
        seeds=seeds,
        training_arms=result["training_arms"],
        world_gates=result["world_gates"],
        hardware={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": cpu_brand(),
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        extra={
            "command": (
                "UV_CACHE_DIR=.uv-cache uv run python examples/r1_parity.py "
                f"--seeds {' '.join(str(seed) for seed in seeds)} --iters {args.iters} "
                f"--finest-resolution {args.finest_resolution} "
                f"--base-resolution {args.base_resolution}"
            ),
            "cache": _relative_path(cache_path, root),
            "resolution": [cache.width, cache.height],
            "segments": cache.segment_count,
            "training_config": {
                k: v for k, v in base_cfg.items() if k not in {"out_dir", "seed", "cache"}
            },
            "resolution_ladder": {
                "base_resolution": args.base_resolution,
                "finest_resolution": args.finest_resolution,
            },
            "arm_models": arm_models,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "validation": {
                "lights": result["validation_specs"],
                "fingerprints": result["validation_fingerprints"],
            },
            "comparisons": result["comparisons"],
        },
    )

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": {arm: g["pass"] for arm, g in report["gate"].items()}}, indent=2))
    print(f"wrote {report_path}")
    if not report["any_world_arm_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
