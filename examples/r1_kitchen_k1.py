"""K1: does lowering `world_sparse`'s finest resolution close the Kitchen parity gap?

`docs/plans/2026-08-27-kitchen-parity-next-steps.md` proposes a single decisive test
for the vertex-support hypothesis behind the Kitchen negative: on Country Kitchen
128^2, a majority of finest-level world-space grid vertices are touched by exactly
one observed pixel (59.1% at finest_resolution=128, vs. 33.7% on toy 64^2), so most
of `world_sparse`'s finest-level parameters are under-determined free variables that
can memorize their one sample without being forced to generalize.

This sweep varies `world_sparse`'s `finest_resolution` (32/48/64/96/128 by default),
holding everything else fixed, and compares every setting against the SAME fixed,
already-committed `pixel2d` control at finest_resolution=128
(`out/r1-parity-kitchen/report.json`). The control is read from that report, never
re-trained: the question is whether lowering the world arm's resolution closes the
gap against the existing screen-space baseline, not whether lowering both arms
together narrows it.

Alongside each setting's parity gate this records the measured finest-level
vertex-support distribution at that resolution (via examples/vertex_support.py, the
same occupancy machinery `nrp/torch_backend/occupancy.py` uses), so the gate result
and the quantity the hypothesis is about are reported side by side.

Prediction: the parity delta improves monotonically (or close to it) as
finest_resolution falls, best near where median vertex support approaches pixel2d's
~4 pixels/vertex.
Falsifier: a delta that is flat across the sweep, or worse as resolution falls,
refutes the hypothesis as stated -- K2-K4 must not then be run.

Usage:
    uv run python examples/r1_kitchen_k1.py \\
        --cache out/kitchen/path_cache.npz \\
        --control-report out/r1-parity-kitchen/report.json \\
        --out-dir out/r1-kitchen-parity-k1 \\
        --seeds 0 1 2 3 4 --resolutions 32 48 64 96 128 \\
        --iters 3000 --base-resolution 4 --denoise-method oidn
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

from examples.r1_parity import (  # noqa: E402
    BASE_TRAIN_CONFIG,
    CONTROL_ARM,
    GATE_DELTA_DB,
    arm_gate_verdict,
    build_arm_models,
    evaluate_model,
    make_arm_config,
)
from examples.r1a_variance import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    build_frozen_validation_sets,
    cpu_brand,
    pair_validation_metrics,
    summarize_values,
    validation_fingerprint,
)
from examples.vertex_support import cache_vertex_support  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.train import load_trained_model, train  # noqa: E402

#: The arm under test. K1 sweeps only this arm; the control is fixed and pre-committed.
SWEPT_ARM = "world_sparse"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_RESOLUTIONS = (32, 48, 64, 96, 128)
DEFAULT_CACHE = "out/kitchen/path_cache.npz"
#: The pre-determinism-fix control (`docs/performance.md`'s "Kitchen parity
#: re-measured under the deterministic denoiser" section retracts this run's
#: per-seed values and arm ranking, though not its aggregate "no arm passes"
#: verdict). Kept as the default only for backward compatibility with reports
#: already committed against it; a re-measurement should pass
#: `--control-report out/r1-parity-kitchen-det/report.json` instead.
DEFAULT_CONTROL_REPORT = "out/r1-parity-kitchen/report.json"
DEFAULT_BASE_RESOLUTION = 4

#: `training_config` keys that must match exactly between the fixed control and this
#: sweep's runs. A control built with a different pool size, learning rate,
#: batch_pixels, model width, sampling strategy, light type/bounds, or validation-
#: light count is not a valid fixed baseline even though its cache, base resolution,
#: and denoiser match -- silently comparing against it would attribute a training-
#: config difference to the swept resolution.
_STRICT_TRAINING_CONFIG_KEYS = (
    "pool",
    "lr",
    "batch_pixels",
    "model",
    "sampling",
    "light_type",
    "light_bounds",
    "n_val_lights",
)

#: `world_sparse`'s level count in the parity arm definition. Recorded here only so the
#: vertex-support diagnostic describes the same ladder the swept arm actually queries;
#: it is asserted against `build_arm_models` rather than trusted as a duplicate.
SWEPT_ARM_LEVELS = 8


def control_metrics_by_seed(control_report: dict, seeds: tuple[int, ...]) -> dict[int, list[dict]]:
    """Pull the fixed `pixel2d` control's per-seed validation metrics out of a report.

    Raises rather than silently comparing against a partial control: every requested
    seed must be present exactly once for the control arm. Comparing a swept arm
    against a control that is missing seeds -- or, worse, against a re-derived one --
    is exactly the confound this experiment is designed to avoid.
    """
    if not seeds:
        raise ValueError("cannot extract a control with zero seeds")
    control_arm = control_report.get("control_arm")
    if control_arm != CONTROL_ARM:
        raise ValueError(
            f"control report's control_arm is {control_arm!r}, expected {CONTROL_ARM!r}"
        )
    by_seed: dict[int, list[dict]] = {}
    for entry in control_report.get("training_arms", []):
        if entry.get("arm") != CONTROL_ARM:
            continue
        seed = int(entry["seed"])
        if seed in by_seed:
            raise ValueError(f"control report has duplicate {CONTROL_ARM} entries for seed {seed}")
        by_seed[seed] = entry["validation"]
    missing = [seed for seed in seeds if seed not in by_seed]
    if missing:
        raise ValueError(f"control report is missing {CONTROL_ARM} seeds {missing}")
    return {seed: by_seed[seed] for seed in seeds}


def check_control_compatibility(
    control_report: dict,
    *,
    cache: str,
    base_resolution: int,
    run_training_config: dict | None = None,
) -> dict:
    """Verify the fixed control was measured on the scene/ladder this sweep assumes.

    Returns the control's provenance for the report. Raises on a mismatch: a control
    trained on a different cache, on a different base resolution, or (when
    `run_training_config` is given) differing in any of `_STRICT_TRAINING_CONFIG_KEYS`
    (pool, lr, batch_pixels, model, sampling, light_type, light_bounds, n_val_lights)
    is not a valid fixed baseline for these runs no matter how convenient the numbers
    look -- a control built with a different pool size, lr, batch_pixels, or model
    width would otherwise compare cleanly and silently attribute a training-config
    difference to the swept resolution. `iters` and `denoise` are intentionally NOT
    checked here: `iters` is a caller-side warning (a shorter control is a legitimate
    smoke-test case) and `denoise.method` is checked separately, before targets are
    ever built, since it changes what the pool images ARE, not just how training ran.
    """
    control_cache = control_report.get("cache")
    if control_cache != cache:
        raise ValueError(f"control report was measured on cache {control_cache!r}, not {cache!r}")
    ladder = control_report.get("resolution_ladder", {})
    if ladder.get("base_resolution") != base_resolution:
        raise ValueError(
            f"control report's base_resolution is {ladder.get('base_resolution')!r}, "
            f"not {base_resolution!r}"
        )
    control_training_config = control_report.get("training_config") or {}
    if run_training_config is not None:
        for key in _STRICT_TRAINING_CONFIG_KEYS:
            control_value = control_training_config.get(key)
            run_value = run_training_config.get(key)
            if control_value != run_value:
                raise ValueError(
                    f"control report's training_config[{key!r}] is {control_value!r}, "
                    f"this run's is {run_value!r} -- a control built with a different "
                    f"{key} is not a valid fixed baseline"
                )
    return {
        "cache": control_cache,
        "arm": CONTROL_ARM,
        "finest_resolution": ladder.get("finest_resolution"),
        "base_resolution": ladder.get("base_resolution"),
        "command": control_report.get("command"),
        "hardware": control_report.get("hardware"),
        "training_config": control_report.get("training_config"),
    }


def check_validation_fingerprints(
    control_report: dict, fingerprints: dict[str, str], seeds: tuple[int, ...]
) -> None:
    """Assert the rebuilt validation lights are bit-identical to the control's.

    The whole comparison rests on the swept arm being scored on the exact held-out
    lights the fixed control was scored on. `pair_validation_metrics` also checks
    light-by-light identity downstream, but failing here fails before ~2 hours of
    training rather than after it.
    """
    recorded = (control_report.get("validation") or {}).get("fingerprints") or {}
    for seed in seeds:
        key = str(seed)
        if key not in recorded:
            raise ValueError(f"control report has no validation fingerprint for seed {seed}")
        if recorded[key] != fingerprints[key]:
            raise ValueError(
                f"validation-light fingerprint mismatch for seed {seed}: "
                f"control {recorded[key]}, rebuilt {fingerprints[key]} -- the swept arm "
                "would be scored on different lights than the fixed control"
            )


def spearman(xs: list[float] | np.ndarray, ys: list[float] | np.ndarray) -> float:
    """Spearman rank correlation, average ranks for ties (no scipy dependency)."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size != y.size:
        raise ValueError("spearman inputs must have equal length")
    if x.size < 2:
        raise ValueError("spearman needs at least two observations")
    rx, ry = _average_ranks(x), _average_ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    for value in np.unique(values):
        tied = values == value
        ranks[tied] = ranks[tied].mean()
    return ranks


def prediction_verdict(per_resolution: list[dict]) -> dict:
    """Evaluate K1's stated prediction against the sweep's mean deltas.

    The prediction is directional -- the delta improves as `finest_resolution` falls --
    so it is scored as a rank correlation between resolution and mean delta (predicted
    NEGATIVE) plus a strict-monotonicity check, not as a pass/fail on any one setting.
    A near-zero correlation is the "flat" falsifier; a positive one is the "worse as
    resolution falls" falsifier. This function reports; it does not decide for the
    reader, and it never converts a falsified prediction into a pass.
    """
    if len(per_resolution) < 2:
        raise ValueError("a prediction verdict needs at least two swept resolutions")
    resolutions = [float(row["finest_resolution"]) for row in per_resolution]
    means = [float(row["gate"]["across_seed_summary"]["mean_db"]) for row in per_resolution]
    order = np.argsort(resolutions)
    ascending_means = [means[i] for i in order]
    rho = spearman(resolutions, means)
    # Ordered by ascending resolution, the predicted pattern is a mean delta that never
    # improves as resolution rises -- i.e. a non-increasing sequence.
    monotonic = all(a >= b for a, b in zip(ascending_means[:-1], ascending_means[1:], strict=True))
    best_index = int(np.argmax(means))
    return {
        "prediction": (
            "the mean paired delta vs. the fixed pixel2d control improves monotonically "
            "(or close to it) as world_sparse's finest_resolution falls"
        ),
        "spearman_resolution_vs_mean_delta": rho,
        "predicted_sign": "negative",
        "direction_supports_prediction": bool(rho < 0.0),
        "monotonic_in_predicted_direction": bool(monotonic),
        "best_resolution": int(per_resolution[best_index]["finest_resolution"]),
        "best_mean_delta_db": means[best_index],
        "any_resolution_passes_gate": bool(any(row["gate"]["pass"] for row in per_resolution)),
        "falsifier": (
            "a flat (|rho| small) or positive correlation refutes the vertex-support "
            "hypothesis as stated; K2-K4 must not be run in that case"
        ),
    }


def build_report(
    *,
    seeds: tuple[int, ...],
    resolutions: tuple[int, ...],
    per_resolution: list[dict],
    control: dict,
    hardware: dict,
    extra: dict | None = None,
) -> dict:
    """Assemble the K1 report from already-computed results (no I/O)."""
    report = {
        "experiment": (
            "K1: world_sparse finest-resolution sweep on Country Kitchen 128^2 vs. the "
            "fixed committed pixel2d control at finest_resolution=128"
        ),
        "rung": "K1 (vertex-support hypothesis, decisive test)",
        "plan": "docs/plans/2026-08-27-kitchen-parity-next-steps.md",
        "swept_arm": SWEPT_ARM,
        "control_arm": CONTROL_ARM,
        "control": control,
        "seeds": list(seeds),
        "resolutions": list(resolutions),
        "hardware": hardware,
        "per_resolution": per_resolution,
        "gate": {str(row["finest_resolution"]): row["gate"]["pass"] for row in per_resolution},
        # A directional verdict needs a sweep. With fewer than two settings (a smoke run),
        # say so rather than raising and discarding the runs that did complete -- and
        # never substitute a single setting's gate result for the swept direction.
        "verdict": (
            prediction_verdict(per_resolution)
            if len(per_resolution) >= 2
            else {
                "prediction_not_evaluable": (
                    "fewer than two swept resolutions: K1's prediction is about the "
                    "direction of the delta across resolutions and cannot be scored here"
                ),
                "resolutions_measured": [int(row["finest_resolution"]) for row in per_resolution],
            }
        ),
    }
    if extra:
        report.update(extra)
    return report


def _relative_path(path: str | Path, root: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def run_sweep(
    base_cfg: dict,
    cache: PathCache,
    *,
    root: Path,
    out_root: Path,
    seeds: tuple[int, ...],
    resolutions: tuple[int, ...],
    base_resolution: int,
    control_by_seed: dict[int, list[dict]],
    validation_sets: dict[int, list[dict]],
    resamples: int,
    bootstrap_seed: int,
) -> list[dict]:
    per_resolution: list[dict] = []
    for res_index, finest in enumerate(resolutions):
        arm_models = build_arm_models(base_resolution=base_resolution, finest_resolution=finest)
        levels = arm_models[SWEPT_ARM]["encoding"]["levels"]
        support = cache_vertex_support(cache, levels, base_resolution, finest)
        runs: list[dict] = []
        seed_mean_deltas: list[float] = []
        comparisons: list[dict] = []
        for seed_index, seed in enumerate(seeds):
            arm_dir = out_root / "train" / f"finest{finest}" / f"seed{seed}"
            cfg = make_arm_config(base_cfg, SWEPT_ARM, seed, arm_dir, arm_models=arm_models)
            train_report = train(cfg)
            model = load_trained_model(str(arm_dir / "model.pt"), cache)
            metrics = evaluate_model(model, cache, validation_sets[seed])
            capacity_report = model.encoding.capacity_report() if model.encoding else None
            del model

            per_light = pair_validation_metrics(control_by_seed[seed], metrics)
            deltas = np.asarray([row["delta_db"] for row in per_light], dtype=np.float64)
            seed_mean_deltas.append(float(deltas.mean()))
            psnrs = np.asarray([row["psnr_db_vs_raw"] for row in metrics], dtype=np.float64)
            runs.append(
                {
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
            comparisons.append(
                {
                    "seed": seed,
                    "per_light_deltas": per_light,
                    "summary": summarize_values(
                        deltas,
                        resamples=resamples,
                        bootstrap_seed=bootstrap_seed + res_index * 100 + seed_index,
                    ),
                }
            )
            print(
                f"finest {finest} seed {seed}: {psnrs.mean():.2f} dB, "
                f"delta {deltas.mean():+.3f} dB, {train_report['parameter_count']} params",
                flush=True,
            )

        # K1's seed count (5, documented and defaulted) is not one of the equivalence
        # gate's scheduled looks, and K1 compares every resolution against a fixed
        # external control rather than running its own adaptive-stopping schedule, so
        # the equivalence rule's schedule does not apply here -- bind on the legacy
        # per-seed rule explicitly. K1's published results were measured under that
        # rule. Passing the default `binding="equivalence"` would raise
        # `ValueError: n=5 is not a scheduled look` only after this resolution's seeds
        # have already finished training.
        gate = arm_gate_verdict(seed_mean_deltas, seeds, binding="per_seed")
        gate["across_seed_summary"] = summarize_values(
            np.asarray(seed_mean_deltas, dtype=np.float64),
            resamples=resamples,
            bootstrap_seed=bootstrap_seed + 1000 + res_index,
        )
        per_resolution.append(
            {
                "finest_resolution": finest,
                "base_resolution": base_resolution,
                "levels": levels,
                "model": arm_models[SWEPT_ARM],
                "vertex_support": support,
                "finest_vertex_support": support["finest"],
                "runs": runs,
                "gate": gate,
                "comparisons": comparisons,
            }
        )
        # Checkpoint after every resolution: this sweep is hours long, and a failure at
        # the last setting must not throw away the settings already measured.
        (out_root / "partial.json").write_text(
            json.dumps({"per_resolution": per_resolution}, indent=2) + "\n"
        )
        print(
            f"finest {finest}: mean delta "
            f"{gate['across_seed_summary']['mean_db']:+.3f} dB, "
            f"gate {'PASS' if gate['pass'] else 'FAIL'}, "
            f"median vertex support {support['finest']['median_support']}, "
            f"<=1px {support['finest']['fraction_touched_by_le1_pixel']:.3f}",
            flush=True,
        )
    return per_resolution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--control-report", default=DEFAULT_CONTROL_REPORT)
    parser.add_argument("--out-dir", default="out/r1-kitchen-parity-k1")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--resolutions", nargs="+", type=int, default=list(DEFAULT_RESOLUTIONS))
    parser.add_argument("--iters", type=int, default=BASE_TRAIN_CONFIG["iters"])
    parser.add_argument("--base-resolution", type=int, default=DEFAULT_BASE_RESOLUTION)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--denoise-method",
        default="oidn",
        choices=["bilateral", "oidn"],
        help="Pool/validation-target denoiser; must match the fixed control's (oidn).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = root / cache_path
    if not cache_path.exists():
        raise SystemExit(f"cache not found: {cache_path}")

    control_path = Path(args.control_report)
    if not control_path.is_absolute():
        control_path = root / control_path
    if not control_path.exists():
        raise SystemExit(f"control report not found: {control_path}")
    control_report = json.loads(control_path.read_text())

    seeds = tuple(args.seeds)
    if len(set(seeds)) != len(seeds):
        raise SystemExit("--seeds must not contain duplicates")
    resolutions = tuple(args.resolutions)
    if len(set(resolutions)) != len(resolutions):
        raise SystemExit("--resolutions must not contain duplicates")
    if any(res < args.base_resolution for res in resolutions):
        raise SystemExit("every --resolutions entry must be >= --base-resolution")
    if args.bootstrap_resamples < 1:
        raise SystemExit("--bootstrap-resamples must be positive")

    control_cache = _relative_path(cache_path, root)
    control = check_control_compatibility(
        control_report,
        cache=control_cache,
        base_resolution=args.base_resolution,
        run_training_config=BASE_TRAIN_CONFIG,
    )
    control["report"] = _relative_path(control_path, root)
    control_by_seed = control_metrics_by_seed(control_report, seeds)

    control_denoise = (control.get("training_config") or {}).get("denoise", {})
    if control_denoise.get("method") != args.denoise_method:
        raise SystemExit(
            f"--denoise-method {args.denoise_method!r} does not match the fixed control's "
            f"{control_denoise.get('method')!r}; the swept arm's targets must be built the "
            "same way the control's were"
        )
    control_iters = (control.get("training_config") or {}).get("iters")
    if control_iters != args.iters:
        print(
            f"WARNING: --iters {args.iters} differs from the fixed control's {control_iters}",
            file=sys.stderr,
        )

    cache = PathCache.load(str(cache_path))

    base_cfg = copy.deepcopy(BASE_TRAIN_CONFIG)
    base_cfg["cache"] = str(cache_path)
    base_cfg["iters"] = args.iters
    base_cfg["denoise"] = copy.deepcopy(base_cfg["denoise"])
    base_cfg["denoise"]["method"] = args.denoise_method
    if args.denoise_method == "oidn":
        base_cfg["denoise"].pop("radius", None)

    validation_sets, validation_specs = build_frozen_validation_sets(cache, base_cfg, seeds)
    fingerprints = {
        str(seed): validation_fingerprint(validation_specs[str(seed)]) for seed in seeds
    }
    check_validation_fingerprints(control_report, fingerprints, seeds)

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    per_resolution = run_sweep(
        base_cfg,
        cache,
        root=root,
        out_root=out_root,
        seeds=seeds,
        resolutions=resolutions,
        base_resolution=args.base_resolution,
        control_by_seed=control_by_seed,
        validation_sets=validation_sets,
        resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )

    report = build_report(
        seeds=seeds,
        resolutions=resolutions,
        per_resolution=per_resolution,
        control=control,
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
                "UV_CACHE_DIR=.uv-cache uv run python examples/r1_kitchen_k1.py "
                f"--cache {args.cache} --control-report {args.control_report} "
                f"--out-dir {args.out_dir} "
                f"--seeds {' '.join(str(seed) for seed in seeds)} "
                f"--resolutions {' '.join(str(res) for res in resolutions)} "
                f"--iters {args.iters} --base-resolution {args.base_resolution} "
                f"--denoise-method {args.denoise_method}"
            ),
            "cache": control_cache,
            "resolution": [cache.width, cache.height],
            "segments": cache.segment_count,
            "training_config": {
                k: v for k, v in base_cfg.items() if k not in {"out_dir", "seed", "cache"}
            },
            "gate_threshold_db": GATE_DELTA_DB,
            "bootstrap_resamples": args.bootstrap_resamples,
            "bootstrap_seed": args.bootstrap_seed,
            "validation": {"lights": validation_specs, "fingerprints": fingerprints},
        },
    )

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["verdict"], indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
