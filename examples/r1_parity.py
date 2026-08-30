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
  uv run python examples/r1_parity.py --out-dir out/r1-parity
  # --seeds bypasses the look schedule and needs --gate per-seed (an explicit seed
  # count is not a scheduled look under the default --gate equivalence, which fails
  # fast rather than raising only after training finishes):
  uv run python examples/r1_parity.py --seeds 0 --iters 50 --gate per-seed \\
      --out-dir out/r1-parity-smoke
  uv run python examples/r1_parity.py --seeds 0 1 2 3 4 --gate per-seed \\
      --out-dir out/r1-parity

Exit codes: 0 if any world arm passed; 3 if no arm passed but every non-passing
arm's verdict is `underpowered` (the equivalence gate needs more seeds to answer
either way); 2 for any other non-pass (a real `fail`, or a non-equivalence binding
rule with no `underpowered` concept). CI can therefore tell "no" from "unknown".
1 for argument-validation failures (e.g. a missing cache, duplicate --seeds, or an
explicit --seeds list that is not a scheduled look under --gate equivalence) --
these fail before training starts.
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
from nrp.experiment_gate import EquivalenceGate, per_seed_verdict  # noqa: E402
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
    # The GATE's held-out set. Separate from n_val_lights (the training-time
    # checkpoint set) because only this one sets the precision of the per-seed delta
    # the equivalence gate consumes. At 12 lights the light-sampling standard error
    # on Kitchen 128 is +/-0.94 dB against a -0.5 dB threshold -- larger than the
    # between-seed spread the gate is trying to resolve. 96 puts it at 0.30 dB.
    "n_gate_lights": 96,
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


def plan_seed_batches(gate: EquivalenceGate, max_seeds: int) -> list[tuple[int, ...]]:
    """Seed batches, one per scheduled look, in fixed ascending order.

    Ascending order matters: it makes an n=16 run a strict prefix of an n=48 run, so
    a longer run reuses a shorter one's trainings and two runs at different caps stay
    comparable.
    """
    if max_seeds < gate.looks[0]:
        raise ValueError(
            f"max_seeds={max_seeds} is below the first look ({gate.looks[0]}); the gate "
            "cannot reach a verdict"
        )
    batches = []
    previous = 0
    for look in gate.looks:
        if look > max_seeds:
            break
        batches.append(tuple(range(previous, look)))
        previous = look
    return batches


def arm_gate_verdict(
    per_seed_deltas: list[float] | np.ndarray,
    seeds: tuple[int, ...],
    gate: EquivalenceGate | None = None,
    binding: str = "equivalence",
) -> dict:
    """Both gate verdicts for one arm, with `binding` naming the decisive one.

    Reporting both is nearly free and lets any report be read against either
    convention; naming the binding rule explicitly keeps a reader from having to
    infer which number decided the promotion.

    Raises rather than silently reporting a pass when there is nothing to gate:
    zero seeds, or a seed/delta count mismatch. A gate that reports a pass with no
    seeds evaluated is the exact recurring defect this experiment must not repeat.
    """
    if len(seeds) == 0:
        raise ValueError("cannot compute a gate verdict with zero seeds")
    deltas = [float(d) for d in per_seed_deltas]
    if len(deltas) != len(seeds):
        raise ValueError(
            f"per_seed_deltas has {len(deltas)} entries, expected one per seed ({len(seeds)})"
        )
    if binding not in ("equivalence", "per_seed"):
        raise ValueError(f"unknown binding rule {binding!r}")
    gate = gate or EquivalenceGate()
    # The equivalence verdict is only defined AT a scheduled look (EquivalenceGate.evaluate
    # raises off schedule -- that is the alpha correction's whole point). The legacy
    # per-seed rule has no such restriction and is the only rule the historical
    # `--seeds` mode (arbitrary seed counts, e.g. 5) can use, so it must not be blocked
    # by an off-schedule n. Reporting both stays "nearly free" only on schedule; off
    # schedule, equivalence is reported as unavailable rather than raised past the
    # caller when it isn't the binding rule.
    if len(seeds) in gate.looks:
        equivalence = gate.evaluate(deltas)
    elif binding == "equivalence":
        raise ValueError(
            f"n={len(seeds)} is not a scheduled look {gate.looks}; the equivalence rule "
            "cannot be binding off schedule"
        )
    else:
        equivalence = None
    legacy = per_seed_verdict(deltas, threshold_db=gate.threshold_db)
    decisive = equivalence["verdict"] == "pass" if binding == "equivalence" else legacy["pass"]
    return {
        "binding": binding,
        "equivalence": equivalence,
        "per_seed": legacy,
        "pass": bool(decisive),
    }


def check_seed_binding_compatibility(
    forced_seeds: tuple[int, ...] | None, binding: str, gate: EquivalenceGate
) -> None:
    """Fail fast when an explicit seed list can never reach a verdict under `binding`.

    `--seeds` bypasses the look schedule entirely (`run_experiment` trains exactly
    that one batch), so under `binding="equivalence"` the run trains every arm and
    only then discovers, inside `arm_gate_verdict`, that the seed count is not a
    scheduled look and the equivalence rule cannot be binding off schedule -- after
    however many hours of training that took. Raise here, before any training
    starts, instead. `binding="per_seed"` has no such restriction and is unaffected.
    """
    if forced_seeds is None or binding != "equivalence":
        return
    if len(forced_seeds) not in gate.looks:
        raise ValueError(
            f"--seeds has {len(forced_seeds)} seeds, which is not a scheduled look "
            f"{gate.looks}, but --gate equivalence (the default) is selected; the "
            "equivalence rule cannot be evaluated off schedule. Either pass "
            "--gate per-seed, or choose a --seeds count matching one of the "
            f"scheduled looks {gate.looks}."
        )


def reproduce_command(args: argparse.Namespace, seeds: tuple[int, ...]) -> str:
    """The exact command line that reproduces this run.

    Must record every argument that distinguishes this run's RESULT, not merely
    its output location -- --cache and --out-dir locate the run, but
    --denoise-method, --bootstrap-seed, --bootstrap-resamples, and --gate-lights
    change the numbers themselves. A command string missing any of those replays a
    DIFFERENT measurement (e.g. the bilateral-denoiser default instead of the
    oidn run actually made, or the pre-96 12-light gate set) while appearing to
    reproduce this one.
    """
    return (
        "UV_CACHE_DIR=.uv-cache uv run python examples/r1_parity.py "
        f"--cache {args.cache} --out-dir {args.out_dir} "
        f"--seeds {' '.join(str(seed) for seed in seeds)} --iters {args.iters} "
        f"--finest-resolution {args.finest_resolution} "
        f"--base-resolution {args.base_resolution} "
        f"--denoise-method {args.denoise_method} "
        f"--gate {args.gate} --max-seeds {args.max_seeds} "
        f"--bootstrap-seed {args.bootstrap_seed} "
        f"--bootstrap-resamples {args.bootstrap_resamples} "
        f"--gate-lights {args.gate_lights}"
    )


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


def gate_exit_code(world_gates: dict[str, dict]) -> int:
    """0 if any world arm passed; 3 if every non-passing arm is `underpowered`
    (CI's `equivalence` verdict); 2 for a real failure (at least one arm's
    equivalence verdict is `fail`, or the binding rule has no `underpowered`
    concept, e.g. `per_seed`).

    Distinguishing 2 from 3 lets CI tell "no" (a clear failure) from "unknown"
    (every arm needs more seeds to say either way) instead of collapsing both
    into the same exit code.
    """
    if any_world_arm_passes(world_gates):
        return 0
    verdicts = []
    for gate in world_gates.values():
        equivalence = gate.get("equivalence")
        if equivalence is None:
            # No equivalence verdict was computed (e.g. off-schedule under
            # `per_seed` binding) -- nothing to call "underpowered", so this is a
            # plain failure.
            return 2
        verdicts.append(equivalence["verdict"])
    if verdicts and all(verdict == "underpowered" for verdict in verdicts):
        return 3
    return 2


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


def seed_mean_delta(control_metrics: list[dict], candidate_metrics: list[dict]) -> float:
    """Mean paired PSNR delta over the validation lights, for one arm and one seed."""
    per_light = pair_validation_metrics(control_metrics, candidate_metrics)
    return float(np.mean([row["delta_db"] for row in per_light]))


def run_experiment(
    base_cfg: dict,
    cache: PathCache,
    *,
    root: Path,
    out_root: Path,
    seeds: tuple[int, ...] | None = None,
    resamples: int,
    bootstrap_seed: int,
    arm_models: dict[str, dict] = ARM_MODELS,
    gate: EquivalenceGate | None = None,
    binding: str = "equivalence",
    max_seeds: int | None = None,
) -> dict:
    """Train every arm over the gate's look schedule, stopping at the first verdict.

    `seeds` forces an explicit seed list (the per-seed rule's mode, and how a caller
    reproduces a historical run); otherwise seeds come from the look schedule. Early
    stopping only ever happens AT a look, which is what keeps the alpha correction
    honest.
    """
    gate = gate or EquivalenceGate()
    if seeds is not None:
        seed_batches = [tuple(seeds)]
    else:
        seed_batches = plan_seed_batches(gate, max_seeds or gate.cap)

    validation_sets: dict[int, list[dict]] = {}
    validation_specs: dict[str, list[dict]] = {}
    metrics_by_arm: dict[tuple[str, int], list[dict]] = {}
    training_arms: list[dict] = []
    trained_seeds: list[int] = []
    # One definition of "this arm's mean delta at this seed" (`seed_mean_delta`),
    # computed once per (arm, seed) and shared between the look-boundary check and
    # the final summary below instead of each recomputing it independently.
    mean_delta_cache: dict[tuple[str, int], float] = {}

    def cached_mean_delta(arm: str, seed: int) -> float:
        key = (arm, seed)
        if key not in mean_delta_cache:
            mean_delta_cache[key] = seed_mean_delta(
                metrics_by_arm[(CONTROL_ARM, seed)], metrics_by_arm[(arm, seed)]
            )
        return mean_delta_cache[key]

    for batch in seed_batches:
        batch_sets, batch_specs = build_frozen_validation_sets(
            cache, base_cfg, batch, n_gate_lights=base_cfg.get("n_gate_lights")
        )
        validation_sets.update(batch_sets)
        validation_specs.update(batch_specs)

        for seed in batch:
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
            trained_seeds.append(seed)

        # Look boundary: stop as soon as every world arm has a terminal verdict.
        if binding == "equivalence" and len(trained_seeds) in gate.looks:
            verdicts = []
            for arm in WORLD_ARMS:
                deltas = [cached_mean_delta(arm, seed) for seed in trained_seeds]
                verdicts.append(gate.evaluate(deltas)["verdict"])
            if all(verdict in ("pass", "fail") for verdict in verdicts):
                break

    seeds_run = tuple(trained_seeds)
    world_gates = {}
    per_arm_comparisons = {}
    for arm_index, arm in enumerate(WORLD_ARMS):
        per_seed = []
        seed_mean_deltas = []
        for seed_index, seed in enumerate(seeds_run):
            control = metrics_by_arm[(CONTROL_ARM, seed)]
            candidate = metrics_by_arm[(arm, seed)]
            per_light = pair_validation_metrics(control, candidate)
            deltas = np.asarray([row["delta_db"] for row in per_light], dtype=np.float64)
            per_seed.append(
                {
                    "seed": seed,
                    "per_light_deltas": per_light,
                    # Bootstrap stays in the report as a DESCRIPTIVE statistic; the
                    # gate's Student-t interval is what binds (see the module docstring).
                    "summary": summarize_values(
                        deltas,
                        resamples=resamples,
                        bootstrap_seed=bootstrap_seed + arm_index * 100 + seed_index,
                    ),
                }
            )
            seed_mean_deltas.append(cached_mean_delta(arm, seed))
        verdict = arm_gate_verdict(seed_mean_deltas, seeds_run, gate=gate, binding=binding)
        verdict["across_seed_summary"] = summarize_values(
            np.asarray(seed_mean_deltas, dtype=np.float64),
            resamples=resamples,
            bootstrap_seed=bootstrap_seed + 1000 + arm_index,
        )
        world_gates[arm] = verdict
        per_arm_comparisons[arm] = per_seed

    return {
        "seeds_run": list(seeds_run),
        "validation_specs": validation_specs,
        "validation_fingerprints": {
            str(seed): validation_fingerprint(validation_specs[str(seed)]) for seed in seeds_run
        },
        "training_arms": training_arms,
        "world_gates": world_gates,
        "comparisons": per_arm_comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", default="out/r1-parity")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Force an explicit seed list, bypassing the look schedule (how a caller "
        "reproduces a historical run; the only mode --gate per-seed uses). Omit to "
        "draw seeds from the look schedule under --gate equivalence.",
    )
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
    parser.add_argument(
        "--gate",
        default="equivalence",
        choices=["equivalence", "per-seed"],
        help="Which rule is binding for promotion (default: equivalence). Both verdicts "
        "are always recorded in the report.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=EquivalenceGate().cap,
        help="Seed cap for the adaptive look schedule (default: 48).",
    )
    parser.add_argument(
        "--gate-lights",
        type=int,
        default=BASE_TRAIN_CONFIG["n_gate_lights"],
        help="held-out lights per seed for the promotion gate (default 96)",
    )
    args = parser.parse_args()
    if args.gate_lights <= 0:
        raise SystemExit(f"--gate-lights must be positive, got {args.gate_lights}")

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
    base_cfg["n_gate_lights"] = args.gate_lights

    out_root = Path(args.out_dir)
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    forced_seeds = tuple(args.seeds) if args.seeds else None
    if forced_seeds is not None and len(set(forced_seeds)) != len(forced_seeds):
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

    gate = EquivalenceGate()
    binding = args.gate.replace("-", "_")
    try:
        check_seed_binding_compatibility(forced_seeds, binding, gate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    result = run_experiment(
        base_cfg,
        cache,
        root=root,
        out_root=out_root,
        seeds=forced_seeds,
        resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        arm_models=arm_models,
        gate=gate,
        binding=binding,
        max_seeds=args.max_seeds,
    )
    seeds = tuple(result["seeds_run"])

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
            "command": reproduce_command(args, seeds),
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
            "gate_lights": args.gate_lights,
            "validation": {
                "lights": result["validation_specs"],
                "fingerprints": result["validation_fingerprints"],
            },
            "comparisons": result["comparisons"],
            "gate_rule": binding,
            "gate_schedule": {
                "looks": list(gate.looks),
                "cap": gate.cap,
                "alpha_overall": gate.alpha,
                "confidence_per_look": gate.confidence_per_look,
            },
        },
    )

    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": {arm: g["pass"] for arm, g in report["gate"].items()}}, indent=2))
    print(f"wrote {report_path}")
    if not report["any_world_arm_pass"]:
        raise SystemExit(gate_exit_code(report["gate"]))


if __name__ == "__main__":
    main()
