"""Re-score a completed run at a different held-out light count.

Trains nothing. A run directory already holds one `model.pt` per (arm, seed); the
number of held-out lights those checkpoints were SCORED against is an evaluation
choice, not a training one, so it can be revisited after the fact for the cost of a
forward pass. On Kitchen 128 that is ~16 s per seed for all four arms, against ~548 s
of training per seed -- which is why every committed verdict on this track can be
re-read for free. See docs/superpowers/plans/2026-08-29-heldout-light-estimator.md.

Because `build_val_set` draws lights one at a time from `default_rng([seed, 0x5EED])`,
a larger set extends the committed one rather than replacing it: re-scoring at the
original count must reproduce the original numbers exactly, and the test asserts it.

`rescore_encoding_redesign` does the same for the held-out-camera encoding-redesign
campaign (`examples/r1_encoding_redesign.py`), whose schema is different: it has no
single cache, scores per (arm, seed, rotation, held-out camera), and renders one
image per 8-light group. Every step of its evaluation path is imported from that
runner rather than reimplemented here.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.r1_encoding_redesign import (  # noqa: E402
    ARM_NAMES,
    N_EVAL_LIGHTS,
    N_HELD_OUT_CAMERAS,
    N_TRAIN_CAMERAS,
    _sparse_collision_fraction,
    camera_arc,
    campaign_peak,
    evaluate_camera,
    frozen_lights,
    load_conditioned_model,
    nearest_trained_camera,
    rotated_caches,
    rotated_camera,
    rotated_lights,
)
from examples.r1_promotion import rotation_matrix_y  # noqa: E402
from nrp.experiment_gate import EquivalenceGate  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.encoder_registry import SPATIAL_ENCODERS  # noqa: E402
from nrp.torch_backend.encoding_gates import (  # noqa: E402
    g1_generalization,
    g2_capacity_context,
    g3_stability,
    g4_frame_robustness,
    g5_fallback_decomposition,
    stop_reason,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.train import (  # noqa: E402
    build_val_set,
    evaluate,
    load_trained_model,
    model_tensors,
)


def rescore(
    run_dir: Path,
    cache: PathCache,
    *,
    seeds,
    arms,
    control_arm: str,
    base_cfg: dict,
    n_gate_lights: int,
) -> dict:
    device = torch.device("cpu")
    psnr: dict[tuple[str, int], np.ndarray] = {}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        val_set = build_val_set(cache, cfg, n_gate_lights)
        for arm in arms:
            model_path = run_dir / "train" / arm / f"seed{seed}" / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"no checkpoint for {arm} seed {seed}: {model_path}")
            # load_trained_model, not TorchNRP.load: occupancy-allocated arms
            # (world_sparse, occupancy world3d) cannot round-trip without it.
            model = load_trained_model(str(model_path), cache)
            spatial, aux = model_tensors(cache, model, device)
            rows = evaluate(model, val_set, spatial, aux, device)
            psnr[(arm, seed)] = np.asarray([r["psnr_db_vs_raw"] for r in rows], dtype=np.float64)
            del model

    gate = EquivalenceGate()
    out = {"n_gate_lights": int(n_gate_lights), "seeds": list(seeds), "arms": {}}
    for arm in arms:
        if arm == control_arm:
            continue
        per_light = [psnr[(arm, s)] - psnr[(control_arm, s)] for s in seeds]
        means = [float(d.mean()) for d in per_light]
        within = float(np.mean([d.var(ddof=1) for d in per_light]))
        out["arms"][arm] = {
            "per_seed_mean_delta_db": means,
            "mean_db": float(st.mean(means)),
            "between_seed_sd_db": float(st.pstdev(means)),
            "light_sem_db": math.sqrt(within / n_gate_lights),
            "gate": gate.evaluate(means),
        }
    return out


def rescore_sweep(
    sweep_dir: Path,
    control_dir: Path,
    cache: PathCache,
    *,
    seeds,
    resolutions,
    control_arm: str,
    base_cfg: dict,
    n_gate_lights: int,
) -> dict:
    """Re-score a K1-style resolution sweep against a fixed control run.

    The sweep trains only the swept arm; its baseline is the `pixel2d` arm of a
    separate control run. Both are scored against the same per-seed held-out set, so
    the control's checkpoints are reloaded here rather than its numbers reused --
    reusing numbers scored at a different light count is the confound this whole
    change exists to remove.
    """
    device = torch.device("cpu")
    gate = EquivalenceGate()
    out = {"n_gate_lights": int(n_gate_lights), "seeds": list(seeds), "resolutions": {}}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        val_set = build_val_set(cache, cfg, n_gate_lights)
        control_path = control_dir / "train" / control_arm / f"seed{seed}" / "model.pt"
        model = load_trained_model(str(control_path), cache)
        spatial, aux = model_tensors(cache, model, device)
        control_psnr = np.asarray(
            [r["psnr_db_vs_raw"] for r in evaluate(model, val_set, spatial, aux, device)],
            dtype=np.float64,
        )
        del model
        for res in resolutions:
            path = sweep_dir / "train" / f"finest{res}" / f"seed{seed}" / "model.pt"
            model = load_trained_model(str(path), cache)
            spatial, aux = model_tensors(cache, model, device)
            arm_psnr = np.asarray(
                [r["psnr_db_vs_raw"] for r in evaluate(model, val_set, spatial, aux, device)],
                dtype=np.float64,
            )
            del model
            row = out["resolutions"].setdefault(str(res), {"per_seed": [], "per_light_var": []})
            delta = arm_psnr - control_psnr
            row["per_seed"].append(float(delta.mean()))
            row["per_light_var"].append(float(delta.var(ddof=1)))
    for _res, row in out["resolutions"].items():
        row["mean_db"] = float(st.mean(row["per_seed"]))
        row["between_seed_sd_db"] = float(st.pstdev(row["per_seed"]))
        row["light_sem_db"] = math.sqrt(st.mean(row["per_light_var"]) / n_gate_lights)
        row["gate"] = gate.evaluate(row["per_seed"])
    return out


def redesign_light_groups(cache: PathCache, seed: int, n_gate_lights: int) -> list[list]:
    """Split one seed's held-out light draw into campaign-sized evaluation groups.

    The encoding-redesign campaign does not score one light per row the way the
    r1_parity schema does: `evaluate_camera` renders a *single* image lit by the whole
    `frozen_lights` set (N_EVAL_LIGHTS = 8) and reports one PSNR for it. Scoring that
    row against 96 lights summed into one image would change the estimand -- a
    96-emitter image is a different, far brighter physical configuration, not a
    lower-variance estimate of the same thing. So the extra light budget buys
    *independent draws of the campaign's own 8-light configuration* instead:
    `n_gate_lights` lights are split into `n_gate_lights // N_EVAL_LIGHTS` groups of 8,
    each group scored exactly as the campaign scores its one group, and the row's
    delta is the mean over groups. The estimand is unchanged; only the number of
    samples of it goes up.

    `frozen_lights` draws lights one at a time from `default_rng([seed, 0xE1C0DE])`,
    so a larger draw *extends* the committed one: group 0 is bit-identical to the
    committed campaign's light set, which is what makes the reproduction check at
    `n_gate_lights = 8` exact.
    """
    if n_gate_lights <= 0:
        raise ValueError(f"n_gate_lights must be positive, got {n_gate_lights}")
    if n_gate_lights % N_EVAL_LIGHTS != 0:
        raise ValueError(
            f"n_gate_lights={n_gate_lights} must be a multiple of the campaign's "
            f"group size N_EVAL_LIGHTS={N_EVAL_LIGHTS}; a partial group would score "
            "a row against a light configuration the campaign never defined"
        )
    lights = frozen_lights(cache, seed, n=n_gate_lights)
    return [lights[i : i + N_EVAL_LIGHTS] for i in range(0, n_gate_lights, N_EVAL_LIGHTS)]


def _mean_or_none(values: list) -> float | None:
    present = [float(v) for v in values if v is not None]
    return float(st.mean(present)) if present else None


def _aggregate_groups(group_rows: list[dict]) -> dict:
    """Average one row's per-group scores into the single row shape the gates read."""
    row = dict(group_rows[0])
    row["psnr_db"] = float(st.mean([r["psnr_db"] for r in group_rows]))
    row["baseline_psnr_db"] = float(st.mean([r["baseline_psnr_db"] for r in group_rows]))
    row["delta_db"] = float(st.mean([r["delta_db"] for r in group_rows]))
    row["in_occupancy_psnr_db"] = _mean_or_none([r["in_occupancy_psnr_db"] for r in group_rows])
    row["out_occupancy_psnr_db"] = _mean_or_none([r["out_occupancy_psnr_db"] for r in group_rows])
    row["n_light_groups"] = len(group_rows)
    row["per_group_delta_db"] = [float(r["delta_db"]) for r in group_rows]
    if len(group_rows) > 1:
        row["delta_sem_db"] = float(
            st.stdev(row["per_group_delta_db"]) / math.sqrt(len(group_rows))
        )
    else:
        row["delta_sem_db"] = None
    return row


def rescore_encoding_redesign(
    run_dir: Path,
    *,
    seeds,
    arms,
    rotations,
    n_gate_lights: int,
    threshold_db: float = 1.0,
    absolute_floor_db: float = 15.0,
) -> dict:
    """Re-score the R1 encoding-redesign campaign's committed checkpoints.

    Trains nothing. Every step of the evaluation path is imported from
    `examples/r1_encoding_redesign.py` itself -- the camera arc, the cache rotation,
    the camera rotation, the *light* rotation, the per-seed PSNR peak, the row
    construction, and the gate functions -- rather than reconstructed here. Run 2 of
    this campaign was invalidated by reusing unrotated evaluation lights at 90/180
    degrees; `rotated_lights` is the fix, and calling the campaign's own helper is the
    only way to be sure this re-read does not repeat it.
    """
    trained, held_out = camera_arc(N_TRAIN_CAMERAS, N_HELD_OUT_CAMERAS)
    rows_by_arm: dict[str, list[dict]] = {arm: [] for arm in arms}
    capacity_rows: list[dict] = []
    latest_capacity_report: dict[str, dict] = {}
    collision_by_arm: dict[str, float] = {}
    peak_by_seed: dict[int, list[float]] = {}

    for seed in seeds:
        seed_dir = run_dir / f"seed{seed}"
        cache_paths = {c["name"]: seed_dir / f"{c['name']}.npz" for c in trained + held_out}
        missing = [str(p) for p in cache_paths.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"seed {seed} is missing caches: {missing}")
        groups = redesign_light_groups(
            PathCache.load(str(cache_paths[trained[0]["name"]])), seed, n_gate_lights
        )
        trained_unrotated = [PathCache.load(str(cache_paths[c["name"]])) for c in trained]
        # One peak per light group, by the campaign's own recipe (trained cameras
        # only). Group 0's peak reproduces the committed `peak_by_seed` entry.
        peaks = [campaign_peak(trained_unrotated, group, seed=seed) for group in groups]
        peak_by_seed[seed] = peaks
        del trained_unrotated

        for rotation in rotations:
            rot_dir = seed_dir if rotation == 0.0 else seed_dir / f"rot{rotation:g}"
            rotation_matrix = rotation_matrix_y(rotation)
            caches = rotated_caches(cache_paths, rotation)
            cameras_in_frame = {
                camera["name"]: rotated_camera(camera, rotation_matrix)
                for camera in trained + held_out
            }
            groups_in_frame = [
                group if rotation == 0.0 else rotated_lights(group, rotation_matrix)
                for group in groups
            ]
            baseline_models = {
                camera["name"]: TorchNRP.load(
                    str(rot_dir / "pixel2d" / camera["name"] / "model.pt")
                ).eval()
                for camera in trained
            }
            for arm in arms:
                model = load_conditioned_model(
                    rot_dir / arm / "model.pt", [caches[camera["name"]] for camera in trained]
                )
                if model.encoding is not None and hasattr(model.encoding, "capacity_report"):
                    report = model.encoding.capacity_report()
                    latest_capacity_report[arm] = report
                    capacity_rows.append(
                        {
                            "arm": arm,
                            "seed": seed,
                            "rotation_degrees": float(rotation),
                            "capacity_report": report,
                        }
                    )
                    if getattr(SPATIAL_ENCODERS[arm], "guarantees_zero_collisions", False):
                        collision_by_arm[arm] = max(
                            collision_by_arm.get(arm, 0.0),
                            _sparse_collision_fraction(report, arm),
                        )
                for camera in held_out:
                    baseline_camera = nearest_trained_camera(camera, trained)
                    group_rows = [
                        evaluate_camera(
                            model,
                            arm,
                            camera,
                            cameras_in_frame[camera["name"]],
                            caches[camera["name"]],
                            baseline_models[baseline_camera["name"]],
                            group,
                            peaks[i],
                        )
                        for i, group in enumerate(groups_in_frame)
                    ]
                    row = _aggregate_groups(group_rows)
                    row["seed"] = seed
                    row["rotation_degrees"] = float(rotation)
                    row["baseline_camera"] = baseline_camera["name"]
                    rows_by_arm[arm].append(row)
                del model
            del baseline_models, caches

    expected_seeds = set(seeds)
    expected_cameras = {camera["name"] for camera in held_out}
    zero_collision_arms = frozenset(
        a for a in arms if getattr(SPATIAL_ENCODERS[a], "guarantees_zero_collisions", False)
    )
    arms_report = {}
    for arm in arms:
        rows = rows_by_arm[arm]
        arm_collisions = {arm: collision_by_arm[arm]} if arm in collision_by_arm else {}
        arms_report[arm] = {
            "rows": rows,
            "rows_count": len(rows),
            "capacity_report": latest_capacity_report.get(arm),
            "g1": g1_generalization(
                rows,
                threshold_db,
                expected_seeds=expected_seeds,
                expected_cameras=expected_cameras,
                absolute_floor_db=absolute_floor_db,
            ),
            "g3": g3_stability(rows, arm_collisions, threshold_db, zero_collision_arms),
            "g4": g4_frame_robustness(rows, threshold_db),
            "g5": g5_fallback_decomposition(rows),
        }

    report = {
        "n_gate_lights": int(n_gate_lights),
        "n_light_groups": int(n_gate_lights // N_EVAL_LIGHTS),
        "light_group_size": int(N_EVAL_LIGHTS),
        "seeds": list(seeds),
        "rotations": [float(r) for r in rotations],
        "absolute_floor_db": absolute_floor_db,
        "threshold_db": threshold_db,
        "peak_by_seed": {str(s): peaks for s, peaks in peak_by_seed.items()},
        "arms": arms_report,
        "g2_capacity_context": g2_capacity_context(capacity_rows),
        "promoted": False,
    }
    report["stop_reason"] = stop_reason(report)
    report["promoted"] = report["stop_reason"] is None
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--control-dir", type=Path)
    parser.add_argument("--redesign-dir", type=Path)
    parser.add_argument("--resolutions", type=int, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--gate-lights", type=int, default=96)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.redesign_dir is not None:
        # The encoding-redesign campaign carries its caches inside its own run
        # directory (one per camera per seed), so --cache does not apply here.
        result = rescore_encoding_redesign(
            args.redesign_dir,
            seeds=args.seeds,
            arms=ARM_NAMES,
            rotations=[0.0, 90.0, 180.0],
            n_gate_lights=args.gate_lights,
        )
        result["source_report"] = str(args.redesign_dir / "report.json")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        for arm, row in result["arms"].items():
            g1 = row["g1"]
            print(
                f"{arm:24s} mean={g1['mean_delta_db']:+.3f} worst={g1['worst_delta_db']:+.3f} "
                f"g1_failures={len(g1['failures'])}/{row['rows_count']} "
                f"seeds_passing={row['g3']['seeds_passing']}/{row['g3']['seeds_total']}"
            )
        print(f"stop_reason: {result['stop_reason']}")
        return 0 if result["promoted"] else 2

    if args.cache is None:
        parser.error("--cache is required for --run-dir and --sweep-dir modes")

    if args.sweep_dir is not None:
        if args.control_dir is None or args.resolutions is None:
            parser.error("--sweep-dir requires --control-dir and --resolutions")
        sweep_report = json.loads((args.sweep_dir / "report.json").read_text())
        control_report = json.loads((args.control_dir / "report.json").read_text())
        cache = PathCache.load(str(args.cache))
        result = rescore_sweep(
            args.sweep_dir,
            args.control_dir,
            cache,
            seeds=sweep_report["seeds"],
            resolutions=args.resolutions,
            control_arm=sweep_report["control_arm"],
            base_cfg=control_report["training_config"],
            n_gate_lights=args.gate_lights,
        )
        result["source_sweep_report"] = str(args.sweep_dir / "report.json")
        result["source_control_report"] = str(args.control_dir / "report.json")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        for res, row in result["resolutions"].items():
            print(
                f"finest{res:<6s} mean={row['mean_db']:+.3f} sd={row['between_seed_sd_db']:.3f} "
                f"light_sem={row['light_sem_db']:.3f} verdict={row['gate']['verdict']} "
                f"seeds_needed={row['gate'].get('seeds_needed')}"
            )
        return 0

    if args.run_dir is None:
        parser.error("either --run-dir or --sweep-dir is required")

    report = json.loads((args.run_dir / "report.json").read_text())
    cache = PathCache.load(str(args.cache))
    result = rescore(
        args.run_dir,
        cache,
        seeds=report["seeds"],
        arms=[report["control_arm"], *report["world_arms"]],
        control_arm=report["control_arm"],
        base_cfg=report["training_config"],
        n_gate_lights=args.gate_lights,
    )
    result["source_report"] = str(args.run_dir / "report.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for arm, row in result["arms"].items():
        print(
            f"{arm:24s} mean={row['mean_db']:+.3f} sd={row['between_seed_sd_db']:.3f} "
            f"light_sem={row['light_sem_db']:.3f} verdict={row['gate']['verdict']} "
            f"seeds_needed={row['gate'].get('seeds_needed')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
