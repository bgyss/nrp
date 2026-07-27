"""R1 failure analysis: provenance, seed stability, collisions, and tri-plane follow-up.

This does not loosen the original 0.5 dB gate. It repairs the comparison so every
candidate is evaluated against a same-run pixel2d control on the exact same held-out
lights, then tests three hypotheses:

1. historical-baseline drift (validation-light provenance + output initialization),
2. 3D hash-collision pressure at matched parameter budget,
3. whether a world-anchored tri-plane allocation is a better next representation.
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

from nrp.gather_light import gather_light
from nrp.metrics import psnr
from nrp.path_cache import PathCache
from nrp.torch_backend.denoise import denoise_image
from nrp.torch_backend.model import TorchNRP
from nrp.torch_backend.sampling import sample_light
from nrp.torch_backend.train import (
    build_val_set,
    evaluate,
    light_param_vector,
    load_config,
    model_tensors,
    train,
)

GATE_DELTA_DB = -0.5
HASH_PRIMES = (1, 2654435761, 805459861)

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
    # 106,239 params vs pixel2d's 106,085 (+0.15%). Three relatively large
    # 2D tables reduce collision pressure while remaining world-anchored.
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
    # Diagnostic capacity arm, deliberately unmatched: same scale schedule as
    # pixel2d but 3D tables (199,841 params). If this wins while matched world3d
    # fails, collision/capacity pressure is implicated.
    "world3d_expanded": {
        "spatial_encoding": "world3d",
        "encoding": {
            "levels": 8,
            "features_per_level": 2,
            "table_size_log2": 14,
            "base_resolution": 4,
            "finest_resolution": 128,
        },
    },
}


def legacy_validation_set(cache: PathCache, cfg: dict) -> list[dict]:
    """Reconstruct the exact pre-checkpoint-era validation stream.

    The historical kitchen artifact drew validation lights from the training RNG
    after the initial pool plus every replacement. Later code moved validation to a
    dedicated RNG. Replaying the old stream is required for an honest cross-era
    comparison.
    """
    rng = np.random.default_rng(cfg.get("seed", 0))
    pool = cfg["pool"]
    consumed = pool["size"] + pool["replace_count"] * (cfg["iters"] // pool["replace_every"])
    for _ in range(consumed):
        sample_light(
            cache,
            rng,
            cfg["light_type"],
            cfg["light_bounds"],
            cfg.get("sampling", "segments"),
        )
    result = []
    for _ in range(cfg.get("n_val_lights", 12)):
        light = sample_light(
            cache,
            rng,
            cfg["light_type"],
            cfg["light_bounds"],
            cfg.get("sampling", "segments"),
        )
        raw = gather_light(cache, light).reshape(-1, 3)
        denoised = denoise_image(
            raw.reshape(cache.height, cache.width, 3),
            cache.albedo,
            cache.normal,
            cache.depth,
            method=cfg.get("denoise", {}).get("method", "bilateral"),
        ).reshape(-1, 3)
        result.append({"light": light, "raw": raw, "denoised": denoised})
    return result


def summarize_metrics(metrics: list[dict]) -> dict:
    psnrs = np.asarray([entry["psnr_db_vs_raw"] for entry in metrics])
    return {
        "psnr_db_mean": float(psnrs.mean()),
        "psnr_db_median": float(np.median(psnrs)),
        "psnr_db_min": float(psnrs.min()),
        "psnr_db_max": float(psnrs.max()),
    }


def exact_light_match_count(first: list[dict], second: list[dict]) -> int:
    first_params = [light_param_vector(entry["light"]) for entry in first]
    second_params = [light_param_vector(entry["light"]) for entry in second]
    return sum(
        any(np.array_equal(params, candidate) for candidate in second_params)
        for params in first_params
    )


def evaluate_model(model: TorchNRP, cache: PathCache, val_set: list[dict]) -> dict:
    device = torch.device("cpu")
    model = model.to(device).eval()
    spatial, aux = model_tensors(cache, model, device)
    return summarize_metrics(evaluate(model, val_set, spatial, aux, device))


def make_arm_config(base: dict, arm: str, seed: int, out_dir: Path) -> dict:
    cfg = copy.deepcopy(base)
    cfg["seed"] = seed
    cfg["device"] = "cpu"
    cfg["out_dir"] = str(out_dir)
    source = arm.removesuffix("_legacy_init")
    cfg["model"].update(copy.deepcopy(ARM_MODELS[source]))
    cfg["model"].pop("init_output_scale", None)
    if arm.endswith("_legacy_init"):
        cfg["model"]["init_output_scale"] = False
    return cfg


def sanitize_report(report: dict, root: Path, path: Path) -> dict:
    cache_path = Path(report["config"]["cache"])
    try:
        report["config"]["cache"] = cache_path.relative_to(root).as_posix()
    except ValueError:
        pass
    path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def train_or_reuse(
    base_cfg: dict,
    arm: str,
    seed: int,
    out_root: Path,
    root: Path,
    reuse: bool,
) -> tuple[TorchNRP, dict, str]:
    if seed == 0 and arm in {"pixel2d", "world3d"}:
        existing = root / "out" / "r1-worldgrid" / "kitchen" / "cpu" / arm
        if reuse and (existing / "model.pt").exists() and (
            existing / "torch_train_report.json"
        ).exists():
            report = json.loads((existing / "torch_train_report.json").read_text())
            return (
                TorchNRP.load(str(existing / "model.pt")),
                report,
                f"out/r1-worldgrid/kitchen/cpu/{arm}/torch_train_report.json",
            )

    arm_dir = out_root / "train" / f"{arm}-seed{seed}"
    report_path = arm_dir / "torch_train_report.json"
    model_path = arm_dir / "model.pt"
    if reuse and report_path.exists() and model_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = train(make_arm_config(base_cfg, arm, seed, arm_dir))
    report = sanitize_report(report, root, report_path)
    return (
        TorchNRP.load(str(model_path)),
        report,
        f"out/r1-followup/train/{arm}-seed{seed}/torch_train_report.json",
    )


def _unique_vertices(coords: np.ndarray, resolution: int) -> np.ndarray:
    dims = coords.shape[1]
    base = np.floor(coords * resolution).astype(np.int64).clip(0, resolution)
    corners = []
    for corner in np.ndindex(*(2,) * dims):
        corners.append(np.minimum(base + np.asarray(corner), resolution))
    return np.unique(np.concatenate(corners, axis=0), axis=0)


def _grid_occupancy(coords: np.ndarray, encoding, dims: int) -> dict:
    levels = []
    for level, resolution in enumerate(encoding.resolutions):
        vertices = _unique_vertices(coords, resolution)
        if dims == 2:
            dense_index = vertices[:, 1] * (resolution + 1) + vertices[:, 0]
            hashed = (vertices[:, 0] * HASH_PRIMES[0]) ^ (
                vertices[:, 1] * HASH_PRIMES[1]
            )
        else:
            dense_index = (
                (vertices[:, 2] * (resolution + 1) + vertices[:, 1])
                * (resolution + 1)
                + vertices[:, 0]
            )
            hashed = (
                (vertices[:, 0] * HASH_PRIMES[0])
                ^ (vertices[:, 1] * HASH_PRIMES[1])
                ^ (vertices[:, 2] * HASH_PRIMES[2])
            )
        dense = encoding._dense[level]
        slots = dense_index if dense else hashed & (encoding.table_size - 1)
        unique_slots, loads = np.unique(slots, return_counts=True)
        levels.append(
            {
                "level": level,
                "resolution": resolution,
                "dense": dense,
                "queried_vertices": int(vertices.shape[0]),
                "occupied_slots": int(unique_slots.shape[0]),
                "collision_fraction": float(1.0 - unique_slots.shape[0] / vertices.shape[0]),
                "max_unique_vertices_per_slot": int(loads.max()),
            }
        )
    queried = sum(level["queried_vertices"] for level in levels)
    occupied = sum(level["occupied_slots"] for level in levels)
    return {
        "weighted_collision_fraction": float(1.0 - occupied / queried),
        "levels": levels,
    }


def occupancy_report(model: TorchNRP, cache: PathCache) -> dict:
    h, w = cache.height, cache.width
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    world = cache.position.reshape(-1, 3)
    if model.spatial_encoding != "pixel2d":
        world = np.clip(
            (world - model.world_min.numpy()) / model.world_extent.numpy(), 0.0, 1.0
        )
    if model.spatial_encoding == "pixel2d":
        grids = {"pixel_xy": _grid_occupancy(xy, model.encoding, 2)}
    elif model.spatial_encoding == "world3d":
        grids = {"world_xyz": _grid_occupancy(world, model.encoding, 3)}
    else:
        projections = {
            "xy": world[:, (0, 1)],
            "xz": world[:, (0, 2)],
            "yz": world[:, (1, 2)],
        }
        grids = {
            name: _grid_occupancy(coords, plane, 2)
            for (name, coords), plane in zip(
                projections.items(), model.encoding.planes, strict=True
            )
        }
    queried = sum(
        sum(level["queried_vertices"] for level in grid["levels"]) for grid in grids.values()
    )
    occupied = sum(
        sum(level["occupied_slots"] for level in grid["levels"]) for grid in grids.values()
    )
    return {
        "spatial_encoding": model.spatial_encoding,
        "parameter_count": model.parameter_count,
        "unique_world_positions": int(
            np.unique(cache.position.reshape(-1, 3), axis=0).shape[0]
        ),
        "pixel_count": cache.width * cache.height,
        "weighted_collision_fraction": float(1.0 - occupied / queried),
        "grids": grids,
    }


def depth_region_metrics(
    model: TorchNRP, cache: PathCache, val_set: list[dict]
) -> list[dict]:
    model = model.cpu().eval()
    depth = cache.depth.reshape(-1)
    edges = np.quantile(depth, [0.0, 0.25, 0.5, 0.75, 1.0])
    spatial, aux = model_tensors(cache, model, torch.device("cpu"))
    rows = []
    with torch.no_grad():
        for quartile in range(4):
            lo, hi = edges[quartile], edges[quartile + 1]
            mask = (depth >= lo) & (depth <= hi if quartile == 3 else depth < hi)
            scores = []
            for entry in val_set:
                params = torch.as_tensor(
                    light_param_vector(entry["light"]), dtype=torch.float32
                ).expand(spatial.shape[0], -1)
                pred = model(spatial, aux, params).numpy().astype(np.float64)
                scores.append(psnr(pred[mask], entry["raw"][mask]))
            rows.append(
                {
                    "depth_quartile": quartile + 1,
                    "depth_min": float(lo),
                    "depth_max": float(hi),
                    "pixels": int(mask.sum()),
                    "psnr_db_mean": float(np.mean(scores)),
                }
            )
    return rows


def aggregate_seed_deltas(rows: list[dict], candidate: str) -> dict:
    by_seed = {}
    for row in rows:
        if row["arm"] in {"pixel2d", candidate}:
            by_seed.setdefault(row["seed"], {})[row["arm"]] = row
    deltas = []
    per_seed = []
    for seed in sorted(by_seed):
        pair = by_seed[seed]
        if set(pair) != {"pixel2d", candidate}:
            continue
        delta = (
            pair[candidate]["fixed_validation"]["psnr_db_mean"]
            - pair["pixel2d"]["fixed_validation"]["psnr_db_mean"]
        )
        deltas.append(delta)
        per_seed.append({"seed": seed, "delta_db": delta, "gate_pass": delta >= GATE_DELTA_DB})
    return {
        "candidate": candidate,
        "per_seed": per_seed,
        "delta_db_mean": float(np.mean(deltas)),
        "delta_db_std": float(np.std(deltas)),
        "gate_pass_count": sum(row["gate_pass"] for row in per_seed),
        "gate_total": len(per_seed),
        "promotion_pass": bool(deltas) and all(delta >= GATE_DELTA_DB for delta in deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", default="out/r1-followup")
    parser.add_argument(
        "--historical-root",
        default=".",
        help="checkout containing the pre-R1 out/kitchen-torch artifact",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_root = root / args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(str(root / "examples" / "kitchen_torch.json"))
    cache = PathCache.load(base_cfg["cache"])

    rows = []
    models: dict[tuple[str, int], TorchNRP] = {}
    for seed in args.seeds:
        arms = ["pixel2d", "world3d", "world_triplane"]
        if seed == 0:
            arms += [
                "pixel2d_legacy_init",
                "world3d_legacy_init",
                "world3d_expanded",
            ]
        cfg_for_seed = copy.deepcopy(base_cfg)
        cfg_for_seed["seed"] = seed
        fixed = build_val_set(cache, cfg_for_seed)
        for arm in arms:
            model, train_report, report_path = train_or_reuse(
                base_cfg, arm, seed, out_root, root, args.reuse
            )
            models[(arm, seed)] = model
            metrics = evaluate_model(model, cache, fixed)
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "parameter_count": model.parameter_count,
                    "iters_per_second": train_report["iters_per_second"],
                    "report": report_path,
                    "fixed_validation": metrics,
                }
            )
            print(
                f"{arm} seed {seed}: {metrics['psnr_db_mean']:.2f} dB, "
                f"{model.parameter_count} params"
            )

    seed0_cfg = copy.deepcopy(base_cfg)
    seed0_cfg["seed"] = 0
    fixed0 = build_val_set(cache, seed0_cfg)
    legacy0 = legacy_validation_set(cache, seed0_cfg)
    historical_root = Path(args.historical_root).resolve()
    historical = TorchNRP.load(
        str(historical_root / "out" / "kitchen-torch" / "model.pt")
    )
    cross_generation = []
    for name, model in [
        ("historical_pixel2d", historical),
        ("current_pixel2d", models[("pixel2d", 0)]),
        ("current_world3d", models[("world3d", 0)]),
    ]:
        cross_generation.append(
            {
                "model": name,
                "legacy_validation": evaluate_model(model, cache, legacy0),
                "fixed_validation": evaluate_model(model, cache, fixed0),
            }
        )

    occupancy = {
        arm: occupancy_report(models[(arm, 0)], cache)
        for arm in ("pixel2d", "world3d", "world_triplane", "world3d_expanded")
    }
    regions = {
        arm: depth_region_metrics(models[(arm, 0)], cache, fixed0)
        for arm in ("pixel2d", "world3d", "world_triplane", "world3d_expanded")
    }
    stability = {
        candidate: aggregate_seed_deltas(rows, candidate)
        for candidate in ("world3d", "world_triplane")
    }

    by_arm_seed = {(row["arm"], row["seed"]): row for row in rows}
    init_effects = {}
    for spatial in ("pixel2d", "world3d"):
        default = by_arm_seed[(spatial, 0)]["fixed_validation"]["psnr_db_mean"]
        legacy = by_arm_seed[(f"{spatial}_legacy_init", 0)]["fixed_validation"][
            "psnr_db_mean"
        ]
        init_effects[spatial] = {
            "default_scale_init_psnr_db": default,
            "legacy_default_init_psnr_db": legacy,
            "delta_db_default_scale_minus_legacy": default - legacy,
        }

    report = {
        "experiment": "R1 failure analysis and world-anchored follow-up",
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_version": torch.__version__,
            "device": "cpu",
        },
        "scene": {
            "name": "Country Kitchen",
            "resolution": [cache.width, cache.height],
            "segments": cache.segment_count,
            "denoiser": base_cfg["denoise"]["method"],
        },
        "gate_threshold_db": GATE_DELTA_DB,
        "promotion_rule": (
            "every paired-seed candidate delta must be at least -0.5 dB; "
            "mean-only parity cannot promote an arm"
        ),
        "validation_provenance_audit": {
            "legacy_light_count": len(legacy0),
            "fixed_light_count": len(fixed0),
            "exact_light_match_count": exact_light_match_count(legacy0, fixed0),
            "legacy_training_rng_samples_before_validation": (
                base_cfg["pool"]["size"]
                + base_cfg["pool"]["replace_count"]
                * (base_cfg["iters"] // base_cfg["pool"]["replace_every"])
            ),
            "fixed_validation_rng": "numpy SeedSequence([training seed, 0x5EED])",
            "historical_output_scale_initialization": False,
            "current_output_scale_initialization": True,
        },
        "cross_generation_validation_provenance": cross_generation,
        "training_arms": rows,
        "seed_stability": stability,
        "output_initialization_effects": init_effects,
        "hash_occupancy": occupancy,
        "depth_region_metrics": regions,
        "conclusions": {
            "historical_25db_comparison_valid": False,
            "reason": (
                "the historical and R1 reports used different validation-light streams "
                "and different output initialization; controlled same-run deltas are binding"
            ),
            "original_world3d_promotion_pass": stability["world3d"]["promotion_pass"],
            "world_triplane_promotion_pass": stability["world_triplane"][
                "promotion_pass"
            ],
        },
    }
    report_path = out_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["conclusions"], indent=2))
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
