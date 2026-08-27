"""Representation-track R1 redesign: three encoding arms against a held-out-camera gate.

Supersedes the experimental design of docs/plans/2026-07-27-r1-next-experiments.md.
The previous gate compared world-anchored encodings against a pixel2d control that is
fully dense below its finest level and one vertex per pixel at it -- a per-pixel lookup
table, optimal at single-view reconstruction and unable to render any other camera. This
runner instead measures what world anchoring is for: quality at a camera never trained on.

Writes out/r1-encoding-redesign/report.json and exits nonzero when the binding gate
fails, after writing all evidence. That nonzero exit is expected for a recorded negative.

Usage:
  uv run python examples/r1_encoding_redesign.py --out out/r1-encoding-redesign/report.json
  uv run python examples/r1_encoding_redesign.py --seeds 0 --arms world_sparse  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.r1_promotion import rotation_matrix_y, transform_cache  # noqa: E402
from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.conditioned_multiview import (  # noqa: E402
    camera_direction,
    train_conditioned,
)
from nrp.torch_backend.encoder_registry import (  # noqa: E402
    SPATIAL_ENCODERS,
    encoder_schedule_params,
)
from nrp.torch_backend.encoding_gates import (  # noqa: E402
    g1_generalization,
    g2_capacity_context,
    g3_stability,
    g4_frame_robustness,
    g5_fallback_decomposition,
    stop_reason,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.occupancy import (  # noqa: E402
    grid_occupancy,
    level_resolutions,
    normalize_positions,
)
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.torch_backend.train import light_param_vector, model_tensors  # noqa: E402
from nrp.torch_backend.train import train as train_single  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

#: The toy box is the unit cube; cameras sit inside it on a shallow arc around the
#: centre, all aimed at the same interior point so the sphere stays in frame.
ARC_CENTRE = np.array([0.5, 0.5, 0.55])
ARC_RADIUS = 0.42
ARC_SPAN_DEG = 70.0
ARC_HEIGHT = 0.5
ARM_NAMES = ("world_sparse", "world_normal_triplane", "world3d")
N_TRAIN_CAMERAS = 8
N_HELD_OUT_CAMERAS = 4
N_EVAL_LIGHTS = 8

#: Per-arm encoding overrides. Deliberately minimal: any key not listed here falls
#: through to the registered encoder class's own constructor default, resolved by
#: `encoder_schedule_params`/`build_encoder` -- re-declaring a schedule default here
#: that merely repeats the class default was the exact duplication defect fixed in
#: the previous task, so only genuine overrides (like world3d's occupancy opt-in)
#: belong in this dict.
ARM_ENCODING_CONFIG: dict[str, dict] = {
    "world_sparse": {},
    "world_normal_triplane": {},
    "world3d": {"allocation": "occupancy"},
}


def _camera_at(angle_deg: float, name: str) -> dict:
    theta = np.radians(angle_deg)
    origin = np.array(
        [
            ARC_CENTRE[0] + ARC_RADIUS * np.sin(theta),
            ARC_HEIGHT,
            ARC_CENTRE[2] - ARC_RADIUS * np.cos(theta),
        ]
    )
    return {"name": name, "origin": origin.tolist(), "target": ARC_CENTRE.tolist()}


def camera_arc(n_train: int, n_held_out: int) -> tuple[list[dict], list[dict]]:
    """Trained cameras evenly spaced on the arc; held-out cameras strictly between them.

    Held-out cameras interpolate rather than extrapolate: extrapolation beyond the arc
    is R3's question, and mixing the two would confound this gate.
    """
    if n_train < 2:
        raise ValueError("need at least two trained cameras")
    if n_held_out < 1 or n_held_out > n_train - 1:
        raise ValueError("held-out cameras must fit strictly between trained cameras")
    angles = np.linspace(-ARC_SPAN_DEG / 2.0, ARC_SPAN_DEG / 2.0, n_train)
    trained = [_camera_at(float(a), f"train{i}") for i, a in enumerate(angles)]
    gaps = np.linspace(0, n_train - 2, n_held_out).round().astype(int)
    held_out = [
        _camera_at(float((angles[g] + angles[g + 1]) / 2.0), f"held{i}") for i, g in enumerate(gaps)
    ]
    return trained, held_out


def nearest_trained_camera(held_out: dict, trained: list[dict]) -> dict:
    """G1's baseline: the trained view whose pixel2d proxy gets reused at this camera."""
    origin = np.asarray(held_out["origin"], dtype=np.float64)
    return min(trained, key=lambda c: float(np.linalg.norm(np.asarray(c["origin"]) - origin)))


def rotated_camera(camera: dict, rotation: np.ndarray) -> dict:
    """Rotate a camera's origin/target the same way `transform_cache` rotates a cache.

    G4 tests robustness to an arbitrary choice of world coordinate frame, not to a
    different physical viewpoint: the rendered pixels are unchanged (same cache
    segments), only the coordinate frame the world-anchored encodings read positions
    in changes. The camera-conditioned model's view-direction input is computed from
    this dict, so it must be expressed in that same rotated frame or it would silently
    describe the pre-rotation camera while the spatial encoding reads rotated
    positions -- an inconsistency the gate is supposed to be measuring the absence of.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    origin = np.asarray(camera["origin"], dtype=np.float64) @ rotation.T
    target = np.asarray(camera["target"], dtype=np.float64) @ rotation.T
    return {"name": camera["name"], "origin": origin.tolist(), "target": target.tolist()}


def export_arc(cameras: list[dict], seed_dir: Path, args) -> dict[str, Path]:
    """Trace one cache per camera. Held-out caches are exported for evaluation only."""
    paths = {}
    seed_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        out_path = seed_dir / f"{camera['name']}.npz"
        if not (args.skip_export and out_path.exists()):
            cache = trace_path_cache(
                width=args.width,
                height=args.width,
                spp=args.spp,
                max_bounces=args.bounces,
                seed=args.trace_seed,
                camera_pos=np.asarray(camera["origin"], dtype=np.float64),
                camera_target=np.asarray(camera["target"], dtype=np.float64),
            )
            cache.validate()
            cache.save(str(out_path))
        paths[camera["name"]] = out_path
    return paths


def rotated_caches(paths: dict[str, Path], rotation_deg: float) -> dict[str, PathCache]:
    """Load every camera's cache and, if non-zero, rotate it into a new world frame."""
    rotation = rotation_matrix_y(rotation_deg)
    out = {}
    for name, path in paths.items():
        cache = PathCache.load(str(path))
        out[name] = cache if rotation_deg == 0.0 else transform_cache(cache, rotation)
    return out


def frozen_lights(cache: PathCache, seed: int, n: int = N_EVAL_LIGHTS) -> list[SphereLight]:
    """One fixed, seed-derived light set shared by every arm and camera at this seed."""
    rng = np.random.default_rng([int(seed), 0xE1C0DE])
    positions = cache.position.reshape(-1, 3)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    lights = []
    for _ in range(n):
        center = lo + rng.random(3) * (hi - lo)
        radius = 0.1 + rng.random() * 0.3
        lights.append(SphereLight(center=center, radius=float(radius)))
    return lights


def baseline_config(cache_path: Path, out_dir: Path, seed: int, args) -> dict:
    """Config for one per-camera pixel2d baseline (`train_single`'s schema)."""
    return {
        "cache": str(cache_path),
        "out_dir": str(out_dir),
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.4},
        "sampling": "segments",
        "pool": {"size": 32, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "method": args.denoise_method},
        "iters": args.iters,
        "batch_pixels": 4096,
        "lr": 0.005,
        "n_val_lights": 4,
        "seed": seed,
        "device": args.device,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {"levels": 8, "features_per_level": 2, "table_size_log2": 14},
        },
    }


def arm_config(arm: str, seed: int, manifest: Path, out_dir: Path, args) -> dict:
    """Config for one camera-conditioned arm. Matches train_conditioned's schema."""
    return {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.4},
        "sampling": "segments",
        "pool": {"size": 32, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "method": args.denoise_method},
        "iters": args.iters,
        "batch_pixels": 4096,
        "lr": 0.005,
        "n_val_lights": 12,
        "seed": seed,
        "device": args.device,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "camera_conditioned": True,
            "spatial_encoding": arm,
            "encoding": ARM_ENCODING_CONFIG[arm],
        },
    }


def load_conditioned_model(model_path: Path, occupancy_caches: list[PathCache]) -> TorchNRP:
    """Reload a camera-conditioned model, rebuilding occupancy if its arm needs one.

    `TorchNRP.save`/`load` deliberately excludes occupancy from the persisted config
    (it isn't JSON/tensor serializable), so an occupancy-allocated arm cannot be
    reloaded from its checkpoint alone. `train_conditioned` built that occupancy from
    the union of the manifest's training-view first-hit positions and the model's own
    stored `world_bounds`; reproducing that exact recipe here (not a fresh guess at
    bounds or a different resolution schedule) is what makes the reloaded encoder's
    table layout identical to the one the checkpoint's weights were trained against.
    """
    blob = torch.load(str(model_path), map_location="cpu", weights_only=True)
    config = dict(blob["config"])
    encoding_name = config["spatial_encoding"]
    encoding_cfg = config.get("encoding") or {}
    encoder_cls = SPATIAL_ENCODERS[encoding_name]
    needs_occupancy = (
        getattr(encoder_cls, "needs_occupancy", False)
        or encoding_cfg.get("allocation") == "occupancy"
    )
    occupancy = None
    if needs_occupancy:
        stacked = np.concatenate(
            [cache.position.reshape(-1, 3) for cache in occupancy_caches], axis=0
        )
        levels, base_resolution, finest_resolution = encoder_schedule_params(
            encoding_name, encoding_cfg
        )
        occupancy = grid_occupancy(
            normalize_positions(stacked, config["world_bounds"]),
            level_resolutions(levels, base_resolution, finest_resolution),
        )
    model = TorchNRP(**config, occupancy=occupancy)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def _predict(model: TorchNRP, cache: PathCache, lights: list, view_dir=None) -> np.ndarray:
    """`relight`'s loop, but able to pass a camera-conditioned model's `view_dir`."""
    device = next(model.parameters()).device
    n_px = cache.height * cache.width
    spatial, aux = model_tensors(cache, model, device)
    image = torch.zeros((n_px, 3), device=device)
    with torch.no_grad():
        for light in lights:
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            rgb = torch.as_tensor(light.rgb, dtype=torch.float32, device=device)
            kwargs = {"view_dir": view_dir} if view_dir is not None else {}
            image += model(spatial, aux, params, **kwargs) * rgb
    return image.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)


def evaluate_camera(
    model: TorchNRP,
    arm: str,
    camera: dict,
    conditioned_camera: dict,
    cache: PathCache,
    baseline_model: TorchNRP,
    lights: list,
    peak: float,
) -> dict:
    """One gate row: conditioned proxy vs the nearest trained view's pixel2d proxy.

    `camera` is the physical (unrotated-name-carrying) camera record used for row
    identity; `conditioned_camera` is the same camera expressed in whatever world
    frame `cache` itself is in (see `rotated_camera`) and is what the model's
    view-direction input is computed from. `peak` is the campaign-fixed PSNR peak
    (see `campaign_peak`) applied to every PSNR computed here, so the absolute
    numbers mean the same thing at every held-out camera.
    """
    reference = gather_lights(cache, lights)
    baseline = relight(baseline_model, cache, lights)
    view_dir = None
    if model.camera_conditioned:
        direction = camera_direction(conditioned_camera)
        n_px = cache.height * cache.width
        device = next(model.parameters()).device
        view_dir = (
            torch.as_tensor(direction, dtype=torch.float32, device=device)
            .reshape(1, 3)
            .expand(n_px, -1)
        )
    predicted = _predict(model, cache, lights, view_dir=view_dir)
    row = {
        "arm": arm,
        "camera": camera["name"],
        "psnr_db": psnr(predicted, reference, peak=peak),
        "baseline_psnr_db": psnr(baseline, reference, peak=peak),
        "out_of_occupancy_fraction": 0.0,
        "in_occupancy_psnr_db": None,
        "out_occupancy_psnr_db": None,
    }
    row["delta_db"] = row["psnr_db"] - row["baseline_psnr_db"]
    encoder = model.encoding
    if hasattr(encoder, "out_of_occupancy_fraction"):
        positions = torch.as_tensor(cache.position.reshape(-1, 3), dtype=torch.float32)
        normalized = ((positions - model.world_min) / model.world_extent).clamp(0.0, 1.0)
        row["out_of_occupancy_fraction"] = encoder.out_of_occupancy_fraction(normalized)
        level = encoder.levels - 1
        res = encoder.resolutions[level]
        pos0 = torch.floor(normalized * res).long().clamp_(0, res - 1)
        _, hit = encoder._lookup(pos0, level)
        mask = hit.numpy().reshape(cache.height, cache.width)
        if mask.any():
            row["in_occupancy_psnr_db"] = psnr(predicted[mask], reference[mask], peak=peak)
        if (~mask).any():
            row["out_occupancy_psnr_db"] = psnr(predicted[~mask], reference[~mask], peak=peak)
    return row


def campaign_peak(trained_caches: list[PathCache], lights: list) -> float:
    """The single fixed PSNR peak for one seed's whole campaign.

    `psnr` defaults its peak to *the reference image's own max* (documented HDR
    convention in `nrp/metrics.py`) -- fine for the comparative delta in G1, which is
    peak-independent (prediction and baseline share the same reference at the same
    camera, so the peak cancels exactly), but NOT fine for G1's absolute floor: a
    held-out camera whose reference happens to contain one very bright pixel gets an
    inflated per-image peak and clears the floor more easily than a dimmer camera, so
    "15 dB absolute" would not mean the same thing at every camera. Fixing one peak
    per seed -- the max GATHERLIGHT radiance over the *trained* cameras only, so the
    held-out references never influence the scale the held-out cameras are judged on
    -- makes the absolute floor comparable across cameras while leaving the delta
    numerically unchanged (it never depended on the peak to begin with).
    """
    return max(float(gather_lights(cache, lights).max()) for cache in trained_caches)


def _sparse_collision_fraction(capacity_report: dict) -> float:
    """Worst per-level collision fraction reported for a sparse-table encoder."""
    levels = capacity_report.get("levels", [])
    fractions = [lvl["collision_fraction"] for lvl in levels if "collision_fraction" in lvl]
    return max(fractions) if fractions else 0.0


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/r1-encoding-redesign/report.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=list(ARM_NAMES))
    parser.add_argument("--rotations", nargs="+", type=float, default=[0.0, 90.0, 180.0])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--trace-seed", type=int, default=0)
    parser.add_argument("--denoise-method", choices=["oidn", "bilateral"], default="bilateral")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold-db", type=float, default=1.0)
    parser.add_argument(
        "--absolute-floor-db",
        type=float,
        default=15.0,
        help="G1's absolute PSNR floor: a comparative win over the baseline is not "
        "enough on its own (review finding; see nrp/torch_backend/encoding_gates.py).",
    )
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trained, held_out = camera_arc(N_TRAIN_CAMERAS, N_HELD_OUT_CAMERAS)
    rows_by_arm: dict[str, list[dict]] = {arm: [] for arm in args.arms}
    capacity_rows: list[dict] = []
    latest_capacity_report: dict[str, dict] = {}
    collision_by_arm: dict[str, float] = {}

    peak_by_seed: dict[int, float] = {}
    for seed in args.seeds:
        seed_dir = out_path.parent / f"seed{seed}"
        cache_paths = export_arc(trained + held_out, seed_dir, args)
        eval_lights = frozen_lights(PathCache.load(str(cache_paths[trained[0]["name"]])), seed)
        seed_peak = campaign_peak(
            [PathCache.load(str(cache_paths[camera["name"]])) for camera in trained], eval_lights
        )
        peak_by_seed[seed] = seed_peak

        for rotation in args.rotations:
            rot_dir = seed_dir if rotation == 0.0 else seed_dir / f"rot{rotation:g}"
            rot_dir.mkdir(parents=True, exist_ok=True)
            rotation_matrix = rotation_matrix_y(rotation)
            caches = rotated_caches(cache_paths, rotation)
            cameras_in_frame = {
                camera["name"]: rotated_camera(camera, rotation_matrix)
                for camera in trained + held_out
            }

            trained_cache_paths: dict[str, Path] = {}
            manifest_entries = []
            for camera in trained:
                cache_path = rot_dir / f"{camera['name']}.npz"
                if not (args.skip_export and cache_path.exists()):
                    caches[camera["name"]].save(str(cache_path))
                trained_cache_paths[camera["name"]] = cache_path
                manifest_entries.append(
                    {
                        "name": camera["name"],
                        "cache": str(cache_path.resolve()),
                        "camera": cameras_in_frame[camera["name"]],
                    }
                )
            manifest_path = rot_dir / "views.json"
            manifest_path.write_text(json.dumps({"views": manifest_entries}, indent=2))

            baseline_models: dict[str, TorchNRP] = {}
            for camera in trained:
                baseline_dir = rot_dir / "pixel2d" / camera["name"]
                model_path = baseline_dir / "model.pt"
                if not (args.skip_export and model_path.exists()):
                    train_single(
                        baseline_config(
                            trained_cache_paths[camera["name"]], baseline_dir, seed, args
                        )
                    )
                baseline_models[camera["name"]] = TorchNRP.load(str(model_path)).eval()

            for arm in args.arms:
                arm_dir = rot_dir / arm
                model_path = arm_dir / "model.pt"
                if not (args.skip_export and model_path.exists()):
                    train_conditioned(arm_config(arm, seed, manifest_path, arm_dir, args))
                model = load_conditioned_model(
                    model_path, [caches[camera["name"]] for camera in trained]
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
                    if arm == "world_sparse":
                        collision_by_arm[arm] = max(
                            collision_by_arm.get(arm, 0.0),
                            _sparse_collision_fraction(report),
                        )

                for camera in held_out:
                    baseline_camera = nearest_trained_camera(camera, trained)
                    row = evaluate_camera(
                        model,
                        arm,
                        camera,
                        cameras_in_frame[camera["name"]],
                        caches[camera["name"]],
                        baseline_models[baseline_camera["name"]],
                        eval_lights,
                        seed_peak,
                    )
                    row["seed"] = seed
                    row["rotation_degrees"] = float(rotation)
                    row["baseline_camera"] = baseline_camera["name"]
                    rows_by_arm[arm].append(row)

    expected_seeds = set(args.seeds)
    expected_cameras = {camera["name"] for camera in held_out}
    arms_report = {}
    for arm in args.arms:
        rows = rows_by_arm[arm]
        g1 = g1_generalization(
            rows,
            args.threshold_db,
            expected_seeds=expected_seeds,
            expected_cameras=expected_cameras,
            absolute_floor_db=args.absolute_floor_db,
        )
        g3 = g3_stability(rows, collision_by_arm, args.threshold_db)
        g4 = g4_frame_robustness(rows, args.threshold_db)
        g5 = g5_fallback_decomposition(rows)
        arms_report[arm] = {
            "rows": rows,
            "rows_count": len(rows),
            "capacity_report": latest_capacity_report.get(arm),
            "g1": g1,
            "g3": g3,
            "g4": g4,
            "g5": g5,
        }

    report = {
        "scene": "toy_box",
        "width": args.width,
        "height": args.width,
        "spp": args.spp,
        "iters": args.iters,
        "seeds": list(args.seeds),
        "cameras": {"trained": trained, "held_out": held_out},
        "hardware": {"platform": sys.platform, "device": args.device},
        "absolute_floor_db": args.absolute_floor_db,
        "peak_by_seed": {str(seed): peak for seed, peak in peak_by_seed.items()},
        "arms": arms_report,
        "g2_capacity_context": g2_capacity_context(capacity_rows),
        "promoted": False,
    }
    report["stop_reason"] = stop_reason(report)
    report["promoted"] = report["stop_reason"] is None

    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    return 0 if report["promoted"] else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
