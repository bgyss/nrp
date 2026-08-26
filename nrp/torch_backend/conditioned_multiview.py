"""Shared data contracts for the R2 camera-conditioned NRP.

The representation-track R2 experiment trains one world-anchored proxy over several
fixed-camera path caches.  This module owns the camera-aware manifest and geometry
helpers; training, pool construction, and shared inference are added below this
contract so all R2 paths agree on paths, camera directions, and scene bounds.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..gather_light import gather_light
from ..path_cache import PathCache
from .denoise import denoise_image
from .device import autocast, resolve_device, resolve_precision
from .gather import TorchPathCache
from .model import TorchNRP, relative_mse_loss
from .sampling import sample_light
from .train import evaluate, light_param_dim_from_cfg, light_param_vector, spatial_tensors


def _finite_vector(value, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,):
        raise ValueError(f"camera {name} must contain exactly three values")
    if not np.isfinite(vector).all():
        raise ValueError(f"camera {name} must be finite")
    return vector


def camera_direction(camera: dict) -> np.ndarray:
    """Return the normalized camera forward direction ``target - origin``."""
    if not isinstance(camera, dict):
        raise ValueError("camera metadata must be an object")
    if "origin" not in camera or "target" not in camera:
        raise ValueError("camera metadata requires origin and target")
    origin = _finite_vector(camera["origin"], "origin")
    target = _finite_vector(camera["target"], "target")
    direction = target - origin
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        raise ValueError("camera origin and target must define a non-zero direction")
    return direction / length


@dataclass(frozen=True)
class CameraView:
    """One resolved path cache and its physical camera metadata."""

    name: str
    cache_path: Path
    camera: dict

    @property
    def view_dir(self) -> np.ndarray:
        return camera_direction(self.camera)


def load_camera_manifest(path: str | Path) -> list[CameraView]:
    """Load and validate an R2 manifest with relative cache paths."""
    manifest_path = Path(path).resolve()
    with manifest_path.open() as handle:
        payload = json.load(handle)
    entries = payload.get("views") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or not entries:
        raise ValueError("camera manifest must contain a non-empty views list")

    views: list[CameraView] = []
    resolution: tuple[int, int] | None = None
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"view {index} must be an object")
        raw_cache = entry.get("cache")
        if not isinstance(raw_cache, str) or not raw_cache:
            raise ValueError(f"view {index} requires a cache path")
        camera = entry.get("camera")
        if not isinstance(camera, dict):
            raise ValueError(f"view {index} requires camera metadata")
        # Validate before loading so malformed camera records fail consistently even
        # when an optional cache is not available.
        camera_direction(camera)
        cache_path = Path(raw_cache)
        if not cache_path.is_absolute():
            cache_path = manifest_path.parent / cache_path
        cache = PathCache.load(str(cache_path))
        current_resolution = (cache.height, cache.width)
        if resolution is None:
            resolution = current_resolution
        elif current_resolution != resolution:
            raise ValueError(
                "all camera-manifest caches must have the same resolution "
                f"({resolution[1]}x{resolution[0]}), got "
                f"{current_resolution[1]}x{current_resolution[0]} for {cache_path}"
            )
        views.append(
            CameraView(
                name=str(entry.get("name", f"view{index}")),
                cache_path=cache_path.resolve(),
                camera=camera,
            )
        )
    return views


def global_world_bounds(caches: Sequence[PathCache]) -> dict:
    """Return one finite, non-degenerate world-position bound for all views."""
    if not caches:
        raise ValueError("at least one cache is required for global world bounds")
    positions = []
    for cache in caches:
        position = np.asarray(cache.position, dtype=np.float64).reshape(-1, 3)
        if not np.isfinite(position).all():
            raise ValueError("cache.position contains non-finite values")
        positions.append(position)
    all_positions = np.concatenate(positions, axis=0)
    lower = all_positions.min(axis=0)
    upper = all_positions.max(axis=0)
    if np.any(upper <= lower):
        raise ValueError("global cache positions must span a non-zero range on every axis")
    return {"min": lower.tolist(), "max": upper.tolist()}


def camera_tensor(view: CameraView, n_pixels: int, device) -> torch.Tensor:
    """Broadcast one manifest camera direction to a device-resident pixel batch."""
    if n_pixels <= 0:
        raise ValueError("n_pixels must be positive")
    direction = torch.as_tensor(view.view_dir, dtype=torch.float32, device=device)
    return direction.reshape(1, 3).expand(n_pixels, -1)


def _render_target(
    cache: PathCache, light, cfg: dict, torch_cache: TorchPathCache | None
) -> np.ndarray:
    """Render one unit-emission training target using the configured gather backend."""
    if torch_cache is None:
        image = gather_light(cache, light)
    else:
        image = torch_cache.gather_light(light).detach().cpu().numpy().astype(np.float64)
    denoise = cfg.get("denoise", {})
    if denoise.get("enabled", True):
        image = denoise_image(
            image,
            cache.albedo,
            cache.normal,
            cache.depth,
            method=denoise.get("method", "bilateral"),
            **{
                key: value
                for key, value in denoise.items()
                if key not in {"enabled", "method"}
            },
        )
    return np.asarray(image, dtype=np.float64).reshape(-1, 3)


class MultiViewImagePool:
    """Shared light-shape pool with one target image per camera view."""

    def __init__(self, caches: Sequence[PathCache], cfg: dict, rng, device, fill: bool = True):
        if not caches:
            raise ValueError("at least one cache is required for a multi-view pool")
        resolution = (caches[0].height, caches[0].width)
        if any((cache.height, cache.width) != resolution for cache in caches[1:]):
            raise ValueError("all multi-view pool caches must have the same resolution")
        self.caches = list(caches)
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.size = int(cfg["pool"]["size"])
        if self.size <= 0:
            raise ValueError("pool.size must be positive")
        self._next_replace = 0
        n_pixels = caches[0].height * caches[0].width
        light_dim = light_param_dim_from_cfg(cfg)
        self.params = torch.empty((self.size, light_dim), dtype=torch.float32, device=device)
        self.targets = torch.empty(
            (len(caches), self.size, n_pixels, 3), dtype=torch.float32, device=device
        )
        self.used_params: list[np.ndarray] = []
        self.supervision_seconds = 0.0
        self._torch_caches = (
            [TorchPathCache(cache, device) for cache in self.caches]
            if cfg.get("gather_backend", "numpy") == "torch"
            else None
        )
        if cfg.get("gather_backend", "numpy") not in {"numpy", "torch"}:
            raise ValueError("gather_backend must be 'numpy' or 'torch'")
        if fill:
            for slot in range(self.size):
                self.fill(slot)

    def fill(self, slot: int) -> None:
        if not 0 <= slot < self.size:
            raise IndexError(f"pool slot {slot} is outside [0, {self.size})")
        started = time.perf_counter()
        light = sample_light(
            self.caches[0],
            self.rng,
            self.cfg["light_type"],
            self.cfg["light_bounds"],
            self.cfg.get("sampling", "segments"),
        )
        vector = np.asarray(light_param_vector(light), dtype=np.float64)
        self.params[slot] = torch.as_tensor(vector, dtype=torch.float32, device=self.device)
        for view_index, cache in enumerate(self.caches):
            torch_cache = self._torch_caches[view_index] if self._torch_caches is not None else None
            self.targets[view_index, slot] = torch.as_tensor(
                _render_target(cache, light, self.cfg, torch_cache),
                dtype=torch.float32,
                device=self.device,
            )
        self.used_params.append(vector.copy())
        self.supervision_seconds += time.perf_counter() - started

    def replace_round(self) -> None:
        replace_count = int(self.cfg["pool"]["replace_count"])
        if replace_count < 0:
            raise ValueError("pool.replace_count must be non-negative")
        for _ in range(replace_count):
            self.fill(self._next_replace)
            self._next_replace = (self._next_replace + 1) % self.size

    @property
    def supervision_images(self) -> int:
        return len(self.used_params) * len(self.caches)


def build_validation_sets(
    caches: Sequence[PathCache], cfg: dict, seed: int | None = None
) -> list[list[dict]]:
    """Build independent, per-view held-out light sets and physical references."""
    if not caches:
        raise ValueError("at least one cache is required for validation")
    base_seed = int(cfg.get("seed", 0) if seed is None else seed)
    count = int(cfg.get("n_val_lights", 12))
    if count < 0:
        raise ValueError("n_val_lights must be non-negative")
    result: list[list[dict]] = []
    for view_index, cache in enumerate(caches):
        rng = np.random.default_rng([base_seed, 0x5EED, view_index])
        entries = []
        for _ in range(count):
            light = sample_light(
                cache,
                rng,
                cfg["light_type"],
                cfg["light_bounds"],
                cfg.get("sampling", "segments"),
            )
            raw = gather_light(cache, light).reshape(-1, 3)
            denoised = _render_target(cache, light, cfg, None)
            entries.append(
                {
                    "light": light,
                    "params": np.asarray(light_param_vector(light), dtype=np.float64),
                    "raw": raw,
                    "denoised": denoised,
                }
            )
        result.append(entries)
    return result


def validation_disjointness(
    training_params: Sequence[np.ndarray], validation_sets: Sequence[Sequence[dict]]
) -> list[bool]:
    """Return one exact parameter-disjointness result for each view's validation set."""
    training_keys = {
        tuple(np.asarray(vector, dtype=np.float64).reshape(-1).tolist())
        for vector in training_params
    }
    result = []
    for entries in validation_sets:
        result.append(
            all(
                tuple(np.asarray(entry["params"], dtype=np.float64).reshape(-1).tolist())
                not in training_keys
                for entry in entries
            )
        )
    return result


def train_conditioned(cfg: dict, resume: bool = False) -> dict:
    """Train one world-anchored, camera-conditioned proxy over all manifest views."""
    if resume:
        raise ValueError("camera-conditioned R2 training does not support resume yet")
    model_cfg = cfg.get("model")
    if not isinstance(model_cfg, dict) or not model_cfg.get("camera_conditioned", False):
        raise ValueError("R2 training requires model.camera_conditioned=true")
    if model_cfg.get("spatial_encoding", "pixel2d") != "world3d":
        raise ValueError("R2 training requires model.spatial_encoding='world3d'")

    views = load_camera_manifest(cfg["manifest"])
    caches = [PathCache.load(str(view.cache_path)) for view in views]
    device = resolve_device(cfg.get("device"))
    precision = resolve_precision(cfg.get("precision"))
    seed = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_pixels = caches[0].height * caches[0].width
    bounds = global_world_bounds(caches)

    model = TorchNRP(
        light_type=cfg["light_type"],
        light_param_dim=light_param_dim_from_cfg(cfg),
        hidden_width=model_cfg.get("hidden_width", 128),
        hidden_layers=model_cfg.get("hidden_layers", 4),
        encoding=model_cfg.get("encoding"),
        spatial_encoding="world3d",
        world_bounds=bounds,
        camera_conditioned=True,
        use_encoding=model_cfg.get("use_encoding", True),
        use_aux=model_cfg.get("use_aux", True),
        texture_kernel=model_cfg.get("texture_conditioning") == "kernel",
    ).to(device)
    spatial_rows, aux_rows = zip(
        *(spatial_tensors(cache, device, "world3d") for cache in caches), strict=True
    )
    spatial = torch.stack(spatial_rows)
    aux = torch.stack(aux_rows)
    view_dirs = torch.stack(
        [camera_tensor(view, n_pixels, device) for view in views], dim=0
    )

    pool_started = time.perf_counter()
    pool = MultiViewImagePool(caches, cfg, rng, device)
    pool_seconds = time.perf_counter() - pool_started
    validation_sets = build_validation_sets(caches, cfg, seed=seed)
    disjoint_by_view = validation_disjointness(pool.used_params, validation_sets)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-2))
    iters = int(cfg["iters"])
    batch = int(cfg.get("batch_pixels", 4096))
    if iters <= 0 or batch <= 0:
        raise ValueError("iters and batch_pixels must be positive")
    if model_cfg.get("init_output_scale", True):
        model.init_output_scale(float(pool.targets.mean(dim=-1).median().item()))

    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" and device.type == "cuda" else None
    loss_curve: list[float] = []
    train_started = time.perf_counter()
    replace_every = int(cfg["pool"]["replace_every"])
    if replace_every <= 0:
        raise ValueError("pool.replace_every must be positive")
    model.train()
    for iteration in range(iters):
        view_ids = torch.randint(0, len(caches), (batch,), generator=generator).to(device)
        pool_ids = torch.randint(0, pool.size, (batch,), generator=generator).to(device)
        pixel_ids = torch.randint(0, n_pixels, (batch,), generator=generator).to(device)
        with autocast(device, precision):
            pred = model(
                spatial[view_ids, pixel_ids],
                aux[view_ids, pixel_ids],
                pool.params[pool_ids],
                view_dir=view_dirs[view_ids, pixel_ids],
            )
            target = pool.targets[view_ids, pool_ids, pixel_ids]
            loss = relative_mse_loss(pred, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        loss_curve.append(float(loss.detach().item()))
        if (iteration + 1) % replace_every == 0:
            pool.replace_round()
    train_seconds = time.perf_counter() - train_started

    model.eval()
    view_reports = []
    for index, (view, cache, entries) in enumerate(
        zip(views, caches, validation_sets, strict=True)
    ):
        metrics = evaluate(
            model,
            entries,
            spatial[index],
            aux[index],
            device,
            hw=(cache.height, cache.width),
            view_dir=view_dirs[index],
        )
        view_reports.append(
            {
                "name": view.name,
                "camera": view.camera,
                "validation_light_params": [entry["params"].tolist() for entry in entries],
                "val_lights": metrics,
                "val_psnr_db_vs_raw_mean": float(
                    np.mean([metric["psnr_db_vs_raw"] for metric in metrics])
                ),
                "val_smape_vs_raw_mean": float(
                    np.mean([metric["smape_vs_raw"] for metric in metrics])
                ),
            }
        )

    model_dir = Path(cfg["out_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"
    model.save(str(model_path))
    with torch.no_grad():
        fixed = pool.params[0].expand(n_pixels, -1)
        started = time.perf_counter()
        for _ in range(5):
            model(spatial[0], aux[0], fixed, view_dir=view_dirs[0])
        inference_ms = (time.perf_counter() - started) / 5.0 * 1000.0
    report = {
        "config": {key: value for key, value in cfg.items() if key != "out_dir"},
        "view_count": len(views),
        "views": view_reports,
        "resolution": [caches[0].width, caches[0].height],
        "parameter_count": model.parameter_count,
        "model_bytes": os.path.getsize(model_path),
        "model_path": "model.pt",
        "path_cache_segments": [cache.segment_count for cache in caches],
        "global_world_bounds": bounds,
        "shared_training_light_params": [vector.tolist() for vector in pool.used_params],
        "validation_light_params": [
            [entry["params"].tolist() for entry in entries] for entries in validation_sets
        ],
        "validation_disjoint_by_view": disjoint_by_view,
        "pool_build_seconds": pool_seconds,
        "supervision_images": pool.supervision_images,
        "supervision_seconds": pool.supervision_seconds,
        "train_seconds": train_seconds,
        "iters_per_second": iters / train_seconds if train_seconds > 0 else None,
        "inference_ms_per_frame": inference_ms,
        "inference_hz": 1000.0 / inference_ms if inference_ms > 0 else None,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "loss_curve": loss_curve,
        "validation_disjoint": all(disjoint_by_view),
        "val_psnr_db_vs_raw_mean": float(
            np.mean([row["val_psnr_db_vs_raw_mean"] for row in view_reports])
        ),
        "val_smape_vs_raw_mean": float(
            np.mean([row["val_smape_vs_raw_mean"] for row in view_reports])
        ),
    }
    report_path = model_dir / "conditioned_train_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(
        f"trained one conditioned model for {len(views)} views ({model.parameter_count} params) "
        f"in {train_seconds:.1f}s; held-out PSNR {report['val_psnr_db_vs_raw_mean']:.2f} dB"
    )
    print(f"wrote {model_path} and {report_path}")
    return report
