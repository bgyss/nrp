"""Train the compact neural proxy against reference GATHERLIGHT targets (NRP M4).

Config-driven CLI (see examples/toy_sphere.json). If the configured path
cache does not exist yet, it is traced first with the config's `trace` block, so the
smoke path is self-contained and CPU-only. All paths in the config are resolved
relative to the config file's directory.

Outputs into `out_dir`: model.npz (weights), train_report.json (loss curve, held-out
PSNR/SMAPE per light, parameter count, sizes, wall-clock timings).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from .dataset import build_inputs, light_targets, pixel_feature_block, sample_lights
from .metrics import psnr, smape
from .model import Adam, ProxyMLP, relative_mse_loss
from .path_cache import PathCache
from .toy_tracer import trace_path_cache


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    for key in ("cache", "out_dir"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.normpath(os.path.join(base, cfg[key]))
    return cfg


def ensure_cache(cfg: dict) -> PathCache:
    if os.path.exists(cfg["cache"]):
        return PathCache.load(cfg["cache"])
    if "trace" not in cfg:
        raise SystemExit(
            f"path cache {cfg['cache']} does not exist and the config has no 'trace' "
            "block; export it first (e.g. python -m nrp.mitsuba_exporter, see examples/)"
        )
    t = cfg["trace"]
    print(f"cache missing; tracing {t['width']}x{t['height']} @ {t['spp']} spp ...")
    cache = trace_path_cache(
        t["width"],
        t["height"],
        t["spp"],
        t["bounces"],
        t["seed"],
        medium=t.get("medium"),
        layer=t.get("layer"),
    )
    os.makedirs(os.path.dirname(cfg["cache"]), exist_ok=True)
    cache.save(cfg["cache"])
    return cache


def train(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg.get("seed", 0))
    cache = ensure_cache(cfg)
    px = pixel_feature_block(cache)
    n_px = cache.height * cache.width

    t_data0 = time.perf_counter()
    train_lights = sample_lights(cfg["light_bounds"], cfg["n_train_lights"], rng)
    val_lights = sample_lights(cfg["light_bounds"], cfg["n_val_lights"], rng)
    train_targets = light_targets(cache, train_lights)
    val_targets = light_targets(cache, val_lights)
    train_x = np.concatenate([build_inputs(px, c, r) for c, r in train_lights])
    train_y = train_targets.reshape(-1, 3)
    val_x = np.concatenate([build_inputs(px, c, r) for c, r in val_lights])
    val_y = val_targets.reshape(-1, 3)
    data_seconds = time.perf_counter() - t_data0

    model = ProxyMLP(hidden=tuple(cfg.get("hidden", [64, 64])), seed=cfg.get("seed", 0))
    opt = Adam(model, lr=cfg.get("lr", 1e-3))
    batch = cfg.get("batch_size", 4096)
    epochs = cfg["epochs"]

    def val_loss() -> float:
        pred = model.forward(val_x)
        loss, _ = relative_mse_loss(pred, val_y)
        return loss

    history = {"train_loss": [], "val_loss": [val_loss()]}
    t_train0 = time.perf_counter()
    n_rows = train_x.shape[0]
    for _epoch in range(epochs):
        order = rng.permutation(n_rows)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_rows, batch):
            idx = order[start : start + batch]
            pred = model.forward(train_x[idx])
            loss, dpred = relative_mse_loss(pred, train_y[idx])
            _, d_w, d_b = model.backward(dpred)
            opt.step(d_w, d_b)
            epoch_loss += loss
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss())
    train_seconds = time.perf_counter() - t_train0

    # Held-out per-light quality vs GATHERLIGHT (pre-emission-scaling contribution).
    val_metrics = []
    for i, (center, radius) in enumerate(val_lights):
        pred = model.forward(build_inputs(px, center, radius))
        ref = val_targets[i]
        val_metrics.append(
            {
                "center": np.asarray(center).tolist(),
                "radius": radius,
                "psnr_db": psnr(pred, ref),
                "smape": smape(pred, ref),
            }
        )

    # Single-frame inference latency at the cache's resolution (feature build + forward).
    n_bench = 20
    t_inf0 = time.perf_counter()
    for _ in range(n_bench):
        model.forward(build_inputs(px, val_lights[0][0], val_lights[0][1]))
    inference_ms = (time.perf_counter() - t_inf0) / n_bench * 1000.0

    os.makedirs(cfg["out_dir"], exist_ok=True)
    model_path = os.path.join(cfg["out_dir"], "model.npz")
    model.save(model_path)

    report = {
        "config": {k: v for k, v in cfg.items() if k != "out_dir"},
        "resolution": [cache.width, cache.height],
        "path_cache_segments": cache.segment_count,
        "path_cache_bytes": os.path.getsize(cfg["cache"]),
        "n_pixels": n_px,
        "parameter_count": model.parameter_count,
        "model_bytes": os.path.getsize(model_path),
        "dataset_seconds": data_seconds,
        "train_seconds": train_seconds,
        "inference_ms_per_frame": inference_ms,
        "inference_hz": 1000.0 / inference_ms if inference_ms > 0 else None,
        "history": history,
        "val_lights": val_metrics,
        "val_psnr_db_mean": float(np.mean([m["psnr_db"] for m in val_metrics])),
        "val_smape_mean": float(np.mean([m["smape"] for m in val_metrics])),
    }
    report_path = os.path.join(cfg["out_dir"], "train_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"trained {model.parameter_count} params in {train_seconds:.1f}s; "
        f"val loss {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f}; "
        f"held-out PSNR {report['val_psnr_db_mean']:.2f} dB, "
        f"SMAPE {report['val_smape_mean']:.4f}; "
        f"inference {inference_ms:.1f} ms/frame ({report['inference_hz']:.1f} Hz) "
        f"at {cache.width}x{cache.height}"
    )
    print(f"wrote {model_path} and {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
