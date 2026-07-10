"""T2: packed-shard streaming proof on the 512x512 Country Kitchen cache."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.streamed_torchnrp_train import (  # noqa: E402
    train_monolithic,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.mitsuba_exporter import _hardware_context  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight_tiled  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.streamed_train import (  # noqa: E402
    gather_sphere_streamed,
    load_sharded_gbuffer,
    train_streamed,
)


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if sys.platform.startswith("linux") else value


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def config(iters: int, width: int) -> dict:
    return {
        "seed": 0,
        "device": "cpu",
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.6},
        "sampling": "bbox",
        "denoise": {"enabled": False},
        "pool": {"size": 8, "replace_count": 1, "replace_every": 25},
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 10,
                "features_per_level": 2,
                "table_size_log2": 16,
                "base_resolution": 4,
                "finest_resolution": width,
            },
        },
        "lr": 5e-3,
        "batch_pixels": 8192,
        "iters": iters,
    }


def save_model(model: TorchNRP, cfg: dict, path: Path) -> None:
    torch.save({"model": model.state_dict(), "config": cfg}, path)


def load_model(path: Path) -> TorchNRP:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    model = TorchNRP(
        hidden_width=cfg["model"]["hidden_width"],
        hidden_layers=cfg["model"]["hidden_layers"],
        encoding=cfg["model"]["encoding"],
        light_type="sphere",
    )
    model.load_state_dict(payload["model"])
    return model


def worker_shard(cache_path: str, shard_dir: str, tile_size: int, queue) -> None:
    t0 = time.perf_counter()
    cache = PathCache.load(cache_path)
    Path(shard_dir).mkdir(parents=True, exist_ok=True)
    cache.save_sharded(shard_dir, tile_size=tile_size, packed=True)
    queue.put(
        {
            "seconds": time.perf_counter() - t0,
            "peak_rss_bytes": rss_bytes(),
            "segments": cache.segment_count,
            "packed_shard_bytes": directory_bytes(Path(shard_dir)),
        }
    )


def worker_train(kind: str, cache_path: str, shard_dir: str, cfg: dict, model_path: str, queue):
    t0 = time.perf_counter()
    if kind == "monolithic":
        cache = PathCache.load(cache_path)
        model, stats = train_monolithic(cache, cfg)
    else:
        cache = load_sharded_gbuffer(Path(shard_dir))
        model, stats = train_streamed(Path(shard_dir), cache, cfg)
    save_model(model, cfg, Path(model_path))
    queue.put(
        {
            "seconds": time.perf_counter() - t0,
            "peak_rss_bytes": rss_bytes(),
            "pool_seconds": stats["pool_seconds"],
            "train_seconds": stats["train_seconds"],
            "loss_first": stats["loss_curve"][0],
            "loss_last": stats["loss_curve"][-1],
            "peak_segment_bytes_loaded": stats.get("peak_segment_bytes_loaded"),
            "segments_per_second": stats.get("segments_per_second"),
        }
    )


def worker_quality(cache_path: str, shard_dir: str, cfg: dict, queue) -> None:
    cache = PathCache.load(cache_path)
    gbuffer = load_sharded_gbuffer(Path(shard_dir))
    rng = np.random.default_rng([cfg["seed"], 0xCA5E])
    scores = []
    t0 = time.perf_counter()
    for _ in range(6):
        light = sample_light(gbuffer, rng, "sphere", cfg["light_bounds"], "bbox")
        packed, _ = gather_sphere_streamed(
            Path(shard_dir), gbuffer.n_paths, light.center, light.radius
        )
        scores.append(psnr(packed, gather_light(cache, light)))
    queue.put(
        {
            "n_lights": len(scores),
            "gather_psnr_db_min": float(np.min(scores)),
            "gather_psnr_db_mean": float(np.mean(scores)),
            "seconds": time.perf_counter() - t0,
            "peak_rss_bytes": rss_bytes(),
        }
    )


def worker_eval(shard_dir: str, mono_path: str, stream_path: str, cfg: dict, queue):
    cache = load_sharded_gbuffer(Path(shard_dir))
    mono, streamed = load_model(Path(mono_path)), load_model(Path(stream_path))
    xy, aux = __import__(
        "nrp.torch_backend.streamed_train", fromlist=["_pixel_tensors"]
    )._pixel_tensors(cache, "cpu")
    rng = np.random.default_rng([cfg["seed"], 0x5EED])
    mono_scores, stream_scores = [], []
    inference = []
    for _ in range(6):
        light = sample_light(cache, rng, "sphere", cfg["light_bounds"], "bbox")
        target, _ = gather_sphere_streamed(
            Path(shard_dir), cache.n_paths, light.center, light.radius
        )
        params = torch.as_tensor(
            np.concatenate([light.center, [light.radius]]), dtype=torch.float32
        ).expand(xy.shape[0], -1)
        with torch.no_grad():
            a = mono(xy, aux, params).numpy().reshape(cache.height, cache.width, 3)
            b = streamed(xy, aux, params).numpy().reshape(cache.height, cache.width, 3)
        mono_scores.append(psnr(a, target))
        stream_scores.append(psnr(b, target))
    light = sample_light(cache, rng, "sphere", cfg["light_bounds"], "bbox")
    for model in (mono, streamed):
        t0 = time.perf_counter()
        relight_tiled(model, cache, [light], tile_pixels=16384)
        inference.append(1000 * (time.perf_counter() - t0))
    queue.put(
        {
            "monolithic_psnr_db": float(np.mean(mono_scores)),
            "streamed_psnr_db": float(np.mean(stream_scores)),
            "psnr_gap_db": abs(float(np.mean(mono_scores)) - float(np.mean(stream_scores))),
            "inference_ms": {"monolithic": inference[0], "streamed": inference[1]},
            "peak_rss_bytes": rss_bytes(),
        }
    )


def run_worker(target, *args):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, queue))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"{target.__name__} failed with exit code {proc.exitcode}")
    return queue.get()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="out/kitchen-512/path_cache.npz")
    parser.add_argument("--out", default="out/t2-streaming/report.json")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument(
        "--export-full-report", default="out/t2-streaming/export_512x512_64spp.json"
    )
    parser.add_argument(
        "--export-ceiling-report", default="out/t2-streaming/export_128x128_64spp.json"
    )
    parser.add_argument(
        "--compose-only",
        action="store_true",
        help=(
            "refresh export evidence and hardware in an existing report without "
            "rerunning training"
        ),
    )
    args = parser.parse_args()
    cache_path, out_path = Path(args.cache), Path(args.out)
    budget = 8 * 1024**3
    export_full = json.loads(Path(args.export_full_report).read_text())
    export_ceiling = json.loads(Path(args.export_ceiling_report).read_text())
    export_full["within_8gb"] = export_full["peak_rss_bytes"] <= budget
    export_ceiling["within_8gb"] = export_ceiling["peak_rss_bytes"] <= budget
    export_measurements = {
        "t1_full": export_full,
        "verified_in_budget_ceiling": export_ceiling,
        "finding": (
            "512x512/64spp exceeds the budget; 128x128/64spp is the verified "
            "passing ceiling. No intermediate configuration is inferred."
        ),
    }
    if args.compose_only:
        report = json.loads(out_path.read_text())
        report["export_measurements"] = export_measurements
        report["passes"]["full_export_within_8gb"] = export_full["within_8gb"]
        report["passes"]["reduced_export_within_8gb"] = export_ceiling["within_8gb"]
        report["hardware"] = _hardware_context()
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if not cache_path.exists():
        raise SystemExit(f"missing T1 cache: {cache_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out_path.parent / "kitchen-512-packed-shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard = run_worker(worker_shard, str(cache_path), str(shard_dir), args.tile_size)
    cfg = config(args.iters, 512)
    quality = run_worker(worker_quality, str(cache_path), str(shard_dir), cfg)
    mono_path = out_path.parent / "monolithic.pt"
    stream_path = out_path.parent / "streamed.pt"
    mono = run_worker(
        worker_train, "monolithic", str(cache_path), str(shard_dir), cfg, str(mono_path)
    )
    streamed = run_worker(
        worker_train, "streamed", str(cache_path), str(shard_dir), cfg, str(stream_path)
    )
    evaluation = run_worker(
        worker_eval, str(shard_dir), str(mono_path), str(stream_path), cfg
    )
    report = {
        "rung": "T2",
        "scene": "Mitsuba Country Kitchen (T1)",
        "resolution": [512, 512],
        "spp": 64,
        "config": cfg,
        "memory_budget_bytes": budget,
        "export_measurements": export_measurements,
        "packed_cache_conversion": {
            "peak_rss_bytes": shard["peak_rss_bytes"],
        },
        "cache_size_quality_curve": [
            {"layout": "float64_npz", "bytes": cache_path.stat().st_size, "reference": True},
            {
                "layout": "fp16_rgb9e5_packed_shards",
                "bytes": shard["packed_shard_bytes"],
                "gather_psnr_db_min": quality["gather_psnr_db_min"],
                "gather_psnr_db_mean": quality["gather_psnr_db_mean"],
            },
        ],
        "packing_quality": quality,
        "sharding": shard,
        "monolithic": mono,
        "streamed": streamed,
        "evaluation": evaluation,
        "passes": {
            "psnr_parity_within_0p1_db": evaluation["psnr_gap_db"] <= 0.1,
            "full_export_within_8gb": export_full["within_8gb"],
            "reduced_export_within_8gb": export_ceiling["within_8gb"],
            "packed_cache_conversion_within_8gb": shard["peak_rss_bytes"] <= budget,
            "monolithic_within_8gb": mono["peak_rss_bytes"] <= budget,
            "streamed_training_within_8gb": streamed["peak_rss_bytes"] <= budget,
            "inference_within_8gb": evaluation["peak_rss_bytes"] <= budget,
        },
        "hardware": _hardware_context(),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
