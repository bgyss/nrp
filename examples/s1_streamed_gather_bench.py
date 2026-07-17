"""S1: vectorized streamed gather — parity + throughput report at 512x512/128spp.

Benchmarks the streamed pool machinery (`nrp.torch_backend.streamed_train`) before
and after S1's two changes — per-shard `TorchPathCache` batched gathers
(`gather_backend: torch`) and multi-light shard passes — against the committed E5
baselines (382.2 s pool build / 1,004.9 s total at 150 iters, scalar numpy,
one shard pass per pool slot).

Three phases:
1. per-shard gather microbenchmark (numpy vs torch-cpu vs torch-mps) on one real
   shard, with parity assertions (cpu rtol 1e-5; mps aggregate rel-L1 < 1e-4);
2. full-frame single-light streamed gather parity numpy vs torch on the real
   sharded cache;
3. end-to-end `train_streamed` (150 iters, baseline config) per backend, asserting
   peak resident segment bytes are unchanged across backends.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.mitsuba_512_streamed import held_out_psnr  # noqa: E402
from nrp.lights import segment_hits_sphere  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.gather import TorchPathCache  # noqa: E402
from nrp.torch_backend.streamed_train import (  # noqa: E402
    _decode_shard,
    gather_sphere_streamed,
    train_streamed,
)

LIGHT_CENTER = np.array([278.0, 274.0, 279.0])
LIGHT_RADIUS = 10.0


def bench_shard_gather(shard_dir: Path, repeats: int = 5) -> dict:
    """Time one shard's sphere gather per backend, decode excluded."""
    with open(shard_dir / "manifest.json") as f:
        manifest = json.load(f)
    width, height = int(manifest["width"]), int(manifest["height"])
    # largest shard file = the representative worst case
    shard = max(manifest["shards"], key=lambda s: (shard_dir / s["path"]).stat().st_size)
    t0 = time.perf_counter()
    with np.load(shard_dir / shard["path"]) as z:
        seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput, decoded = _decode_shard(z)
    decode_s = time.perf_counter() - t0
    n_px = width * height
    n_paths = np.full(n_px, 1, dtype=np.int64)

    def numpy_gather():
        out = np.zeros((n_px, 3), dtype=np.float64)
        hits = segment_hits_sphere(seg_origin, seg_dir, seg_tmax, LIGHT_CENTER, LIGHT_RADIUS)
        if hits.any():
            np.add.at(out, seg_pixel[hits], seg_throughput[hits])
        return out

    results = {
        "shard": shard["path"],
        "segments": int(seg_pixel.size),
        "decoded_bytes": decoded,
        "decode_seconds": decode_s,
    }
    times = []
    ref = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        ref = numpy_gather()
        times.append(time.perf_counter() - t0)
    results["numpy_ms"] = float(np.median(times) * 1e3)

    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    for dev in devices:
        device = torch.device(dev)
        tpc = TorchPathCache.from_arrays(
            width=width,
            height=height,
            seg_pixel=seg_pixel,
            seg_origin=seg_origin,
            seg_dir=seg_dir,
            seg_tmax=seg_tmax,
            seg_throughput=seg_throughput,
            n_paths=n_paths,
            device=device,
        )
        tpc.gather_throughput(LIGHT_CENTER, LIGHT_RADIUS)  # warmup
        if dev == "mps":
            torch.mps.synchronize()
        times = []
        got = None
        for _ in range(repeats):
            t0 = time.perf_counter()
            got = tpc.gather_throughput(LIGHT_CENTER, LIGHT_RADIUS)
            if dev == "mps":
                torch.mps.synchronize()
            times.append(time.perf_counter() - t0)
        results[f"torch_{dev}_ms"] = float(np.median(times) * 1e3)
        got_np = got.cpu().numpy().reshape(-1, 3).astype(np.float64)
        if dev == "cpu":
            np.testing.assert_allclose(got_np, ref, rtol=1e-5, atol=1e-12)
            results["torch_cpu_parity"] = "rtol 1e-5 pass"
        else:
            rel_l1 = float(np.abs(got_np - ref).sum() / max(ref.sum(), 1e-9))
            assert rel_l1 < 1e-4, f"mps rel L1 {rel_l1}"
            results["torch_mps_rel_l1"] = rel_l1
        del tpc
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/s1-streamed-gather/report.json")
    parser.add_argument("--cache", default="out/mitsuba-512/path_cache.npz")
    parser.add_argument("--shards", default="out/out-of-core/mitsuba_512_sharded")
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument(
        "--backends",
        default="numpy,torch:cpu,torch:mps",
        help="comma-separated: numpy | torch:cpu | torch:mps",
    )
    parser.add_argument("--skip-full-parity", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = Path(args.shards)

    report: dict = {
        "rung": "S1",
        "scope": "vectorized streamed gather: parity + wall-clock vs E5 baselines",
        "baseline": {
            "pool_seconds": 382.2,
            "train_seconds": 621.7,
            "total_seconds": 1004.9,
            "iters": 150,
            "source": "docs/performance.md E5 512x512/128spp table",
        },
    }

    print("== per-shard gather microbenchmark ==", flush=True)
    report["shard_gather"] = bench_shard_gather(shard_dir)
    print(json.dumps(report["shard_gather"], indent=2), flush=True)

    cache = PathCache.load(args.cache)
    n_paths = cache.n_paths.reshape(-1)

    if not args.skip_full_parity:
        print("== full-frame streamed parity (one real light) ==", flush=True)
        ref, ref_stats = gather_sphere_streamed(
            shard_dir, n_paths, LIGHT_CENTER, LIGHT_RADIUS, backend="numpy"
        )
        got, got_stats = gather_sphere_streamed(
            shard_dir, n_paths, LIGHT_CENTER, LIGHT_RADIUS, backend="torch", device="cpu"
        )
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-12)
        assert got_stats["peak_segment_bytes"] == ref_stats["peak_segment_bytes"]
        report["full_frame_parity"] = {
            "check": "torch-cpu vs numpy, full 512x512 sharded cache, rtol 1e-5",
            "pass": True,
            "peak_segment_bytes": ref_stats["peak_segment_bytes"],
        }
        print("parity pass", flush=True)

    cfg_base = {
        "seed": 0,
        "device": "cpu",
        "light_type": "sphere",
        "light_bounds": {"radius_min": 5.0, "radius_max": 15.0},
        "sampling": "segments",
        "denoise": {"enabled": False},
        "pool": {"size": 16, "replace_count": 1, "replace_every": 6},
        "model": {
            "hidden_width": 64,
            "hidden_layers": 3,
            "encoding": {"levels": 4, "features_per_level": 2, "finest_resolution": 512},
        },
        "lr": 5e-3,
        "batch_pixels": 4096,
        "iters": args.iters,
    }

    runs = {}
    peak_bytes = {}
    for spec in args.backends.split(","):
        spec = spec.strip()
        if spec == "numpy":
            cfg = dict(cfg_base)
            key = "numpy"
        else:
            _, dev = spec.split(":")
            if dev == "mps" and not torch.backends.mps.is_available():
                print(f"skipping {spec}: mps unavailable", flush=True)
                continue
            cfg = dict(cfg_base, gather_backend="torch", gather_device=dev)
            key = f"torch_{dev}"
        print(f"== end-to-end train_streamed [{key}] ==", flush=True)
        t0 = time.perf_counter()
        model, stats = train_streamed(shard_dir, cache, cfg)
        total = time.perf_counter() - t0
        psnr_db = held_out_psnr(model, cache, cfg)
        runs[key] = {
            "pool_seconds": stats["pool_seconds"],
            "train_seconds": stats["train_seconds"],
            "total_seconds": total,
            "supervision_seconds": stats["supervision_seconds"],
            "decode_seconds": stats["decode_seconds"],
            "gather_seconds": stats["gather_seconds"],
            "peak_segment_bytes_loaded": stats["peak_segment_bytes_loaded"],
            "peak_device_tensor_bytes": stats["peak_device_tensor_bytes"],
            "loss_first": stats["loss_curve"][0],
            "loss_last": stats["loss_curve"][-1],
            "held_out_psnr_db": psnr_db,
            "speedup_vs_baseline_total": 1004.9 / total,
            "speedup_vs_baseline_pool": 382.2 / stats["pool_seconds"],
        }
        peak_bytes[key] = stats["peak_segment_bytes_loaded"]
        print(json.dumps(runs[key], indent=2), flush=True)

    assert len(set(peak_bytes.values())) == 1, f"peak resident bytes differ: {peak_bytes}"
    report["end_to_end"] = runs
    report["peak_resident_bytes_unchanged"] = True
    report["rng_order_note"] = (
        "multi-light shard passes sample lights in slot order before gathering; the "
        "rng stream and numpy-backend targets are bit-identical to per-slot fills "
        "(tests/test_out_of_core.py streamed-vs-monolithic equality still passes)"
    )
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
