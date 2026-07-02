"""Packed-cache benchmark (roadmap item 5, paper §4.2): size, decode cost, quality.

For each available exported cache (toy, Mitsuba cornell box) this script writes the
fp16+rgb9e5 packed twin next to it under `out/pack/`, then measures cache size,
load time (packing moves the decode cost to load; gather itself sees float64 arrays
either way), per-image GATHERLIGHT time, and GATHERLIGHT fidelity (PSNR of the
packed cache's image against the float64 cache's) over random sphere lights.
With `--train` it also trains the toy torch proxy twice — identical config and seed,
float64 vs packed cache — to quantify the end-to-end held-out PSNR cost. One command,
deterministic:

  uv run python examples/pack_bench.py --train --out out/pack/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCENES = {
    "toy": ROOT / "out" / "toy" / "path_cache.npz",
    "mitsuba": ROOT / "out" / "mitsuba" / "path_cache.npz",
}


def timed_load(path: str, repeats: int = 3) -> tuple[PathCache, float]:
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        cache = PathCache.load(path)
        best = min(best, time.perf_counter() - t0)
    return cache, best


def random_lights(rng: np.random.Generator, n: int) -> list[SphereLight]:
    lights = []
    for _ in range(n):
        center = rng.uniform([-0.6, 0.3, -0.6], [0.6, 1.6, 0.6])
        lights.append(
            SphereLight(
                center=center,
                radius=float(rng.uniform(0.1, 0.3)),
                rgb=rng.uniform(1.0, 6.0, size=3),
            )
        )
    return lights


def bench_scene(name: str, cache_path: Path, out_dir: Path, n_lights: int) -> dict:
    packed_path = out_dir / f"{name}_packed.npz"
    cache, load_s_full = timed_load(str(cache_path))
    cache.save(str(packed_path), compressed=True)
    packed, load_s_packed = timed_load(str(packed_path))

    rng = np.random.default_rng(7)
    lights = random_lights(rng, n_lights)
    psnrs, t_full, t_packed = [], [], []
    for light in lights:
        t0 = time.perf_counter()
        ref = gather_light(cache, light)
        t_full.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        img = gather_light(packed, light)
        t_packed.append(time.perf_counter() - t0)
        psnrs.append(psnr(img, ref))

    size_full = os.path.getsize(cache_path)
    size_packed = os.path.getsize(packed_path)
    # A light no cached segment ever crosses yields two identical all-zero images
    # (PSNR = inf); count those separately so the mean stays finite JSON.
    finite = [p for p in psnrs if np.isfinite(p)]
    row = {
        "cache": str(cache_path),
        "resolution": [cache.width, cache.height],
        "segments": cache.segment_count,
        "size_mb_float64": size_full / 2**20,
        "size_mb_packed": size_packed / 2**20,
        "size_ratio": size_full / size_packed,
        "load_seconds_float64": load_s_full,
        "load_seconds_packed": load_s_packed,
        "gather_ms_float64": 1e3 * float(np.mean(t_full)),
        "gather_ms_packed": 1e3 * float(np.mean(t_packed)),
        "n_lights": n_lights,
        "gather_psnr_db_min": float(np.min(psnrs)),
        "gather_psnr_db_mean_finite": float(np.mean(finite)),
        "n_lights_identical_image": n_lights - len(finite),
    }
    print(
        f"{name}: {row['size_mb_float64']:.2f} MB -> {row['size_mb_packed']:.2f} MB "
        f"({row['size_ratio']:.2f}x); gather PSNR min {row['gather_psnr_db_min']:.1f} dB "
        f"over {n_lights} lights; load {load_s_full * 1e3:.0f} -> {load_s_packed * 1e3:.0f} ms"
    )
    return row


def train_comparison(out_dir: Path, seeds: list[int]) -> dict:
    """Train the toy torch proxy from the float64 vs packed cache over several seeds.

    A single seeded run is not a fair comparison: the caches' tiny numerical
    differences perturb the training trajectory chaotically over thousands of
    iterations, so per-seed deltas measure trajectory sensitivity, not packing
    damage. Following the repo's testing convention, the claim is about *means*
    across seeds.
    """
    from nrp.torch_backend.train import train
    from nrp.train import load_config

    cfg = load_config(str(ROOT / "examples" / "toy_sphere_torch.json"))
    packed_cache = out_dir / "toy_packed.npz"
    runs: dict = {}
    for tag, cache_path in [("float64", cfg["cache"]), ("packed", str(packed_cache))]:
        psnrs, smapes, secs = [], [], []
        for seed in seeds:
            run_cfg = dict(cfg)
            run_cfg["cache"] = cache_path
            run_cfg["seed"] = seed
            run_cfg["out_dir"] = str(out_dir / f"train-{tag}-seed{seed}")
            report = train(run_cfg)
            psnrs.append(report["val_psnr_db_vs_raw_mean"])
            smapes.append(report["val_smape_vs_raw_mean"])
            secs.append(report["train_seconds"])
        runs[tag] = {
            "cache": cache_path,
            "seeds": seeds,
            "train_seconds_mean": float(np.mean(secs)),
            "val_psnr_db_per_seed": psnrs,
            "val_psnr_db_mean": float(np.mean(psnrs)),
            "val_psnr_db_std": float(np.std(psnrs)),
            "val_smape_mean": float(np.mean(smapes)),
        }
    runs["psnr_delta_db_mean"] = (
        runs["packed"]["val_psnr_db_mean"] - runs["float64"]["val_psnr_db_mean"]
    )
    print(
        f"train comparison over seeds {seeds}: packed - float64 held-out PSNR = "
        f"{runs['psnr_delta_db_mean']:+.3f} dB "
        f"(float64 {runs['float64']['val_psnr_db_mean']:.2f} "
        f"± {runs['float64']['val_psnr_db_std']:.2f}, "
        f"packed {runs['packed']['val_psnr_db_mean']:.2f} "
        f"± {runs['packed']['val_psnr_db_std']:.2f})"
    )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "out" / "pack" / "report.json"))
    parser.add_argument("--n-lights", type=int, default=20)
    parser.add_argument(
        "--train",
        action="store_true",
        help="also train the toy torch proxy from float64 vs packed caches (same seeds)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="training seeds for the --train comparison (means are compared)",
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"scenes": {}}
    for name, cache_path in SCENES.items():
        if not cache_path.exists():
            print(f"{name}: {cache_path} missing, skipping (export it first)")
            continue
        report["scenes"][name] = bench_scene(name, cache_path, out_dir, args.n_lights)
    if args.train:
        report["train"] = train_comparison(out_dir, args.seeds)

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
