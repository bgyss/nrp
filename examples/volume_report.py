"""Volume-vs-surface measurement report (roadmap item 2).

Traces the toy scene twice at identical resolution/spp/bounces — surface-only and
with the homogeneous medium from `toy_sphere_volume_torch.json` — and measures cache
growth (segments, MB), trace time, and GATHERLIGHT time over random sphere lights.
If the two torch training reports exist (`mise run train-torch` and
`uv run python -m nrp.torch_backend.train --config examples/toy_sphere_volume_torch.json`),
their held-out quality is embedded for the proxy-quality comparison.

Usage:
  uv run python examples/volume_report.py --out out/volume-report.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

MEDIUM = {"sigma_t": 2.0, "albedo": 0.8}
TRACE = {"width": 48, "height": 48, "spp": 24, "bounces": 3, "seed": 1}


def measure(medium: dict | None, rng: np.random.Generator) -> dict:
    t0 = time.perf_counter()
    cache = trace_path_cache(
        TRACE["width"], TRACE["height"], TRACE["spp"], TRACE["bounces"], TRACE["seed"], medium
    )
    trace_seconds = time.perf_counter() - t0
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "cache.npz")
        cache.save(path)
        size_mb = os.path.getsize(path) / 1e6
    gather_times = []
    for _ in range(20):
        light = SphereLight(
            center=rng.uniform(0.1, 0.9, 3), radius=float(rng.uniform(0.05, 0.25)), rgb=[1, 1, 1]
        )
        t0 = time.perf_counter()
        gather_light(cache, light)
        gather_times.append(time.perf_counter() - t0)
    return {
        "medium": medium,
        "segments": cache.segment_count,
        "cache_mb": size_mb,
        "trace_seconds": trace_seconds,
        "gather_ms_mean": float(np.mean(gather_times) * 1000),
    }


def training_quality(report_path: str) -> dict | None:
    if not os.path.exists(report_path):
        return None
    with open(report_path) as f:
        r = json.load(f)
    return {
        "report": report_path,
        "val_psnr_db_vs_raw_mean": r.get("val_psnr_db_vs_raw_mean"),
        "val_smape_vs_raw_mean": r.get("val_smape_vs_raw_mean"),
        "train_seconds": r.get("train_seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output JSON report")
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    surface = measure(None, rng)
    volume = measure(MEDIUM, rng)
    report = {
        "platform": platform.platform(),
        "trace_config": TRACE,
        "surface": surface,
        "volume": volume,
        "growth": {
            "segments_ratio": volume["segments"] / surface["segments"],
            "mb_ratio": volume["cache_mb"] / surface["cache_mb"],
            "gather_ms_ratio": volume["gather_ms_mean"] / surface["gather_ms_mean"],
        },
        "training": {
            "surface": training_quality("out/toy-torch/torch_train_report.json"),
            "volume": training_quality("out/toy-volume-torch/torch_train_report.json"),
        },
    }
    for name, row in [("surface", surface), ("volume", volume)]:
        print(
            f"{name:8s}: {row['segments']} segments, {row['cache_mb']:.1f} MB, "
            f"trace {row['trace_seconds']:.1f}s, gather {row['gather_ms_mean']:.1f} ms"
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
