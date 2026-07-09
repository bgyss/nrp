"""E9's remaining criterion: a production-scale final-frame trust verdict.

`examples/quality_tiers.py` proves the tier plumbing and residual identity at toy
scale (16x16). This reuses the same tier semantics — (1) proxy preview, (2) cached
GATHERLIGHT at export spp, (3) GATHERLIGHT from a fresh high-spp cache (converged
reference), (4) proxy + cached residual — at 512x512 on two real Mitsuba cornell-box
caches: a 32spp "export" cache (tier 1/2/4 source) and the 128spp cache already used
for E5's production-scale report (tier 3, the converged reference). The proxy is
trained via the streamed pipeline (`nrp.torch_backend.streamed_train`) so training
itself doesn't require the full export cache resident in memory.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.quality_tiers import supervisor_trust_verdict  # noqa: E402
from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import flip, psnr, ssim, tonemap_srgb  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.relight import render_quality_tier, write_image_with_metadata  # noqa: E402
from nrp.torch_backend.streamed_train import train_streamed  # noqa: E402


def finite_or_inf(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def quality_metrics(image: np.ndarray, reference: np.ndarray) -> dict:
    image_ldr = tonemap_srgb(image)
    reference_ldr = tonemap_srgb(reference)
    return {
        "psnr_vs_final_db": finite_or_inf(psnr(image, reference)),
        "ssim_vs_final": ssim(image_ldr, reference_ldr, data_range=1.0),
        "flip_vs_final": flip(image_ldr, reference_ldr),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/quality/production_report.json")
    parser.add_argument("--export-cache", default="out/mitsuba-512-draft/path_cache.npz")
    parser.add_argument("--final-cache", default="out/mitsuba-512/path_cache.npz")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    export_path = Path(args.export_cache)
    final_path = Path(args.final_cache)
    if not export_path.exists() or not final_path.exists():
        raise SystemExit(
            f"expected both {export_path} and {final_path} to exist; export them with "
            f"nrp.mitsuba_exporter first."
        )

    cache = PathCache.load(str(export_path))
    shard_dir = base / "production_export_sharded"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    t0 = time.perf_counter()
    cache.save_sharded(str(shard_dir), tile_size=args.tile_size)
    shard_s = time.perf_counter() - t0

    approved = [SphereLight(center=[5.0, 3.0, 0.0], radius=8.0, rgb=[3.0, 2.2, 1.6])]

    cfg = {
        "seed": 0,
        "device": "cpu",
        "light_type": "sphere",
        "light_bounds": {"radius_min": 5.0, "radius_max": 15.0},
        "sampling": "segments",
        "denoise": {"enabled": False},
        "pool": {"size": 8, "replace_count": 1, "replace_every": 6},
        "model": {
            "hidden_width": 64,
            "hidden_layers": 3,
            "encoding": {"levels": 4, "features_per_level": 2, "finest_resolution": cache.width},
        },
        "lr": 5e-3,
        "batch_pixels": 4096,
        "iters": args.iters,
    }
    t0 = time.perf_counter()
    model, train_stats = train_streamed(shard_dir, cache, cfg)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    final_cache = PathCache.load(str(final_path))
    final_load_s = time.perf_counter() - t0

    tiers = {}

    def timed(fn):
        t0 = time.perf_counter()
        value = fn()
        return value, (time.perf_counter() - t0) * 1000.0

    (final_image, final_meta), final_ms = timed(
        lambda: render_quality_tier(
            model, cache, approved, quality="final", final_cache=final_cache
        )
    )
    write_image_with_metadata(str(base / "production_final.npy"), final_image, final_meta)
    tiers["final"] = {"ms": final_ms, "metadata": final_meta}

    for quality in ("preview", "draft"):
        (image, metadata), ms = timed(
            lambda q=quality: render_quality_tier(model, cache, approved, quality=q)
        )
        write_image_with_metadata(str(base / f"production_{quality}.npy"), image, metadata)
        tiers[quality] = {
            "ms": ms,
            "metadata": metadata,
            **quality_metrics(image, final_image),
        }

    (residual_image, residual_meta), residual_ms = timed(
        lambda: render_quality_tier(
            model, cache, approved, quality="preview", residual_lights=approved
        )
    )
    write_image_with_metadata(
        str(base / "production_preview_residual.npy"), residual_image, residual_meta
    )
    tiers["preview_plus_residual"] = {
        "ms": residual_ms,
        "metadata": residual_meta,
        **quality_metrics(residual_image, final_image),
    }
    draft_image = gather_lights(cache, approved)
    residual_identity = float(np.max(np.abs(residual_image - draft_image)))

    decay = []
    for dx in (0.0, 5.0, 10.0, 20.0):
        moved = [SphereLight(center=[5.0 + dx, 3.0, 0.0], radius=8.0, rgb=[3.0, 2.2, 1.6])]
        corrected, _ = render_quality_tier(
            model, cache, moved, quality="preview", residual_lights=approved
        )
        reference = gather_lights(cache, moved)
        decay.append(
            {
                "center_dx": dx,
                "psnr_db_vs_cached_gather": finite_or_inf(psnr(corrected, reference)),
            }
        )

    verdict = supervisor_trust_verdict(residual_identity, decay)
    verdict["scope"] = (
        "production-scale (512x512, real Mitsuba cornell-box) cached-residual trust verdict"
    )
    verdict["production_claim"] = True

    report = {
        "extension": "E9",
        "scope": "production-scale quality-tier ladder and trust verdict",
        "resolution": [cache.width, cache.height],
        "export_cache_segments": cache.segment_count,
        "final_cache_segments": final_cache.segment_count,
        "final_cache_load_seconds": final_load_s,
        "save_sharded_seconds": shard_s,
        "streamed_train_seconds": train_s,
        "streamed_train_stats": {
            "pool_seconds": train_stats["pool_seconds"],
            "train_seconds": train_stats["train_seconds"],
            "loss_first": train_stats["loss_curve"][0],
            "loss_last": train_stats["loss_curve"][-1],
            "peak_segment_bytes_loaded": train_stats["peak_segment_bytes_loaded"],
        },
        "tiers": tiers,
        "display_metric_preprocess": "Reinhard tonemap + sRGB before SSIM/FLIP",
        "residual_identity_max_abs_diff": residual_identity,
        "residual_decay": decay,
        "supervisor_trust_verdict": verdict,
        "outputs": {
            "final": "production_final.npy",
            "preview": "production_preview.npy",
            "draft": "production_draft.npy",
            "preview_plus_residual": "production_preview_residual.npy",
        },
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
