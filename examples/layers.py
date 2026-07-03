"""Per-layer compositing NRP experiment (roadmap item 8, paper §6.1 / Fig. 11).

Splits the toy scene into first-hit layers (foreground sphere / background box),
traces one path cache per layer plus the full scene (same seed: the layers partition
the full cache's paths exactly), then verifies and measures the compositing claims:

- **Verify:** the two layers' GATHERLIGHT images sum to the full-scene GATHERLIGHT
  image (exactly, since the layers partition one traced path set — reported as PSNR
  and allclose); a composited edit demo image is written by this committed command.
- **Measure:** per-layer training time and held-out PSNR; composite edit latency
  (one layer's proxy + fixed image add) vs full-scene proxy relight; whether each
  layer proxy matches the full-scene proxy's quality on the layer's own pixels
  (within 1 dB — recorded either way).

Results land in out/layers/report.json, the demo image in out/layers/composite_demo.npy
(sphere layer relit under a new light, box layer held fixed under the original light).

Usage:
  uv run python examples/layers.py --out out/layers/report.json
  uv run python examples/layers.py --skip-train   # re-measure with existing models
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import light_from_dict  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.composite import composite  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight_multiview import ViewProxy, edit_latency_ms  # noqa: E402
from nrp.torch_backend.sampling import sample_light  # noqa: E402
from nrp.torch_backend.train import train as train_torch  # noqa: E402
from nrp.toy_tracer import LAYERS, layer_ownership_mask, trace_path_cache  # noqa: E402

#: Demo edit: the box layer stays lit by the warm light; the sphere layer is relit
#: by the cool one. Coordinates are inside the toy tracer's unit box.
DEMO_FIXED_LIGHT = {
    "type": "sphere",
    "center": [0.75, 0.8, 0.5],
    "radius": 0.12,
    "rgb": [6.0, 4.5, 3.0],
}
DEMO_EDIT_LIGHT = {
    "type": "sphere",
    "center": [0.2, 0.75, 0.35],
    "radius": 0.1,
    "rgb": [2.0, 3.5, 6.0],
}


def train_cfg(cache_path: str, out_dir: str, iters: int) -> dict:
    """Identical training config for the full-scene and both layer proxies (matches
    examples/toy_sphere_torch.json)."""
    return {
        "cache": cache_path,
        "out_dir": out_dir,
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
        "sampling": "segments",
        "pool": {"size": 64, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "radius": 2},
        "iters": iters,
        "batch_pixels": 4096,
        "lr": 0.005,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 8,
                "features_per_level": 2,
                "table_size_log2": 12,
                "base_resolution": 4,
                "finest_resolution": 48,
            },
        },
        "n_val_lights": 12,
        "seed": 0,
        "device": "cpu",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/layers/report.json")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--spp", type=int, default=24)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--skip-trace", action="store_true", help="reuse existing caches")
    parser.add_argument("--skip-train", action="store_true", help="reuse existing models")
    args = parser.parse_args()
    base = Path(args.out).resolve().parent
    names = ("full", *LAYERS)

    # 1. Trace the full cache and both layer caches from the same seed.
    caches: dict[str, PathCache] = {}
    for name in names:
        cache_path = base / name / "path_cache.npz"
        if args.skip_trace and cache_path.exists():
            caches[name] = PathCache.load(str(cache_path))
            print(f"[{name}] reusing {cache_path} ({caches[name].segment_count} segments)")
            continue
        layer = None if name == "full" else name
        cache = trace_path_cache(
            args.width, args.height, args.spp, args.bounces, args.seed, layer=layer
        )
        os.makedirs(cache_path.parent, exist_ok=True)
        cache.save(str(cache_path))
        caches[name] = cache
        print(f"[{name}] traced {cache.segment_count} segments -> {cache_path}")

    # 2. Linearity: the layers partition the full path set, so their GATHERLIGHT
    #    images must sum to the full-scene image for any light.
    val_rng = np.random.default_rng([args.seed, 0x1A7E5])
    val_lights = [
        sample_light(
            caches["full"], val_rng, "sphere", {"radius_min": 0.05, "radius_max": 0.25}, "segments"
        )
        for _ in range(12)
    ]
    linearity_psnrs = []
    linearity_allclose = True
    for light in val_lights:
        full_img = gather_light(caches["full"], light)
        layer_sum = gather_light(caches["sphere"], light) + gather_light(caches["box"], light)
        linearity_allclose &= bool(np.allclose(layer_sum, full_img))
        linearity_psnrs.append(min(psnr(layer_sum, full_img), 999.0))
    seg_partition = (
        caches["sphere"].segment_count + caches["box"].segment_count == caches["full"].segment_count
    )
    print(
        f"linearity over {len(val_lights)} lights: min PSNR {min(linearity_psnrs):.1f} dB, "
        f"allclose {linearity_allclose}, segment partition {seg_partition}"
    )

    # 3. Train the full-scene proxy and one proxy per layer, identical configs.
    rows = {}
    for name in names:
        out_dir = base / name
        cfg = train_cfg(str(out_dir / "path_cache.npz"), str(out_dir), args.iters)
        report_path = out_dir / "torch_train_report.json"
        if args.skip_train and (out_dir / "model.pt").exists() and report_path.exists():
            print(f"[{name}] reusing {out_dir / 'model.pt'}")
            report = json.loads(report_path.read_text())
        else:
            print(f"[{name}] training ({args.iters} iters) ...")
            report = train_torch(cfg)
        rows[name] = {
            "segments": caches[name].segment_count,
            "train_seconds": report["train_seconds"],
            "val_psnr_db_mean": report["val_psnr_db_vs_raw_mean"],
            "val_smape_mean": report["val_smape_vs_raw_mean"],
        }

    # 4. Owned-pixel quality: on each layer's own pixels, does the layer proxy match
    #    the full-scene proxy (each judged against its own GATHERLIGHT reference,
    #    shared peak so the PSNRs are comparable)?
    device = torch.device("cpu")
    proxies = {
        name: ViewProxy(name, TorchNRP.load(str(base / name / "model.pt")), caches[name], device)
        for name in names
    }
    masks = {layer: layer_ownership_mask(args.width, args.height, layer) for layer in LAYERS}
    owned = {}
    for layer in LAYERS:
        mask = masks[layer]
        layer_psnrs, full_psnrs = [], []
        for light in val_lights:
            full_ref = gather_light(caches["full"], light)
            layer_ref = gather_light(caches[layer], light)
            peak = float(full_ref[mask].max())
            layer_psnrs.append(psnr(proxies[layer].render([light])[mask], layer_ref[mask], peak))
            full_psnrs.append(psnr(proxies["full"].render([light])[mask], full_ref[mask], peak))
        owned[layer] = {
            "own_pixels": int(mask.sum()),
            "layer_proxy_psnr_db": float(np.mean(layer_psnrs)),
            "full_proxy_psnr_db": float(np.mean(full_psnrs)),
            "delta_db": float(np.mean(layer_psnrs) - np.mean(full_psnrs)),
        }
        print(
            f"owned pixels [{layer}] ({owned[layer]['own_pixels']} px): layer proxy "
            f"{owned[layer]['layer_proxy_psnr_db']:.2f} dB vs full proxy "
            f"{owned[layer]['full_proxy_psnr_db']:.2f} dB (delta {owned[layer]['delta_db']:+.2f})"
        )

    # 5. Composited edit demo (the committed command's artifact): box layer held
    #    fixed under the warm light, sphere layer relit by its proxy under the cool
    #    light. Written via the same composite() the CLI uses.
    fixed_light = light_from_dict(DEMO_FIXED_LIGHT)
    edit_light = light_from_dict(DEMO_EDIT_LIGHT)
    fixed_image = gather_light(caches["box"], fixed_light)
    np.save(base / "box_fixed.npy", fixed_image)
    demo = composite(proxies["sphere"], [edit_light], fixed_image)
    np.save(base / "composite_demo.npy", demo)
    print(f"wrote {base / 'composite_demo.npy'} (mean radiance {demo.mean():.6f})")

    # 6. Latency: composite edit (one layer forward + image add) vs full relight.
    n_frames = 30
    for _ in range(3):
        composite(proxies["sphere"], [edit_light], fixed_image)
    t0 = time.perf_counter()
    for _ in range(n_frames):
        composite(proxies["sphere"], [edit_light], fixed_image)
    composite_ms = (time.perf_counter() - t0) / n_frames * 1000.0
    full_ms = edit_latency_ms([proxies["full"]], [edit_light], frames=n_frames, warmup=3)
    print(f"composite edit {composite_ms:.2f} ms vs full-scene relight {full_ms:.2f} ms")

    checks = {
        "segments_partition_exactly": bool(seg_partition),
        "layer_sum_allclose_full": bool(linearity_allclose),
        "layer_sum_psnr_above_30db": bool(min(linearity_psnrs) > 30.0),
        "layer_proxies_within_1db_on_own_pixels": bool(
            all(abs(o["delta_db"]) <= 1.0 for o in owned.values())
        ),
    }
    report = {
        "scene": "toy cornell-style box + sphere",
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "bounces": args.bounces,
        "iters": args.iters,
        "linearity_psnr_db_min": min(linearity_psnrs),
        "proxies": rows,
        "owned_pixel_quality": owned,
        "demo": {"fixed_light": DEMO_FIXED_LIGHT, "edit_light": DEMO_EDIT_LIGHT},
        "composite_edit_ms": composite_ms,
        "full_relight_ms": full_ms,
        "checks": checks,
    }
    os.makedirs(base, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.out}")

    # The 1 dB owned-pixel comparison is recorded, not enforced (roadmap: "record
    # whether"); the linearity property and exact partition are hard requirements.
    required = [
        "segments_partition_exactly",
        "layer_sum_allclose_full",
        "layer_sum_psnr_above_30db",
    ]
    failed = [name for name in required if not checks[name]]
    if failed:
        raise SystemExit(f"verification checks failed: {failed}")
    print("all verification checks passed")


if __name__ == "__main__":
    main()
