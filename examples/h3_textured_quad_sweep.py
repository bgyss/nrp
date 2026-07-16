"""H3 report: iteration-budget (and, if that plateaus, capacity) sweep for the V1
rig's two TexturedQuadLight proxies (`neon_sign`, `tv_glow`) -- V1's genuinely-
contributing low scorers (12.35 / 13.35 dB at 800 iters, 3.4x slower per iter than
sphere/quad because textured_quad pool targets fall back to the numpy gather
backend, see `examples/v1_rig.py`'s `build_per_light_config` docstring).

Reuses `examples.v1_rig.default_rig_lights`/`build_per_light_config` so every swept
run shares the rig's real light geometry/texture and base training config; only
`iters` (and, for the capacity arm, `model.hidden_width`) varies.

Usage:
  uv run python examples/h3_textured_quad_sweep.py \
      --cache out/kitchen-512/path_cache.npz --out-dir out/h3-textured-quad
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from examples.v1_rig import build_per_light_config, default_rig_lights  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.train import light_param_vector, pixel_tensors, train  # noqa: E402

TEXTURED_QUAD_LIGHTS = ("neon_sign", "tv_glow")

# (label, iters, hidden_width) -- hidden_width=None keeps the rig's default (128).
# The capacity arm (2x width) only runs if the iteration sweep plateaus below the
# rig's sphere/quad quality envelope (per this rung's own fallback instruction).
ITERS_SWEEP = [800, 1600, 3200]
CAPACITY_HIDDEN_WIDTH = 256


def _light_full_image_gate(model_path: str, cache, cfg: dict, rl) -> dict:
    """Full-image preview-tier gate at the light's own authored params, mirroring
    v1_rig.py's additivity-gate reference convention (raw gather, since this sweep
    doesn't retrain the other 7 rig lights needed for a denoised multi-light sum)."""
    model = TorchNRP.load(model_path)
    device = torch.device("cpu")
    xy, aux = pixel_tensors(cache, device)
    params = torch.as_tensor(
        light_param_vector(rl.light), dtype=torch.float32, device=device
    ).expand(xy.shape[0], -1)
    model.eval()
    with torch.no_grad():
        pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
    pred = pred.reshape(cache.height, cache.width, 3)
    ref = gather_light(cache, rl.light)
    return evaluate_gate(pred, ref, tier="preview")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", default="out/h3-textured-quad")
    parser.add_argument(
        "--kernel-iters",
        type=int,
        nargs="*",
        default=[800, 3200],
        help="iteration budgets for the texture-kernel conditioning arm "
        "(model.texture_conditioning='kernel': the MLP predicts a per-texel "
        "throughput kernel contracted with the texture at the output, instead "
        "of consuming the flattened texture as input -- the 'different input "
        "scheme' this rung's honest-negative finding pointed to)",
    )
    parser.add_argument(
        "--sphere-quad-envelope-db",
        type=float,
        default=17.0,
        help="rig's post-H1-fix sphere/quad val PSNR floor (out/h1-quad-fix "
        "measured 11.65-16.22 dB at 800 iters; H2 retrains at a higher budget -- "
        "this default is the H1 floor, override once H2's report lands)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache = PathCache.load(args.cache)
    lights_by_name = {rl.name: rl for rl in default_rig_lights()}

    base_cfg = {
        "cache": os.path.abspath(args.cache),
        "sampling": "segments",
        "gather_backend": "torch",
        "pool": {"size": 64, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "method": "bilateral"},
        "batch_pixels": 8192,
        "lr": 0.005,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 10,
                "features_per_level": 2,
                "table_size_log2": 16,
                "base_resolution": 4,
                "finest_resolution": 512,
            },
        },
        "n_val_lights": 12,
        "seed": 0,
        "device": "cpu",
    }

    rows = []
    for name in TEXTURED_QUAD_LIGHTS:
        rl = lights_by_name[name]
        for iters in ITERS_SWEEP:
            run_dir = os.path.join(args.out_dir, "train", f"{name}_iters{iters}")
            cfg = build_per_light_config(base_cfg, rl, run_dir, iters)
            model_path = os.path.join(run_dir, "model.pt")
            report_path = os.path.join(run_dir, "torch_train_report.json")
            if os.path.exists(model_path) and os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)
                print(f"skipping {name} iters={iters}: reusing existing {model_path}")
            else:
                t0 = time.perf_counter()
                report = train(cfg)
                report["_measured_train_seconds"] = time.perf_counter() - t0
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
            gate = _light_full_image_gate(model_path, cache, cfg, rl)
            rows.append(
                {
                    "light": name,
                    "iters": iters,
                    "hidden_width": 128,
                    "val_psnr_db_vs_raw_mean": report["val_psnr_db_vs_raw_mean"],
                    "val_ssim_vs_raw_mean": report["val_ssim_vs_raw_mean"],
                    "val_flip_vs_raw_mean": report["val_flip_vs_raw_mean"],
                    "train_seconds": report.get(
                        "train_seconds", report.get("_measured_train_seconds")
                    ),
                    "full_image_gate": gate,
                }
            )
            print(
                f"{name} iters={iters}: val_psnr={report['val_psnr_db_vs_raw_mean']:.2f} dB, "
                f"train_s={rows[-1]['train_seconds']:.1f}"
            )

    plateaued = all(
        r["val_psnr_db_vs_raw_mean"] < args.sphere_quad_envelope_db
        for r in rows
        if r["iters"] == ITERS_SWEEP[-1]
    )
    capacity_rows = []
    if plateaued:
        print(
            f"iteration sweep plateaued below {args.sphere_quad_envelope_db} dB -- "
            "running the capacity fallback arm (2x hidden_width)"
        )
        cap_cfg_base = copy.deepcopy(base_cfg)
        cap_cfg_base["model"]["hidden_width"] = CAPACITY_HIDDEN_WIDTH
        for name in TEXTURED_QUAD_LIGHTS:
            rl = lights_by_name[name]
            iters = ITERS_SWEEP[-1]
            run_dir = os.path.join(args.out_dir, "train", f"{name}_capacity{CAPACITY_HIDDEN_WIDTH}")
            cfg = build_per_light_config(cap_cfg_base, rl, run_dir, iters)
            model_path = os.path.join(run_dir, "model.pt")
            report_path = os.path.join(run_dir, "torch_train_report.json")
            if os.path.exists(model_path) and os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)
            else:
                t0 = time.perf_counter()
                report = train(cfg)
                report["_measured_train_seconds"] = time.perf_counter() - t0
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
            capacity_rows.append(
                {
                    "light": name,
                    "iters": iters,
                    "hidden_width": CAPACITY_HIDDEN_WIDTH,
                    "val_psnr_db_vs_raw_mean": report["val_psnr_db_vs_raw_mean"],
                    "val_ssim_vs_raw_mean": report["val_ssim_vs_raw_mean"],
                    "val_flip_vs_raw_mean": report["val_flip_vs_raw_mean"],
                    "train_seconds": report.get(
                        "train_seconds", report.get("_measured_train_seconds")
                    ),
                }
            )
            print(
                f"{name} capacity(hidden_width={CAPACITY_HIDDEN_WIDTH}): "
                f"val_psnr={report['val_psnr_db_vs_raw_mean']:.2f} dB"
            )

    # Texture-kernel conditioning arm (H3 finding follow-up): gather_textured_quad
    # is linear in the texture, so the model predicts a per-texel kernel and
    # contracts it with the texture at the output (TorchNRP texture_kernel=True).
    kernel_rows = []
    for name in TEXTURED_QUAD_LIGHTS:
        rl = lights_by_name[name]
        for iters in args.kernel_iters:
            kernel_cfg_base = copy.deepcopy(base_cfg)
            kernel_cfg_base["model"]["texture_conditioning"] = "kernel"
            run_dir = os.path.join(args.out_dir, "train", f"{name}_kernel_iters{iters}")
            cfg = build_per_light_config(kernel_cfg_base, rl, run_dir, iters)
            model_path = os.path.join(run_dir, "model.pt")
            report_path = os.path.join(run_dir, "torch_train_report.json")
            if os.path.exists(model_path) and os.path.exists(report_path):
                with open(report_path) as f:
                    report = json.load(f)
                print(f"skipping {name} kernel iters={iters}: reusing existing {model_path}")
            else:
                t0 = time.perf_counter()
                report = train(cfg)
                report["_measured_train_seconds"] = time.perf_counter() - t0
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)
            gate = _light_full_image_gate(model_path, cache, cfg, rl)
            kernel_rows.append(
                {
                    "light": name,
                    "iters": iters,
                    "hidden_width": 128,
                    "texture_conditioning": "kernel",
                    "val_psnr_db_vs_raw_mean": report["val_psnr_db_vs_raw_mean"],
                    "val_ssim_vs_raw_mean": report["val_ssim_vs_raw_mean"],
                    "val_flip_vs_raw_mean": report["val_flip_vs_raw_mean"],
                    "train_seconds": report.get(
                        "train_seconds", report.get("_measured_train_seconds")
                    ),
                    "full_image_gate": gate,
                }
            )
            print(
                f"{name} kernel iters={iters}: "
                f"val_psnr={report['val_psnr_db_vs_raw_mean']:.2f} dB, "
                f"train_s={kernel_rows[-1]['train_seconds']:.1f}"
            )

    best_by_light = {}
    for name in TEXTURED_QUAD_LIGHTS:
        candidates = [r for r in rows + capacity_rows + kernel_rows if r["light"] == name]
        best_by_light[name] = max(candidates, key=lambda r: r["val_psnr_db_vs_raw_mean"])

    report = {
        "rung": "H3",
        "scope": (
            "iteration-budget (+ capacity fallback) sweep for the V1 rig's 2 textured_quad proxies"
        ),
        "sphere_quad_envelope_db": args.sphere_quad_envelope_db,
        "iters_sweep": rows,
        "capacity_sweep": capacity_rows,
        "kernel_conditioning_sweep": kernel_rows,
        "iteration_sweep_plateaued": plateaued,
        "chosen_operating_point": {
            name: {
                "iters": best_by_light[name]["iters"],
                "hidden_width": best_by_light[name]["hidden_width"],
                "texture_conditioning": best_by_light[name].get("texture_conditioning", "flat"),
                "val_psnr_db_vs_raw_mean": best_by_light[name]["val_psnr_db_vs_raw_mean"],
                "meets_envelope": bool(
                    best_by_light[name]["val_psnr_db_vs_raw_mean"] >= args.sphere_quad_envelope_db
                ),
            }
            for name in TEXTURED_QUAD_LIGHTS
        },
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
