"""H5 report: G1's dynamic-geometry regime comparison at real kitchen scale.

G1's residual result (`examples/residual_dynamic.py`) is toy-scale only (32x32
sphere-in-box, `nrp.toy_tracer`) because until H5 the Mitsuba exporter had no way to
re-trace an edited scene. `nrp.mitsuba_exporter.apply_shape_translation` (H5) adds
that: translate one named shape, re-export. This script re-traces the T1 kitchen
scene with `ChoppingBoard` moved, computes both invalidation masks
(`nrp.dynamic_geometry`) against the shipped `out/kitchen-512/path_cache.npz`, and
reruns G1's four regimes -- (a) full retrace + retrain, (b) incremental fine-tune,
(c) stale, (d) frozen-base + shard residual -- for the V1 rig's "key" `SphereLight`
against its own already-trained proxy (`out/h2-rig/train/key/model.pt`), at a
deliberately reduced iteration budget (kitchen-scale per-iteration cost is far above
toy scale; see module docstring for the tradeoff this rung records).

Usage:
  uv run python examples/h5_kitchen_dynamic.py \
      --base-cache out/kitchen-512/path_cache.npz \
      --edited-cache out/h5-kitchen/edited_path_cache.npz \
      --base-model out/h2-rig/train/key/model.pt \
      --out-dir out/h5-kitchen
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nrp.dynamic_geometry import (  # noqa: E402
    primary_visibility_invalidation_mask,
    swept_volume_invalidation_mask,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.denoise import denoise_image  # noqa: E402
from nrp.torch_backend.model import TorchNRP, relative_mse_loss  # noqa: E402
from nrp.torch_backend.residual_dynamic import (  # noqa: E402
    ResidualNRP,
    composite_predict,
    invalidated_shards,
    train_residual,
)
from nrp.torch_backend.train import light_param_vector, pixel_tensors  # noqa: E402

RECOVERY_TARGET_DB = 1.0  # E2/G1's criterion: within 1 dB of regime (a)

# The V1 rig's "key" light and ChoppingBoard's own geometry (measured once via
# mi.traverse on the loaded kitchen scene -- see the H5 commit for the probe).
KEY_LIGHT = SphereLight(center=[-0.64, 2.19, -0.22], radius=0.3, rgb=[3.0, 3.0, 3.0])
CHOPPING_BOARD_CENTROID = np.array([0.15986572, 1.01368979, 1.0702121])
CHOPPING_BOARD_RADIUS = 0.3
CHOPPING_BOARD_TRANSLATE = np.array([0.2, 0.0, 0.0])


def _finetune(model: TorchNRP, cache, target, mask, light_params, iters, lr) -> list[float]:
    """Warm-started fine-tune of `model`'s own weights, restricted to `mask` pixels
    (E2's regime (b): the same mechanism as `examples.dynamic_geometry
    .TorchNRPWarmStartProxy.fine_tune`, inlined here so it runs against the real
    kitchen-scale TorchNRP architecture rather than that helper's fixed toy config)."""
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return []
    device = torch.device("cpu")
    xy_all, aux_all = pixel_tensors(cache, device)
    xy = xy_all[idx]
    aux = aux_all[idx]
    params = torch.as_tensor(light_params, dtype=torch.float32, device=device).expand(idx.size, -1)
    y = torch.as_tensor(
        target.reshape(-1, 3)[idx].astype(np.float32), dtype=torch.float32, device=device
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(iters):
        pred = model(xy, aux, params)
        loss = torch.mean((pred - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    model.eval()
    return losses


def _predict(model: TorchNRP, cache, light_params) -> np.ndarray:
    device = torch.device("cpu")
    xy, aux = pixel_tensors(cache, device)
    params = torch.as_tensor(light_params, dtype=torch.float32, device=device).expand(
        xy.shape[0], -1
    )
    model.eval()
    with torch.no_grad():
        out = model(xy, aux, params).cpu().numpy().astype(np.float64)
    return out.reshape(cache.height, cache.width, 3)


def _train_from_scratch(cache, target, light_params, iters, lr, arch_like: TorchNRP) -> TorchNRP:
    """Regime (a): a fresh proxy of the same architecture as `arch_like`, trained
    full-image on `target` -- the "full retrace + full retrain" ceiling regime.

    Must reuse H1's fix (`init_output_scale`) and the paper's relative-MSE loss
    (Eq. 4, `relative_mse_loss`), not a naive from-scratch MSE loop: a first version
    of this function did exactly that and silently reproduced H1's zero-collapse
    bug -- nn.Linear's default output bias vs. this cache's dim true target scale
    drove the pre-softplus logit to permanent zero within the first ~50-100
    iterations, after which the network was frozen and 300 vs. 1600 iterations
    produced bit-identical output (caught by comparing `out/h5-kitchen/report.json`
    against `out/h5-kitchen-converged/report.json` -- both reported the exact same
    PSNR to 15 significant digits, which is not something converged floating-point
    gradient descent does by chance)."""
    device = torch.device("cpu")
    model = TorchNRP(**arch_like.config)
    xy, aux = pixel_tensors(cache, device)
    params = torch.as_tensor(light_params, dtype=torch.float32, device=device).expand(
        xy.shape[0], -1
    )
    y = torch.as_tensor(target.reshape(-1, 3).astype(np.float32), dtype=torch.float32)
    model.init_output_scale(float(y.mean(dim=-1).median().item()))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(iters):
        pred = model(xy, aux, params)
        loss = relative_mse_loss(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--edited-cache", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--iters", type=int, default=300, help="regime (a)/(b)/(d) budget")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--shard-size", type=int, default=32)
    parser.add_argument("--denoise", default="bilateral", choices=["oidn", "bilateral", "none"])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t_load0 = time.perf_counter()
    before = PathCache.load(args.base_cache)
    after = PathCache.load(args.edited_cache)
    base_model = TorchNRP.load(args.base_model)
    load_seconds = time.perf_counter() - t_load0

    light_params = light_param_vector(KEY_LIGHT)

    t_mask0 = time.perf_counter()
    primary_mask = primary_visibility_invalidation_mask(before, after)
    swept_mask = swept_volume_invalidation_mask(
        after,
        CHOPPING_BOARD_CENTROID,
        CHOPPING_BOARD_CENTROID + CHOPPING_BOARD_TRANSLATE,
        CHOPPING_BOARD_RADIUS,
    )
    union_mask = primary_mask | swept_mask
    mask_seconds = time.perf_counter() - t_mask0

    region_mask, tiles = invalidated_shards(union_mask, args.shard_size)

    t_ref0 = time.perf_counter()
    raw_after = gather_light(after, KEY_LIGHT)
    if args.denoise == "none":
        full_after = raw_after
    else:
        full_after = denoise_image(
            raw_after, after.albedo, after.normal, after.depth, method=args.denoise
        )
    ref_seconds = time.perf_counter() - t_ref0

    # Regime (c): stale -- the frozen base proxy evaluated as-is on the edited scene's
    # own (moved) G-buffer aux, no update at all.
    t_c0 = time.perf_counter()
    regime_c_pred = _predict(base_model, after, light_params)
    regime_c_seconds = time.perf_counter() - t_c0

    # Regime (b): incremental fine-tune, warm-started from the base weights, updated
    # only on the invalidated (shard-aggregated) region.
    model_b = copy.deepcopy(base_model)
    t_b0 = time.perf_counter()
    _finetune(model_b, after, full_after, region_mask, light_params, args.iters, args.lr)
    regime_b_pred = _predict(model_b, after, light_params)
    regime_b_seconds = time.perf_counter() - t_b0

    # Regime (a): full retrace + full retrain -- the ceiling every other regime is
    # measured against, same architecture as the shipped base proxy.
    t_a0 = time.perf_counter()
    model_a = _train_from_scratch(
        after, full_after, light_params, args.iters, args.lr, arch_like=base_model
    )
    regime_a_pred = _predict(model_a, after, light_params)
    regime_a_seconds = time.perf_counter() - t_a0

    # Regime (d): G1's fix -- base frozen, small signed residual trained on the
    # shard-aggregated invalidated region only, composited additively at inference.
    residual = ResidualNRP(hidden_width=32, hidden_layers=2, light_param_dim=len(light_params))
    t_d0 = time.perf_counter()
    train_residual(
        residual,
        base_model,
        after,
        full_after,
        region_mask,
        light_params,
        iters=args.iters,
        lr=args.lr,
    )
    regime_d_pred = composite_predict(base_model, residual, after, light_params, region_mask)
    regime_d_seconds = time.perf_counter() - t_d0

    def _masked_psnr(pred, mask):
        if not mask.any():
            return None
        return psnr(pred[mask], full_after[mask])

    report = {
        "rung": "H5",
        "scene": "examples/scenes/kitchen/scene.xml (T1 Country Kitchen)",
        "moved_shape": "ChoppingBoard",
        "translate": CHOPPING_BOARD_TRANSLATE.tolist(),
        "resolution": [before.width, before.height],
        "light": "key (V1 rig SphereLight)",
        "masks": {
            "primary_visibility_invalid_pixels": int(primary_mask.sum()),
            "swept_volume_invalid_pixels": int(swept_mask.sum()),
            "union_invalid_pixels": int(union_mask.sum()),
            "total_pixels": int(union_mask.size),
            "shard_size": args.shard_size,
            "shard_region_pixels": int(region_mask.sum()),
            "n_invalidated_shards": len(tiles),
        },
        "mask_correctness_spot_check": {
            "out_of_mask_depth_max_abs_diff": float(
                np.abs(before.depth[~union_mask] - after.depth[~union_mask]).max()
            )
            if (~union_mask).any()
            else None,
            "in_mask_depth_mean_abs_diff": float(
                np.abs(before.depth[union_mask] - after.depth[union_mask]).mean()
            )
            if union_mask.any()
            else None,
        },
        "budget": {"iters": args.iters, "lr": args.lr, "denoise": args.denoise},
        "regimes": {
            "a_full_retrace_retrain": {
                "psnr_vs_full_db": psnr(regime_a_pred, full_after),
                "psnr_in_mask_db": _masked_psnr(regime_a_pred, union_mask),
                "seconds": regime_a_seconds,
            },
            "b_incremental_finetune": {
                "psnr_vs_full_db": psnr(regime_b_pred, full_after),
                "psnr_in_mask_db": _masked_psnr(regime_b_pred, union_mask),
                "seconds": regime_b_seconds,
            },
            "c_stale": {
                "psnr_vs_full_db": psnr(regime_c_pred, full_after),
                "psnr_in_mask_db": _masked_psnr(regime_c_pred, union_mask),
                "seconds": regime_c_seconds,
            },
            "d_frozen_base_plus_residual": {
                "psnr_vs_full_db": psnr(regime_d_pred, full_after),
                "psnr_in_mask_db": _masked_psnr(regime_d_pred, union_mask),
                "seconds": regime_d_seconds,
                "residual_parameter_count": residual.parameter_count,
            },
        },
        "wall_clock_seconds": {
            "load": load_seconds,
            "mask_compute": mask_seconds,
            "reference_gather_denoise": ref_seconds,
            "regime_a_full_retrace_retrain": regime_a_seconds,
            "invalidate_and_recover_regime_d": mask_seconds + regime_d_seconds,
        },
        "hardware": {
            "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
        },
    }
    mean_a = report["regimes"]["a_full_retrace_retrain"]["psnr_vs_full_db"]
    mean_b = report["regimes"]["b_incremental_finetune"]["psnr_vs_full_db"]
    mean_d = report["regimes"]["d_frozen_base_plus_residual"]["psnr_vs_full_db"]
    report["gap_a_minus_b_db"] = mean_a - mean_b
    report["gap_a_minus_d_db"] = mean_a - mean_d
    report["regime_b_within_target"] = bool(report["gap_a_minus_b_db"] <= RECOVERY_TARGET_DB)
    report["regime_d_within_target"] = bool(report["gap_a_minus_d_db"] <= RECOVERY_TARGET_DB)
    report["recovery_target_db"] = RECOVERY_TARGET_DB

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")
    print(
        f"regime (a) {mean_a:.2f} dB | (b) {mean_b:.2f} dB (gap {report['gap_a_minus_b_db']:.2f}) "
        f"| (d) {mean_d:.2f} dB (gap {report['gap_a_minus_d_db']:.2f})"
    )


if __name__ == "__main__":
    main()
