"""Differentiable inverse light optimization through the proxy (NRP M5/M6/M7).

Optimizes one sphere light's center, radius, and RGB intensity against an image-space
target, by gradient descent (Adam) through the trained proxy: GATHERLIGHT itself has
zero gradient almost everywhere in center/radius (hard visibility), so the smooth proxy
provides the differentiable path. The result is always *re-rendered through reference
GATHERLIGHT* as well, and the report separates proxy-space loss from
reference/GATHERLIGHT-space error, per the goal prompt.

Target modes:
  --target-light JSON   the target image is GATHERLIGHT of a hidden "true" light
                        (M5: parameter recovery)
  --target FILE.npy     an arbitrary (H,W,3) image — painted or generated (M6). The
                        target is an *objective*, not truth: no light configuration may
                        be able to reproduce it, and the report says how close the
                        physically re-rendered result actually gets.

Masks (both optional, both (H,W) float .npy):
  --mask FILE           objective weights: where to match the target (0 = don't care)
  --protect FILE        constraint mask: regions that must stay close to --protect-base
                        (default: the image under the *initial* light parameters). This
                        is the AGGR protected-region hook (M7): protected tiles keep
                        their appearance while the light is optimized elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .dataset import build_inputs, pixel_feature_block
from .gather_light import gather_light
from .lights import SphereLight
from .metrics import psnr, smape
from .model import ProxyMLP, light_param_gradient
from .path_cache import PathCache


class FlatAdam:
    """Adam over one flat parameter vector."""

    def __init__(self, n: int, lr: float):
        self.lr = lr
        self.t = 0
        self.m = np.zeros(n)
        self.v = np.zeros(n)

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = 0.9 * self.m + 0.1 * grad
        self.v = 0.999 * self.v + 0.001 * grad * grad
        m_hat = self.m / (1.0 - 0.9**self.t)
        v_hat = self.v / (1.0 - 0.999**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def proxy_image(model: ProxyMLP, px: dict, center: np.ndarray, radius: float, rgb: np.ndarray):
    """Returns (image (N,3), raw contribution f (N,3), inputs x) for gradient reuse."""
    x = build_inputs(px, center, radius)
    f = model.forward(x)
    return f * rgb, f, x


def optimize(
    model: ProxyMLP,
    cache: PathCache,
    target: np.ndarray,
    init: dict,
    bounds: dict,
    steps: int,
    lr: float,
    weight_mask: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
    protect_base: np.ndarray | None = None,
    protect_lambda: float = 10.0,
) -> dict:
    px = pixel_feature_block(cache)
    n_px = cache.height * cache.width
    target = target.reshape(n_px, 3)
    w = np.ones((n_px, 1)) if weight_mask is None else weight_mask.reshape(n_px, 1)
    prot = None if protect_mask is None else protect_mask.reshape(n_px, 1)

    params = np.concatenate(
        [np.asarray(init["center"], dtype=np.float64), [init["radius"]], np.asarray(init["rgb"])]
    )
    lo = np.concatenate([bounds["center_min"], [bounds["radius_min"]], [0.0, 0.0, 0.0]])
    hi = np.concatenate([bounds["center_max"], [bounds["radius_max"]], [np.inf] * 3])

    if prot is not None and protect_base is None:
        img0, _, _ = proxy_image(model, px, params[:3], params[3], params[4:7])
        protect_base = img0.copy()
    base = None if protect_base is None else protect_base.reshape(n_px, 3)

    opt = FlatAdam(7, lr)
    loss_curve = []
    for _step in range(steps):
        center, radius, rgb = params[:3], params[3], params[4:7]
        pred, f, x = proxy_image(model, px, center, radius, rgb)

        diff = pred - target
        w_sum = max(float(w.sum()) * 3.0, 1.0)
        loss = float(np.sum(w * diff**2) / w_sum)
        dpred = 2.0 * w * diff / w_sum
        if prot is not None:
            pdiff = pred - base
            p_sum = max(float(prot.sum()) * 3.0, 1.0)
            loss += protect_lambda * float(np.sum(prot * pdiff**2) / p_sum)
            dpred += protect_lambda * 2.0 * prot * pdiff / p_sum
        loss_curve.append(loss)

        d_rgb = np.sum(dpred * f, axis=0)
        d_f = dpred * rgb
        dx, _, _ = model.backward(d_f)
        d_center, d_radius = light_param_gradient(dx, px["position"], center)
        grad = np.concatenate([d_center, [d_radius], d_rgb])
        params = np.clip(opt.step(params, grad), lo, hi)

    result_light = SphereLight(center=params[:3], radius=float(params[3]), rgb=params[4:7])
    pred_final, _, _ = proxy_image(model, px, params[:3], params[3], params[4:7])
    gather_final = gather_light(cache, result_light).reshape(n_px, 3)

    report = {
        "optimized_light": result_light.to_dict(),
        "steps": steps,
        "proxy_loss_first": loss_curve[0],
        "proxy_loss_last": loss_curve[-1],
        "proxy_loss_curve": loss_curve[:: max(1, steps // 50)],
        "proxy_vs_target_psnr_db": psnr(pred_final, target),
        "proxy_vs_target_smape": smape(pred_final, target),
        "gather_vs_target_psnr_db": psnr(gather_final, target),
        "gather_vs_target_smape": smape(gather_final, target),
    }
    if prot is not None:
        pmask = prot[:, 0] > 0.0
        report["protected_region_pixels"] = int(pmask.sum())
        report["protected_region_mse_vs_base_proxy"] = float(
            np.mean((pred_final[pmask] - base[pmask]) ** 2)
        )
        report["protected_region_mse_vs_base_gather"] = float(
            np.mean((gather_final[pmask] - base[pmask]) ** 2)
        )
    report["_images"] = {
        "proxy": pred_final.reshape(cache.height, cache.width, 3),
        "gather": gather_final.reshape(cache.height, cache.width, 3),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-light", help="JSON spec of the hidden true light (M5)")
    parser.add_argument("--target", help=".npy (H,W,3) painted/generated target image (M6)")
    parser.add_argument("--mask", help=".npy (H,W) objective weight mask")
    parser.add_argument("--protect", help=".npy (H,W) protected-region constraint mask (M7)")
    parser.add_argument("--protect-base", help=".npy (H,W,3) protected-region base image")
    parser.add_argument("--protect-lambda", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-2)
    parser.add_argument("--seed", type=int, default=0, help="random initial light parameters")
    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
        help="random restarts (seeds seed..seed+N-1); the run with the lowest final "
        "proxy loss wins — the optimization landscape has real local minima "
        "(observed empirically: some seeds stall at ~20x the best seed's loss)",
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if bool(args.target_light) == bool(args.target):
        parser.error("exactly one of --target-light / --target is required")

    model = ProxyMLP.load(args.model)
    cache = PathCache.load(args.cache)

    bounds = {
        "center_min": np.array([0.1, 0.1, 0.15]),
        "center_max": np.array([0.9, 0.9, 0.9]),
        "radius_min": 0.03,
        "radius_max": 0.3,
    }
    true_light = None
    if args.target_light:
        try:
            spec = json.loads(args.target_light)
        except json.JSONDecodeError:
            with open(args.target_light) as f:
                spec = json.load(f)
        true_light = SphereLight.from_dict(spec)
        target = gather_light(cache, true_light)
    else:
        target = np.load(args.target)

    weight_mask = np.load(args.mask) if args.mask else None
    protect_mask = np.load(args.protect) if args.protect else None
    protect_base = np.load(args.protect_base) if args.protect_base else None

    report = None
    init = None
    restart_summary = []
    for restart in range(args.restarts):
        rng = np.random.default_rng(args.seed + restart)
        candidate_init = {
            "center": bounds["center_min"]
            + rng.random(3) * (bounds["center_max"] - bounds["center_min"]),
            "radius": float(
                bounds["radius_min"] + rng.random() * (bounds["radius_max"] - bounds["radius_min"])
            ),
            "rgb": 0.5 + rng.random(3),
        }
        candidate = optimize(
            model,
            cache,
            target,
            candidate_init,
            bounds,
            args.steps,
            args.lr,
            weight_mask=weight_mask,
            protect_mask=protect_mask,
            protect_base=protect_base,
            protect_lambda=args.protect_lambda,
        )
        restart_summary.append(
            {"seed": args.seed + restart, "proxy_loss_last": candidate["proxy_loss_last"]}
        )
        if report is None or candidate["proxy_loss_last"] < report["proxy_loss_last"]:
            report, init = candidate, candidate_init
    images = report.pop("_images")
    report["restarts"] = restart_summary
    report["initial_light"] = {
        "center": init["center"].tolist(),
        "radius": init["radius"],
        "rgb": np.asarray(init["rgb"]).tolist(),
    }
    if true_light is not None:
        opt_light = report["optimized_light"]
        report["true_light"] = true_light.to_dict()
        report["center_error"] = float(
            np.linalg.norm(np.array(opt_light["center"]) - true_light.center)
        )
        report["radius_error"] = float(abs(opt_light["radius"] - true_light.radius))
        report["rgb_error"] = float(np.linalg.norm(np.array(opt_light["rgb"]) - true_light.rgb))

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "optimized_proxy.npy"), images["proxy"])
    np.save(os.path.join(args.out_dir, "optimized_gather.npy"), images["gather"])
    with open(os.path.join(args.out_dir, "optimize_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "proxy_loss_curve"}, indent=2))
    print(f"wrote {args.out_dir}/optimize_report.json and optimized images")


if __name__ == "__main__":
    main()
