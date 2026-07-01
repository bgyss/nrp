"""Paper-faithful inverse lighting through the torch NRP (§5.3).

Optimizes N sphere lights' (center, radius, rgb) to match a target image:
- Predicted image: I(px) = sum_i E_i * N_sphere(px, l_i, r_i)   (Eq. 5),
  accumulated sequentially per light.
- Loss: MSE on Reinhard-tonemapped values, T(I) = I / (1 + I)   (Eq. 6).
- Reparameterization to unconstrained space: center and radius through the logit of
  their bounded domains (recovered via sigmoid), color through inverse softplus from
  R>0 (recovered via softplus). Adam runs in unconstrained space (paper defaults:
  lr 0.05, 500 iterations).
- Mini-batch SGD: --pixel-fraction alpha evaluates the loss on a random pixel subset of
  size floor(alpha * H * W) each iteration, drawn without replacement (Table 3).

Target modes and masks mirror the numpy backend: --target-light JSON (single spec or
list; the target is its GATHERLIGHT re-render), or --target FILE.npy; --mask weights
the objective, --protect/--protect-base constrain protected regions (both tonemapped).
The optimized configuration is always re-rendered through reference GATHERLIGHT so
proxy-space and physically-gathered errors are reported separately.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from ..gather_light import gather_lights
from ..lights import SphereLight, light_from_dict
from ..metrics import psnr, smape
from ..path_cache import PathCache
from .model import TorchNRP, sphere_params
from .train import pixel_tensors

DEFAULT_BOUNDS = {
    "center_min": [0.1, 0.1, 0.15],
    "center_max": [0.9, 0.9, 0.9],
    "radius_min": 0.03,
    "radius_max": 0.3,
}


def reinhard(x: torch.Tensor) -> torch.Tensor:
    return x / (1.0 + x)


def logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1.0 - x))


def inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x))


class ReparamSphereLights:
    """N sphere lights in unconstrained space (logit position/radius, inv-softplus rgb)."""

    def __init__(self, init: list[dict], bounds: dict, device: torch.device):
        self.lo = torch.tensor(
            list(bounds["center_min"]) + [bounds["radius_min"]], dtype=torch.float32, device=device
        )
        self.hi = torch.tensor(
            list(bounds["center_max"]) + [bounds["radius_max"]], dtype=torch.float32, device=device
        )
        geo = torch.tensor(
            [list(light["center"]) + [light["radius"]] for light in init],
            dtype=torch.float32,
            device=device,
        )
        frac = ((geo - self.lo) / (self.hi - self.lo)).clamp(1e-4, 1.0 - 1e-4)
        rgb = torch.tensor([light["rgb"] for light in init], dtype=torch.float32, device=device)
        self.u_geo = logit(frac).requires_grad_(True)
        self.u_rgb = inv_softplus(rgb.clamp(min=1e-4)).requires_grad_(True)

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.u_geo, self.u_rgb]

    def constrained(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        geo = self.lo + torch.sigmoid(self.u_geo) * (self.hi - self.lo)
        return geo[:, :3], geo[:, 3], torch.nn.functional.softplus(self.u_rgb)

    def to_lights(self) -> list[SphereLight]:
        centers, radii, rgbs = self.constrained()
        return [
            SphereLight(
                center=c.detach().cpu().numpy().astype(np.float64),
                radius=float(r),
                rgb=e.detach().cpu().numpy().astype(np.float64),
            )
            for c, r, e in zip(centers, radii, rgbs, strict=True)
        ]


def predicted_image(
    model: TorchNRP, lights: ReparamSphereLights, xy: torch.Tensor, aux: torch.Tensor
) -> torch.Tensor:
    """Eq. 5 over a pixel subset: sequential accumulation over lights."""
    n = xy.shape[0]
    centers, radii, rgbs = lights.constrained()
    image = torch.zeros((n, 3), device=xy.device)
    for i in range(centers.shape[0]):
        image = image + model(xy, aux, sphere_params(centers[i], radii[i], n)) * rgbs[i]
    return image


def optimize(
    model: TorchNRP,
    cache: PathCache,
    target: np.ndarray,
    init: list[dict],
    bounds: dict,
    steps: int,
    lr: float,
    pixel_fraction: float = 1.0,
    weight_mask: np.ndarray | None = None,
    protect_mask: np.ndarray | None = None,
    protect_base: np.ndarray | None = None,
    protect_lambda: float = 10.0,
    seed: int = 0,
) -> dict:
    device = next(model.parameters()).device
    n_px = cache.height * cache.width
    xy, aux = pixel_tensors(cache, device)
    tgt = torch.as_tensor(target.reshape(n_px, 3), dtype=torch.float32, device=device)
    w = None
    if weight_mask is not None:
        w = torch.as_tensor(weight_mask.reshape(n_px, 1), dtype=torch.float32, device=device)
    prot = base = None
    if protect_mask is not None:
        prot = torch.as_tensor(protect_mask.reshape(n_px, 1), dtype=torch.float32, device=device)

    lights = ReparamSphereLights(init, bounds, device)
    if prot is not None:
        if protect_base is None:
            with torch.no_grad():
                base = predicted_image(model, lights, xy, aux)
        else:
            base = torch.as_tensor(
                protect_base.reshape(n_px, 3), dtype=torch.float32, device=device
            )

    opt = torch.optim.Adam(lights.parameters, lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = max(int(pixel_fraction * n_px), 1)
    loss_curve = []
    for _step in range(steps):
        idx = torch.randperm(n_px, generator=gen)[:k].to(device) if k < n_px else slice(None)
        pred = predicted_image(model, lights, xy[idx], aux[idx])
        diff = reinhard(pred) - reinhard(tgt[idx])
        if w is not None:
            wk = w[idx]
            loss = (wk * diff**2).sum() / (wk.sum() * 3.0).clamp(min=1.0)
        else:
            loss = (diff**2).mean()
        if prot is not None:
            pk, bk = prot[idx], base[idx]
            pdiff = reinhard(pred) - reinhard(bk)
            loss = loss + protect_lambda * (pk * pdiff**2).sum() / (pk.sum() * 3.0).clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(loss.detach().item())

    result_lights = lights.to_lights()
    with torch.no_grad():
        pred_final = predicted_image(model, lights, xy, aux).cpu().numpy().astype(np.float64)
    gather_final = gather_lights(cache, result_lights).reshape(n_px, 3)
    target_flat = target.reshape(n_px, 3)

    report = {
        "optimized_lights": [light.to_dict() for light in result_lights],
        "steps": steps,
        "pixel_fraction": pixel_fraction,
        "proxy_loss_first": loss_curve[0],
        "proxy_loss_last": loss_curve[-1],
        "proxy_loss_curve": loss_curve[:: max(1, steps // 50)],
        "proxy_vs_target_psnr_db": psnr(pred_final, target_flat),
        "proxy_vs_target_smape": smape(pred_final, target_flat),
        "gather_vs_target_psnr_db": psnr(gather_final, target_flat),
        "gather_vs_target_smape": smape(gather_final, target_flat),
        "_images": {
            "proxy": pred_final.reshape(cache.height, cache.width, 3),
            "gather": gather_final.reshape(cache.height, cache.width, 3),
        },
    }
    if prot is not None:
        pmask = prot.cpu().numpy()[:, 0] > 0.0
        base_np = base.cpu().numpy().astype(np.float64)
        report["protected_region_pixels"] = int(pmask.sum())
        report["protected_region_mse_vs_base_proxy"] = float(
            np.mean((pred_final[pmask] - base_np[pmask]) ** 2)
        )
        report["protected_region_mse_vs_base_gather"] = float(
            np.mean((gather_final[pmask] - base_np[pmask]) ** 2)
        )
    return report


def _load_spec(arg: str):
    try:
        return json.loads(arg)
    except json.JSONDecodeError:
        with open(arg) as f:
            return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="trained torch model .pt (sphere)")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--target-light", help="JSON: hidden true light spec, or a list of specs")
    parser.add_argument("--target", help=".npy (H,W,3) painted/generated target image")
    parser.add_argument(
        "--n-lights",
        type=int,
        default=None,
        help="lights to optimize (default: match target lights, else 1)",
    )
    parser.add_argument("--mask", help=".npy (H,W) objective weight mask")
    parser.add_argument("--protect", help=".npy (H,W) protected-region constraint mask")
    parser.add_argument("--protect-base", help=".npy (H,W,3) protected-region base image")
    parser.add_argument("--protect-lambda", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pixel-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--restarts", type=int, default=1, help="random restarts; lowest final loss wins"
    )
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    if bool(args.target_light) == bool(args.target):
        parser.error("exactly one of --target-light / --target is required")

    model = TorchNRP.load(args.model)
    if model.light_type != "sphere":
        parser.error("inverse optimization currently supports sphere-light models (paper §5.3)")
    cache = PathCache.load(args.cache)
    bounds = DEFAULT_BOUNDS

    true_lights = None
    if args.target_light:
        spec = _load_spec(args.target_light)
        specs = spec if isinstance(spec, list) else [spec]
        true_lights = [light_from_dict(s) for s in specs]
        target = gather_lights(cache, true_lights)
    else:
        target = np.load(args.target)
    n_lights = args.n_lights or (len(true_lights) if true_lights else 1)

    weight_mask = np.load(args.mask) if args.mask else None
    protect_mask = np.load(args.protect) if args.protect else None
    protect_base = np.load(args.protect_base) if args.protect_base else None

    best = None
    best_init = None
    restart_summary = []
    lo = np.asarray(bounds["center_min"])
    hi = np.asarray(bounds["center_max"])
    for restart in range(args.restarts):
        rng = np.random.default_rng(args.seed + restart)
        init = [
            {
                "center": (lo + rng.random(3) * (hi - lo)).tolist(),
                "radius": float(
                    bounds["radius_min"]
                    + rng.random() * (bounds["radius_max"] - bounds["radius_min"])
                ),
                "rgb": (0.5 + rng.random(3)).tolist(),
            }
            for _ in range(n_lights)
        ]
        candidate = optimize(
            model,
            cache,
            target,
            init,
            bounds,
            args.steps,
            args.lr,
            pixel_fraction=args.pixel_fraction,
            weight_mask=weight_mask,
            protect_mask=protect_mask,
            protect_base=protect_base,
            protect_lambda=args.protect_lambda,
            seed=args.seed + restart,
        )
        restart_summary.append(
            {"seed": args.seed + restart, "proxy_loss_last": candidate["proxy_loss_last"]}
        )
        if best is None or candidate["proxy_loss_last"] < best["proxy_loss_last"]:
            best, best_init = candidate, init

    images = best.pop("_images")
    best["restarts"] = restart_summary
    best["initial_lights"] = best_init
    if true_lights is not None and len(true_lights) == n_lights and n_lights == 1:
        true = true_lights[0]
        opt_light = best["optimized_lights"][0]
        best["true_light"] = true.to_dict()
        best["center_error"] = float(np.linalg.norm(np.array(opt_light["center"]) - true.center))
        best["radius_error"] = float(abs(opt_light["radius"] - true.radius))
        best["rgb_error"] = float(np.linalg.norm(np.array(opt_light["rgb"]) - true.rgb))

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "optimized_proxy.npy"), images["proxy"])
    np.save(os.path.join(args.out_dir, "optimized_gather.npy"), images["gather"])
    with open(os.path.join(args.out_dir, "torch_optimize_report.json"), "w") as f:
        json.dump(best, f, indent=2)
    print(json.dumps({k: v for k, v in best.items() if k != "proxy_loss_curve"}, indent=2))
    print(f"wrote {args.out_dir}/torch_optimize_report.json and optimized images")


if __name__ == "__main__":
    main()
