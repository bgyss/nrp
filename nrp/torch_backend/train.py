"""Train the torch NRP with the paper's pool-of-denoised-images scheme (§4.4).

Because the denoiser needs a full image under a single light, per-pixel light diversity
comes from a pool: the pool holds `pool.size` denoised GATHERLIGHT images (each for one
random light configuration), every training pixel samples its target uniformly from the
pool, and `pool.replace_count` images are replaced with fresh configurations every
`pool.replace_every` iterations (paper: pool 300, replace 2 every 5). Light positions
are sampled on recorded path segments or in the visible bbox (§4.4); the loss is the
relative MSE of Eq. 4 with a stop-gradient denominator.

Config-driven CLI (see examples/toy_sphere_torch.json). Paths resolve relative to the
config file; the path cache is traced first if missing. Outputs into `out_dir`:
model.pt and torch_train_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..gather_light import gather_light
from ..metrics import psnr, smape
from ..path_cache import PathCache
from ..train import ensure_cache, load_config
from .denoise import denoise_image
from .gather import TorchPathCache
from .model import LIGHT_PARAM_DIMS, TorchNRP, relative_mse_loss
from .sampling import sample_light


def light_param_vector(light) -> np.ndarray:
    if hasattr(light, "radius"):
        return np.concatenate([light.center, [light.radius]])
    return np.concatenate([light.center, light.normal, [light.width], [light.height]])


def pixel_tensors(cache: PathCache, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """((N,2) pixel xy in [0,1]^2, (N,7) aux features albedo+depth+normal)."""
    h, w = cache.height, cache.width
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    aux = np.concatenate(
        [cache.albedo.reshape(-1, 3), cache.depth.reshape(-1, 1), cache.normal.reshape(-1, 3)],
        axis=1,
    )
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return to(xy), to(aux)


class ImagePool:
    """Pool of (light params, denoised target image) rows, with periodic replacement."""

    def __init__(self, cache: PathCache, cfg: dict, rng: np.random.Generator, device):
        self.cache = cache
        self.cfg = cfg
        self.rng = rng
        self.device = device
        # gather_backend "torch" builds pool targets with the batched device gather
        # (nrp/torch_backend/gather.py); "numpy" (default) keeps the reference path.
        self.torch_cache = (
            TorchPathCache(cache, device) if cfg.get("gather_backend", "numpy") == "torch" else None
        )
        self.size = cfg["pool"]["size"]
        n_px = cache.height * cache.width
        self.params = torch.empty(
            (self.size, LIGHT_PARAM_DIMS[cfg["light_type"]]), dtype=torch.float32, device=device
        )
        self.targets = torch.empty((self.size, n_px, 3), dtype=torch.float32, device=device)
        self._next_replace = 0
        for i in range(self.size):
            self.fill(i)

    def _render_target(self, light) -> np.ndarray:
        # Unit emission: the pre-emission contribution the proxy learns.
        if self.torch_cache is not None:
            image = self.torch_cache.gather_light(light).cpu().numpy().astype(np.float64)
        else:
            image = gather_light(self.cache, light)
        dn = self.cfg.get("denoise", {})
        if dn.get("enabled", True):
            image = denoise_image(
                image,
                self.cache.albedo,
                self.cache.normal,
                self.cache.depth,
                method=dn.get("method", "bilateral"),
                **{k: v for k, v in dn.items() if k not in ("enabled", "method")},
            )
        return image.reshape(-1, 3)

    def fill(self, slot: int) -> None:
        light = sample_light(
            self.cache,
            self.rng,
            self.cfg["light_type"],
            self.cfg["light_bounds"],
            self.cfg.get("sampling", "segments"),
        )
        self.params[slot] = torch.as_tensor(
            light_param_vector(light), dtype=torch.float32, device=self.device
        )
        self.targets[slot] = torch.as_tensor(
            self._render_target(light), dtype=torch.float32, device=self.device
        )

    def replace_round(self) -> None:
        for _ in range(self.cfg["pool"]["replace_count"]):
            self.fill(self._next_replace)
            self._next_replace = (self._next_replace + 1) % self.size


def train(cfg: dict) -> dict:
    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))
    cache = ensure_cache(cfg)
    n_px = cache.height * cache.width
    xy, aux = pixel_tensors(cache, device)

    model = TorchNRP(
        light_type=cfg["light_type"],
        hidden_width=cfg["model"].get("hidden_width", 128),
        hidden_layers=cfg["model"].get("hidden_layers", 4),
        encoding=cfg["model"].get("encoding"),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-2))

    t_pool0 = time.perf_counter()
    pool = ImagePool(cache, cfg, rng, device)
    pool_seconds = time.perf_counter() - t_pool0

    iters = cfg["iters"]
    batch = cfg.get("batch_pixels", 4096)
    replace_every = cfg["pool"]["replace_every"]
    loss_curve = []
    t_train0 = time.perf_counter()
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0))
    for it in range(iters):
        pool_ids = torch.randint(0, pool.size, (batch,), generator=gen).to(device)
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen).to(device)
        pred = model(xy[pixel_ids], aux[pixel_ids], pool.params[pool_ids])
        loss = relative_mse_loss(pred, pool.targets[pool_ids, pixel_ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(loss.detach().item())
        if (it + 1) % replace_every == 0:
            pool.replace_round()
    train_seconds = time.perf_counter() - t_train0

    # Held-out validation: fresh configurations, never in the pool; reference is the
    # *raw* GATHERLIGHT estimate (the physically grounded target, per the numpy backend)
    # and the denoised one (what the network was actually supervised with).
    model.eval()
    val_metrics = []
    with torch.no_grad():
        for _ in range(cfg.get("n_val_lights", 12)):
            light = sample_light(
                cache, rng, cfg["light_type"], cfg["light_bounds"], cfg.get("sampling", "segments")
            )
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
            raw = gather_light(cache, light).reshape(n_px, 3)
            den = denoise_image(
                raw.reshape(cache.height, cache.width, 3),
                cache.albedo,
                cache.normal,
                cache.depth,
                method=cfg.get("denoise", {}).get("method", "bilateral"),
            ).reshape(n_px, 3)
            val_metrics.append(
                {
                    "light": light.to_dict() if hasattr(light, "to_dict") else None,
                    "psnr_db_vs_raw": psnr(pred, raw),
                    "smape_vs_raw": smape(pred, raw),
                    "psnr_db_vs_denoised": psnr(pred, den),
                    "smape_vs_denoised": smape(pred, den),
                }
            )

    # Single-frame inference latency (full image forward, no gather).
    n_bench = 20
    fixed = pool.params[0].expand(n_px, -1)
    with torch.no_grad():
        t_inf0 = time.perf_counter()
        for _ in range(n_bench):
            model(xy, aux, fixed)
        inference_ms = (time.perf_counter() - t_inf0) / n_bench * 1000.0

    os.makedirs(cfg["out_dir"], exist_ok=True)
    model_path = os.path.join(cfg["out_dir"], "model.pt")
    model.save(model_path)
    report = {
        "config": {k: v for k, v in cfg.items() if k != "out_dir"},
        "resolution": [cache.width, cache.height],
        "path_cache_segments": cache.segment_count,
        "parameter_count": model.parameter_count,
        "model_bytes": os.path.getsize(model_path),
        "pool_build_seconds": pool_seconds,
        "train_seconds": train_seconds,
        "inference_ms_per_frame": inference_ms,
        "inference_hz": 1000.0 / inference_ms if inference_ms > 0 else None,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "loss_curve": loss_curve[:: max(1, iters // 100)],
        "val_lights": val_metrics,
        "val_psnr_db_vs_raw_mean": float(np.mean([m["psnr_db_vs_raw"] for m in val_metrics])),
        "val_smape_vs_raw_mean": float(np.mean([m["smape_vs_raw"] for m in val_metrics])),
        "val_psnr_db_vs_denoised_mean": float(
            np.mean([m["psnr_db_vs_denoised"] for m in val_metrics])
        ),
    }
    report_path = os.path.join(cfg["out_dir"], "torch_train_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(
        f"trained {model.parameter_count} params ({cfg['light_type']}) in {train_seconds:.1f}s "
        f"(+{pool_seconds:.1f}s pool); loss {loss_curve[0]:.4f} -> {loss_curve[-1]:.4f}; "
        f"held-out PSNR {report['val_psnr_db_vs_raw_mean']:.2f} dB vs raw "
        f"({report['val_psnr_db_vs_denoised_mean']:.2f} dB vs denoised), "
        f"SMAPE {report['val_smape_vs_raw_mean']:.4f}; "
        f"inference {inference_ms:.1f} ms/frame ({report['inference_hz']:.1f} Hz) "
        f"at {cache.width}x{cache.height}"
    )
    print(f"wrote {model_path} and {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--gather-backend",
        choices=["numpy", "torch"],
        default=None,
        help="pool-target gather implementation (overrides the config; numpy is the "
        "authoritative reference, torch runs batched on the training device)",
    )
    parser.add_argument(
        "--device", default=None, help="override the config's device (cpu/mps/cuda)"
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.gather_backend is not None:
        cfg["gather_backend"] = args.gather_backend
    if args.device is not None:
        cfg["device"] = args.device
    train(cfg)


if __name__ == "__main__":
    main()
