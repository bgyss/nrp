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

Long runs (roadmap item 6) add two optional config keys. `lr_schedule: "cosine"`
decays the LR from `lr` to `lr_min` (default lr/100) over `iters` with cosine
annealing. `checkpoint: {"every": N}` saves `out_dir/checkpoint.pt` every N
iterations with the *full* training state — model, optimizer, scheduler, both RNGs,
and the image pool — so `--resume` continues the exact trajectory the uninterrupted
run would have taken (bit-identical on CPU; the unit suite asserts the loss curves
match). Each checkpoint also evaluates held-out PSNR on a fixed validation light
set (drawn from a dedicated RNG so it never perturbs pool sampling), which is what
produces the PSNR-vs-iteration convergence curve in the report.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from ..gather_light import gather_light
from ..lights import TexturedQuadLight
from ..metrics import flip, psnr, smape, ssim
from ..path_cache import PathCache
from ..train import ensure_cache, load_config
from .denoise import denoise_image
from .device import autocast, resolve_device, resolve_precision
from .gather import TorchPathCache
from .model import LIGHT_PARAM_DIMS, TorchNRP, relative_mse_loss
from .sampling import sample_light


def light_param_vector(light) -> np.ndarray:
    if isinstance(light, TexturedQuadLight):
        return np.concatenate(
            [
                light.center,
                light.normal,
                [light.width, light.height],
                light.texture.reshape(-1),
            ]
        )
    if hasattr(light, "radius"):
        return np.concatenate([light.center, [light.radius]])
    return np.concatenate([light.center, light.normal, [light.width], [light.height]])


def light_param_dim_from_cfg(cfg: dict) -> int:
    if cfg["light_type"] == "textured_quad":
        tex_h, tex_w = cfg["light_bounds"].get("texture_size", [2, 2])
        return 8 + int(tex_h) * int(tex_w) * 3
    return LIGHT_PARAM_DIMS[cfg["light_type"]]


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

    def __init__(self, cache: PathCache, cfg: dict, rng: np.random.Generator, device, fill=True):
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
            (self.size, light_param_dim_from_cfg(cfg)), dtype=torch.float32, device=device
        )
        self.targets = torch.empty((self.size, n_px, 3), dtype=torch.float32, device=device)
        self._next_replace = 0
        # Supervision accounting (roadmap item 9): every light configuration ever
        # rendered into the pool, and the cumulative gather+denoise cost of doing so.
        self.used_params: list[np.ndarray] = []
        self.supervision_seconds = 0.0
        if fill:
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
        t0 = time.perf_counter()
        light = sample_light(
            self.cache,
            self.rng,
            self.cfg["light_type"],
            self.cfg["light_bounds"],
            self.cfg.get("sampling", "segments"),
        )
        vec = light_param_vector(light)
        self.used_params.append(vec)
        self.params[slot] = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        self.targets[slot] = torch.as_tensor(
            self._render_target(light), dtype=torch.float32, device=self.device
        )
        self.supervision_seconds += time.perf_counter() - t0

    @property
    def supervision_images(self) -> int:
        return len(self.used_params)

    def replace_round(self) -> None:
        for _ in range(self.cfg["pool"]["replace_count"]):
            self.fill(self._next_replace)
            self._next_replace = (self._next_replace + 1) % self.size


def build_val_set(cache: PathCache, cfg: dict) -> list[dict]:
    """Fixed held-out validation set: fresh light configurations from a dedicated RNG
    (never the training RNG, so evaluating cannot perturb pool sampling), each with
    its raw GATHERLIGHT reference (physically grounded) and the denoised one (what
    the network is supervised with), computed once and reused at every checkpoint."""
    val_rng = np.random.default_rng([cfg.get("seed", 0), 0x5EED])
    val_set = []
    for _ in range(cfg.get("n_val_lights", 12)):
        light = sample_light(
            cache, val_rng, cfg["light_type"], cfg["light_bounds"], cfg.get("sampling", "segments")
        )
        raw = gather_light(cache, light).reshape(-1, 3)
        den = denoise_image(
            raw.reshape(cache.height, cache.width, 3),
            cache.albedo,
            cache.normal,
            cache.depth,
            method=cfg.get("denoise", {}).get("method", "bilateral"),
        ).reshape(-1, 3)
        val_set.append({"light": light, "raw": raw, "denoised": den})
    return val_set


def _tonemap_srgb(img: np.ndarray) -> np.ndarray:
    """Reinhard + gamma 1/2.2 tonemap of an HDR image into [0,1] sRGB-ish space,
    the display-referred domain SSIM (data_range=1) and FLIP are defined over."""
    img = np.maximum(np.asarray(img, dtype=np.float64), 0.0)
    return np.clip((img / (1.0 + img)) ** (1.0 / 2.2), 0.0, 1.0)


def evaluate(model, val_set, xy, aux, device, hw: tuple[int, int] | None = None) -> list[dict]:
    model.eval()
    metrics = []
    n_px = xy.shape[0]
    with torch.no_grad():
        for entry in val_set:
            params = torch.as_tensor(
                light_param_vector(entry["light"]), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            pred = model(xy, aux, params).cpu().numpy().astype(np.float64)
            light = entry["light"]
            m = {
                "light": light.to_dict() if hasattr(light, "to_dict") else None,
                "psnr_db_vs_raw": psnr(pred, entry["raw"]),
                "smape_vs_raw": smape(pred, entry["raw"]),
                "psnr_db_vs_denoised": psnr(pred, entry["denoised"]),
                "smape_vs_denoised": smape(pred, entry["denoised"]),
            }
            if hw is not None:
                h, w = hw
                pred_tm = _tonemap_srgb(pred.reshape(h, w, 3))
                raw_tm = _tonemap_srgb(entry["raw"].reshape(h, w, 3))
                m["ssim_vs_raw"] = ssim(pred_tm, raw_tm, data_range=1.0)
                m["flip_vs_raw"] = flip(pred_tm, raw_tm)
            metrics.append(m)
    model.train()
    return metrics


def save_checkpoint(path, iteration, model, opt, sched, gen, rng, pool, state) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict() if sched is not None else None,
            "torch_gen": gen.get_state(),
            "numpy_rng": rng.bit_generator.state,
            "pool_params": pool.params.detach().cpu(),
            "pool_targets": pool.targets.detach().cpu(),
            "pool_next_replace": pool._next_replace,
            "pool_used_params": pool.used_params,
            "pool_supervision_seconds": pool.supervision_seconds,
            **state,
        },
        path,
    )


def train(cfg: dict, resume: bool = False) -> dict:
    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    device = resolve_device(cfg.get("device"))
    precision = resolve_precision(cfg.get("precision"))
    cache = ensure_cache(cfg)
    n_px = cache.height * cache.width
    xy, aux = pixel_tensors(cache, device)

    model = TorchNRP(
        light_type=cfg["light_type"],
        light_param_dim=light_param_dim_from_cfg(cfg),
        hidden_width=cfg["model"].get("hidden_width", 128),
        hidden_layers=cfg["model"].get("hidden_layers", 4),
        encoding=cfg["model"].get("encoding"),
        use_encoding=cfg["model"].get("use_encoding", True),
        use_aux=cfg["model"].get("use_aux", True),
        texture_kernel=cfg["model"].get("texture_conditioning") == "kernel",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-2))
    iters = cfg["iters"]
    sched = None
    if cfg.get("lr_schedule") == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=iters, eta_min=cfg.get("lr_min", cfg.get("lr", 1e-2) / 100.0)
        )

    ckpt_cfg = cfg.get("checkpoint")
    ckpt_path = os.path.join(cfg["out_dir"], "checkpoint.pt")
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0))
    val_set = build_val_set(cache, cfg)

    start_iter = 0
    loss_curve: list[float] = []
    checkpoint_metrics: list[dict] = []
    pool_seconds = 0.0
    train_seconds = 0.0
    if resume:
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        start_iter = ck["iteration"]
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if sched is not None:
            sched.load_state_dict(ck["sched"])
        gen.set_state(ck["torch_gen"])
        rng.bit_generator.state = ck["numpy_rng"]
        pool = ImagePool(cache, cfg, rng, device, fill=False)
        pool.params = ck["pool_params"].to(device)
        pool.targets = ck["pool_targets"].to(device)
        pool._next_replace = ck["pool_next_replace"]
        pool.used_params = ck.get("pool_used_params", [])
        pool.supervision_seconds = ck.get("pool_supervision_seconds", 0.0)
        loss_curve = ck["loss_curve"]
        checkpoint_metrics = ck["checkpoint_metrics"]
        pool_seconds = ck["pool_seconds"]
        train_seconds = ck["train_seconds"]
        print(f"resumed from {ckpt_path} at iteration {start_iter}")
    else:
        t_pool0 = time.perf_counter()
        pool = ImagePool(cache, cfg, rng, device)
        pool_seconds = time.perf_counter() - t_pool0
        # H1 fix (docs/hardening-track.md): start the output head near the pool's
        # actual target scale instead of nn.Linear's default ~0.69, which is far
        # brighter than typical QuadLight targets on some caches and otherwise
        # drives a zero-collapse (see TorchNRP.init_output_scale docstring).
        model.init_output_scale(
            float(pool.targets.mean(dim=-1).median().item()),
            mean_texture_value=(
                float(pool.params[:, 8:].mean().item()) if model.texture_kernel else None
            ),
        )

    batch = cfg.get("batch_pixels", 4096)
    replace_every = cfg["pool"]["replace_every"]
    t_train0 = time.perf_counter()
    # S5/S6 precision lever: forward + loss run under torch.autocast for
    # fp16/bf16; fp32 keeps the eager path (autocast() is a no-op context then).
    # fp16 on CUDA additionally needs loss scaling (GradScaler); bf16 and the
    # fp32 default do not.
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" and device.type == "cuda" else None
    # S5 torch.compile lever: the compiled wrapper is used only for training
    # forwards; `model` (shared parameters) keeps handling checkpointing,
    # evaluation, and save, so the on-disk format is identical to eager runs.
    run_model = (
        torch.compile(model, **cfg.get("compile_options", {})) if cfg.get("compile") else model
    )
    for it in range(start_iter, iters):
        pool_ids = torch.randint(0, pool.size, (batch,), generator=gen).to(device)
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen).to(device)
        with autocast(device, precision):
            pred = run_model(xy[pixel_ids], aux[pixel_ids], pool.params[pool_ids])
            loss = relative_mse_loss(pred, pool.targets[pool_ids, pixel_ids])
        opt.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if sched is not None:
            sched.step()
        loss_curve.append(loss.detach().item())
        if (it + 1) % replace_every == 0:
            pool.replace_round()
        if ckpt_cfg is not None and (it + 1) % ckpt_cfg["every"] == 0:
            # Checkpoint evaluation and I/O are excluded from the training clock.
            train_seconds += time.perf_counter() - t_train0
            metrics = evaluate(model, val_set, xy, aux, device)
            checkpoint_metrics.append(
                {
                    "iteration": it + 1,
                    "val_psnr_db_vs_raw_mean": float(
                        np.mean([m["psnr_db_vs_raw"] for m in metrics])
                    ),
                    "val_smape_vs_raw_mean": float(np.mean([m["smape_vs_raw"] for m in metrics])),
                    "train_seconds": train_seconds,
                }
            )
            os.makedirs(cfg["out_dir"], exist_ok=True)
            save_checkpoint(
                ckpt_path,
                it + 1,
                model,
                opt,
                sched,
                gen,
                rng,
                pool,
                {
                    "loss_curve": loss_curve,
                    "checkpoint_metrics": checkpoint_metrics,
                    "pool_seconds": pool_seconds,
                    "train_seconds": train_seconds,
                },
            )
            c = checkpoint_metrics[-1]
            print(
                f"[{it + 1}/{iters}] loss {loss_curve[-1]:.4f} "
                f"val PSNR {c['val_psnr_db_vs_raw_mean']:.2f} dB "
                f"({train_seconds:.0f}s train)"
            )
            t_train0 = time.perf_counter()
    train_seconds += time.perf_counter() - t_train0

    val_metrics = evaluate(model, val_set, xy, aux, device, hw=(cache.height, cache.width))
    model.eval()

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
        "supervision_images": pool.supervision_images,
        "supervision_seconds": pool.supervision_seconds,
        "train_seconds": train_seconds,
        "iters_per_second": iters / train_seconds if train_seconds > 0 else None,
        "checkpoint_metrics": checkpoint_metrics,
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
        # SSIM/FLIP are computed on Reinhard+gamma-tonemapped images (FLIP is
        # defined over display-referred sRGB in [0,1]); data_range=1 for SSIM.
        "val_ssim_vs_raw_mean": float(np.mean([m["ssim_vs_raw"] for m in val_metrics])),
        "val_flip_vs_raw_mean": float(np.mean([m["flip_vs_raw"] for m in val_metrics])),
    }
    if cfg.get("record_supervision_lights"):
        # Opt-in (roadmap item 9): the exact light-parameter vectors of every
        # supervision image, so experiments can assert validation disjointness.
        report["supervision_light_params"] = [v.tolist() for v in pool.used_params]
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
    parser.add_argument(
        "--precision",
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="override the config's precision (autocast dtype; fp32 = eager default)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from out_dir/checkpoint.pt (requires a 'checkpoint' config block)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.gather_backend is not None:
        cfg["gather_backend"] = args.gather_backend
    if args.device is not None:
        cfg["device"] = args.device
    if args.precision is not None:
        cfg["precision"] = args.precision
    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
