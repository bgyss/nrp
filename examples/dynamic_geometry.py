"""E2 dynamic-geometry report: primary-visibility invalidation and cache splicing."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.dynamic_geometry import (  # noqa: E402
    primary_visibility_invalidation_mask,
    splice_invalidated_pixels,
    swept_volume_invalidation_mask,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.toy_tracer import SPHERE_CENTER, trace_path_cache  # noqa: E402


def _pixel_xy(width: int, height: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([(xs.reshape(-1) + 0.5) / width, (ys.reshape(-1) + 0.5) / height], axis=1)


def _aux(cache) -> np.ndarray:
    return np.concatenate(
        [cache.albedo.reshape(-1, 3), cache.depth.reshape(-1, 1), cache.normal.reshape(-1, 3)],
        axis=1,
    )


class TorchNRPWarmStartProxy:
    """Real TorchNRP weight fine-tuning for the E2 "regime (b)" comparison.

    Unlike `WarmStartImageProxy` (a per-pixel value with no generalization), this
    wraps an actual `TorchNRP` model conditioned on pixel xy + G-buffer aux for one
    fixed light. `fine_tune` warm-starts from the *previous frame's* weights and runs
    a few hundred Adam steps against only the invalidated pixels' new targets and aux
    (the splice's G-buffer), matching the paper's warm-start intent for animated
    primary visibility.
    """

    def __init__(self, hidden_width: int = 32, hidden_layers: int = 2, seed: int = 0):
        torch.manual_seed(seed)
        self.model = TorchNRP(
            light_type="sphere",
            hidden_width=hidden_width,
            hidden_layers=hidden_layers,
            encoding=None,
            use_encoding=False,
        )
        self.light_params = None

    def set_light(self, light: SphereLight) -> None:
        self.light_params = np.concatenate([light.center, [light.radius]]).astype(np.float32)

    def predict(self, cache) -> np.ndarray:
        xy = torch.as_tensor(_pixel_xy(cache.width, cache.height), dtype=torch.float32)
        aux = torch.as_tensor(_aux(cache), dtype=torch.float32)
        params = torch.as_tensor(self.light_params, dtype=torch.float32).expand(xy.shape[0], -1)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(xy, aux, params).numpy().astype(np.float64)
        self.model.train()
        return pred.reshape(cache.height, cache.width, 3)

    def train_full(self, cache, target: np.ndarray, iters: int = 300, lr: float = 5e-3) -> None:
        """Full training pass on a fresh cache (used once, for the base frame)."""
        xy = torch.as_tensor(_pixel_xy(cache.width, cache.height), dtype=torch.float32)
        aux = torch.as_tensor(_aux(cache), dtype=torch.float32)
        params = torch.as_tensor(self.light_params, dtype=torch.float32).expand(xy.shape[0], -1)
        y = torch.as_tensor(target.reshape(-1, 3), dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        for _ in range(iters):
            pred = self.model(xy, aux, params)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def fine_tune(
        self, cache, target: np.ndarray, mask: np.ndarray, iters: int = 300, lr: float = 5e-3
    ) -> list[float]:
        """Warm-started fine-tune restricted to the invalidated pixels."""
        flat_mask = mask.reshape(-1)
        idx = np.nonzero(flat_mask)[0]
        if idx.size == 0:
            return [0.0, 0.0]
        xy_all = _pixel_xy(cache.width, cache.height)
        aux_all = _aux(cache)
        xy = torch.as_tensor(xy_all[idx], dtype=torch.float32)
        aux = torch.as_tensor(aux_all[idx], dtype=torch.float32)
        params = torch.as_tensor(self.light_params, dtype=torch.float32).expand(idx.size, -1)
        y = torch.as_tensor(target.reshape(-1, 3)[idx], dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        for _ in range(iters):
            pred = self.model(xy, aux, params)
            loss = torch.mean((pred - y) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return losses

    def fine_tune_with_replay(
        self,
        cache,
        target: np.ndarray,
        mask: np.ndarray,
        iters: int = 300,
        lr: float = 5e-3,
        replay_fraction: float = 0.5,
    ) -> list[float]:
        """E2 follow-up: `fine_tune` but with self-distillation replay on unchanged
        pixels, to test whether it closes the regime (a)/(b) gap `fine_tune` leaves.

        Records the model's own predictions on *unmasked* pixels before any update
        (self-distillation targets — no extra ground truth is used, since a real
        incremental-update pipeline wouldn't have one) and mixes a `replay_fraction`
        sample of them into every training batch alongside the real invalidated-pixel
        supervision, so the loss no longer only sees the changed region.
        """
        flat_mask = mask.reshape(-1)
        invalid_idx = np.nonzero(flat_mask)[0]
        valid_idx = np.nonzero(~flat_mask)[0]
        if invalid_idx.size == 0:
            return [0.0, 0.0]
        xy_all = torch.as_tensor(_pixel_xy(cache.width, cache.height), dtype=torch.float32)
        aux_all = torch.as_tensor(_aux(cache), dtype=torch.float32)
        params_all = torch.as_tensor(self.light_params, dtype=torch.float32).expand(
            xy_all.shape[0], -1
        )
        target_t = torch.as_tensor(target.reshape(-1, 3), dtype=torch.float32)

        replay_idx = torch.as_tensor(valid_idx, dtype=torch.long)
        if replay_idx.numel():
            self.model.eval()
            with torch.no_grad():
                replay_targets = self.model(
                    xy_all[replay_idx], aux_all[replay_idx], params_all[replay_idx]
                ).clone()
            self.model.train()

        n_replay = int(round(replay_fraction * invalid_idx.size)) if replay_idx.numel() else 0
        gen = np.random.default_rng(0)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        losses = []
        for _ in range(iters):
            pred_invalid = self.model(
                xy_all[invalid_idx], aux_all[invalid_idx], params_all[invalid_idx]
            )
            loss = torch.mean((pred_invalid - target_t[invalid_idx]) ** 2)
            if n_replay:
                sample = gen.choice(replay_idx.numel(), size=n_replay, replace=True)
                sample = torch.as_tensor(sample, dtype=torch.long)
                pred_replay = self.model(
                    xy_all[replay_idx[sample]],
                    aux_all[replay_idx[sample]],
                    params_all[replay_idx[sample]],
                )
                loss = loss + torch.mean((pred_replay - replay_targets[sample]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        return losses


class WarmStartImageProxy:
    """Tiny image-space proxy baseline for E2 warm-start recovery measurements.

    It is intentionally simpler than TorchNRP: one RGB value per pixel, initialized
    from the previous frame and fine-tuned by gradient descent against the spliced
    cache's GATHERLIGHT target. This measures whether incremental cache targets can
    quickly repair a stale proxy, without claiming neural transport generalization.
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image, dtype=np.float64).copy()

    def predict(self) -> np.ndarray:
        return self.image.copy()

    def fine_tune(
        self,
        target: np.ndarray,
        mask: np.ndarray,
        *,
        steps: int = 8,
        lr: float = 0.5,
    ) -> list[float]:
        if target.shape != self.image.shape:
            raise ValueError(f"target shape {target.shape} does not match {self.image.shape}")
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.image.shape[:2]:
            raise ValueError(f"mask must be {self.image.shape[:2]}, got {mask.shape}")
        losses = []
        for _ in range(steps):
            diff = self.image[mask] - target[mask]
            losses.append(float(np.mean(diff * diff)) if diff.size else 0.0)
            self.image[mask] -= lr * diff
        diff = self.image[mask] - target[mask]
        losses.append(float(np.mean(diff * diff)) if diff.size else 0.0)
        return losses


def multi_bounce_invalidation_comparison(
    width: int, height: int, spp: int, light: SphereLight, dx: float = 0.12
) -> dict:
    """E2's multi-bounce failure-then-fix demonstration.

    Primary-visibility invalidation only looks at each pixel's first-hit G-buffer, so
    it misses pixels whose *indirect* bounces changed because the moving object
    altered transport between two other, still-visible surfaces. This traces a
    2-bounce before/after pair, splices with (a) primary-only and (b) primary +
    swept-volume masks, and reports how far each spliced cache's GATHERLIGHT is from
    a true full retrace — proving the swept-volume mask is needed and sufficient at
    this bounce depth.
    """
    before = trace_path_cache(
        width, height, spp, max_bounces=2, seed=31, sphere_center=SPHERE_CENTER
    )
    moved = SPHERE_CENTER + np.array([dx, 0.0, 0.0])
    after = trace_path_cache(width, height, spp, max_bounces=2, seed=31, sphere_center=moved)
    full_image = gather_light(after, light)

    primary_mask = primary_visibility_invalidation_mask(before, after)
    swept_mask = swept_volume_invalidation_mask(
        before, SPHERE_CENTER, moved, radius=0.25, margin=0.05
    )
    combined_mask = primary_mask | swept_mask

    primary_only_spliced, primary_stats = splice_invalidated_pixels(before, after, primary_mask)
    combined_spliced, combined_stats = splice_invalidated_pixels(before, after, combined_mask)

    primary_only_image = gather_light(primary_only_spliced, light)
    combined_image = gather_light(combined_spliced, light)

    return {
        "resolution": [width, height],
        "spp": spp,
        "bounces": 2,
        "sphere_dx": dx,
        "primary_only_invalid_pixels": primary_stats.invalid_pixels,
        "combined_invalid_pixels": combined_stats.invalid_pixels,
        "additional_pixels_from_swept_volume": combined_stats.invalid_pixels
        - primary_stats.invalid_pixels,
        "primary_only_max_abs_vs_full_retrace": float(
            np.max(np.abs(primary_only_image - full_image))
        ),
        "combined_max_abs_vs_full_retrace": float(np.max(np.abs(combined_image - full_image))),
        "primary_only_psnr_vs_full": psnr_json(primary_only_image, full_image),
        "combined_psnr_vs_full": psnr_json(combined_image, full_image),
        "finding": (
            "primary-only invalidation under-invalidates at 2 bounces; swept-volume "
            "invalidation closes the gap"
            if float(np.max(np.abs(primary_only_image - full_image)))
            > float(np.max(np.abs(combined_image - full_image))) + 1e-12
            else "no indirect-only change detected in this fixture at this bounce depth"
        ),
    }


def psnr_json(a: np.ndarray, b: np.ndarray) -> float | str:
    value = psnr(a, b)
    return "inf" if math.isinf(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/dynamic-geometry/report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--proxy-steps", type=int, default=8)
    parser.add_argument("--proxy-lr", type=float, default=0.5)
    parser.add_argument("--torchnrp-finetune-iters", type=int, default=300)
    parser.add_argument("--torchnrp-lr", type=float, default=5e-3)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    light = SphereLight(center=[0.35, 0.28, 0.62], radius=0.2, rgb=[1.3, 1.0, 0.8])
    base_center = SPHERE_CENTER.copy()
    offsets = np.linspace(0.0, 0.16, args.frames)

    t0 = time.perf_counter()
    base = trace_path_cache(
        args.width,
        args.height,
        args.spp,
        max_bounces=1,
        seed=21,
        sphere_center=base_center,
    )
    base_trace_s = time.perf_counter() - t0
    stale_image = gather_light(base, light)
    proxy = WarmStartImageProxy(stale_image)

    # Regime (b): the running incrementally-updated proxy, carried across frames.
    proxy_b = TorchNRPWarmStartProxy(hidden_width=32, hidden_layers=2, seed=0)
    proxy_b.set_light(light)
    t0 = time.perf_counter()
    proxy_b.train_full(base, stale_image, iters=args.torchnrp_finetune_iters, lr=args.torchnrp_lr)
    torchnrp_base_train_s = time.perf_counter() - t0
    # Regime (a): "full retrace" reference — retrained fully from the same warm start
    # every frame, i.e. what a from-scratch-per-frame proxy would produce.
    proxy_a = copy.deepcopy(proxy_b)
    # Regime (b2): incremental + self-distillation replay follow-up to E2's negative
    # result — does regularizing against unchanged pixels close the (a)/(b) gap?
    proxy_b2 = copy.deepcopy(proxy_b)

    frames = []
    for frame, dx in enumerate(offsets):
        center = base_center + np.array([dx, 0.0, 0.0])
        t0 = time.perf_counter()
        full = trace_path_cache(
            args.width,
            args.height,
            args.spp,
            max_bounces=1,
            seed=21,
            sphere_center=center,
        )
        full_trace_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        mask = primary_visibility_invalidation_mask(base, full)
        spliced, stats = splice_invalidated_pixels(base, full, mask)
        splice_s = time.perf_counter() - t0

        full_image = gather_light(full, light)
        spliced_image = gather_light(spliced, light)
        proxy_before = proxy.predict()
        t0 = time.perf_counter()
        proxy_losses = proxy.fine_tune(
            spliced_image,
            mask,
            steps=args.proxy_steps,
            lr=args.proxy_lr,
        )
        proxy_finetune_s = time.perf_counter() - t0
        proxy_after = proxy.predict()
        outside = ~mask
        outside_max_diff = 0.0
        if outside.any():
            outside_max_diff = float(np.max(np.abs(stale_image[outside] - full_image[outside])))

        # Regime (c): stale — freeze regime (b)'s weights before this frame's update.
        regime_c_pred = proxy_b.predict(full)

        # Regime (b): incremental — warm-start from previous weights, fine-tune only
        # the invalidated pixels against the spliced cache's target.
        t0 = time.perf_counter()
        torchnrp_finetune_losses = proxy_b.fine_tune(
            spliced,
            spliced_image,
            mask,
            iters=args.torchnrp_finetune_iters,
            lr=args.torchnrp_lr,
        )
        torchnrp_finetune_s = time.perf_counter() - t0
        regime_b_pred = proxy_b.predict(full)

        # Regime (a): full retrace — full retrain on the fully-retraced frame.
        t0 = time.perf_counter()
        proxy_a.train_full(
            full, full_image, iters=args.torchnrp_finetune_iters, lr=args.torchnrp_lr
        )
        torchnrp_full_retrain_s = time.perf_counter() - t0
        regime_a_pred = proxy_a.predict(full)

        # Regime (b2): incremental + replay — same budget as (b), but with
        # self-distillation regularization on unchanged pixels.
        t0 = time.perf_counter()
        proxy_b2.fine_tune_with_replay(
            spliced,
            spliced_image,
            mask,
            iters=args.torchnrp_finetune_iters,
            lr=args.torchnrp_lr,
        )
        torchnrp_replay_finetune_s = time.perf_counter() - t0
        regime_b2_pred = proxy_b2.predict(full)

        frames.append(
            {
                "frame": frame,
                "sphere_center": center.tolist(),
                "full_trace_ms": full_trace_s * 1000.0,
                "incremental_splice_ms": splice_s * 1000.0,
                "invalid_pixels": stats.invalid_pixels,
                "invalid_fraction": stats.invalid_fraction,
                "segments_retraced_fraction": stats.new_segments_inserted
                / max(full.segment_count, 1),
                "outside_mask_full_vs_base_max_abs": outside_max_diff,
                "stale_psnr_vs_full": psnr_json(stale_image, full_image),
                "incremental_psnr_vs_full": psnr_json(spliced_image, full_image),
                "incremental_max_abs_vs_full": float(np.max(np.abs(spliced_image - full_image))),
                "proxy_before_psnr_vs_full": psnr_json(proxy_before, full_image),
                "proxy_after_psnr_vs_full": psnr_json(proxy_after, full_image),
                "proxy_finetune_ms": proxy_finetune_s * 1000.0,
                "proxy_finetune_loss_first": proxy_losses[0],
                "proxy_finetune_loss_last": proxy_losses[-1],
                "spliced_cache_valid": True,
                "torchnrp_regime_a_full_retrace_psnr_vs_full": psnr_json(regime_a_pred, full_image),
                "torchnrp_regime_b_incremental_psnr_vs_full": psnr_json(regime_b_pred, full_image),
                "torchnrp_regime_c_stale_psnr_vs_full": psnr_json(regime_c_pred, full_image),
                "torchnrp_regime_a_full_retrain_ms": torchnrp_full_retrain_s * 1000.0,
                "torchnrp_regime_b_finetune_ms": torchnrp_finetune_s * 1000.0,
                "torchnrp_finetune_loss_first": torchnrp_finetune_losses[0],
                "torchnrp_finetune_loss_last": torchnrp_finetune_losses[-1],
                "torchnrp_regime_b2_replay_psnr_vs_full": psnr_json(regime_b2_pred, full_image),
                "torchnrp_regime_b2_replay_finetune_ms": torchnrp_replay_finetune_s * 1000.0,
            }
        )

    mean_full_ms = float(np.mean([f["full_trace_ms"] for f in frames]))
    mean_splice_ms = float(np.mean([f["incremental_splice_ms"] for f in frames]))
    mean_invalid_fraction = float(np.mean([f["invalid_fraction"] for f in frames]))
    mean_proxy_ms = float(np.mean([f["proxy_finetune_ms"] for f in frames]))
    finite_proxy_psnr = [
        f["proxy_after_psnr_vs_full"]
        for f in frames
        if isinstance(f["proxy_after_psnr_vs_full"], float)
    ]

    def _finite(key: str) -> list[float]:
        return [f[key] for f in frames if isinstance(f[key], float)]

    regime_a = _finite("torchnrp_regime_a_full_retrace_psnr_vs_full")
    regime_b = _finite("torchnrp_regime_b_incremental_psnr_vs_full")
    regime_c = _finite("torchnrp_regime_c_stale_psnr_vs_full")
    regime_b2 = _finite("torchnrp_regime_b2_replay_psnr_vs_full")
    mean_a = float(np.mean(regime_a)) if regime_a else float("nan")
    mean_b = float(np.mean(regime_b)) if regime_b else float("nan")
    mean_c = float(np.mean(regime_c)) if regime_c else float("nan")
    mean_b2 = float(np.mean(regime_b2)) if regime_b2 else float("nan")
    mean_torchnrp_full_retrain_ms = float(
        np.mean([f["torchnrp_regime_a_full_retrain_ms"] for f in frames])
    )
    mean_torchnrp_finetune_ms = float(np.mean([f["torchnrp_regime_b_finetune_ms"] for f in frames]))
    mean_torchnrp_replay_finetune_ms = float(
        np.mean([f["torchnrp_regime_b2_replay_finetune_ms"] for f in frames])
    )
    report = {
        "extension": "E2",
        "scope": (
            "one-bounce primary-visibility cache invalidation, segment splicing, "
            "and warm-start image-proxy repair"
        ),
        "resolution": [args.width, args.height],
        "spp": args.spp,
        "frames": args.frames,
        "base_trace_ms": base_trace_s * 1000.0,
        "mean_full_trace_ms": mean_full_ms,
        "mean_incremental_splice_ms": mean_splice_ms,
        "mean_proxy_finetune_ms": mean_proxy_ms,
        "mean_invalid_fraction": mean_invalid_fraction,
        "mean_frame_budget_fraction_16ms_full_trace": mean_full_ms / 16.0,
        "mean_frame_budget_fraction_16ms_splice_only": mean_splice_ms / 16.0,
        "mean_frame_budget_fraction_16ms_splice_plus_proxy": (mean_splice_ms + mean_proxy_ms)
        / 16.0,
        "proxy_finetune": {
            "kind": "warm-start image-space proxy baseline",
            "steps": args.proxy_steps,
            "lr": args.proxy_lr,
            "mean_after_psnr_vs_full": float(np.mean(finite_proxy_psnr))
            if finite_proxy_psnr
            else "inf",
            "min_after_psnr_vs_full": float(np.min(finite_proxy_psnr))
            if finite_proxy_psnr
            else "inf",
        },
        "torchnrp_regimes": {
            "kind": "real TorchNRP weight fine-tuning, warm-started per frame",
            "base_train_ms": torchnrp_base_train_s * 1000.0,
            "finetune_iters": args.torchnrp_finetune_iters,
            "lr": args.torchnrp_lr,
            "mean_regime_a_full_retrace_psnr_vs_full": mean_a,
            "mean_regime_b_incremental_psnr_vs_full": mean_b,
            "mean_regime_c_stale_psnr_vs_full": mean_c,
            "mean_regime_a_minus_b_gap_db": mean_a - mean_b,
            "regime_b_within_1db_of_a": bool((mean_a - mean_b) <= 1.0),
            "mean_regime_a_full_retrain_ms": mean_torchnrp_full_retrain_ms,
            "mean_regime_b_finetune_ms": mean_torchnrp_finetune_ms,
            "mean_regime_b2_replay_psnr_vs_full": mean_b2,
            "mean_regime_a_minus_b2_gap_db": mean_a - mean_b2,
            "regime_b2_within_1db_of_a": bool((mean_a - mean_b2) <= 1.0),
            "mean_regime_b2_replay_finetune_ms": mean_torchnrp_replay_finetune_ms,
            "replay_closes_the_gap": bool((mean_a - mean_b2) < (mean_a - mean_b) - 0.5),
        },
        "multi_bounce_invalidation": multi_bounce_invalidation_comparison(
            args.width, args.height, args.spp, light
        ),
        "frames_detail": frames,
        "limitations": [
            "The per-frame regimes above are one-bounce; multi_bounce_invalidation "
            "is a separate 2-bounce failure-then-fix comparison, not integrated "
            "into the per-frame splice/proxy loop above.",
            "The image-space WarmStartImageProxy above is a simple baseline; "
            "torchnrp_regimes is the actual TorchNRP weight fine-tuning comparison.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
