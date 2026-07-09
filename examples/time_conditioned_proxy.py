"""E1's harder axis: a single TorchNRP proxy conditioned on a normalized time input.

`examples/time_conditioned_camera.py` is the image-space interpolation baseline this
extension explicitly calls "not the full time-conditioned neural proxy." This script
is that proxy: one `TorchNRP` model, trained jointly on K traced camera-keyframe
caches, with the light-parameter input extended by one extra scalar (`time`). Each
keyframe's own G-buffer aux (albedo/depth/normal — camera-pose-dependent) supervises
training; held-out intermediate camera times are evaluated against a freshly traced
cache at that exact camera pose (real ground truth, not an interpolated proxy for
ground truth), giving a fair PSNR comparison per the E1 criterion.
"""

from __future__ import annotations

import argparse
import json
import sys
import time as time_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.time_conditioned_camera import camera_at  # noqa: E402
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def _pixel_xy(width: int, height: int) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return np.stack([(xs.reshape(-1) + 0.5) / width, (ys.reshape(-1) + 0.5) / height], axis=1)


def _aux(cache) -> np.ndarray:
    return np.concatenate(
        [cache.albedo.reshape(-1, 3), cache.depth.reshape(-1, 1), cache.normal.reshape(-1, 3)],
        axis=1,
    )


def _light_time_params(light: SphereLight, t: float, n: int) -> np.ndarray:
    vec = np.concatenate([light.center, [light.radius], [t]]).astype(np.float32)
    return np.tile(vec, (n, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/time-camera/proxy_report.json")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--offset-extent", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--iters", type=int, default=800)
    parser.add_argument("--lr", type=float, default=5e-3)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    train_times = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    heldout_times = np.array([0.25, 0.75], dtype=np.float64)
    light = SphereLight(
        center=np.array([0.52, 0.72, 0.45]),
        radius=0.08,
        rgb=np.array([3.0, 2.2, 1.6]),
    )

    n_px = args.width * args.height
    xy_np = _pixel_xy(args.width, args.height)

    train_aux = []
    train_targets = []
    t0 = time_module.perf_counter()
    for idx, t in enumerate(train_times):
        cache = trace_path_cache(
            args.width,
            args.height,
            args.spp,
            args.bounces,
            seed=args.seed + idx,
            camera_pos=camera_at(float(t), args.offset_extent),
        )
        train_aux.append(_aux(cache))
        train_targets.append(gather_light(cache, light).reshape(-1, 3))
    trace_seconds = time_module.perf_counter() - t0

    torch.manual_seed(0)
    model = TorchNRP(
        light_type="sphere",
        light_param_dim=5,
        hidden_width=64,
        hidden_layers=3,
        encoding={"levels": 3, "features_per_level": 2, "finest_resolution": args.width},
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    gen = torch.Generator(device="cpu").manual_seed(0)

    xy_t = torch.as_tensor(xy_np, dtype=torch.float32)
    aux_t = [torch.as_tensor(a, dtype=torch.float32) for a in train_aux]
    target_t = [torch.as_tensor(y, dtype=torch.float32) for y in train_targets]
    params_t = [
        torch.as_tensor(_light_time_params(light, float(t), n_px), dtype=torch.float32)
        for t in train_times
    ]

    batch = 256
    loss_curve = []
    t0 = time_module.perf_counter()
    for _ in range(args.iters):
        k = int(torch.randint(0, len(train_times), (1,), generator=gen).item())
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen)
        pred = model(xy_t[pixel_ids], aux_t[k][pixel_ids], params_t[k][pixel_ids])
        loss = torch.mean((pred - target_t[k][pixel_ids]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
    train_seconds = time_module.perf_counter() - t0

    def predict_full(aux: np.ndarray, t: float) -> np.ndarray:
        aux_tensor = torch.as_tensor(aux, dtype=torch.float32)
        params = torch.as_tensor(_light_time_params(light, t, n_px), dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            pred = model(xy_t, aux_tensor, params).numpy().astype(np.float64)
        model.train()
        return pred.reshape(args.height, args.width, 3)

    train_psnrs = []
    for idx, t in enumerate(train_times):
        pred = predict_full(train_aux[idx], float(t))
        train_psnrs.append(psnr(pred, train_targets[idx].reshape(args.height, args.width, 3)))

    heldout_reports = []
    latencies_ms = []
    for idx, t in enumerate(heldout_times):
        heldout_cache = trace_path_cache(
            args.width,
            args.height,
            args.spp,
            args.bounces,
            seed=args.seed + 100 + idx,
            camera_pos=camera_at(float(t), args.offset_extent),
        )
        direct = gather_light(heldout_cache, light)
        aux_h = _aux(heldout_cache)
        t0 = time_module.perf_counter()
        pred = predict_full(aux_h, float(t))
        latencies_ms.append((time_module.perf_counter() - t0) * 1000.0)
        heldout_reports.append(
            {
                "time": float(t),
                "psnr_db_vs_direct": psnr(pred, direct),
            }
        )

    mean_train_psnr = float(np.mean(train_psnrs))
    mean_heldout_psnr = float(np.mean([e["psnr_db_vs_direct"] for e in heldout_reports]))
    gap_db = mean_train_psnr - mean_heldout_psnr

    report = {
        "extension": "E1",
        "scope": (
            "single TorchNRP proxy conditioned on normalized time + light params, "
            "trained jointly on K camera keyframes"
        ),
        "width": args.width,
        "height": args.height,
        "spp": args.spp,
        "bounces": args.bounces,
        "train_times": train_times.tolist(),
        "heldout_times": heldout_times.tolist(),
        "trace_seconds": trace_seconds,
        "train_seconds": train_seconds,
        "iters": args.iters,
        "loss_first": loss_curve[0],
        "loss_last": loss_curve[-1],
        "train_keyframe_psnr_db": train_psnrs,
        "mean_train_keyframe_psnr_db": mean_train_psnr,
        "heldout_intermediate": heldout_reports,
        "mean_heldout_psnr_db": mean_heldout_psnr,
        "heldout_minus_train_gap_db": gap_db,
        "within_3db_criterion": bool(gap_db <= 3.0),
        "mean_inference_latency_ms": float(np.mean(latencies_ms)),
        "per_frame_inference_latency_ms": latencies_ms,
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
