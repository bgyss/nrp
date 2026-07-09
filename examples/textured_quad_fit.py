"""E4 textured-quad inverse recovery and texture-resolution scaling report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import TexturedQuadLight, quad_tangent_frame, segment_quad_uv  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.texture_fit import (  # noqa: E402
    fit_textured_quad_light,
    textured_quad_reconstruction_error,
)
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import relight  # noqa: E402
from nrp.torch_backend.train import train as train_torchnrp  # noqa: E402

CENTER = np.array([0.0, 0.0, 1.0], dtype=np.float64)
NORMAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)
WIDTH = 2.0
HEIGHT = 2.0


def make_full_rank_cache(texture_size: int, samples_per_texel: int = 3) -> PathCache:
    """Build one synthetic pixel observation per texel sample crossing the quad."""
    rng = np.random.default_rng(100 + texture_size)
    tex_h = tex_w = texture_size
    n_pixels = tex_h * tex_w * samples_per_texel
    seg_pixel = np.arange(n_pixels, dtype=np.int64)
    origins = np.zeros((n_pixels, 3), dtype=np.float64)
    dirs = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (n_pixels, 1))
    tmax = np.full(n_pixels, 2.0, dtype=np.float64)
    throughput = np.zeros((n_pixels, 3), dtype=np.float64)
    u_axis, v_axis = quad_tangent_frame(NORMAL)

    idx = 0
    for y in range(tex_h):
        for x in range(tex_w):
            uv = np.array([(x + 0.5) / tex_w, (y + 0.5) / tex_h], dtype=np.float64)
            hit = CENTER + (uv[0] - 0.5) * WIDTH * u_axis + (uv[1] - 0.5) * HEIGHT * v_axis
            for _ in range(samples_per_texel):
                origins[idx] = hit - dirs[idx]
                throughput[idx] = rng.uniform(0.35, 1.25, size=3)
                idx += 1

    return PathCache(
        width=n_pixels,
        height=1,
        n_paths=np.ones(n_pixels, dtype=np.int64),
        seg_pixel=seg_pixel,
        seg_origin=origins,
        seg_dir=dirs,
        seg_tmax=tmax,
        seg_throughput=throughput,
        albedo=np.full((1, n_pixels, 3), 0.5, dtype=np.float64),
        position=np.zeros((1, n_pixels, 3), dtype=np.float64),
        depth=np.ones((1, n_pixels), dtype=np.float64),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, n_pixels, 1)),
    )


def make_random_uv_cache(texture_size: int, samples: int, seed: int) -> PathCache:
    """Synthetic cache with random quad-hit UVs for held-out proxy-scaling tests."""
    rng = np.random.default_rng(seed)
    seg_pixel = np.arange(samples, dtype=np.int64)
    origins = np.zeros((samples, 3), dtype=np.float64)
    dirs = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (samples, 1))
    tmax = np.full(samples, 2.0, dtype=np.float64)
    throughput = rng.uniform(0.35, 1.25, size=(samples, 3))
    u_axis, v_axis = quad_tangent_frame(NORMAL)
    uv = rng.random((samples, 2))
    for idx, coord in enumerate(uv):
        hit = CENTER + (coord[0] - 0.5) * WIDTH * u_axis + (coord[1] - 0.5) * HEIGHT * v_axis
        origins[idx] = hit - dirs[idx]

    return PathCache(
        width=samples,
        height=1,
        n_paths=np.ones(samples, dtype=np.int64),
        seg_pixel=seg_pixel,
        seg_origin=origins,
        seg_dir=dirs,
        seg_tmax=tmax,
        seg_throughput=throughput,
        albedo=np.full((1, samples, 3), 0.5, dtype=np.float64),
        position=np.zeros((1, samples, 3), dtype=np.float64),
        depth=np.ones((1, samples), dtype=np.float64),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, samples, 1)),
    )


def make_reference_texture(texture_size: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, texture_size),
        np.linspace(0.0, 1.0, texture_size),
        indexing="ij",
    )
    return np.stack(
        [
            0.2 + 0.8 * x,
            0.3 + 0.6 * y,
            0.4 + 0.2 * np.sin(np.pi * (x + y)),
        ],
        axis=2,
    )


def make_random_texture_bank(texture_size: int, count: int, seed: int) -> np.ndarray:
    """Smooth-ish RGB texture bank for learned texture-conditioning experiments."""
    rng = np.random.default_rng(seed)
    base = make_reference_texture(texture_size)
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, texture_size),
        np.linspace(0.0, 1.0, texture_size),
        indexing="ij",
    )
    textures = []
    for _ in range(count):
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        freq = rng.integers(1, 4, size=3)
        waves = np.stack(
            [
                np.sin(freq[0] * np.pi * x + phase[0]),
                np.cos(freq[1] * np.pi * y + phase[1]),
                np.sin(freq[2] * np.pi * (x + y) + phase[2]),
            ],
            axis=2,
        )
        gain = rng.uniform(0.6, 1.4, size=(1, 1, 3))
        bias = rng.uniform(-0.08, 0.08, size=(1, 1, 3))
        texture = np.clip(base * gain + bias + 0.12 * waves, 0.02, 1.5)
        textures.append(texture.astype(np.float32))
    return np.stack(textures, axis=0)


def cache_observation_features(cache: PathCache) -> np.ndarray:
    """Return per-observation UV plus RGB throughput for textured-quad hits."""
    hits, uv = segment_quad_uv(
        cache.seg_origin,
        cache.seg_dir,
        cache.seg_tmax,
        CENTER,
        NORMAL,
        WIDTH,
        HEIGHT,
    )
    if not np.all(hits):
        raise ValueError("learned texture proxy cache must contain only quad-hit segments")
    return np.concatenate([uv, cache.seg_throughput], axis=1).astype(np.float32)


class LearnedTextureProxy(nn.Module):
    """Small learned texture-embedding proxy for E4 scaling evidence."""

    def __init__(self, texture_size: int, embedding_dim: int = 8, hidden_width: int = 48):
        super().__init__()
        flat_dim = texture_size * texture_size * 3
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, embedding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim + 5, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 3),
        )

    def forward(self, texture_flat: torch.Tensor, obs_features: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(texture_flat)
        emb = emb[:, None, :].expand(-1, obs_features.shape[1], -1)
        x = torch.cat([obs_features, emb], dim=2)
        return F.softplus(self.decoder(x))


def texture_targets(cache: PathCache, textures: np.ndarray) -> np.ndarray:
    targets = []
    for texture in textures:
        light = TexturedQuadLight(
            center=CENTER,
            normal=NORMAL,
            width=WIDTH,
            height=HEIGHT,
            texture=texture,
        )
        targets.append(gather_light(cache, light).reshape(cache.width * cache.height, 3))
    return np.stack(targets, axis=0).astype(np.float32)


def run_case(texture_size: int, out_dir: Path) -> dict:
    cache = make_full_rank_cache(texture_size)
    reference = TexturedQuadLight(
        center=CENTER,
        normal=NORMAL,
        width=WIDTH,
        height=HEIGHT,
        texture=make_reference_texture(texture_size),
    )
    target = gather_light(cache, reference)
    fit = fit_textured_quad_light(
        cache,
        target,
        CENTER,
        NORMAL,
        WIDTH,
        HEIGHT,
        (texture_size, texture_size),
        reference=reference,
    )
    reconstructed = gather_light(cache, fit.light)
    np.save(out_dir / f"texture_{texture_size}x{texture_size}_reference.npy", reference.texture)
    np.save(out_dir / f"texture_{texture_size}x{texture_size}_recovered.npy", fit.light.texture)
    return {
        "texture_size": [texture_size, texture_size],
        "parameter_count": int(texture_size * texture_size * 3),
        "pixels": cache.width * cache.height,
        "segments": cache.segment_count,
        "ranks": list(fit.ranks),
        "full_rank": all(rank == texture_size * texture_size for rank in fit.ranks),
        "relative_texture_error": fit.relative_texture_error,
        "relative_image_error": textured_quad_reconstruction_error(cache, target, fit.light),
        "reconstruction_psnr_db": psnr(reconstructed, target),
        "min_singular_value": float(min(svals[-1] for svals in fit.singular_values)),
    }


def run_proxy_scaling_case(texture_size: int, out_dir: Path, train_samples: int = 48) -> dict:
    """Fit from equal train observations and score on a dense held-out UV cache."""
    train_cache = make_random_uv_cache(texture_size, train_samples, seed=200 + texture_size)
    heldout_cache = make_random_uv_cache(texture_size, 256, seed=300 + texture_size)
    reference = TexturedQuadLight(
        center=CENTER,
        normal=NORMAL,
        width=WIDTH,
        height=HEIGHT,
        texture=make_reference_texture(texture_size),
    )
    train_target = gather_light(train_cache, reference)
    heldout_target = gather_light(heldout_cache, reference)
    fit = fit_textured_quad_light(
        train_cache,
        train_target,
        CENTER,
        NORMAL,
        WIDTH,
        HEIGHT,
        (texture_size, texture_size),
        reference=reference,
    )
    heldout_pred = gather_light(heldout_cache, fit.light)
    np.save(out_dir / f"proxy_scaling_{texture_size}x{texture_size}_pred.npy", heldout_pred)
    return {
        "texture_size": [texture_size, texture_size],
        "texture_parameter_count": int(texture_size * texture_size * 3),
        "train_observations": train_samples,
        "heldout_observations": heldout_cache.width,
        "rank_per_channel": list(fit.ranks),
        "unknowns_per_channel": int(texture_size * texture_size),
        "underdetermined": any(rank < texture_size * texture_size for rank in fit.ranks),
        "relative_texture_error": fit.relative_texture_error,
        "heldout_psnr_db": psnr(heldout_pred, heldout_target),
    }


def run_learned_texture_proxy_case(
    texture_size: int,
    out_dir: Path,
    train_textures: int = 24,
    heldout_textures: int = 6,
    observations: int = 96,
    steps: int = 600,
    embedding_dim: int = 8,
) -> dict:
    """Train a torch proxy conditioned on a learned texture embedding."""
    torch.manual_seed(400 + texture_size)
    train_cache = make_random_uv_cache(texture_size, observations, seed=500 + texture_size)
    heldout_cache = make_random_uv_cache(texture_size, 256, seed=600 + texture_size)
    train_tex = make_random_texture_bank(texture_size, train_textures, seed=700 + texture_size)
    heldout_tex = make_random_texture_bank(texture_size, heldout_textures, seed=800 + texture_size)

    train_x = torch.as_tensor(cache_observation_features(train_cache)[None, :, :])
    heldout_x = torch.as_tensor(cache_observation_features(heldout_cache)[None, :, :])
    train_x = train_x.expand(train_textures, -1, -1)
    heldout_x = heldout_x.expand(heldout_textures, -1, -1)
    train_tex_flat = torch.as_tensor(train_tex.reshape(train_textures, -1))
    heldout_tex_flat = torch.as_tensor(heldout_tex.reshape(heldout_textures, -1))
    train_y = torch.as_tensor(texture_targets(train_cache, train_tex))
    heldout_y = texture_targets(heldout_cache, heldout_tex)

    model = LearnedTextureProxy(texture_size, embedding_dim=embedding_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(steps):
        pred = model(train_tex_flat, train_x)
        loss = F.mse_loss(pred, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        heldout_pred = model(heldout_tex_flat, heldout_x).cpu().numpy()
    per_texture_psnr = [
        psnr(heldout_pred[i].reshape(1, -1, 3), heldout_y[i].reshape(1, -1, 3))
        for i in range(heldout_textures)
    ]
    np.save(
        out_dir / f"learned_texture_proxy_{texture_size}x{texture_size}_heldout.npy",
        heldout_pred,
    )
    return {
        "texture_size": [texture_size, texture_size],
        "texture_parameter_count": int(texture_size * texture_size * 3),
        "train_textures": train_textures,
        "heldout_textures": heldout_textures,
        "observations_per_train_texture": observations,
        "heldout_observations": heldout_cache.width,
        "embedding_dim": embedding_dim,
        "steps": steps,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "heldout_psnr_db_mean": float(np.mean(per_texture_psnr)),
        "heldout_psnr_db_min": float(np.min(per_texture_psnr)),
    }


def run_first_class_torchnrp_textured_quad_case(out_dir: Path) -> dict:
    """Train and relight a first-class textured_quad TorchNRP config."""
    cache = make_random_uv_cache(2, 96, seed=900)
    cache_path = out_dir / "torchnrp_textured_quad_cache.npz"
    model_dir = out_dir / "torchnrp-textured-quad"
    cache.save(cache_path)
    cfg = {
        "cache": str(cache_path),
        "out_dir": str(model_dir),
        "light_type": "textured_quad",
        "light_bounds": {
            "center": CENTER.tolist(),
            "normal": NORMAL.tolist(),
            "width": WIDTH,
            "height": HEIGHT,
            "texture_size": [2, 2],
            "texture_min": 0.05,
            "texture_max": 1.25,
        },
        "pool": {"size": 10, "replace_every": 10, "replace_count": 2},
        "denoise": {"enabled": False},
        "iters": 120,
        "batch_pixels": 256,
        "lr": 0.01,
        "model": {
            "hidden_width": 24,
            "hidden_layers": 2,
            "encoding": {"levels": 2, "table_size_log2": 8, "finest_resolution": cache.width},
        },
        "n_val_lights": 3,
        "seed": 11,
    }
    train_report = train_torchnrp(cfg)
    model = TorchNRP.load(str(model_dir / "model.pt"))
    heldout_texture = make_random_texture_bank(2, 1, seed=901)[0]
    heldout_light = TexturedQuadLight(
        center=CENTER,
        normal=NORMAL,
        width=WIDTH,
        height=HEIGHT,
        texture=heldout_texture,
    )
    pred = relight(model, cache, [heldout_light])
    ref = gather_light(cache, heldout_light)
    return {
        "texture_size": [2, 2],
        "light_param_dim": model.light_param_dim,
        "model_light_type": model.light_type,
        "parameter_count": train_report["parameter_count"],
        "iters": cfg["iters"],
        "loss_first": train_report["loss_first"],
        "loss_last": train_report["loss_last"],
        "val_psnr_db_vs_raw_mean": train_report["val_psnr_db_vs_raw_mean"],
        "heldout_relight_psnr_db": psnr(pred, ref),
        "model_path": "torchnrp-textured-quad/model.pt",
        "cache_path": "torchnrp_textured_quad_cache.npz",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/textured-quad-fit/report.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [run_case(size, out_path.parent) for size in (2, 4, 8)]
    scaling = [run_proxy_scaling_case(size, out_path.parent) for size in (2, 4, 8)]
    learned = [run_learned_texture_proxy_case(size, out_path.parent) for size in (2, 4, 8)]
    first_class = run_first_class_torchnrp_textured_quad_case(out_path.parent)
    report = {
        "extension": "E4",
        "scope": (
            "textured quad inverse recovery, linear proxy scaling, and learned "
            "texture-embedding proxy scaling"
        ),
        "cases": cases,
        "linear_proxy_scaling": {
            "description": (
                "Least-squares texture proxy fitted from an equal 48 random quad-hit "
                "observations per texture size and evaluated on a separate 256-sample cache."
            ),
            "cases": scaling,
        },
        "learned_texture_proxy_scaling": {
            "description": (
                "Torch MLP proxy that encodes each flattened RGB texture into an 8D "
                "learned embedding, then predicts per-observation textured-quad "
                "GATHERLIGHT from UV, throughput, and the embedding."
            ),
            "cases": learned,
        },
        "first_class_torchnrp_textured_quad": first_class,
        "completion_note": (
            "This satisfies the textured-quad reference inverse-recovery slice. "
            "The learned scaling table is a compact texture-conditioned torch proxy, "
            "and the first-class case verifies the main TorchNRP train/relight path "
            "accepts a small TexturedQuadLight parameterization."
        ),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    worst = max(case["relative_texture_error"] for case in cases)
    print(f"wrote {out_path}; worst relative texture error {worst:.3e}")


if __name__ == "__main__":
    main()
