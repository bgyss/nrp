"""Partitioned residual retraining for dynamic geometry (production-track rung G1).

E2's settled negative result: fine-tuning a shared TorchNRP's weights on only the
invalidated pixels lets the weights drift on the unchanged pixels (11-20 dB below the
1 dB recovery target even with replay regularization). This module tests the changed
hypothesis: keep the base proxy *frozen*, aggregate the invalidation mask to a fixed
shard grid (the cache-shard analogue at toy scale), and train a small signed-output
*residual* proxy over only the invalidated region, composited additively at
inference. Outside the region the composite equals the frozen base bitwise, so E2's
failure mode — global forgetting — is structurally impossible; any failure must be a
different one (residual underfit inside the region).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..path_cache import PathCache
from .model import TorchNRP


def invalidated_shards(
    mask: np.ndarray, shard_size: int
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Aggregate a per-pixel invalidation mask to a fixed shard (tile) grid.

    Returns the region mask (the union of every whole ``shard_size`` x ``shard_size``
    tile containing at least one invalid pixel, clipped to image bounds) and the
    sorted list of invalidated tile coordinates ``(tile_y, tile_x)``.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    height, width = mask.shape
    region = np.zeros_like(mask)
    tiles: list[tuple[int, int]] = []
    for ty in range((height + shard_size - 1) // shard_size):
        for tx in range((width + shard_size - 1) // shard_size):
            y0, x0 = ty * shard_size, tx * shard_size
            y1, x1 = min(y0 + shard_size, height), min(x0 + shard_size, width)
            if mask[y0:y1, x0:x1].any():
                region[y0:y1, x0:x1] = True
                tiles.append((ty, tx))
    return region, tiles


def pixel_features(cache: PathCache) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel (xy, aux) inputs, same convention as torch_backend.train.pixel_tensors."""
    h, w = cache.height, cache.width
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    aux = np.concatenate(
        [cache.albedo.reshape(-1, 3), cache.depth.reshape(-1, 1), cache.normal.reshape(-1, 3)],
        axis=1,
    )
    return xy.astype(np.float32), aux.astype(np.float32)


class ResidualNRP(nn.Module):
    """Signed-output residual proxy: raw pixel xy + aux + light params -> RGB delta.

    Same input signature as ``TorchNRP`` with ``use_encoding=False``, but a *linear*
    head — a residual must be able to go negative, which TorchNRP's softplus head
    cannot.
    """

    def __init__(self, hidden_width: int = 32, hidden_layers: int = 2, light_param_dim: int = 4):
        super().__init__()
        self.light_param_dim = int(light_param_dim)
        self.config = {
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "light_param_dim": self.light_param_dim,
        }
        in_dim = 2 + 7 + self.light_param_dim
        layers: list[nn.Module] = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_width, hidden_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_width if hidden_layers else in_dim, 3))
        self.mlp = nn.Sequential(*layers)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, pixel_xy: torch.Tensor, aux: torch.Tensor, light_params: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([pixel_xy, aux, light_params], dim=1))

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> ResidualNRP:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


def _base_prediction(
    base: TorchNRP, xy: np.ndarray, aux: np.ndarray, light_params: np.ndarray
) -> torch.Tensor:
    params = torch.as_tensor(light_params, dtype=torch.float32).expand(xy.shape[0], -1)
    base.eval()
    with torch.no_grad():
        pred = base(torch.as_tensor(xy), torch.as_tensor(aux), params)
    return pred


def train_residual(
    residual: ResidualNRP,
    base: TorchNRP,
    cache: PathCache,
    target_image: np.ndarray,
    region_mask: np.ndarray,
    light_params: np.ndarray,
    iters: int = 300,
    lr: float = 5e-3,
) -> list[float]:
    """Fit ``residual`` to ``target - base`` on the region pixels only; base frozen."""
    region = np.asarray(region_mask, dtype=bool).reshape(-1)
    idx = np.flatnonzero(region)
    if idx.size == 0:
        return []
    xy, aux = pixel_features(cache)
    base_pred = _base_prediction(base, xy, aux, light_params)
    target = torch.as_tensor(
        target_image.reshape(-1, 3)[idx].astype(np.float32), dtype=torch.float32
    )
    delta = target - base_pred[idx]
    xy_t = torch.as_tensor(xy[idx])
    aux_t = torch.as_tensor(aux[idx])
    params = torch.as_tensor(light_params, dtype=torch.float32).expand(idx.size, -1)
    opt = torch.optim.Adam(residual.parameters(), lr=lr)
    residual.train()
    losses: list[float] = []
    for _ in range(iters):
        pred = residual(xy_t, aux_t, params)
        loss = torch.mean((pred - delta) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
    residual.eval()
    return losses


def composite_predict(
    base: TorchNRP,
    residual: ResidualNRP,
    cache: PathCache,
    light_params: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    """Frozen-base prediction everywhere, plus the residual on region pixels only."""
    region = np.asarray(region_mask, dtype=bool)
    if region.shape != (cache.height, cache.width):
        raise ValueError(f"region_mask must be {(cache.height, cache.width)}, got {region.shape}")
    xy, aux = pixel_features(cache)
    image = _base_prediction(base, xy, aux, light_params).numpy().astype(np.float64)
    idx = np.flatnonzero(region.reshape(-1))
    if idx.size:
        params = torch.as_tensor(light_params, dtype=torch.float32).expand(idx.size, -1)
        residual.eval()
        with torch.no_grad():
            delta = residual(torch.as_tensor(xy[idx]), torch.as_tensor(aux[idx]), params).numpy()
        image[idx] += delta.astype(np.float64)
    return image.reshape(cache.height, cache.width, 3)
