"""E5: stream TorchNRP pool-training targets from a sharded cache (§7 out-of-core gap).

`nrp.torch_backend.train.ImagePool` renders each pool slot's GATHERLIGHT target from a
fully-resident `PathCache`. This module builds the same targets by visiting on-disk
tile shards (`PathCache.save_sharded`) one at a time, so the resident segment memory
is bounded by the largest shard rather than the whole cache. G-buffer (albedo/depth/
normal) and the shard manifest stay resident — they are per-pixel, not per-segment,
and small — only the segment arrays are streamed. Sphere lights only (the config
shape used by the toy training example); other light types would need the analogous
`segment_hits_*` predicate wired in the same way.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..lights import SphereLight
from ..path_cache import PathCache
from ..rgb9e5 import rgb9e5_decode
from .denoise import denoise_image
from .model import TorchNRP, relative_mse_loss
from .sampling import sample_light
from .train import light_param_vector


def load_sharded_gbuffer(shard_dir: Path) -> PathCache:
    """Load only per-pixel data from shards, leaving all segment arrays on disk."""
    with open(shard_dir / "manifest.json") as f:
        manifest = json.load(f)
    width, height = int(manifest["width"]), int(manifest["height"])
    n_paths = np.zeros((height, width), dtype=np.int64)
    albedo = np.zeros((height, width, 3), dtype=np.float64)
    position = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.zeros((height, width), dtype=np.float64)
    normal = np.zeros((height, width, 3), dtype=np.float64)
    for shard in manifest["shards"]:
        with np.load(shard_dir / shard["path"]) as z:
            y0, y1 = int(z["y0"]), int(z["y1"])
            x0, x1 = int(z["x0"]), int(z["x1"])
            n_paths[y0:y1, x0:x1] = z["n_paths"]
            albedo[y0:y1, x0:x1] = z["albedo"]
            position[y0:y1, x0:x1] = z["position"]
            depth[y0:y1, x0:x1] = z["depth"]
            normal[y0:y1, x0:x1] = z["normal"]
    empty = np.zeros(0, dtype=np.float64)
    cache = PathCache(
        width=width,
        height=height,
        n_paths=n_paths.reshape(-1),
        seg_pixel=np.zeros(0, dtype=np.int64),
        seg_origin=np.zeros((0, 3), dtype=np.float64),
        seg_dir=np.zeros((0, 3), dtype=np.float64),
        seg_tmax=empty,
        seg_throughput=np.zeros((0, 3), dtype=np.float64),
        albedo=albedo,
        position=position,
        depth=depth,
        normal=normal,
        medium=manifest.get("medium"),
    )
    cache.validate()
    return cache


def gather_sphere_streamed(
    shard_dir: Path, n_paths: np.ndarray, center: np.ndarray, radius: float
) -> tuple[np.ndarray, dict]:
    """GATHERsphere computed by streaming shard tiles; returns (H, W, 3) plus stats.

    `n_paths` (flat, H*W) is the per-pixel path count used to normalize accumulated
    throughput, matching `nrp.gather_light._accumulate_hits`.
    """
    with open(shard_dir / "manifest.json") as f:
        manifest = json.load(f)
    width, height = int(manifest["width"]), int(manifest["height"])
    from ..lights import segment_hits_sphere  # local import: avoid cycle at module load

    out = np.zeros((height * width, 3), dtype=np.float64)
    peak_segment_bytes = 0
    segments_processed = 0
    for shard in manifest["shards"]:
        z = np.load(shard_dir / shard["path"])
        seg_pixel = z["seg_pixel"].astype(np.int64)
        segments_processed += int(seg_pixel.size)
        seg_origin = z["seg_origin"].astype(np.float64)
        seg_dir = z["seg_dir"].astype(np.float64)
        if "packed_layout" in z:
            norms = np.linalg.norm(seg_dir, axis=1, keepdims=True)
            seg_dir = np.divide(seg_dir, norms, out=seg_dir, where=norms > 0)
        seg_tmax = z["seg_tmax"].astype(np.float64)
        seg_throughput = (
            rgb9e5_decode(z["seg_throughput_rgb9e5"])
            if "packed_layout" in z
            else z["seg_throughput"].astype(np.float64)
        )
        peak_segment_bytes = max(
            peak_segment_bytes,
            int(
                seg_pixel.nbytes
                + seg_origin.nbytes
                + seg_dir.nbytes
                + seg_tmax.nbytes
                + seg_throughput.nbytes
            ),
        )
        if seg_pixel.size:
            hits = segment_hits_sphere(seg_origin, seg_dir, seg_tmax, center, radius)
            if hits.any():
                np.add.at(out, seg_pixel[hits], seg_throughput[hits])
    denom = np.maximum(n_paths, 1).astype(np.float64)
    out /= denom[:, None]
    return out.reshape(height, width, 3), {
        "peak_segment_bytes": peak_segment_bytes,
        "segments_processed": segments_processed,
    }


class StreamedImagePool:
    """`ImagePool` equivalent whose targets are rendered by streaming shard tiles."""

    def __init__(
        self,
        shard_dir: Path,
        gbuffer_cache: PathCache,
        cfg: dict,
        rng: np.random.Generator,
        device,
    ):
        self.shard_dir = shard_dir
        self.gbuffer_cache = gbuffer_cache
        self.cfg = cfg
        self.rng = rng
        self.device = device
        self.size = cfg["pool"]["size"]
        n_px = gbuffer_cache.height * gbuffer_cache.width
        self.params = torch.empty((self.size, 4), dtype=torch.float32, device=device)
        self.targets = torch.empty((self.size, n_px, 3), dtype=torch.float32, device=device)
        self._next_replace = 0
        self.used_params: list[np.ndarray] = []
        self.supervision_seconds = 0.0
        self.peak_segment_bytes = 0
        self.segments_processed = 0
        for i in range(self.size):
            self.fill(i)

    def fill(self, slot: int) -> None:
        t0 = time.perf_counter()
        light = sample_light(
            self.gbuffer_cache,
            self.rng,
            "sphere",
            self.cfg["light_bounds"],
            self.cfg.get("sampling", "segments"),
        )
        assert isinstance(light, SphereLight)
        image, stats = gather_sphere_streamed(
            self.shard_dir,
            self.gbuffer_cache.n_paths.reshape(-1),
            light.center,
            light.radius,
        )
        self.peak_segment_bytes = max(self.peak_segment_bytes, stats["peak_segment_bytes"])
        self.segments_processed += stats["segments_processed"]
        dn = self.cfg.get("denoise", {})
        if dn.get("enabled", True):
            image = denoise_image(
                image,
                self.gbuffer_cache.albedo,
                self.gbuffer_cache.normal,
                self.gbuffer_cache.depth,
                method=dn.get("method", "bilateral"),
                **{k: v for k, v in dn.items() if k not in ("enabled", "method")},
            )
        vec = light_param_vector(light)
        self.used_params.append(vec)
        self.params[slot] = torch.as_tensor(vec, dtype=torch.float32, device=self.device)
        self.targets[slot] = torch.as_tensor(
            image.reshape(-1, 3), dtype=torch.float32, device=self.device
        )
        self.supervision_seconds += time.perf_counter() - t0

    def replace_round(self) -> None:
        for _ in range(self.cfg["pool"]["replace_count"]):
            self.fill(self._next_replace)
            self._next_replace = (self._next_replace + 1) % self.size


def _pixel_tensors(cache: PathCache, device) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = cache.height, cache.width
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    xy = np.stack([(xs.reshape(-1) + 0.5) / w, (ys.reshape(-1) + 0.5) / h], axis=1)
    aux = np.concatenate(
        [cache.albedo.reshape(-1, 3), cache.depth.reshape(-1, 1), cache.normal.reshape(-1, 3)],
        axis=1,
    )
    to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    return to(xy), to(aux)


def train_streamed(shard_dir: Path, gbuffer_cache: PathCache, cfg: dict) -> tuple[TorchNRP, dict]:
    """Train a TorchNRP sphere-light proxy from a streamed pool. Mirrors
    `nrp.torch_backend.train.train`'s core loop closely enough that, given the same
    seed and an in-memory cache built from the same segments, loss curves are directly
    comparable iteration-for-iteration."""
    rng = np.random.default_rng(cfg.get("seed", 0))
    torch.manual_seed(cfg.get("seed", 0))
    device = torch.device(cfg.get("device", "cpu"))
    xy, aux = _pixel_tensors(gbuffer_cache, device)

    t_pool0 = time.perf_counter()
    pool = StreamedImagePool(shard_dir, gbuffer_cache, cfg, rng, device)
    pool_seconds = time.perf_counter() - t_pool0

    model = TorchNRP(
        hidden_width=cfg["model"]["hidden_width"],
        hidden_layers=cfg["model"]["hidden_layers"],
        encoding=cfg["model"]["encoding"],
        light_type="sphere",
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    gen = torch.Generator(device="cpu").manual_seed(cfg.get("seed", 0))

    batch = cfg.get("batch_pixels", 512)
    replace_every = cfg["pool"]["replace_every"]
    n_px = xy.shape[0]
    loss_curve = []
    t0 = time.perf_counter()
    for it in range(cfg["iters"]):
        pixel_ids = torch.randint(0, n_px, (batch,), generator=gen).to(device)
        pool_ids = torch.randint(0, pool.size, (batch,), generator=gen).to(device)
        pred = model(xy[pixel_ids], aux[pixel_ids], pool.params[pool_ids])
        loss = relative_mse_loss(pred, pool.targets[pool_ids, pixel_ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(float(loss.item()))
        if (it + 1) % replace_every == 0:
            pool.replace_round()
    train_seconds = time.perf_counter() - t0

    return model, {
        "pool_seconds": pool_seconds,
        "train_seconds": train_seconds,
        "loss_curve": loss_curve,
        "peak_segment_bytes_loaded": pool.peak_segment_bytes,
        "supervision_seconds": pool.supervision_seconds,
        "segments_processed": pool.segments_processed,
        "segments_per_second": pool.segments_processed / max(pool.supervision_seconds, 1e-12),
    }
