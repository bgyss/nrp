"""E5: stream TorchNRP pool-training targets from a sharded cache (§7 out-of-core gap).

`nrp.torch_backend.train.ImagePool` renders each pool slot's GATHERLIGHT target from a
fully-resident `PathCache`. This module builds the same targets by visiting on-disk
tile shards (`PathCache.save_sharded`) one at a time, so the resident segment memory
is bounded by the largest shard rather than the whole cache. G-buffer (albedo/depth/
normal) and the shard manifest stay resident — they are per-pixel, not per-segment,
and small — only the segment arrays are streamed. Sphere lights only (the config
shape used by the toy training example); other light types would need the analogous
`segment_hits_*` predicate wired in the same way.

S1 additions: `gather_backend: "torch"` (config key) swaps the scalar numpy predicate
for a per-shard `TorchPathCache` batched gather on `gather_device` (cpu/mps/cuda),
and every pool fill batches its lights into one shard pass (`fill_slots`), so a
16-slot pool build decodes the sharded cache once instead of 16 times. Both changes
preserve the bounded-residency guarantee (one shard's segments resident at a time)
and the rng consumption order; numpy-backend targets are bit-identical to per-slot
fills, torch-backend targets match numpy within the established rtol 1e-5 parity.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..lights import SphereLight
from ..path_cache import PathCache
from ..rgb9e5 import rgb9e5_decode
from .denoise import denoise_image
from .device import resolve_device
from .gather import TorchPathCache
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


def _decode_shard(z) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Decode one shard's segment arrays to float64 (packed or float layout).

    Returns (seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput, decoded_bytes);
    the byte count uses the same decoded-numpy accounting as the E5 reports.
    """
    seg_pixel = z["seg_pixel"].astype(np.int64)
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
    decoded_bytes = int(
        seg_pixel.nbytes
        + seg_origin.nbytes
        + seg_dir.nbytes
        + seg_tmax.nbytes
        + seg_throughput.nbytes
    )
    return seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput, decoded_bytes


def _decode_shard_file(
    path: Path, workers: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """`_decode_shard` from a shard path, decompressing members on `workers` threads.

    Each thread opens its own npz handle, so only this one shard's arrays are ever
    resident: parallelism is *within* the shard (zlib inflate and the float64
    conversions release the GIL), never across shards. `workers=1` matches the
    serial `_decode_shard` exactly.
    """
    if workers <= 1:
        with np.load(path) as z:
            return _decode_shard(z)
    with np.load(path) as z:
        packed = "packed_layout" in z.files

    def read(name: str) -> np.ndarray:
        with np.load(path) as z:
            arr = z[name]
        if name == "seg_pixel":
            return arr.astype(np.int64)
        if name == "seg_throughput_rgb9e5":
            return rgb9e5_decode(arr)
        out = arr.astype(np.float64)
        if name == "seg_dir" and packed:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = np.divide(out, norms, out=out, where=norms > 0)
        return out

    names = [
        "seg_pixel",
        "seg_origin",
        "seg_dir",
        "seg_tmax",
        "seg_throughput_rgb9e5" if packed else "seg_throughput",
    ]
    with ThreadPoolExecutor(max_workers=min(workers, len(names))) as pool:
        seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput = list(pool.map(read, names))
    decoded_bytes = int(
        seg_pixel.nbytes
        + seg_origin.nbytes
        + seg_dir.nbytes
        + seg_tmax.nbytes
        + seg_throughput.nbytes
    )
    return seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput, decoded_bytes


def gather_spheres_streamed(
    shard_dir: Path,
    n_paths: np.ndarray,
    lights: list[tuple[np.ndarray, float]],
    backend: str = "numpy",
    device: str = "cpu",
    decode_workers: int = 1,
) -> tuple[list[np.ndarray], dict]:
    """GATHERsphere for several lights in one pass over the shard tiles.

    Each shard is decoded once and tested against every `(center, radius)` in
    `lights`, amortizing decompression across lights; per-light results are identical
    to independent single-light passes. `backend` selects the scalar numpy predicate
    (`"numpy"`, the authoritative reference) or a per-shard `TorchPathCache` batched
    gather (`"torch"`) on `device`. Either way only one shard's segments are resident
    at a time, and `n_paths` (flat, H*W) normalizes accumulated throughput exactly as
    `nrp.gather_light._accumulate_hits` does. Returns ([(H, W, 3)] * len(lights), stats).
    """
    with open(shard_dir / "manifest.json") as f:
        manifest = json.load(f)
    width, height = int(manifest["width"]), int(manifest["height"])
    from ..lights import segment_hits_sphere  # local import: avoid cycle at module load

    use_torch = backend == "torch"
    torch_device = torch.device(device) if use_torch else None
    peak_segment_bytes = 0
    peak_device_tensor_bytes = 0
    segments_processed = 0
    decode_seconds = 0.0
    gather_seconds = 0.0

    if use_torch:
        outs_t: list[torch.Tensor] | None = None
    else:
        outs = [np.zeros((height * width, 3), dtype=np.float64) for _ in lights]

    for shard in manifest["shards"]:
        t0 = time.perf_counter()
        seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput, decoded = _decode_shard_file(
            shard_dir / shard["path"], workers=decode_workers
        )
        decode_seconds += time.perf_counter() - t0
        segments_processed += int(seg_pixel.size)
        peak_segment_bytes = max(peak_segment_bytes, decoded)
        if not seg_pixel.size:
            continue
        t0 = time.perf_counter()
        if use_torch:
            tpc = TorchPathCache.from_arrays(
                width=width,
                height=height,
                seg_pixel=seg_pixel,
                seg_origin=seg_origin,
                seg_dir=seg_dir,
                seg_tmax=seg_tmax,
                seg_throughput=seg_throughput,
                n_paths=n_paths,
                device=torch_device,
            )
            peak_device_tensor_bytes = max(
                peak_device_tensor_bytes,
                sum(
                    t.element_size() * t.nelement()
                    for t in (tpc.origin, tpc.dir, tpc.tmax, tpc.throughput, tpc.pixel)
                ),
            )
            if outs_t is None:
                outs_t = [
                    torch.zeros((height, width, 3), dtype=tpc.dtype, device=torch_device)
                    for _ in lights
                ]
            for i, (center, radius) in enumerate(lights):
                outs_t[i] += tpc.gather_throughput(center, radius)
            del tpc
        else:
            for i, (center, radius) in enumerate(lights):
                hits = segment_hits_sphere(seg_origin, seg_dir, seg_tmax, center, radius)
                if hits.any():
                    np.add.at(outs[i], seg_pixel[hits], seg_throughput[hits])
        gather_seconds += time.perf_counter() - t0

    stats = {
        "peak_segment_bytes": peak_segment_bytes,
        "peak_device_tensor_bytes": peak_device_tensor_bytes,
        "segments_processed": segments_processed,
        "decode_seconds": decode_seconds,
        "gather_seconds": gather_seconds,
        "lights_per_pass": len(lights),
    }
    if use_torch:
        if outs_t is None:
            outs_t = [
                torch.zeros((height, width, 3), dtype=torch.float64, device=torch_device)
                for _ in lights
            ]
        images = [o.cpu().numpy().astype(np.float64) for o in outs_t]
        return images, stats
    denom = np.maximum(n_paths, 1).astype(np.float64)
    images = [(o / denom[:, None]).reshape(height, width, 3) for o in outs]
    return images, stats


def gather_sphere_streamed(
    shard_dir: Path,
    n_paths: np.ndarray,
    center: np.ndarray,
    radius: float,
    backend: str = "numpy",
    device: str = "cpu",
    decode_workers: int = 1,
) -> tuple[np.ndarray, dict]:
    """Single-light wrapper around `gather_spheres_streamed` (kept for E5 callers)."""
    images, stats = gather_spheres_streamed(
        shard_dir,
        n_paths,
        [(np.asarray(center, dtype=np.float64), float(radius))],
        backend=backend,
        device=device,
        decode_workers=decode_workers,
    )
    return images[0], stats


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
        self.gather_backend = cfg.get("gather_backend", "numpy")
        self.gather_device = cfg.get("gather_device", cfg.get("device", "cpu"))
        self.decode_workers = int(cfg.get("decode_workers", 1))
        n_px = gbuffer_cache.height * gbuffer_cache.width
        self.params = torch.empty((self.size, 4), dtype=torch.float32, device=device)
        self.targets = torch.empty((self.size, n_px, 3), dtype=torch.float32, device=device)
        self._next_replace = 0
        self.used_params: list[np.ndarray] = []
        self.supervision_seconds = 0.0
        self.peak_segment_bytes = 0
        self.peak_device_tensor_bytes = 0
        self.segments_processed = 0
        self.decode_seconds = 0.0
        self.gather_seconds = 0.0
        # One shard pass fills every initial slot: lights are sampled in slot order
        # first (identical rng stream to per-slot fills), then each decoded shard is
        # tested against all of them, amortizing decompression across the pool.
        self.fill_slots(list(range(self.size)))

    def fill_slots(self, slots: list[int]) -> None:
        t0 = time.perf_counter()
        lights = []
        for _ in slots:
            light = sample_light(
                self.gbuffer_cache,
                self.rng,
                "sphere",
                self.cfg["light_bounds"],
                self.cfg.get("sampling", "segments"),
            )
            assert isinstance(light, SphereLight)
            lights.append(light)
        images, stats = gather_spheres_streamed(
            self.shard_dir,
            self.gbuffer_cache.n_paths.reshape(-1),
            [(light.center, light.radius) for light in lights],
            backend=self.gather_backend,
            device=self.gather_device,
            decode_workers=self.decode_workers,
        )
        self.peak_segment_bytes = max(self.peak_segment_bytes, stats["peak_segment_bytes"])
        self.peak_device_tensor_bytes = max(
            self.peak_device_tensor_bytes, stats["peak_device_tensor_bytes"]
        )
        self.segments_processed += stats["segments_processed"]
        self.decode_seconds += stats["decode_seconds"]
        self.gather_seconds += stats["gather_seconds"]
        dn = self.cfg.get("denoise", {})
        for slot, light, image in zip(slots, lights, images, strict=True):
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

    def fill(self, slot: int) -> None:
        self.fill_slots([slot])

    def replace_round(self) -> None:
        count = self.cfg["pool"]["replace_count"]
        slots = [(self._next_replace + k) % self.size for k in range(count)]
        self.fill_slots(slots)
        self._next_replace = (self._next_replace + count) % self.size


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
    device = resolve_device(cfg.get("device"))
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
        "peak_device_tensor_bytes": pool.peak_device_tensor_bytes,
        "supervision_seconds": pool.supervision_seconds,
        "decode_seconds": pool.decode_seconds,
        "gather_seconds": pool.gather_seconds,
        "gather_backend": pool.gather_backend,
        "gather_device": pool.gather_device,
        "segments_processed": pool.segments_processed,
        "segments_per_second": pool.segments_processed / max(pool.supervision_seconds, 1e-12),
    }
