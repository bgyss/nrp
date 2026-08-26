"""Multi-view NRPs (§6.1): one resident proxy per camera view, one light edit for all.

The paper's multi-view argument: because proxies are compact and evaluating one needs
no path-cache access, several views of a scene can stay resident simultaneously and a
single light edit re-renders every view at interactive rates — total cost is N times
one view's inference, nothing else. This CLI loads N (model, cache) pairs from a view
manifest, applies one light edit (a light spec or list, as in `relight`) across all
views, and writes one image per view.

Each view's path cache is read once at load time, only for the per-pixel inputs
(pixel coordinates + G-buffer aux); after that an edit touches no cache data. The
manifest is a JSON list of view objects (paths resolve relative to the manifest file):

  [{"name": "front", "model": "front/model.pt", "cache": "front/path_cache.npz"}, ...]

Usage:
  python -m nrp.torch_backend.relight_multiview --views out/multiview/views.json \
      --light '{"type": "sphere", "center": [0, 0.4, 0], "radius": 0.2, "rgb": [8, 8, 8]}' \
      --out-dir out/multiview/edit --bench 20
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch

from ..gather_light import gather_lights
from ..lights import TexturedQuadLight, light_from_dict
from ..metrics import psnr
from ..path_cache import PathCache
from .bench import _synchronize
from .conditioned_multiview import camera_tensor, load_camera_manifest
from .model import TorchNRP
from .train import light_param_vector, model_tensors


@dataclass
class ConditionedView:
    """Resident pixel features for one camera evaluated by a shared R2 model."""

    name: str
    model: TorchNRP
    cache: PathCache
    device: torch.device
    spatial: torch.Tensor
    aux: torch.Tensor
    view_dir: torch.Tensor

    def render(self, lights: list, model: TorchNRP | None = None) -> np.ndarray:
        active_model = self.model if model is None else model
        n_px = self.cache.height * self.cache.width
        image = torch.zeros((n_px, 3), device=self.device)
        with torch.no_grad():
            for light in lights:
                params = torch.as_tensor(
                    light_param_vector(light), dtype=torch.float32, device=self.device
                ).expand(n_px, -1)
                rgb = (
                    torch.ones(3, dtype=torch.float32, device=self.device)
                    if isinstance(light, TexturedQuadLight)
                    else torch.as_tensor(light.rgb, dtype=torch.float32, device=self.device)
                )
                image += active_model(
                    self.spatial, self.aux, params, view_dir=self.view_dir
                ) * rgb
        return image.cpu().numpy().astype(np.float64).reshape(
            self.cache.height, self.cache.width, 3
        )


def load_conditioned_views(
    manifest_path: str, model_path: str, device: str = "cpu"
) -> tuple[TorchNRP, list[ConditionedView]]:
    """Load one camera-conditioned model and N resident camera feature records."""
    dev = torch.device(device)
    views = load_camera_manifest(manifest_path)
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    resolved_model = (
        model_path if os.path.isabs(model_path) else os.path.join(manifest_dir, model_path)
    )
    model = TorchNRP.load(resolved_model).to(dev).eval()
    if not model.camera_conditioned:
        raise ValueError("R2 shared inference requires a camera-conditioned model")
    resident = []
    for view in views:
        cache = PathCache.load(str(view.cache_path))
        spatial, aux = model_tensors(cache, model, dev)
        resident.append(
            ConditionedView(
                name=view.name,
                model=model,
                cache=cache,
                device=dev,
                spatial=spatial,
                aux=aux,
                view_dir=camera_tensor(view, cache.height * cache.width, dev),
            )
        )
    return model, resident


def relight_conditioned_all(
    model: TorchNRP, views: list[ConditionedView], lights: list
) -> dict[str, np.ndarray]:
    """Apply one light edit through the same resident model for every view."""
    return {view.name: view.render(lights, model=model) for view in views}


def conditioned_edit_latency_ms(
    model: TorchNRP,
    views: list[ConditionedView],
    lights: list,
    frames: int = 10,
    warmup: int = 2,
) -> float:
    """Measure one all-view proxy edit without cache/GATHERLIGHT access."""
    if not views:
        raise ValueError("at least one conditioned view is required")
    if frames <= 0 or warmup < 0:
        raise ValueError("frames must be positive and warmup must be non-negative")
    devices = {view.device for view in views}
    for _ in range(warmup):
        relight_conditioned_all(model, views, lights)
    for device in devices:
        _synchronize(device)
    started = time.perf_counter()
    for _ in range(frames):
        relight_conditioned_all(model, views, lights)
    for device in devices:
        _synchronize(device)
    return (time.perf_counter() - started) / frames * 1000.0


class ViewProxy:
    """One view's resident proxy: the model plus precomputed per-pixel input tensors.

    The cache is kept only for reference renders (`gather_reference`); `render` never
    touches it — that is the multi-view latency claim being measured.
    """

    def __init__(self, name: str, model: TorchNRP, cache: PathCache, device: torch.device):
        self.name = name
        self.model = model.to(device).eval()
        self.cache = cache
        self.device = device
        self.xy, self.aux = model_tensors(cache, self.model, device)

    @property
    def model_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self.model.parameters())

    def render(self, lights: list) -> np.ndarray:
        """Eq. 3 for this view: emission-weighted sum of per-light network outputs."""
        n_px = self.cache.height * self.cache.width
        image = torch.zeros((n_px, 3), device=self.device)
        with torch.no_grad():
            for light in lights:
                params = torch.as_tensor(
                    light_param_vector(light), dtype=torch.float32, device=self.device
                ).expand(n_px, -1)
                rgb = torch.as_tensor(light.rgb, dtype=torch.float32, device=self.device)
                image += self.model(self.xy, self.aux, params) * rgb
        return (
            image.cpu().numpy().astype(np.float64).reshape(self.cache.height, self.cache.width, 3)
        )

    def gather_reference(self, lights: list) -> np.ndarray:
        """GATHERLIGHT ground truth for the same lights (cache access; checks only)."""
        return gather_lights(self.cache, lights)


def load_views(manifest_path: str, device: str = "cpu") -> list[ViewProxy]:
    """Load every view's model + cache from a manifest (see module docstring)."""
    with open(manifest_path) as f:
        entries = json.load(f)
    if isinstance(entries, dict):
        entries = entries["views"]
    base = os.path.dirname(os.path.abspath(manifest_path))
    resolve = lambda p: p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))  # noqa: E731
    dev = torch.device(device)
    return [
        ViewProxy(
            e.get("name", f"view{i}"),
            TorchNRP.load(resolve(e["model"])),
            PathCache.load(resolve(e["cache"])),
            dev,
        )
        for i, e in enumerate(entries)
    ]


def relight_all(views: list[ViewProxy], lights: list) -> dict[str, np.ndarray]:
    """Apply one light edit across all views; returns name -> image."""
    return {view.name: view.render(lights) for view in views}


def edit_latency_ms(
    views: list[ViewProxy], lights: list, frames: int = 10, warmup: int = 2
) -> float:
    """Wall-clock ms for one light edit across *all* views (device-synchronized)."""
    devices = {view.device for view in views}
    for _ in range(warmup):
        relight_all(views, lights)
    for device in devices:
        _synchronize(device)
    t0 = time.perf_counter()
    for _ in range(frames):
        relight_all(views, lights)
    for device in devices:
        _synchronize(device)
    return (time.perf_counter() - t0) / frames * 1000.0


def cross_view_consistency(views: list[ViewProxy], lights: list) -> dict:
    """For one light configuration, each view's proxy prediction vs its own
    GATHERLIGHT reference (PSNR, dB), plus the max spread across views — the §6.1
    sanity check that no view is catastrophically worse than the others."""
    rows = []
    for view in views:
        rows.append(
            {
                "view": view.name,
                "psnr_db": psnr(view.render(lights), view.gather_reference(lights)),
            }
        )
    values = [r["psnr_db"] for r in rows]
    return {
        "per_view": rows,
        "psnr_db_min": min(values),
        "psnr_db_max": max(values),
        "psnr_db_spread": max(values) - min(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--views", required=True, help="view manifest JSON (see module doc)")
    parser.add_argument(
        "--light", required=True, help="JSON file or inline JSON: one light spec or a list"
    )
    parser.add_argument("--out-dir", help="write one <view-name>.npy image per view here")
    parser.add_argument("--device", default="cpu", help="cpu/mps/cuda")
    parser.add_argument("--bench", type=int, default=0, help="benchmark N multi-view edits")
    args = parser.parse_args()

    views = load_views(args.views, device=args.device)
    try:
        spec = json.loads(args.light)
    except json.JSONDecodeError:
        with open(args.light) as f:
            spec = json.load(f)
    specs = spec if isinstance(spec, list) else [spec]
    lights = [light_from_dict(s) for s in specs]

    total_mb = sum(view.model_bytes for view in views) / 1e6
    images = relight_all(views, lights)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for name, image in images.items():
            out = os.path.join(args.out_dir, f"{name}.npy")
            np.save(out, image)
            print(f"wrote {out}")
    for name, image in images.items():
        print(f"{name}: mean radiance {image.mean():.6f}")
    print(
        f"{len(views)} resident view proxies, {total_mb:.2f} MB total, "
        f"{len(lights)} light(s), device {args.device}"
    )

    if args.bench:
        ms = edit_latency_ms(views, lights, frames=args.bench)
        print(
            f"multi-view edit latency ({len(views)} views): {ms:.2f} ms/edit "
            f"({1000.0 / ms:.1f} Hz, {ms / len(views):.2f} ms/view) over {args.bench} edits"
        )


if __name__ == "__main__":
    main()
