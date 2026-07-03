"""Exported TorchNRP runtime for engine-shaped inference experiments (E6).

The exported artifact is TorchScript plus a JSON sidecar. Runtime inference loads the
scripted graph with `torch.jit.load`; it does not instantiate `TorchNRP` or depend on
the training checkpoint format.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..lights import light_from_dict
from ..path_cache import PathCache
from .model import TorchNRP
from .train import light_param_vector, pixel_tensors


class RuntimeNRP(torch.nn.Module):
    def __init__(self, model: TorchNRP):
        super().__init__()
        self.model = model

    def forward(
        self, pixel_xy: torch.Tensor, aux: torch.Tensor, light_params: torch.Tensor
    ) -> torch.Tensor:
        return self.model(pixel_xy, aux, light_params)


def export_artifact(model: TorchNRP, artifact_path: str, example_pixels: int = 16) -> dict:
    """Trace `model` into a TorchScript artifact and write `<artifact>.json` metadata."""
    model = model.cpu().eval()
    param_dim = 4 if model.light_type == "sphere" else 8
    xy = torch.rand(example_pixels, 2)
    aux = torch.rand(example_pixels, 7)
    params = torch.rand(example_pixels, param_dim)
    traced = torch.jit.trace(RuntimeNRP(model), (xy, aux, params), strict=True)
    traced.save(artifact_path)
    metadata = {
        "format": "torchscript_nrp_runtime",
        "light_type": model.light_type,
        "model_config": model.config,
        "parameter_count": model.parameter_count,
        "artifact_bytes": os.path.getsize(artifact_path),
    }
    with open(f"{artifact_path}.json", "w") as f:
        json.dump(metadata, f, indent=2)
    return metadata


def load_runtime(artifact_path: str, device: str = "cpu") -> torch.jit.ScriptModule:
    runtime = torch.jit.load(artifact_path, map_location=device)
    runtime.eval()
    return runtime


def runtime_relight(runtime, cache: PathCache, lights: list, device: str = "cpu") -> np.ndarray:
    dev = torch.device(device)
    runtime.to(dev)
    xy, aux = pixel_tensors(cache, dev)
    n_px = cache.width * cache.height
    image = torch.zeros((n_px, 3), device=dev)
    with torch.no_grad():
        for light in lights:
            params = torch.as_tensor(
                light_param_vector(light), dtype=torch.float32, device=dev
            ).expand(n_px, -1)
            rgb = torch.as_tensor(light.rgb, dtype=torch.float32, device=dev)
            image += runtime(xy, aux, params) * rgb
    return image.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)


def runtime_latency_ms(
    runtime, cache: PathCache, lights: list, frames: int, device: str = "cpu"
) -> float:
    if frames <= 0:
        raise ValueError("frames must be positive")
    t0 = time.perf_counter()
    for _ in range(frames):
        runtime_relight(runtime, cache, lights, device=device)
    return (time.perf_counter() - t0) / frames * 1000.0


def _load_light_specs(text_or_path: str) -> list:
    try:
        spec = json.loads(text_or_path)
    except json.JSONDecodeError:
        with open(text_or_path) as f:
            spec = json.load(f)
    specs = spec if isinstance(spec, list) else [spec]
    return [light_from_dict(s) for s in specs]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    export_p = sub.add_parser("export", help="export a TorchNRP checkpoint to TorchScript")
    export_p.add_argument("--model", required=True)
    export_p.add_argument("--artifact", required=True)
    export_p.add_argument("--example-pixels", type=int, default=16)

    run_p = sub.add_parser("run", help="render through an exported TorchScript artifact")
    run_p.add_argument("--artifact", required=True)
    run_p.add_argument("--cache", required=True)
    run_p.add_argument("--light", required=True)
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--device", default="cpu")
    run_p.add_argument("--bench", type=int, default=0)

    args = parser.parse_args()
    if args.cmd == "export":
        Path(args.artifact).parent.mkdir(parents=True, exist_ok=True)
        metadata = export_artifact(TorchNRP.load(args.model), args.artifact, args.example_pixels)
        print(f"wrote {args.artifact} ({metadata['artifact_bytes']} bytes)")
        return

    runtime = load_runtime(args.artifact, args.device)
    cache = PathCache.load(args.cache)
    lights = _load_light_specs(args.light)
    image = runtime_relight(runtime, cache, lights, device=args.device)
    np.save(args.out, image)
    print(f"wrote {args.out}")
    if args.bench:
        ms = runtime_latency_ms(runtime, cache, lights, args.bench, device=args.device)
        print(f"exported runtime latency: {ms:.2f} ms/frame ({1000.0 / ms:.1f} fps)")


if __name__ == "__main__":
    main()
