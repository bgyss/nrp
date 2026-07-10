"""G2 per-frame quality gate: browser demo frames vs controlled GATHERLIGHT.

For every gate-sample frame the trace runner (webgpu/demo_g2.mjs) captured — the raw
linear-HDR buffer the demo's compute shader wrote, controls included — this script
renders the matching GATHERLIGHT reference from the T1 kitchen cache, applies the
*identical* control modulations (per-pixel layer-mask linking, first-hit
linear-distance attenuation, emission tint), and runs the T3 gate at preview tier.
Both sides apply the same pixel-level control algebra, so the gate measures proxy
fidelity, not control approximation. Exits 1 unless every frame passes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.torch_backend.gather import TorchPathCache  # noqa: E402

TIER = "preview"


def effective_light(state: dict) -> SphereLight:
    """The light the shader evaluated: keyframe light with the emission tint folded in."""
    light = state["light"]
    tint = state["controls"]["rgb"]
    rgb = [light["rgb"][i] * tint[i] for i in range(3)]
    return SphereLight(center=light["center"], radius=light["radius"], rgb=rgb)


def apply_controls(
    image: np.ndarray,
    state: dict,
    link_mask: np.ndarray,
    positions: np.ndarray,
    light_center: np.ndarray,
) -> np.ndarray:
    """Apply the demo's two E8 controls to a reference image, exactly as the shader
    does (webgpu/shader_gen.mjs, demo store): linking zeroes the layer-mask pixels;
    attenuation multiplies by max(0, 1 - k * distance(first_hit, light_center))."""
    out = image.copy()
    if state["controls"]["link"]:
        out[link_mask > 0.5] = 0.0
    k = float(state["controls"]["attenuation_k"])
    if k != 0.0:
        dist = np.linalg.norm(positions - np.asarray(light_center)[None, :], axis=1)
        weight = np.maximum(0.0, 1.0 - k * dist)
        out *= weight.reshape(out.shape[0], out.shape[1])[..., None]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames-dir", default="out/g2-demo/frames")
    parser.add_argument("--export-dir", default="out/g2-demo/export")
    parser.add_argument("--cache", default="out/kitchen-512/path_cache.npz")
    parser.add_argument("--out", default="out/g2-demo/gate.json")
    parser.add_argument(
        "--device", default="cpu", help="torch device for the batched gather (cpu/mps)"
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    states = json.loads((frames_dir / "states.json").read_text())
    width, height = states["resolution"]

    export_dir = Path(args.export_dir)
    link_mask = np.fromfile(export_dir / "link_mask.bin", dtype=np.float32).reshape(height, width)
    positions = np.fromfile(export_dir / "positions.bin", dtype=np.float32).reshape(-1, 3)

    t0 = time.perf_counter()
    cache = PathCache.load(args.cache)
    torch_cache = TorchPathCache(cache, torch.device(args.device))
    load_s = time.perf_counter() - t0

    results = []
    for state in states["states"]:
        light = effective_light(state)
        pred = (
            np.fromfile(frames_dir / Path(state["file"]).name, dtype=np.float32)
            .reshape(height, width, 3)
            .astype(np.float64)
        )
        t0 = time.perf_counter()
        reference = torch_cache.gather_light(light).cpu().numpy().astype(np.float64)
        gather_s = time.perf_counter() - t0
        reference = apply_controls(reference, state, link_mask, positions, light.center)
        gate = evaluate_gate(pred, reference, TIER)
        results.append(
            {
                "index": state["index"],
                "t": state["t"],
                "controls": state["controls"],
                "light": {
                    "center": light.center.tolist(),
                    "radius": float(light.radius),
                    "rgb": light.rgb.tolist(),
                },
                "gather_seconds": gather_s,
                "quality_gate": gate,
            }
        )
        print(f"frame {state['index']:2d} t={state['t']:.1f}s: {gate['verdict']}")

    all_passed = all(r["quality_gate"]["passed"] for r in results)
    report = {
        "rung": "G2",
        "scope": (
            "per-frame preview-tier gate: browser demo output (controls applied in-shader) "
            "vs GATHERLIGHT from the T1 cache with identical pixel-level control algebra"
        ),
        "tier": TIER,
        "cache": args.cache,
        "cache_load_seconds": load_s,
        "frames": len(results),
        "all_passed": all_passed,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{'PASS' if all_passed else 'FAIL'}: {sum(r['quality_gate']['passed'] for r in results)}"
        f"/{len(results)} frames pass at {TIER} tier — wrote {args.out}"
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
