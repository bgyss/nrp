"""G2 per-frame quality gate: browser demo frames vs controlled GATHERLIGHT.

For every gate-sample frame the trace runner (webgpu/demo_g2.mjs) captured — the raw
linear-HDR buffer the demo's compute shader wrote, controls included — this script
renders the matching GATHERLIGHT reference from the T1 kitchen cache, denoises it
(OIDN by default — the reference class the proxy was *supervised* on, §4.4; the raw
64-spp gather's MC noise bounds SSIM at ~0.2-0.35 regardless of proxy quality, and
the raw-reference metrics are recorded per frame for that comparison), applies the
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
from nrp.torch_backend.denoise import denoise_image, oidn_available  # noqa: E402
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
    parser.add_argument(
        "--denoise",
        default="oidn",
        choices=["oidn", "bilateral", "none"],
        help="denoiser for the gate reference; the T1 proxy was supervised on "
        "oidn-denoised gathers, so oidn is the apples-to-apples reference "
        "(needs the nix devshell for libtbb — run under `nix develop --command`)",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    states = json.loads((frames_dir / "states.json").read_text())
    width, height = states["resolution"]

    export_dir = Path(args.export_dir)
    link_mask = np.fromfile(export_dir / "link_mask.bin", dtype=np.float32).reshape(height, width)
    positions = np.fromfile(export_dir / "positions.bin", dtype=np.float32).reshape(-1, 3)
    # The denoiser's guides live in the exported pixel blob: per pixel xy(2) +
    # albedo(3) + depth(1) + normal(3).
    pixel_blob = np.fromfile(export_dir / "pixels.bin", dtype=np.float32).reshape(-1, 9)
    albedo = pixel_blob[:, 2:5].astype(np.float64).reshape(height, width, 3)
    depth = pixel_blob[:, 5].astype(np.float64).reshape(height, width)
    normal = pixel_blob[:, 6:9].astype(np.float64).reshape(height, width, 3)
    if args.denoise == "oidn" and not oidn_available():
        raise SystemExit(
            "oidn unavailable — run under `nix develop --command` (libtbb), or pass "
            "--denoise bilateral/none (the gate reference then differs from the "
            "proxy's supervision target)"
        )

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
        raw = torch_cache.gather_light(light).cpu().numpy().astype(np.float64)
        gather_s = time.perf_counter() - t0
        if args.denoise == "none":
            reference = raw
        else:
            reference = denoise_image(raw, albedo, normal, depth, method=args.denoise)
        reference = apply_controls(reference, state, link_mask, positions, light.center)
        raw_controlled = apply_controls(raw, state, link_mask, positions, light.center)
        gate = evaluate_gate(pred, reference, TIER)
        raw_gate = evaluate_gate(pred, raw_controlled, TIER)
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
                "raw_reference_metrics": {
                    name: raw_gate["metrics"][name].get("value")
                    for name in ("psnr_db", "ssim", "flip")
                },
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
        "reference": (
            "raw gather" if args.denoise == "none" else f"{args.denoise}-denoised gather"
        ),
        "raw_reference_note": (
            "raw_reference_metrics per frame quote the same pred against the un-denoised "
            "64-spp gather; its MC noise bounds SSIM well below the preview threshold "
            "independently of proxy quality"
        ),
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
