"""S4: decompose the rig additivity residual per light (names the next quality floor).

The additivity gate compares `LightRig.render` (sum of 8 per-light proxies) against
the denoised 8-light GATHERLIGHT reference. By Eq. 1 linearity the total residual is
exactly the sum of per-light residuals (proxy_i - reference_i), so each light's
share of the residual energy attributes the gate failure. Writes
out/s4-rig/residual_decomposition.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light, gather_lights  # noqa: E402
from nrp.metrics import psnr, ssim  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.denoise import denoise_image, oidn_available  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.rig import LightRig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rig", default="out/s4-rig/rig.json")
    parser.add_argument("--models-dir", default="out/s4-rig/models")
    parser.add_argument("--cache", default="out/kitchen-512/path_cache.npz")
    parser.add_argument("--out", default="out/s4-rig/residual_decomposition.json")
    parser.add_argument("--denoise", default="oidn", choices=["oidn", "bilateral"])
    args = parser.parse_args()

    if args.denoise == "oidn" and not oidn_available():
        raise SystemExit("oidn unavailable -- run under `nix develop --command`")

    with open(args.rig) as f:
        rig_dict = json.load(f)
    models = {
        name: TorchNRP.load(str(Path(args.models_dir) / Path(rel).name))
        for name, rel in rig_dict["models"].items()
    }
    rig = LightRig.from_dict(rig_dict, models)
    cache = PathCache.load(args.cache)

    def dn(img: np.ndarray) -> np.ndarray:
        return denoise_image(img, cache.albedo, cache.normal, cache.depth, method=args.denoise)

    lights = rig.active_lights()
    total_ref = dn(gather_lights(cache, [rl.light for rl in lights]))
    total_pred = rig.render(cache)
    total_residual = total_pred - total_ref

    per_light_pred = rig.render_per_light(cache)
    rows = []
    residual_energy_total = float(np.square(total_residual).sum())
    for rl in lights:
        ref_i = dn(gather_light(cache, rl.light))
        pred_i = per_light_pred[rl.name]
        r_i = pred_i - ref_i
        rows.append(
            {
                "light": rl.name,
                "light_type": type(rl.light).__name__,
                "proxy_vs_own_denoised_gather_psnr_db": float(psnr(pred_i, ref_i)),
                "proxy_vs_own_denoised_gather_ssim": float(ssim(pred_i, ref_i)),
                "residual_mean_abs": float(np.abs(r_i).mean()),
                "residual_energy_share": float(np.square(r_i).sum())
                / max(residual_energy_total, 1e-30),
                "reference_mean": float(ref_i.mean()),
            }
        )
    rows.sort(key=lambda r: -r["residual_energy_share"])
    report = {
        "rung": "S4",
        "scope": "per-light decomposition of the additivity-gate residual",
        "note": (
            "per-light residual energies are computed against per-light denoised "
            "references; denoising is not exactly linear, so shares are approximate "
            "attribution (they need not sum to 1) -- ordering is what matters"
        ),
        "total": {
            "psnr_db": float(psnr(total_pred, total_ref)),
            "ssim": float(ssim(total_pred, total_ref)),
            "residual_mean_abs": float(np.abs(total_residual).mean()),
        },
        "per_light": rows,
    }
    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["total"], indent=2))
    for r in rows:
        print(
            f"{r['light']:14s} {r['light_type']:18s} share {r['residual_energy_share']:.3f} "
            f"psnr {r['proxy_vs_own_denoised_gather_psnr_db']:.2f} "
            f"ssim {r['proxy_vs_own_denoised_gather_ssim']:.3f}"
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
