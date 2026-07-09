"""E9 quality-tier ladder demo: preview, draft, final, and cached residuals.

This is a lightweight toy-scale report for the relight quality-tier plumbing. It does
not claim film/VFX final-frame quality; it verifies the core identity that
proxy-plus-cached-residual equals cached GATHERLIGHT at the approved light config and
measures how that residual decays as the light moves.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import flip, psnr, ssim, tonemap_srgb  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import render_quality_tier, write_image_with_metadata  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000.0


def finite_or_inf(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def quality_metrics(image: np.ndarray, reference: np.ndarray) -> dict:
    image_ldr = tonemap_srgb(image)
    reference_ldr = tonemap_srgb(reference)
    return {
        "psnr_vs_final_db": finite_or_inf(psnr(image, reference)),
        "ssim_vs_final": ssim(image_ldr, reference_ldr, data_range=1.0),
        "flip_vs_final": flip(image_ldr, reference_ldr),
    }


def supervisor_trust_verdict(
    residual_identity_max_abs: float,
    residual_decay: list[dict],
    identity_atol: float = 1e-12,
    psnr_threshold_db: float = 25.0,
) -> dict:
    """Toy-scale trust verdict for the E9 approval-frame ladder.

    The approved frame is trustworthy only if cached residual identity holds. Movement
    away from that approval point is trustworthy up to the largest measured dx whose
    residual-corrected image remains above the configured PSNR threshold.
    """
    approval_exact = residual_identity_max_abs <= identity_atol
    trusted_dx = 0.0 if approval_exact else None
    first_failure = None
    for row in sorted(residual_decay, key=lambda item: float(item["center_dx"])):
        value = row["psnr_db_vs_cached_gather"]
        psnr_db = math.inf if value == "inf" else float(value)
        dx = float(row["center_dx"])
        if approval_exact and psnr_db >= psnr_threshold_db:
            trusted_dx = dx
        elif first_failure is None:
            first_failure = {"center_dx": dx, "psnr_db_vs_cached_gather": value}
    return {
        "scope": "toy-scale cached-residual trust verdict",
        "approved_config_exact": bool(approval_exact),
        "identity_atol": identity_atol,
        "movement_psnr_threshold_db": psnr_threshold_db,
        "trusted_center_dx_radius": trusted_dx,
        "first_untrusted_sample": first_failure,
        "verdict": (
            "trust approved frame only; re-bake residual after any measured light move"
            if trusted_dx == 0.0
            else "trust within measured residual radius"
            if trusted_dx is not None
            else "do not trust approval frame"
        ),
        "production_claim": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/quality/report.json")
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--spp", type=int, default=8)
    parser.add_argument("--final-spp", type=int, default=32)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(args.width, args.height, args.spp, 2, seed=5)
    final_cache = trace_path_cache(args.width, args.height, args.final_spp, 2, seed=6)
    model = TorchNRP(
        hidden_width=16,
        hidden_layers=2,
        encoding={"levels": 2, "features_per_level": 2, "finest_resolution": args.width},
    )
    approved = [SphereLight(center=[0.0, 0.6, 0.0], radius=0.2, rgb=[1.2, 1.0, 0.8])]
    final_reference = gather_lights(final_cache, approved)

    tiers = {}
    for quality in ("preview", "draft", "final"):
        kwargs = {"final_cache": final_cache} if quality == "final" else {}
        (image, metadata), ms = timed(
            lambda q=quality, kw=kwargs: render_quality_tier(
                model, cache, approved, quality=q, **kw
            )
        )
        write_image_with_metadata(str(base / f"{quality}.npy"), image, metadata)
        tiers[quality] = {
            "ms": ms,
            "metadata": metadata,
            **quality_metrics(image, final_reference),
        }

    residual_image, residual_meta = render_quality_tier(
        model,
        cache,
        approved,
        quality="preview",
        residual_lights=approved,
    )
    draft = gather_lights(cache, approved)
    write_image_with_metadata(str(base / "preview_residual.npy"), residual_image, residual_meta)

    decay = []
    for dx in (0.0, 0.05, 0.1, 0.2):
        moved = [
            SphereLight(
                center=[dx, 0.6, 0.0],
                radius=0.2,
                rgb=[1.2, 1.0, 0.8],
            )
        ]
        corrected, _ = render_quality_tier(
            model,
            cache,
            moved,
            quality="preview",
            residual_lights=approved,
        )
        reference = gather_lights(cache, moved)
        decay.append(
            {
                "center_dx": dx,
                "psnr_db_vs_cached_gather": finite_or_inf(psnr(corrected, reference)),
            }
        )

    residual_identity = float(np.max(np.abs(residual_image - draft)))
    report = {
        "resolution": [args.width, args.height],
        "cache_spp": args.spp,
        "final_cache_spp": args.final_spp,
        "tiers": tiers,
        "display_metric_preprocess": "Reinhard tonemap + sRGB before SSIM/FLIP",
        "residual_identity_max_abs_diff": residual_identity,
        "residual_decay": decay,
        "supervisor_trust_verdict": supervisor_trust_verdict(residual_identity, decay),
        "outputs": {
            "preview": "preview.npy",
            "draft": "draft.npy",
            "final": "final.npy",
            "preview_residual": "preview_residual.npy",
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
