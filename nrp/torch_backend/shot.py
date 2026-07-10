"""Shot harness with temporal stability (production track, rung F1).

Renders a keyframed-light shot (the E1 keyframe JSON format, >= 120 frames at
full scale) through the E9 quality-tier ladder with a per-frame T3 trust
verdict, plus the temporal metric the ladder lacks: frame-to-frame FLIP delta
(flicker is the failure mode per-frame PSNR cannot see).

Tier mapping on a single-cache scene (the T1 kitchen ships one 64-spp cache,
no higher-spp final cache): preview = proxy inference, draft = raw cached
GATHERLIGHT, final = denoised cached GATHERLIGHT — the reference class the
proxy is supervised on (Sec. 4.4; the same choice as the G2 gate, where the
raw gather's MC noise bounds SSIM at ~0.2-0.44 regardless of proxy quality).
Raw-reference metrics are recorded per frame alongside the gate.

Temporal check: the light moves, so consecutive frames legitimately differ —
flicker is frame-to-frame perceptual change the *reference* does not have. Per
consecutive pair we compute FLIP between the tonemapped frames for the proxy
sequence and for the final-tier reference sequence; the proxy passes iff its
per-pair delta never exceeds the reference's by more than
``temporal_excess_max``. A deliberately flickering baseline — the reference
plus per-frame independent Gaussian noise scaled to a fixed per-frame PSNR —
must fail the same check (F1's verification requirement: individually the
noised frames hold that PSNR, temporally they flicker).

Frames are processed streamingly (only the previous frame of each sequence
stays resident), so a 120-frame 512x512 shot never holds the ~2 GiB of frames.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..metrics import flip, psnr, tonemap_srgb
from ..path_cache import PathCache
from ..quality.gate import evaluate_gate
from .animate import frame_times, interpolate_light_spec, lights_at
from .denoise import denoise_image, oidn_available
from .gather import TorchPathCache
from .model import TorchNRP
from .relight import relight

TIER_ORDER = ("preview", "draft", "final")


def temporal_flip_delta(prev_hdr: np.ndarray, curr_hdr: np.ndarray) -> float:
    """Frame-to-frame FLIP between two consecutive Reinhard-tonemapped frames."""
    return flip(tonemap_srgb(prev_hdr), tonemap_srgb(curr_hdr))


def delta_stats(deltas: list[float]) -> dict:
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def temporal_check(
    candidate_deltas: list[float],
    reference_deltas: list[float],
    excess_max: float,
) -> dict:
    """Pass iff no per-pair candidate FLIP delta exceeds the reference's by more
    than ``excess_max``. Sequences must be pair-aligned (same delta count)."""
    if len(candidate_deltas) != len(reference_deltas):
        raise ValueError("delta sequences must have equal length")
    excess = [c - r for c, r in zip(candidate_deltas, reference_deltas, strict=True)]
    worst = max(excess) if excess else 0.0
    passed = bool(excess) and worst <= excess_max
    return {
        "excess_max_allowed": excess_max,
        "candidate": delta_stats(candidate_deltas),
        "reference": delta_stats(reference_deltas),
        "excess": delta_stats(excess),
        "passed": passed,
        "verdict": (
            "no temporal-delta pairs evaluated"
            if not excess
            else f"pass: worst excess FLIP delta {worst:.4f} <= {excess_max}"
            if passed
            else f"fail: worst excess FLIP delta {worst:.4f} > {excess_max}"
        ),
    }


def noise_sigma_for_psnr(reference: np.ndarray, psnr_db: float) -> float:
    """Gaussian sigma such that reference+noise has the target PSNR in
    expectation under `nrp.metrics.psnr`'s peak-is-reference-max convention."""
    ref = np.asarray(reference, dtype=np.float64)
    peak = float(ref.max()) if ref.size and ref.max() > 0 else 1.0
    return peak * 10.0 ** (-psnr_db / 20.0)


def flickering_baseline_frame(
    reference_hdr: np.ndarray,
    rng: np.random.Generator,
    psnr_db: float = 30.0,
) -> np.ndarray:
    """One frame of the deliberately flickering baseline: the reference plus
    per-frame *independent* Gaussian noise at a fixed per-frame PSNR."""
    sigma = noise_sigma_for_psnr(reference_hdr, psnr_db)
    return reference_hdr + rng.normal(0.0, sigma, size=reference_hdr.shape)


def light_specs_at(spec: dict, t: float) -> list[dict]:
    """JSON-ready interpolated light specs at time ``t`` (for the report rows;
    `lights_at` builds the light objects from the same interpolation)."""
    tracks = spec.get("lights")
    if tracks is None:
        tracks = [{"keyframes": spec["keyframes"]}]
    return [interpolate_light_spec(track["keyframes"], t) for track in tracks]


def render_shot(
    model: TorchNRP,
    cache: PathCache,
    spec: dict,
    out_dir: Path,
    denoise_method: str = "bilateral",
    device: str = "cpu",
    gate_tier: str = "preview",
    temporal_excess_max: float = 0.02,
    baseline_psnr_db: float = 30.0,
    save_frames: bool = False,
    seed: int = 0,
) -> dict:
    """Render the shot through the tier ladder and write ``out_dir/report.json``.

    Per frame: preview (proxy), draft (raw cached GATHERLIGHT via the torch
    backend), final (denoised draft; ``denoise_method="none"`` keeps the raw
    gather), each timed; the T3 gate at ``gate_tier`` scores preview against
    final, with raw-reference metrics recorded alongside. Temporal FLIP deltas
    are accumulated streamingly for the preview sequence, the final-tier
    reference sequence, and the flickering baseline.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    times = frame_times(int(spec["frames"]))
    torch_cache = TorchPathCache(cache, torch.device(device))
    rng = np.random.default_rng(seed)

    frames_detail: list[dict] = []
    tier_ms: dict[str, list[float]] = {tier: [] for tier in TIER_ORDER}
    deltas: dict[str, list[float]] = {"preview": [], "final": [], "flicker_baseline": []}
    prev: dict[str, np.ndarray | None] = {k: None for k in deltas}
    baseline_frame_psnrs: list[float] = []

    for idx, t in enumerate(times):
        t_f = float(t)
        lights = lights_at(spec, t_f)

        t0 = time.perf_counter()
        preview = relight(model, cache, lights)
        preview_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        draft_t = torch_cache.gather_light(lights[0])
        for light in lights[1:]:
            draft_t = draft_t + torch_cache.gather_light(light)
        draft = draft_t.cpu().numpy().astype(np.float64)
        draft_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if denoise_method == "none":
            final = draft
        else:
            final = denoise_image(
                draft, cache.albedo, cache.normal, cache.depth, method=denoise_method
            )
        denoise_ms = (time.perf_counter() - t0) * 1000.0

        gate = evaluate_gate(preview, final, gate_tier)
        raw_gate = evaluate_gate(preview, draft, gate_tier)

        baseline = flickering_baseline_frame(final, rng, baseline_psnr_db)
        baseline_frame_psnrs.append(psnr(baseline, final))

        for name, frame in (
            ("preview", preview),
            ("final", final),
            ("flicker_baseline", baseline),
        ):
            if prev[name] is not None:
                deltas[name].append(temporal_flip_delta(prev[name], frame))
            prev[name] = frame

        tier_ms["preview"].append(preview_ms)
        tier_ms["draft"].append(draft_ms)
        tier_ms["final"].append(draft_ms + denoise_ms)  # final renders the gather too

        if save_frames:
            np.save(out_dir / f"preview_{idx:04d}.npy", preview)
            np.save(out_dir / f"final_{idx:04d}.npy", final)

        frames_detail.append(
            {
                "index": idx,
                "t": t_f,
                "lights": light_specs_at(spec, t_f),
                "tier_ms": {
                    "preview": preview_ms,
                    "draft": draft_ms,
                    "final": draft_ms + denoise_ms,
                },
                "quality_gate": gate,
                "raw_reference_metrics": {
                    name: raw_gate["metrics"][name].get("value")
                    for name in ("psnr_db", "ssim", "flip")
                },
            }
        )

    proxy_temporal = temporal_check(deltas["preview"], deltas["final"], temporal_excess_max)
    baseline_temporal = temporal_check(
        deltas["flicker_baseline"], deltas["final"], temporal_excess_max
    )
    gate_pass_count = sum(row["quality_gate"]["passed"] for row in frames_detail)
    report = {
        "rung": "F1",
        "scope": (
            "keyframed-light shot through the E9 tier ladder with per-frame T3 "
            "trust verdicts and a frame-to-frame FLIP temporal-stability check"
        ),
        "frames": len(times),
        "resolution": [cache.width, cache.height],
        "gate_tier": gate_tier,
        "denoise": denoise_method,
        "tier_definitions": {
            "preview": "proxy inference (relight)",
            "draft": "raw cached GATHERLIGHT (torch backend)",
            "final": (
                "raw cached GATHERLIGHT"
                if denoise_method == "none"
                else f"{denoise_method}-denoised cached GATHERLIGHT "
                "(single-cache scene; supervision-class reference, Sec. 4.4)"
            ),
        },
        "per_frame_gate_pass_count": int(gate_pass_count),
        "all_frames_pass_gate": bool(gate_pass_count == len(times)),
        "per_tier_render_ms": {tier: delta_stats(tier_ms[tier]) for tier in TIER_ORDER},
        "temporal": {
            "metric": "frame-to-frame FLIP on Reinhard-tonemapped sRGB frames",
            "per_pair_deltas": deltas,
            "proxy_check": proxy_temporal,
            "flicker_baseline_check": baseline_temporal,
            "flicker_baseline_frame_psnr_db": delta_stats(baseline_frame_psnrs),
            "baseline_construction": (
                "final-tier reference + per-frame independent Gaussian noise at "
                f"~{baseline_psnr_db} dB per-frame PSNR (flicker per-frame PSNR cannot see)"
            ),
        },
        "verification": {
            "proxy_passes_temporal_check": proxy_temporal["passed"],
            "flicker_baseline_fails_temporal_check": not baseline_temporal["passed"],
        },
        "frames_detail": frames_detail,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="trained model .pt")
    parser.add_argument("--cache", required=True, help="path cache .npz")
    parser.add_argument("--keyframes", required=True, help="shot keyframe JSON (E1 format)")
    parser.add_argument("--out-dir", required=True, help="directory for report.json")
    parser.add_argument(
        "--denoise",
        default="oidn",
        choices=["oidn", "bilateral", "none"],
        help="final-tier denoiser; the T1 proxy was supervised on oidn-denoised "
        "gathers (needs the nix devshell for libtbb — `nix develop --command`)",
    )
    parser.add_argument("--device", default="cpu", help="torch device for the gather")
    parser.add_argument("--gate-tier", default="preview", help="T3 tier for the per-frame gate")
    parser.add_argument(
        "--temporal-excess-max",
        type=float,
        default=0.02,
        help="max allowed per-pair FLIP delta excess over the reference sequence",
    )
    parser.add_argument(
        "--baseline-psnr-db",
        type=float,
        default=30.0,
        help="per-frame PSNR of the deliberately flickering baseline",
    )
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.denoise == "oidn" and not oidn_available():
        raise SystemExit(
            "oidn unavailable — run under `nix develop --command` (libtbb), or pass "
            "--denoise bilateral/none (the final tier then differs from the proxy's "
            "supervision target)"
        )

    model = TorchNRP.load(args.model)
    cache = PathCache.load(args.cache)
    with open(args.keyframes) as f:
        spec = json.load(f)
    report = render_shot(
        model,
        cache,
        spec,
        Path(args.out_dir),
        denoise_method=args.denoise,
        device=args.device,
        gate_tier=args.gate_tier,
        temporal_excess_max=args.temporal_excess_max,
        baseline_psnr_db=args.baseline_psnr_db,
        save_frames=args.save_frames,
        seed=args.seed,
    )
    ok = all(report["verification"].values())
    print(
        f"{'PASS' if ok else 'FAIL'}: {report['per_frame_gate_pass_count']}/{report['frames']} "
        f"frames pass {report['gate_tier']} tier; proxy temporal "
        f"{report['temporal']['proxy_check']['verdict']}; baseline "
        f"{report['temporal']['flicker_baseline_check']['verdict']} — "
        f"wrote {Path(args.out_dir) / 'report.json'}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
