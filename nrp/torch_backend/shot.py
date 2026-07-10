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

import numpy as np

from ..metrics import flip, tonemap_srgb

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
