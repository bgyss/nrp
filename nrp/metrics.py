"""Quality metrics for NRP experiments.

PSNR and SMAPE are implemented directly. LPIPS and SSIM are *not* implemented (this
project is numpy-only); PSNR/SMAPE serve as the clearly documented substitute, and any
LPIPS number would need torch/LPIPS installed separately and must say so.
"""

from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def psnr(pred: np.ndarray, ref: np.ndarray, peak: float | None = None) -> float:
    """PSNR in dB. For HDR images `peak` defaults to the reference max (documented
    convention; there is no fixed peak for unbounded radiance)."""
    ref = np.asarray(ref, dtype=np.float64)
    if peak is None:
        peak = float(ref.max()) if ref.size and ref.max() > 0 else 1.0
    err = mse(pred, ref)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10(peak**2 / err))


def smape(pred: np.ndarray, ref: np.ndarray, eps: float = 1e-3) -> float:
    """Symmetric mean absolute percentage error in [0, 2]."""
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.mean(2.0 * np.abs(pred - ref) / (np.abs(pred) + np.abs(ref) + eps)))
