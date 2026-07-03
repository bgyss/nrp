"""Linear inverse recovery for degree-2 SH environment lights (extension E4).

GATHERLIGHT is linear in `EnvironmentLight.coeffs`, so the reference inverse problem
for a fixed cache can be solved directly with least squares. This module is deliberately
numpy-only: it validates the richer-light vocabulary and gives the torch optimizer a
closed-form target to compare against later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gather_light import gather_light
from .lights import EnvironmentLight, sh_basis_degree2
from .path_cache import PathCache


@dataclass
class EnvironmentFitResult:
    light: EnvironmentLight
    residuals: np.ndarray
    rank: int
    singular_values: np.ndarray
    relative_coeff_error: float | None = None


def environment_design_matrix(cache: PathCache) -> np.ndarray:
    """Return A such that `A @ coeffs.reshape(27)` equals the gathered image.

    Coefficients are grouped by RGB channel: all 9 red SH coefficients, then green,
    then blue. The output rows follow numpy image flattening order, `(pixel, channel)`.
    """
    n_pixels = cache.height * cache.width
    design = np.zeros((n_pixels * 3, 27), dtype=np.float64)
    if not cache.segment_count:
        return design

    escaped = np.isinf(cache.seg_tmax)
    if not escaped.any():
        return design

    escaped_pixels = cache.seg_pixel[escaped]
    basis = sh_basis_degree2(cache.seg_dir[escaped])
    throughputs = cache.seg_throughput[escaped]
    denom = np.maximum(cache.n_paths, 1).astype(np.float64)

    for segment_index, pixel in enumerate(escaped_pixels):
        weighted_basis = basis[segment_index] / denom[pixel]
        for channel in range(3):
            row = int(pixel) * 3 + channel
            col0 = channel * 9
            design[row, col0 : col0 + 9] += throughputs[segment_index, channel] * weighted_basis
    return design


def fit_environment_light(
    cache: PathCache,
    target: np.ndarray,
    *,
    reference: EnvironmentLight | None = None,
    rcond: float | None = None,
) -> EnvironmentFitResult:
    """Recover SH environment coefficients from a target image for a fixed cache."""
    target = np.asarray(target, dtype=np.float64)
    expected_shape = (cache.height, cache.width, 3)
    if target.shape != expected_shape:
        raise ValueError(f"target must be {expected_shape}, got {target.shape}")

    design = environment_design_matrix(cache)
    solution, residuals, rank, singular_values = np.linalg.lstsq(
        design, target.reshape(-1), rcond=rcond
    )
    coeffs = solution.reshape(3, 9).T
    light = EnvironmentLight(coeffs)
    relative_coeff_error = None
    if reference is not None:
        denom = max(float(np.linalg.norm(reference.coeffs)), 1e-12)
        relative_coeff_error = float(np.linalg.norm(coeffs - reference.coeffs) / denom)
    return EnvironmentFitResult(
        light=light,
        residuals=residuals,
        rank=int(rank),
        singular_values=singular_values,
        relative_coeff_error=relative_coeff_error,
    )


def environment_reconstruction_error(
    cache: PathCache, target: np.ndarray, light: EnvironmentLight
) -> dict:
    reconstructed = gather_light(cache, light)
    delta = reconstructed - target
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
    }
