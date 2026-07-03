"""Linear inverse recovery for textured quad lights (extension E4).

For a fixed path cache and fixed quad geometry, textured-quad GATHERLIGHT is linear in
the texture texel RGB values: each segment crossing the quad contributes its
throughput to exactly one nearest-neighbor texel. This module builds that design
matrix and solves the least-squares inverse problem independently per color channel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gather_light import gather_light
from .lights import TexturedQuadLight, segment_quad_uv
from .path_cache import PathCache


@dataclass
class TextureFitResult:
    light: TexturedQuadLight
    ranks: tuple[int, int, int]
    singular_values: tuple[np.ndarray, np.ndarray, np.ndarray]
    residuals: tuple[np.ndarray, np.ndarray, np.ndarray]
    relative_texture_error: float | None = None


def textured_quad_design_matrices(
    cache: PathCache,
    center: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
    texture_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-channel design matrices mapping flattened texels to pixels."""
    tex_h, tex_w = texture_shape
    if tex_h <= 0 or tex_w <= 0:
        raise ValueError(f"texture_shape must be positive, got {texture_shape}")
    n_pixels = cache.width * cache.height
    n_texels = tex_h * tex_w
    designs = tuple(np.zeros((n_pixels, n_texels), dtype=np.float64) for _ in range(3))
    if not cache.segment_count:
        return designs

    hits, uv = segment_quad_uv(
        cache.seg_origin,
        cache.seg_dir,
        cache.seg_tmax,
        center,
        normal,
        width,
        height,
    )
    if not hits.any():
        return designs

    ij = np.floor(uv[hits] * np.array([tex_w, tex_h])).astype(np.int64)
    ij[:, 0] = np.clip(ij[:, 0], 0, tex_w - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, tex_h - 1)
    texel_ids = ij[:, 1] * tex_w + ij[:, 0]
    pixels = cache.seg_pixel[hits]
    denom = np.maximum(cache.n_paths, 1).astype(np.float64)
    throughputs = cache.seg_throughput[hits]
    for channel, design in enumerate(designs):
        values = throughputs[:, channel] / denom[pixels]
        np.add.at(design, (pixels, texel_ids), values)
    return designs


def fit_textured_quad_light(
    cache: PathCache,
    target: np.ndarray,
    center: np.ndarray,
    normal: np.ndarray,
    width: float,
    height: float,
    texture_shape: tuple[int, int],
    reference: TexturedQuadLight | None = None,
) -> TextureFitResult:
    """Recover a nearest-neighbor RGB texture from a target image for fixed geometry."""
    target = np.asarray(target, dtype=np.float64)
    if target.shape != (cache.height, cache.width, 3):
        raise ValueError(f"target must be {(cache.height, cache.width, 3)}, got {target.shape}")

    designs = textured_quad_design_matrices(cache, center, normal, width, height, texture_shape)
    target_flat = target.reshape(cache.height * cache.width, 3)
    tex_h, tex_w = texture_shape
    texture = np.zeros((tex_h * tex_w, 3), dtype=np.float64)
    ranks: list[int] = []
    singular_values: list[np.ndarray] = []
    residuals: list[np.ndarray] = []
    for channel, design in enumerate(designs):
        coeffs, residual, rank, svals = np.linalg.lstsq(design, target_flat[:, channel], rcond=None)
        texture[:, channel] = coeffs
        ranks.append(int(rank))
        singular_values.append(svals)
        residuals.append(residual)

    light = TexturedQuadLight(
        center=center,
        normal=normal,
        width=width,
        height=height,
        texture=texture.reshape(tex_h, tex_w, 3),
    )
    relative = None
    if reference is not None:
        denom = np.linalg.norm(reference.texture)
        relative = float(np.linalg.norm(light.texture - reference.texture) / max(denom, 1e-12))
    return TextureFitResult(
        light=light,
        ranks=tuple(ranks),
        singular_values=tuple(singular_values),
        residuals=tuple(residuals),
        relative_texture_error=relative,
    )


def textured_quad_reconstruction_error(
    cache: PathCache,
    target: np.ndarray,
    light: TexturedQuadLight,
) -> float:
    """Relative image reconstruction error after fitting a textured quad."""
    pred = gather_light(cache, light)
    return float(np.linalg.norm(pred - target) / max(np.linalg.norm(target), 1e-12))
