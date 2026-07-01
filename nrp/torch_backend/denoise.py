"""Auxiliary-feature-guided joint bilateral denoiser (documented OIDN substitute).

The paper denoises GATHERLIGHT training targets with Intel Open Image Denoise [Áfr26],
guided by the same auxiliary features the network consumes (§4.4). OIDN is a
pretrained CNN and a heavyweight native dependency; this module substitutes a classic
cross/joint bilateral filter guided by albedo, normal, and depth — the same guidance
signal, a much weaker prior. The substitution is documented in the README; anyone
wanting parity with the paper can swap in `oidn` python bindings behind the same
function signature.
"""

from __future__ import annotations

import numpy as np


def joint_bilateral_denoise(
    image: np.ndarray,
    albedo: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    radius: int = 2,
    sigma_spatial: float = 2.0,
    sigma_albedo: float = 0.2,
    sigma_normal: float = 0.3,
    sigma_depth: float = 0.5,
) -> np.ndarray:
    """Denoise an (H,W,3) HDR image with edge-stopping weights from the G-buffer.

    Weights for a neighbor at offset (dy,dx):
      exp(-(dy²+dx²)/2σs²) · exp(-|Δalbedo|²/2σa²) · exp(-|Δnormal|²/2σn²) · exp(-Δdepth²/2σd²)
    computed per pixel via shifted arrays (no Python loop over pixels).
    """
    h, w, _ = image.shape
    acc = np.zeros_like(image)
    wsum = np.zeros((h, w, 1))

    def shifted(arr: np.ndarray, dy: int, dx: int) -> np.ndarray:
        # Clamp-to-edge shift keeps borders usable without shrinking the image.
        ys = np.clip(np.arange(h) + dy, 0, h - 1)
        xs = np.clip(np.arange(w) + dx, 0, w - 1)
        return arr[np.ix_(ys, xs)]

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            w_spatial = np.exp(-(dy * dy + dx * dx) / (2.0 * sigma_spatial**2))
            d_albedo = np.sum((shifted(albedo, dy, dx) - albedo) ** 2, axis=2)
            d_normal = np.sum((shifted(normal, dy, dx) - normal) ** 2, axis=2)
            d_depth = (shifted(depth, dy, dx) - depth) ** 2
            weight = w_spatial * np.exp(
                -d_albedo / (2.0 * sigma_albedo**2)
                - d_normal / (2.0 * sigma_normal**2)
                - d_depth / (2.0 * sigma_depth**2)
            )
            acc += weight[:, :, None] * shifted(image, dy, dx)
            wsum += weight[:, :, None]
    return acc / np.maximum(wsum, 1e-12)
