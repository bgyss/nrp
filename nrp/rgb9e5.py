"""Shared-exponent rgb9e5 HDR packing (paper §4.2, cache compression).

Implements the 32-bit rgb9e5 format from the OpenGL `EXT_texture_shared_exponent`
spec: three 9-bit unsigned mantissas sharing one 5-bit biased exponent
(N=9, E=5, bias B=15), packed little-endian as r | g<<9 | b<<18 | exp<<27.
There is no sign bit and no implicit leading 1, so the format covers
[0, (2^9-1)/2^9 * 2^(31-15)] ≈ [0, 65408] with ~9 bits of relative precision on
the dominant channel; negative inputs clamp to 0, NaN encodes as 0, and values
above the max clamp to it (all per the spec's conversion rules).

The paper stores per-segment path throughput in this format; throughput is
non-negative HDR data whose channels are correlated, which is exactly the case
shared-exponent formats are designed for (4x smaller than float32 per texel with
no visible banding).
"""

from __future__ import annotations

import numpy as np

MANTISSA_BITS = 9
EXPONENT_BITS = 5
EXPONENT_BIAS = 15
_MANTISSA_VALUES = 1 << MANTISSA_BITS  # 512
_MAX_BIASED_EXP = (1 << EXPONENT_BITS) - 1  # 31

# Largest representable value: full mantissa at the largest exponent.
MAX_RGB9E5 = (_MANTISSA_VALUES - 1) / _MANTISSA_VALUES * 2.0 ** (_MAX_BIASED_EXP - EXPONENT_BIAS)


def rgb9e5_encode(rgb: np.ndarray) -> np.ndarray:
    """Pack (..., 3) non-negative floats into (...,) uint32 rgb9e5 words."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.shape[-1] != 3:
        raise ValueError(f"expected trailing dim 3, got shape {rgb.shape}")
    c = np.clip(np.nan_to_num(rgb, nan=0.0), 0.0, MAX_RGB9E5)
    max_c = c.max(axis=-1)

    # Shared exponent from the dominant channel (spec: floor(log2)+1, clamped so
    # the smallest exponent covers the denormal range down to 0).
    with np.errstate(divide="ignore"):
        exp_p = np.floor(np.log2(max_c, where=max_c > 0, out=np.full_like(max_c, -np.inf)))
    exp_shared = np.maximum(-(EXPONENT_BIAS + 1), exp_p) + 1 + EXPONENT_BIAS
    exp_shared = np.where(max_c > 0, exp_shared, 0.0)

    # Spec refinement: if rounding the max channel overflows the mantissa, bump
    # the exponent by one.
    scale = 2.0 ** (exp_shared - EXPONENT_BIAS - MANTISSA_BITS)
    max_s = np.floor(max_c / scale + 0.5)
    exp_shared = np.where(max_s == _MANTISSA_VALUES, exp_shared + 1, exp_shared)
    scale = 2.0 ** (exp_shared - EXPONENT_BIAS - MANTISSA_BITS)

    mant = np.floor(c / scale[..., None] + 0.5).astype(np.uint32)
    mant = np.minimum(mant, _MANTISSA_VALUES - 1)
    e = exp_shared.astype(np.uint32)
    return mant[..., 0] | (mant[..., 1] << 9) | (mant[..., 2] << 18) | (e << 27)


def rgb9e5_decode(packed: np.ndarray) -> np.ndarray:
    """Unpack (...,) uint32 rgb9e5 words to (..., 3) float64."""
    packed = np.asarray(packed, dtype=np.uint32)
    mask = np.uint32(_MANTISSA_VALUES - 1)
    r = packed & mask
    g = (packed >> 9) & mask
    b = (packed >> 18) & mask
    e = (packed >> 27).astype(np.float64)
    scale = 2.0 ** (e - EXPONENT_BIAS - MANTISSA_BITS)
    return np.stack([r, g, b], axis=-1).astype(np.float64) * scale[..., None]
