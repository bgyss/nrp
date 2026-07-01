"""GATHERLIGHT target denoisers: OIDN (paper-exact) and joint bilateral (fallback).

The paper denoises GATHERLIGHT training targets with Intel Open Image Denoise [Áfr26],
guided by the same auxiliary features the network consumes (§4.4). Two methods live
behind `denoise_image`:

- "oidn": the paper's denoiser, via the optional `oidn` python bindings
  (`uv sync --extra oidn`; on macOS the wheel additionally needs `brew install tbb`).
  Uses the RT filter in HDR mode with albedo + normal guides.
- "bilateral": a dependency-free classic cross/joint bilateral filter guided by
  albedo, normal, and depth — the same guidance signal, a much weaker prior.
"""

from __future__ import annotations

import ctypes
import functools

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


@functools.cache
def _oidn_module():
    """Import oidn once; returns (module, set_filter_bool) or raises ImportError."""
    import oidn  # noqa: PLC0415 - optional dependency

    # The 0.2.x wrapper does not expose the boolean parameter setter needed for HDR
    # mode; reach it on the already-loaded dylib (OIDN 1.4.x C API: oidnSetFilter1b).
    lib = None
    for attr in dir(oidn):
        candidate = getattr(oidn, attr)
        if isinstance(candidate, ctypes.CDLL):
            lib = candidate
            break
    set_bool = None
    if lib is not None and hasattr(lib, "oidnSetFilter1b"):
        lib.oidnSetFilter1b.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_bool]
        lib.oidnSetFilter1b.restype = None
        set_bool = lambda handle, name, value: lib.oidnSetFilter1b(  # noqa: E731
            ctypes.c_void_p(handle), name.encode(), value
        )
    return oidn, set_bool


def oidn_available() -> bool:
    try:
        _oidn_module()
        return True
    except (ImportError, OSError):
        return False


def oidn_denoise(image: np.ndarray, albedo: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Denoise an (H,W,3) HDR image with OIDN's RT filter, guided by albedo + normal."""
    oidn, set_bool = _oidn_module()
    h, w, _ = image.shape
    color = np.ascontiguousarray(image, dtype=np.float32)
    alb = np.ascontiguousarray(np.clip(albedo, 0.0, 1.0), dtype=np.float32)
    nrm = np.ascontiguousarray(normal, dtype=np.float32)
    out = np.zeros_like(color)

    device = oidn.NewDevice()
    oidn.CommitDevice(device)
    try:
        flt = oidn.NewFilter(device, "RT")
        try:
            oidn.SetSharedFilterImage(flt, "color", color, oidn.FORMAT_FLOAT3, w, h)
            oidn.SetSharedFilterImage(flt, "albedo", alb, oidn.FORMAT_FLOAT3, w, h)
            oidn.SetSharedFilterImage(flt, "normal", nrm, oidn.FORMAT_FLOAT3, w, h)
            oidn.SetSharedFilterImage(flt, "output", out, oidn.FORMAT_FLOAT3, w, h)
            if set_bool is not None:
                set_bool(flt, "hdr", True)
            oidn.CommitFilter(flt)
            oidn.ExecuteFilter(flt)
            err = oidn.GetDeviceError(device)
            if isinstance(err, tuple):  # (code, message) in some wrapper versions
                code, message = err
            else:
                code, message = err, ""
            if code != oidn.ERROR_NONE:
                raise RuntimeError(f"OIDN error {code}: {message}")
        finally:
            oidn.ReleaseFilter(flt)
    finally:
        oidn.ReleaseDevice(device)
    return out.astype(np.float64)


def denoise_image(
    image: np.ndarray,
    albedo: np.ndarray,
    normal: np.ndarray,
    depth: np.ndarray,
    method: str = "bilateral",
    **kwargs,
) -> np.ndarray:
    """Dispatch on the configured denoiser method ("bilateral" or "oidn")."""
    if method == "oidn":
        return oidn_denoise(image, albedo, normal)
    if method == "bilateral":
        return joint_bilateral_denoise(image, albedo, normal, depth, **kwargs)
    raise ValueError(f"unknown denoise method {method!r}")
