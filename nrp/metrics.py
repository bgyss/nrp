"""Quality metrics for NRP experiments.

PSNR, SMAPE, SSIM, and LDR-FLIP are implemented directly in numpy (roadmap item 10;
the paper's Table 2 / Fig. 7 report SMAPE/PSNR/SSIM/FLIP). LPIPS is *not* implemented;
any LPIPS number would need torch/LPIPS installed separately and must say so.

SSIM follows Wang et al., "Image Quality Assessment: From Error Visibility to
Structural Similarity", IEEE TIP 13(4), 2004 (Eq. 13): 11×11 Gaussian window
(σ = 1.5), K1 = 0.01, K2 = 0.03.

FLIP follows Andersson et al., "FLIP: A Difference Evaluator for Alternating Images",
Proc. ACM CGIT 3(2), 2020 (LDR-FLIP): YCxCz CSF prefiltering, Hunt-adjusted L*a*b*
HyAB color difference with the paper's exponent/redistribution constants
(q_C = 0.7, p_C = 0.4, p_t = 0.95), Gaussian-derivative edge/point feature difference
(w = 0.082, q_F = 0.5), combined as ΔE = ΔE_c^(1−ΔE_f). One documented deviation from
the reference implementation: convolutions use edge-replicate padding instead of
zero-fill, so constant images are preserved exactly at the borders (important at this
repo's tiny resolutions). Inputs are display-encoded sRGB in [0,1] — tonemap HDR
radiance first (see `tonemap_srgb`).
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


def tonemap_srgb(hdr: np.ndarray) -> np.ndarray:
    """Reinhard-tonemap linear HDR radiance and sRGB-encode to display [0,1] — the
    documented preprocessing for the display-referred metrics below (SSIM, FLIP)."""
    x = np.clip(np.asarray(hdr, dtype=np.float64), 0.0, None)
    return _linrgb_to_srgb(x / (1.0 + x))


# ---------------------------------------------------------------------------
# SSIM [Wang et al. 2004]


def _gaussian_kernel(sigma: float, radius: int) -> np.ndarray:
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def _filter_sep(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 2D correlation with edge-replicate padding, (H,W) image."""
    r = len(k) // 2
    p = np.pad(img, ((r, r), (0, 0)), mode="edge")
    img = np.lib.stride_tricks.sliding_window_view(p, len(k), axis=0) @ k
    p = np.pad(img, ((0, 0), (r, r)), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(p, len(k), axis=1) @ k


def ssim(
    pred: np.ndarray,
    ref: np.ndarray,
    data_range: float | None = None,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Mean SSIM (Wang et al. 2004, Eq. 13) over an (H,W) or (H,W,C) image pair.

    `data_range` is the dynamic range L in C1 = (K1·L)², C2 = (K2·L)²; it defaults to
    the reference max (same HDR convention as `psnr`). For tonemapped/display images
    pass 1.0 explicitly.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {ref.shape}")
    if pred.ndim == 2:
        pred, ref = pred[..., None], ref[..., None]
    if data_range is None:
        data_range = float(ref.max()) if ref.size and ref.max() > 0 else 1.0
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    k = _gaussian_kernel(sigma, radius=5)  # 11×11 window, the paper's setting
    means = []
    for c in range(pred.shape[2]):
        x, y = pred[..., c], ref[..., c]
        mu_x, mu_y = _filter_sep(x, k), _filter_sep(y, k)
        var_x = _filter_sep(x * x, k) - mu_x**2
        var_y = _filter_sep(y * y, k) - mu_y**2
        cov = _filter_sep(x * y, k) - mu_x * mu_y
        num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
        den = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
        means.append(np.mean(num / den))
    return float(np.mean(means))


# ---------------------------------------------------------------------------
# LDR-FLIP [Andersson et al. 2020]

# sRGB→XYZ (IEC 61966-2-1, D65); the reference white is the matrix applied to (1,1,1).
_RGB2XYZ = np.array(
    [
        [0.41238656, 0.35759149, 0.18045049],
        [0.21263682, 0.71516870, 0.07219232],
        [0.01933062, 0.11919716, 0.95037259],
    ]
)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE = _RGB2XYZ @ np.ones(3)


def _srgb_to_linrgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linrgb_to_srgb(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)


def _linrgb_to_ycxcz(rgb: np.ndarray) -> np.ndarray:
    xyz = rgb @ _RGB2XYZ.T / _WHITE
    y = 116.0 * xyz[..., 1] - 16.0
    cx = 500.0 * (xyz[..., 0] - xyz[..., 1])
    cz = 200.0 * (xyz[..., 1] - xyz[..., 2])
    return np.stack([y, cx, cz], axis=-1)


def _ycxcz_to_linrgb(ycc: np.ndarray) -> np.ndarray:
    yn = (ycc[..., 0] + 16.0) / 116.0
    xn = yn + ycc[..., 1] / 500.0
    zn = yn - ycc[..., 2] / 200.0
    xyz = np.stack([xn, yn, zn], axis=-1) * _WHITE
    return xyz @ _XYZ2RGB.T


def _linrgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    xyz = np.clip(rgb, 0.0, None) @ _RGB2XYZ.T / _WHITE
    d = 6.0 / 29.0
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d**2) + 4.0 / 29.0)
    lum = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lum, a, b], axis=-1)


def _hunt(lab: np.ndarray) -> np.ndarray:
    """Hunt adjustment (FLIP §3.2): scale chroma by luminance."""
    return np.stack(
        [lab[..., 0], 0.01 * lab[..., 0] * lab[..., 1], 0.01 * lab[..., 0] * lab[..., 2]], axis=-1
    )


def _hyab(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """HyAB distance: |ΔL*| + ‖Δ(a*,b*)‖₂."""
    d = p - q
    return np.abs(d[..., 0]) + np.sqrt(d[..., 1] ** 2 + d[..., 2] ** 2)


def _csf_filter(ppd: float, a1: float, b1: float, a2: float, b2: float) -> np.ndarray:
    """1D spatial CSF filter (FLIP Eqs. 2–3): sum of two Gaussians given in the
    frequency domain, evaluated in the spatial domain on a pixel grid of ppd
    pixels/degree, normalized to unit sum. Separable, so 1D suffices."""
    # Common support across channels so filtered channels stay aligned (radius from
    # the widest spatial Gaussian, b = 0.04).
    radius = int(np.ceil(3 * np.sqrt(0.04 / (2 * np.pi**2)) * ppd))
    x = np.arange(-radius, radius + 1, dtype=np.float64) / ppd
    g = a1 * np.sqrt(np.pi / b1) * np.exp(-(np.pi**2) * x**2 / b1)
    if a2 > 0:
        g = g + a2 * np.sqrt(np.pi / b2) * np.exp(-(np.pi**2) * x**2 / b2)
    return g / g.sum()


# (a1, b1, a2, b2) per YCxCz channel — Table 1 of the FLIP paper.
_CSF_PARAMS = [(1.0, 0.0047, 0.0, 1e-5), (1.0, 0.0053, 0.0, 1e-5), (34.1, 0.04, 13.5, 0.025)]


def _feature_kernels(ppd: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(smoothing, edge, point) 1D kernels: Gaussian (σ = 0.5·w·ppd px, w = 0.082°)
    and its first/second derivatives, positive and negative lobes normalized to ±1
    (FLIP §3.3)."""
    sd = 0.5 * 0.082 * ppd
    radius = int(np.ceil(3 * sd))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(x**2) / (2 * sd**2))
    edge = -x * g
    point = (x**2 / sd**2 - 1) * g

    def norm_lobes(k):
        pos, neg = k[k > 0].sum(), -k[k < 0].sum()
        return np.where(k > 0, k / pos, k / neg if neg > 0 else k)

    return g / g.sum(), norm_lobes(edge), norm_lobes(point)


def _correlate_axis(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    r = len(k) // 2
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="edge")
    return np.lib.stride_tricks.sliding_window_view(p, len(k), axis=axis) @ k


def flip_error_map(pred: np.ndarray, ref: np.ndarray, ppd: float = 67.02) -> np.ndarray:
    """Per-pixel LDR-FLIP ΔE in [0,1] for two (H,W,3) sRGB images in [0,1].

    Default `ppd` (pixels per degree) is the FLIP paper's observer: 0.7 m from a
    0.7 m-wide 4K monitor.
    """
    qc, pc, pt, qf = 0.7, 0.4, 0.95, 0.5
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0.0, 1.0)
    ref = np.clip(np.asarray(ref, dtype=np.float64), 0.0, 1.0)
    if pred.shape != ref.shape or pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError(f"expected matching (H,W,3) images, got {pred.shape} vs {ref.shape}")

    ycc_pred = _linrgb_to_ycxcz(_srgb_to_linrgb(pred))
    ycc_ref = _linrgb_to_ycxcz(_srgb_to_linrgb(ref))

    # --- color pipeline: CSF prefilter -> clamp to gamut -> Hunt-adjusted Lab, HyAB.
    def prefilter(ycc):
        out = np.stack(
            [_filter_sep(ycc[..., c], _csf_filter(ppd, *_CSF_PARAMS[c])) for c in range(3)],
            axis=-1,
        )
        return _hunt(_linrgb_to_lab(np.clip(_ycxcz_to_linrgb(out), 0.0, 1.0)))

    hyab = _hyab(prefilter(ycc_pred), prefilter(ycc_ref)) ** qc
    green = _hunt(_linrgb_to_lab(np.array([0.0, 1.0, 0.0])))
    blue = _hunt(_linrgb_to_lab(np.array([0.0, 0.0, 1.0])))
    cmax = float(_hyab(green, blue)) ** qc
    delta_c = np.where(
        hyab < pc * cmax,
        (pt / (pc * cmax)) * hyab,
        pt + ((hyab - pc * cmax) / (cmax - pc * cmax)) * (1.0 - pt),
    )

    # --- feature pipeline on the achromatic channel, remapped to [0,1].
    smooth, edge, point = _feature_kernels(ppd)

    def magnitudes(y):
        mags = []
        for deriv in (edge, point):
            dx = _correlate_axis(_correlate_axis(y, deriv, 1), smooth, 0)
            dy = _correlate_axis(_correlate_axis(y, deriv, 0), smooth, 1)
            mags.append(np.sqrt(dx**2 + dy**2))
        return mags

    y_pred = (ycc_pred[..., 0] + 16.0) / 116.0
    y_ref = (ycc_ref[..., 0] + 16.0) / 116.0
    (edge_p, point_p), (edge_r, point_r) = magnitudes(y_pred), magnitudes(y_ref)
    delta_f = (np.maximum(np.abs(edge_r - edge_p), np.abs(point_r - point_p)) / np.sqrt(2.0)) ** qf

    return delta_c ** (1.0 - delta_f)


def flip(pred: np.ndarray, ref: np.ndarray, ppd: float = 67.02) -> float:
    """Mean LDR-FLIP error (lower is better, 0 = identical)."""
    return float(np.mean(flip_error_map(pred, ref, ppd)))
