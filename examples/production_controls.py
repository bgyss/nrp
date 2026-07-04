"""E8 gather-time production controls: light linking and custom attenuation.

This report covers the cache/GATHERLIGHT workaround path only. It intentionally does
not claim live proxy-conditioned controls; the report records that as open work.
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

from nrp.gather_light import (  # noqa: E402
    GatherControls,
    gather_light,
    gather_light_controlled,
    segment_hits_sphere,
)
from nrp.lights import SphereLight  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.toy_tracer import layer_ownership_mask, trace_path_cache  # noqa: E402


class BinaryLinkProxy:
    """Tiny E8 proxy for one binary linking toggle.

    It stores the two approved light states (linking inactive/active) and interpolates
    by a scalar control. This is intentionally a table proxy, not a learned neural
    proxy; the report compares its exactness/latency against gather-time fallback and
    an equivalent two-proxy table.
    """

    def __init__(self, inactive: np.ndarray, active: np.ndarray):
        self.inactive = np.asarray(inactive, dtype=np.float64)
        self.active = np.asarray(active, dtype=np.float64)
        if self.inactive.shape != self.active.shape:
            raise ValueError("inactive and active images must have matching shape")

    @property
    def parameter_count(self) -> int:
        return int(self.inactive.size + self.active.size)

    def predict(self, link_active: float) -> np.ndarray:
        t = float(np.clip(link_active, 0.0, 1.0))
        return (1.0 - t) * self.inactive + t * self.active


class LinearAttenuationProxy:
    """Least-squares proxy conditioned on linear-distance attenuation controls.

    The proxy learns image-space coefficients for controls `(intercept, slope)` from
    GATHERLIGHT examples, then predicts held-out attenuation settings without segment
    traversal. It is intentionally narrow: fixed light geometry and one curve family.
    """

    def __init__(self, coeffs: np.ndarray, image_shape: tuple[int, int, int]):
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.image_shape = image_shape
        if self.coeffs.shape != (2, int(np.prod(image_shape))):
            raise ValueError("coeffs must be (2, H*W*C)")

    @property
    def parameter_count(self) -> int:
        return int(self.coeffs.size)

    @classmethod
    def fit(cls, controls: np.ndarray, images: list[np.ndarray]) -> LinearAttenuationProxy:
        controls = np.asarray(controls, dtype=np.float64)
        if controls.ndim != 2 or controls.shape[1] != 2:
            raise ValueError("controls must be (N, 2): intercept and slope")
        if len(images) != controls.shape[0]:
            raise ValueError("number of images must match controls")
        image_shape = images[0].shape
        if any(image.shape != image_shape for image in images):
            raise ValueError("all images must have matching shape")
        y = np.stack([np.asarray(image, dtype=np.float64).reshape(-1) for image in images], axis=0)
        coeffs, _, _, _ = np.linalg.lstsq(controls, y, rcond=None)
        return cls(coeffs, image_shape)

    def predict(self, intercept: float, slope: float) -> np.ndarray:
        features = np.array([intercept, slope], dtype=np.float64)
        return (features @ self.coeffs).reshape(self.image_shape)


class BasisControlProxy:
    """Least-squares proxy for arbitrary masks and polynomial attenuation curves."""

    def __init__(self, coeffs: np.ndarray, image_shape: tuple[int, int, int]):
        self.coeffs = np.asarray(coeffs, dtype=np.float64)
        self.image_shape = image_shape

    @property
    def parameter_count(self) -> int:
        return int(self.coeffs.size)

    @classmethod
    def fit(cls, features: np.ndarray, images: list[np.ndarray]) -> BasisControlProxy:
        features = np.asarray(features, dtype=np.float64)
        image_shape = images[0].shape
        y = np.stack([np.asarray(image, dtype=np.float64).reshape(-1) for image in images], axis=0)
        coeffs, _, _, _ = np.linalg.lstsq(features, y, rcond=None)
        return cls(coeffs, image_shape)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) @ self.coeffs).reshape(self.image_shape)


def mask_basis(width: int, height: int) -> list[np.ndarray]:
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    return [
        np.ones((height, width), dtype=np.float64),
        (xs >= width // 2).astype(np.float64),
        (ys >= height // 2).astype(np.float64),
        (((xs - width / 2.0) ** 2 + (ys - height / 2.0) ** 2) <= (min(width, height) / 4.0) ** 2)
        .astype(np.float64),
    ]


def mask_from_weights(basis: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    raw = sum(float(w) * b for w, b in zip(weights, basis, strict=True))
    return np.clip(raw, 0.0, 1.0)


def apply_soft_link_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return image * (1.0 - mask[..., None])


def polynomial_attenuated_gather(cache, light: SphereLight, coeffs: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(cache.seg_origin - light.center[None, :], axis=1)
    weights = np.maximum(
        0.0,
        sum(float(c) * distance**power for power, c in enumerate(coeffs)),
    )
    hits = segment_hits_sphere(
        cache.seg_origin,
        cache.seg_dir,
        cache.seg_tmax,
        light.center,
        light.radius,
    )
    contrib = np.zeros((cache.height * cache.width, 3), dtype=np.float64)
    if hits.any():
        np.add.at(
            contrib,
            cache.seg_pixel[hits],
            cache.seg_throughput[hits] * weights[hits, None] * light.rgb,
        )
    denom = np.maximum(cache.n_paths, 1).astype(np.float64)
    return (contrib / denom[:, None]).reshape(cache.height, cache.width, 3)


def timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000.0


def finite_or_inf(value: float) -> float | str:
    return value if math.isfinite(value) else "inf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/production-controls/report.json")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--spp", type=int, default=12)
    parser.add_argument("--bounces", type=int, default=2)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cache = trace_path_cache(args.width, args.height, args.spp, args.bounces, seed=8)
    sphere_cache = trace_path_cache(
        args.width, args.height, args.spp, args.bounces, seed=8, layer="sphere"
    )
    box_cache = trace_path_cache(
        args.width, args.height, args.spp, args.bounces, seed=8, layer="box"
    )
    light = SphereLight(center=[0.0, 0.55, 0.0], radius=0.22, rgb=[1.0, 0.8, 0.6])

    full, full_ms = timed(lambda: gather_light(cache, light))
    sphere, _ = timed(lambda: gather_light(sphere_cache, light))
    box, _ = timed(lambda: gather_light(box_cache, light))
    linked, linked_ms = timed(
        lambda: gather_light_controlled(
            cache,
            light,
            GatherControls(
                exclude_pixel_mask=layer_ownership_mask(args.width, args.height, "sphere")
            ),
        )
    )
    attenuated, attenuation_ms = timed(
        lambda: gather_light_controlled(
            cache,
            light,
            GatherControls(
                attenuation={"type": "linear_distance", "intercept": 1.0, "slope": -0.1}
            ),
        )
    )
    conditioned = BinaryLinkProxy(inactive=full, active=linked)
    proxy_inactive, proxy_inactive_ms = timed(lambda: conditioned.predict(0.0))
    proxy_active, proxy_active_ms = timed(lambda: conditioned.predict(1.0))
    two_proxy_parameter_count = int(full.size + linked.size)
    attenuation_train_controls = np.array(
        [
            [0.9, -0.04],
            [1.0, -0.08],
            [1.15, -0.12],
            [1.25, -0.16],
        ],
        dtype=np.float64,
    )
    attenuation_train_images = [
        gather_light_controlled(
            cache,
            light,
            GatherControls(
                attenuation={
                    "type": "linear_distance",
                    "intercept": float(intercept),
                    "slope": float(slope),
                }
            ),
        )
        for intercept, slope in attenuation_train_controls
    ]
    attenuation_proxy = LinearAttenuationProxy.fit(
        attenuation_train_controls, attenuation_train_images
    )
    heldout_intercept, heldout_slope = 1.1, -0.10
    heldout_reference, heldout_gather_ms = timed(
        lambda: gather_light_controlled(
            cache,
            light,
            GatherControls(
                attenuation={
                    "type": "linear_distance",
                    "intercept": heldout_intercept,
                    "slope": heldout_slope,
                }
            ),
        )
    )
    heldout_proxy, heldout_proxy_ms = timed(
        lambda: attenuation_proxy.predict(heldout_intercept, heldout_slope)
    )

    report = {
        "resolution": [args.width, args.height],
        "segments": cache.segment_count,
        "linking": {
            "full_equals_sphere_plus_box_max_abs": float(np.max(np.abs(full - (sphere + box)))),
            "exclude_sphere_psnr_vs_box_layer_db": finite_or_inf(psnr(linked, box)),
            "exclude_sphere_max_abs_vs_box_layer": float(np.max(np.abs(linked - box))),
            "full_gather_ms": full_ms,
            "linked_gather_ms": linked_ms,
        },
        "attenuation": {
            "curve": {"type": "linear_distance", "intercept": 1.0, "slope": -0.1},
            "attenuated_gather_ms": attenuation_ms,
            "mean_radiance_ratio_vs_default": float(attenuated.mean() / max(full.mean(), 1e-12)),
        },
        "proxy_conditioned_controls": {
            "implemented": True,
            "kind": "binary table proxy for linking toggle plus learned linear attenuation proxy",
            "parameter_count": conditioned.parameter_count,
            "two_proxy_parameter_count": two_proxy_parameter_count,
            "inactive_psnr_vs_gather_db": finite_or_inf(psnr(proxy_inactive, full)),
            "active_psnr_vs_gather_db": finite_or_inf(psnr(proxy_active, linked)),
            "inactive_max_abs_vs_gather": float(np.max(np.abs(proxy_inactive - full))),
            "active_max_abs_vs_gather": float(np.max(np.abs(proxy_active - linked))),
            "inactive_predict_ms": proxy_inactive_ms,
            "active_predict_ms": proxy_active_ms,
            "edit_latency_speedup_vs_gather_time": linked_ms / max(proxy_active_ms, 1e-12),
            "attenuation_proxy": {
                "kind": "least-squares image proxy conditioned on intercept and slope",
                "train_controls": attenuation_train_controls.tolist(),
                "heldout_control": {
                    "intercept": heldout_intercept,
                    "slope": heldout_slope,
                },
                "parameter_count": attenuation_proxy.parameter_count,
                "heldout_psnr_vs_gather_db": finite_or_inf(
                    psnr(heldout_proxy, heldout_reference)
                ),
                "heldout_max_abs_vs_gather": float(
                    np.max(np.abs(heldout_proxy - heldout_reference))
                ),
                "heldout_gather_ms": heldout_gather_ms,
                "heldout_predict_ms": heldout_proxy_ms,
                "heldout_speedup_vs_gather_time": heldout_gather_ms
                / max(heldout_proxy_ms, 1e-12),
            },
            "finding": (
                "binary linking and a fixed-family continuous attenuation control can stay "
                "live at proxy speed; arbitrary masks and attenuation curves remain open"
            ),
        },
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
