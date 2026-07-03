"""E4 environment-light inverse recovery report.

This report exercises the closed-form SH recovery path on a deterministic escaped-ray
cache. It is a reference GATHERLIGHT validation, not a trained proxy-quality study.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.environment_fit import (  # noqa: E402
    environment_reconstruction_error,
    fit_environment_light,
)
from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import EnvironmentLight, sh_basis_degree2  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402


def make_escaped_cache(width: int = 48, seed: int = 7) -> PathCache:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(width, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    # Regenerate if the synthetic directions do not span degree-2 SH space.
    for _ in range(8):
        if np.linalg.matrix_rank(sh_basis_degree2(dirs), tol=1e-10) == 9:
            break
        dirs = rng.normal(size=(width, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    throughputs = rng.uniform(0.35, 1.0, size=(width, 3))
    return PathCache(
        width=width,
        height=1,
        n_paths=np.ones(width, dtype=np.int64),
        seg_pixel=np.arange(width, dtype=np.int64),
        seg_origin=np.zeros((width, 3), dtype=np.float64),
        seg_dir=dirs,
        seg_tmax=np.full(width, np.inf, dtype=np.float64),
        seg_throughput=throughputs,
        albedo=np.full((1, width, 3), 0.5, dtype=np.float64),
        position=np.zeros((1, width, 3), dtype=np.float64),
        depth=np.ones((1, width), dtype=np.float64),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, width, 1)),
    )


def make_reference_light(seed: int = 11) -> EnvironmentLight:
    rng = np.random.default_rng(seed)
    coeffs = rng.normal(scale=0.12, size=(9, 3))
    coeffs[0] = [0.8, 0.7, 0.6]
    return EnvironmentLight(coeffs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/environment-fit/report.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cache = make_escaped_cache()
    reference = make_reference_light()
    target = gather_light(cache, reference)
    fit = fit_environment_light(cache, target, reference=reference)
    reconstructed = gather_light(cache, fit.light)
    reconstruction = environment_reconstruction_error(cache, target, fit.light)
    image_psnr = psnr(reconstructed, target)

    np.save(out_path.parent / "target.npy", target)
    np.save(out_path.parent / "reconstructed.npy", reconstructed)
    np.save(out_path.parent / "recovered_coeffs.npy", fit.light.coeffs)

    report = {
        "extension": "E4",
        "scope": "degree-2 SH environment inverse recovery via reference GATHERLIGHT",
        "cache": {
            "width": cache.width,
            "height": cache.height,
            "segments": cache.segment_count,
            "escaped_segments": int(np.isinf(cache.seg_tmax).sum()),
        },
        "least_squares": {
            "rank": fit.rank,
            "unknowns": 27,
            "equations": int(cache.width * cache.height * 3),
            "relative_coeff_error": fit.relative_coeff_error,
            "min_singular_value": float(fit.singular_values[-1]),
            "max_singular_value": float(fit.singular_values[0]),
        },
        "reconstruction": {
            **reconstruction,
            "psnr_db": "inf" if math.isinf(image_psnr) else float(image_psnr),
        },
        "artifacts": {
            "target": "target.npy",
            "reconstructed": "reconstructed.npy",
            "recovered_coeffs": "recovered_coeffs.npy",
        },
        "limitations": [
            "This is SH environment recovery only; textured quad inverse recovery is not included.",
            "This does not measure proxy PSNR versus texture parameter count.",
        ],
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
