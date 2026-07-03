"""E4 textured-quad inverse recovery and texture-resolution scaling report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import TexturedQuadLight, quad_tangent_frame  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.texture_fit import (  # noqa: E402
    fit_textured_quad_light,
    textured_quad_reconstruction_error,
)

CENTER = np.array([0.0, 0.0, 1.0], dtype=np.float64)
NORMAL = np.array([0.0, 0.0, -1.0], dtype=np.float64)
WIDTH = 2.0
HEIGHT = 2.0


def make_full_rank_cache(texture_size: int, samples_per_texel: int = 3) -> PathCache:
    """Build one synthetic pixel observation per texel sample crossing the quad."""
    rng = np.random.default_rng(100 + texture_size)
    tex_h = tex_w = texture_size
    n_pixels = tex_h * tex_w * samples_per_texel
    seg_pixel = np.arange(n_pixels, dtype=np.int64)
    origins = np.zeros((n_pixels, 3), dtype=np.float64)
    dirs = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float64), (n_pixels, 1))
    tmax = np.full(n_pixels, 2.0, dtype=np.float64)
    throughput = np.zeros((n_pixels, 3), dtype=np.float64)
    u_axis, v_axis = quad_tangent_frame(NORMAL)

    idx = 0
    for y in range(tex_h):
        for x in range(tex_w):
            uv = np.array([(x + 0.5) / tex_w, (y + 0.5) / tex_h], dtype=np.float64)
            hit = CENTER + (uv[0] - 0.5) * WIDTH * u_axis + (uv[1] - 0.5) * HEIGHT * v_axis
            for _ in range(samples_per_texel):
                origins[idx] = hit - dirs[idx]
                throughput[idx] = rng.uniform(0.35, 1.25, size=3)
                idx += 1

    return PathCache(
        width=n_pixels,
        height=1,
        n_paths=np.ones(n_pixels, dtype=np.int64),
        seg_pixel=seg_pixel,
        seg_origin=origins,
        seg_dir=dirs,
        seg_tmax=tmax,
        seg_throughput=throughput,
        albedo=np.full((1, n_pixels, 3), 0.5, dtype=np.float64),
        position=np.zeros((1, n_pixels, 3), dtype=np.float64),
        depth=np.ones((1, n_pixels), dtype=np.float64),
        normal=np.tile(np.array([0.0, 0.0, 1.0]), (1, n_pixels, 1)),
    )


def make_reference_texture(texture_size: int) -> np.ndarray:
    y, x = np.meshgrid(
        np.linspace(0.0, 1.0, texture_size),
        np.linspace(0.0, 1.0, texture_size),
        indexing="ij",
    )
    return np.stack(
        [
            0.2 + 0.8 * x,
            0.3 + 0.6 * y,
            0.4 + 0.2 * np.sin(np.pi * (x + y)),
        ],
        axis=2,
    )


def run_case(texture_size: int, out_dir: Path) -> dict:
    cache = make_full_rank_cache(texture_size)
    reference = TexturedQuadLight(
        center=CENTER,
        normal=NORMAL,
        width=WIDTH,
        height=HEIGHT,
        texture=make_reference_texture(texture_size),
    )
    target = gather_light(cache, reference)
    fit = fit_textured_quad_light(
        cache,
        target,
        CENTER,
        NORMAL,
        WIDTH,
        HEIGHT,
        (texture_size, texture_size),
        reference=reference,
    )
    reconstructed = gather_light(cache, fit.light)
    np.save(out_dir / f"texture_{texture_size}x{texture_size}_reference.npy", reference.texture)
    np.save(out_dir / f"texture_{texture_size}x{texture_size}_recovered.npy", fit.light.texture)
    return {
        "texture_size": [texture_size, texture_size],
        "parameter_count": int(texture_size * texture_size * 3),
        "pixels": cache.width * cache.height,
        "segments": cache.segment_count,
        "ranks": list(fit.ranks),
        "full_rank": all(rank == texture_size * texture_size for rank in fit.ranks),
        "relative_texture_error": fit.relative_texture_error,
        "relative_image_error": textured_quad_reconstruction_error(cache, target, fit.light),
        "reconstruction_psnr_db": psnr(reconstructed, target),
        "min_singular_value": float(min(svals[-1] for svals in fit.singular_values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/textured-quad-fit/report.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cases = [run_case(size, out_path.parent) for size in (2, 4, 8)]
    report = {
        "extension": "E4",
        "scope": "textured quad inverse recovery via reference GATHERLIGHT",
        "cases": cases,
        "completion_note": (
            "This satisfies the textured-quad reference inverse-recovery slice. "
            "It does not train a proxy or measure proxy held-out PSNR versus texture size."
        ),
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    worst = max(case["relative_texture_error"] for case in cases)
    print(f"wrote {out_path}; worst relative texture error {worst:.3e}")


if __name__ == "__main__":
    main()
