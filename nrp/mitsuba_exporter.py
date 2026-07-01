"""Export a Mitsuba 3 scene into the light-agnostic path-cache schema (paper §4.1).

The paper renders academic scenes with Mitsuba 3 and records, per path vertex, the 3D
position and throughput weight (plus escape directions), tracing from the camera with
BSDF importance sampling and *no* next-event estimation. This exporter reproduces that
pass with Mitsuba's scalar variant driven from Python: for each pixel sample it follows
BSDF-sampled bounces through the scene, recording one cache segment per path segment
(throughput *before* the segment, as in `toy_tracer`), plus first-hit G-buffer aux
features (albedo via the BSDF's diffuse reflectance, depth, shading normal, position).

Scene emitters are ignored on purpose — the pass is light-agnostic; lights are virtual
and evaluated later via GATHERLIGHT or the trained proxy.

Requires the `mitsuba` extra (`uv sync --extra mitsuba`). The scalar Python loop is
slow (minutes at 64x64 with tens of spp) but has no JIT/backend requirements; it is an
exporter, not a renderer benchmark.

Usage:
  python -m nrp.mitsuba_exporter --scene builtin:cornell-box \
      --width 48 --height 48 --spp 16 --bounces 4 --out out/mitsuba/path_cache.npz
  python -m nrp.mitsuba_exporter --scene path/to/scene.xml ...
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .path_cache import PathCache


def _load_mitsuba():
    try:
        import mitsuba as mi
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "mitsuba is not installed; run `uv sync --extra mitsuba` (see README)"
        ) from err
    mi.set_variant("scalar_rgb")
    return mi


def _load_scene(mi, spec: str, width: int, height: int):
    """Load an XML scene or the built-in cornell box, forcing the film resolution."""
    if spec == "builtin:cornell-box":
        d = mi.cornell_box()
        d["sensor"]["film"]["width"] = width
        d["sensor"]["film"]["height"] = height
        d["sensor"]["film"]["rfilter"] = {"type": "box"}
        return mi.load_dict(d)
    scene = mi.load_file(spec, resx=width, resy=height)
    return scene


def export_path_cache(
    scene,
    mi,
    width: int,
    height: int,
    spp: int,
    max_bounces: int,
    seed: int = 0,
    russian_roulette: bool = True,
    rr_start: int = 2,
) -> PathCache:
    rng = np.random.default_rng(seed)
    sensor = scene.sensors()[0]
    sampler = sensor.sampler().clone()
    sampler.seed(seed, 1)

    n_px = width * height
    n_paths = np.zeros(n_px, dtype=np.int64)
    seg_pixel: list[int] = []
    seg_origin: list[list[float]] = []
    seg_dir: list[list[float]] = []
    seg_tmax: list[float] = []
    seg_throughput: list[list[float]] = []

    albedo = np.zeros((n_px, 3))
    position = np.zeros((n_px, 3))
    depth = np.zeros(n_px)
    normal = np.zeros((n_px, 3))
    aux_weight = np.zeros(n_px)

    ctx = mi.BSDFContext()
    for py in range(height):
        for px_i in range(width):
            pixel = py * width + px_i
            for _s in range(spp):
                n_paths[pixel] += 1
                jitter = rng.random(2)
                pos_sample = [(px_i + jitter[0]) / width, (py + jitter[1]) / height]
                ray, _ = sensor.sample_ray(0.0, 0.5, pos_sample, [0.5, 0.5])
                throughput = np.ones(3)
                for bounce in range(max_bounces):
                    si = scene.ray_intersect(ray)
                    origin = np.array(ray.o, dtype=np.float64)
                    direction = np.array(ray.d, dtype=np.float64)
                    if not si.is_valid():
                        seg_pixel.append(pixel)
                        seg_origin.append(origin.tolist())
                        seg_dir.append(direction.tolist())
                        seg_tmax.append(np.inf)
                        seg_throughput.append(throughput.tolist())
                        break
                    seg_pixel.append(pixel)
                    seg_origin.append(origin.tolist())
                    seg_dir.append(direction.tolist())
                    seg_tmax.append(float(si.t))
                    seg_throughput.append(throughput.tolist())

                    bsdf = si.bsdf(ray)
                    if bounce == 0:
                        # First-hit G-buffer accumulation (averaged over samples).
                        albedo[pixel] += np.array(bsdf.eval_diffuse_reflectance(si))
                        position[pixel] += np.array(si.p, dtype=np.float64)
                        depth[pixel] += float(si.t)
                        normal[pixel] += np.array(si.sh_frame.n, dtype=np.float64)
                        aux_weight[pixel] += 1.0

                    bs, weight = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d())
                    weight = np.array(weight, dtype=np.float64)
                    if bs.pdf <= 0.0 or not np.any(weight > 0.0):
                        break
                    throughput = throughput * weight
                    if russian_roulette and bounce >= rr_start:
                        p_continue = float(np.clip(throughput.max(), 0.05, 0.95))
                        if rng.random() >= p_continue:
                            break
                        throughput = throughput / p_continue
                    ray = si.spawn_ray(si.to_world(bs.wo))

    w = np.maximum(aux_weight, 1.0)[:, None]
    normal_avg = normal / w
    norms = np.linalg.norm(normal_avg, axis=1, keepdims=True)
    normal_avg = normal_avg / np.maximum(norms, 1e-9)
    cache = PathCache(
        width=width,
        height=height,
        n_paths=n_paths,
        seg_pixel=np.asarray(seg_pixel, dtype=np.int64),
        seg_origin=np.asarray(seg_origin, dtype=np.float64).reshape(-1, 3),
        seg_dir=_normalize_rows(np.asarray(seg_dir, dtype=np.float64).reshape(-1, 3)),
        seg_tmax=np.asarray(seg_tmax, dtype=np.float64),
        seg_throughput=np.asarray(seg_throughput, dtype=np.float64).reshape(-1, 3),
        albedo=(albedo / w).reshape(height, width, 3),
        position=(position / w).reshape(height, width, 3),
        depth=(depth / w[:, 0]).reshape(height, width),
        normal=normal_avg.reshape(height, width, 3),
    )
    cache.validate()
    return cache


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scene",
        default="builtin:cornell-box",
        help="Mitsuba scene XML path, or builtin:cornell-box",
    )
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-russian-roulette",
        action="store_true",
        help="disable throughput-based RR (paper uses RR; disable for deterministic path counts)",
    )
    parser.add_argument("--out", required=True, help="output path cache .npz")
    args = parser.parse_args()

    mi = _load_mitsuba()
    scene = _load_scene(mi, args.scene, args.width, args.height)
    t0 = time.perf_counter()
    cache = export_path_cache(
        scene,
        mi,
        args.width,
        args.height,
        args.spp,
        args.bounces,
        seed=args.seed,
        russian_roulette=not args.no_russian_roulette,
    )
    seconds = time.perf_counter() - t0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cache.save(args.out)
    print(
        f"exported {cache.segment_count} segments from {args.scene} "
        f"({args.width}x{args.height} @ {args.spp} spp, {args.bounces} bounces) "
        f"in {seconds:.1f}s -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
