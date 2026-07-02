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

Requires the `mitsuba` extra (`uv sync --extra mitsuba`). Two tracing loops share the
schema and semantics:

- **wavefront** (default when a JIT variant works): a drjit-vectorized loop under
  `llvm_ad_rgb` or `metal_ad_rgb` — all pixels' paths advance one bounce per kernel
  launch, per-bounce results are pulled to numpy and appended to the cache arrays.
- **scalar**: the original pure-Python loop under `scalar_rgb`; slow (minutes at 64x64
  with tens of spp) but has no JIT/backend requirements. Kept as the fallback and the
  semantics reference.

Usage:
  python -m nrp.mitsuba_exporter --scene builtin:cornell-box \
      --width 48 --height 48 --spp 16 --bounces 4 --out out/mitsuba/path_cache.npz
  python -m nrp.mitsuba_exporter --scene path/to/scene.xml --mode scalar ...
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .path_cache import PathCache

#: JIT variants usable for the wavefront loop, in preference order. LLVM is tried
#: first (CPU, works everywhere LLVM is present); Metal covers macOS wheels where
#: drjit finds no libLLVM.
JIT_VARIANTS = ("llvm_ad_rgb", "metal_ad_rgb")


def _import_mitsuba():
    try:
        import mitsuba as mi
    except ImportError as err:  # pragma: no cover - exercised only without the extra
        raise SystemExit(
            "mitsuba is not installed; run `uv sync --extra mitsuba` (see README)"
        ) from err
    return mi


def pick_jit_variant(mi) -> str | None:
    """Return the first JIT variant that actually initializes, or None.

    A variant can be compiled in yet fail at init (e.g. llvm_ad_rgb without a system
    libLLVM), so each candidate is probed with a tiny jitted op.
    """
    for variant in JIT_VARIANTS:
        if variant not in mi.variants():
            continue
        try:
            mi.set_variant(variant)
            import drjit as dr

            dr.eval(dr.arange(mi.UInt32, 8) + 1)
            return variant
        except Exception:
            continue
    return None


def _load_mitsuba(mode: str = "scalar"):
    """Import mitsuba and set the variant for `mode` ('scalar' or 'wavefront')."""
    mi = _import_mitsuba()
    if mode == "scalar":
        mi.set_variant("scalar_rgb")
    elif mode == "wavefront":
        variant = pick_jit_variant(mi)
        if variant is None:
            raise SystemExit(f"no working JIT variant among {JIT_VARIANTS}; use --mode scalar")
        mi.set_variant(variant)
    else:
        raise ValueError(f"unknown mode {mode!r}")
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
                    # Skip degenerate segments (see the wavefront loop for rationale).
                    if float(si.t) > 0.0 and np.isfinite(si.t):
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


def export_path_cache_wavefront(
    scene,
    mi,
    width: int,
    height: int,
    spp: int,
    max_bounces: int,
    seed: int = 0,
    russian_roulette: bool = True,
    rr_start: int = 2,
    max_wavefront: int = 1 << 20,
) -> PathCache:
    """drjit-vectorized SAMPLEPATHS: same semantics as `export_path_cache`.

    All pixels advance one bounce per kernel launch; spp is chunked so the wavefront
    never exceeds `max_wavefront` lanes. Requires a JIT variant to be active
    (`_load_mitsuba("wavefront")`).
    """
    import drjit as dr

    sensor = scene.sensors()[0]
    n_px = width * height
    spp_chunk = max(1, min(spp, max_wavefront // n_px))

    seg_pixel: list[np.ndarray] = []
    seg_origin: list[np.ndarray] = []
    seg_dir: list[np.ndarray] = []
    seg_tmax: list[np.ndarray] = []
    seg_throughput: list[np.ndarray] = []

    albedo = np.zeros((n_px, 3))
    position = np.zeros((n_px, 3))
    depth = np.zeros(n_px)
    normal = np.zeros((n_px, 3))
    aux_weight = np.zeros(n_px)

    ctx = mi.BSDFContext()
    done = 0
    chunk_index = 0
    while done < spp:
        chunk = min(spp_chunk, spp - done)
        n_lanes = n_px * chunk
        sampler = sensor.sampler().fork()
        sampler.seed(seed * 0x10000 + chunk_index, n_lanes)
        chunk_index += 1
        done += chunk

        lane = dr.arange(mi.UInt32, n_lanes)
        pixel = lane // chunk
        pixel_np = np.array(pixel, dtype=np.int64)
        px = mi.Float(pixel % width)
        py = mi.Float(pixel // width)
        pos_sample = mi.Point2f((px + sampler.next_1d()) / width, (py + sampler.next_1d()) / height)
        ray, _ = sensor.sample_ray(mi.Float(0.0), mi.Float(0.5), pos_sample, mi.Point2f(0.5, 0.5))
        throughput = dr.full(mi.Spectrum, 1.0, n_lanes)
        active = dr.full(mi.Bool, True, n_lanes)

        for bounce in range(max_bounces):
            si = scene.ray_intersect(ray, active=active)
            valid = active & si.is_valid()

            active_np = np.array(active)
            if not active_np.any():
                break
            origin_np = np.array(ray.o).T
            dir_np = np.array(ray.d).T
            tmax_np = np.array(si.t, dtype=np.float64)  # inf where si is invalid
            tp_np = np.array(mi.unpolarized_spectrum(throughput)).T
            # Drop degenerate segments (t <= 0 self-intersections, non-finite data
            # from grazing geometry in real scenes): zero-length segments cannot
            # overlap any light, so this loses nothing; the path itself continues.
            record = (
                active_np
                & (tmax_np > 0.0)
                & np.isfinite(origin_np).all(axis=1)
                & np.isfinite(dir_np).all(axis=1)
                & np.isfinite(tp_np).all(axis=1)
            )
            seg_pixel.append(pixel_np[record])
            seg_origin.append(origin_np[record])
            seg_dir.append(dir_np[record])
            seg_tmax.append(tmax_np[record])
            seg_throughput.append(tp_np[record])

            bsdf = si.bsdf(ray)
            if bounce == 0:
                valid_np = np.array(valid)
                pix = pixel_np[valid_np]
                alb_np = np.array(bsdf.eval_diffuse_reflectance(si)).T
                np.add.at(albedo, pix, alb_np[valid_np])
                np.add.at(position, pix, np.array(si.p).T[valid_np])
                np.add.at(depth, pix, tmax_np[valid_np])
                np.add.at(normal, pix, np.array(si.sh_frame.n).T[valid_np])
                np.add.at(aux_weight, pix, 1.0)

            bs, weight = bsdf.sample(
                ctx, si, sampler.next_1d(active), sampler.next_2d(active), valid
            )
            weight_s = mi.unpolarized_spectrum(weight)
            active = valid & (bs.pdf > 0.0) & (dr.max(weight_s) > 0.0)
            throughput = dr.select(active, throughput * weight, throughput)
            if russian_roulette and bounce >= rr_start:
                p_continue = dr.clip(dr.max(mi.unpolarized_spectrum(throughput)), 0.05, 0.95)
                active &= sampler.next_1d(active) < p_continue
                throughput = dr.select(active, throughput / p_continue, throughput)
            ray = si.spawn_ray(si.to_world(bs.wo))

    w = np.maximum(aux_weight, 1.0)[:, None]
    normal_avg = normal / w
    norms = np.linalg.norm(normal_avg, axis=1, keepdims=True)
    normal_avg = normal_avg / np.maximum(norms, 1e-9)
    cache = PathCache(
        width=width,
        height=height,
        n_paths=np.full(n_px, spp, dtype=np.int64),
        seg_pixel=np.concatenate(seg_pixel),
        seg_origin=np.concatenate(seg_origin).astype(np.float64),
        seg_dir=_normalize_rows(np.concatenate(seg_dir).astype(np.float64)),
        seg_tmax=np.concatenate(seg_tmax),
        seg_throughput=np.concatenate(seg_throughput).astype(np.float64),
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
    parser.add_argument(
        "--mode",
        choices=["auto", "scalar", "wavefront"],
        default="auto",
        help="tracing loop: drjit wavefront (JIT), scalar Python, or auto "
        "(wavefront when a JIT variant works, else scalar)",
    )
    parser.add_argument("--out", required=True, help="output path cache .npz")
    args = parser.parse_args()

    mode = args.mode
    if mode == "auto":
        mode = "wavefront" if pick_jit_variant(_import_mitsuba()) else "scalar"
    mi = _load_mitsuba(mode)
    scene = _load_scene(mi, args.scene, args.width, args.height)
    export = export_path_cache_wavefront if mode == "wavefront" else export_path_cache
    t0 = time.perf_counter()
    cache = export(
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
        f"({args.width}x{args.height} @ {args.spp} spp, {args.bounces} bounces, "
        f"{mode}[{mi.variant()}]) "
        f"in {seconds:.1f}s -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
