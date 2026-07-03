"""Minimal educational path tracer + path-cache exporter (NRP M3, preferred-order #2).

Scope: one hard-coded Cornell-style scene (unit-box interior, colored side walls, one
diffuse sphere), pinhole camera fixed inside the box, Lambertian surfaces only,
cosine-weighted hemisphere sampling, no explicit light sampling (lights are virtual and
evaluated later via GATHERLIGHT). This is deliberately a toy: it exists to produce real
(not hand-authored) path caches and an independent rendered reference, not to compete
with any production renderer.

The "direct rendered reference" for a light is the same tracer run with an independent
seed, evaluating the (transparent, purely emissive) sphere light inline along each
traced segment — which for virtual lights is exactly the definition GATHERLIGHT
implements, but over an independent path set, so agreement between the two is a real
Monte Carlo consistency check rather than a tautology.
"""

from __future__ import annotations

import argparse

import numpy as np

from .gather_light import gather_light
from .lights import SphereLight
from .path_cache import PathCache

EPS = 1e-6

# Hard-coded Cornell-style scene: unit box interior.
WALL_ALBEDOS = {
    # axis, side(0=min,1=max) -> albedo
    (0, 0): np.array([0.75, 0.15, 0.15]),  # left wall, red
    (0, 1): np.array([0.15, 0.75, 0.15]),  # right wall, green
    (1, 0): np.array([0.75, 0.75, 0.75]),  # floor
    (1, 1): np.array([0.75, 0.75, 0.75]),  # ceiling
    (2, 0): np.array([0.75, 0.75, 0.75]),  # front (behind camera)
    (2, 1): np.array([0.75, 0.75, 0.75]),  # back
}
SPHERE_CENTER = np.array([0.35, 0.28, 0.62])
SPHERE_RADIUS = 0.22
SPHERE_ALBEDO = np.array([0.55, 0.55, 0.70])

CAM_POS = np.array([0.5, 0.5, 0.08])
CAM_FOV_DEG = 68.0  # horizontal

#: Compositing layers (§6.1, Fig. 11): the scene decomposes into the foreground
#: sphere and the background box by *first-hit* ownership. A layer's paths still
#: bounce off the full scene geometry; the layer only owns the paths (and pixels)
#: whose first hit lands on its object.
LAYERS = ("sphere", "box")


def _camera_rays(
    width: int, height: int, jitter: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole rays looking down +z. jitter is (N,2) in [0,1) or None for pixel centers."""
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    px = xs.reshape(-1).astype(np.float64)
    py = ys.reshape(-1).astype(np.float64)
    if jitter is None:
        jx = jy = 0.5
    else:
        jx, jy = jitter[:, 0], jitter[:, 1]
    half = np.tan(np.radians(CAM_FOV_DEG) / 2.0)
    u = ((px + jx) / width * 2.0 - 1.0) * half
    v = -((py + jy) / height * 2.0 - 1.0) * half * (height / width)
    dirs = np.stack([u, v, np.ones_like(u)], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(CAM_POS, dirs.shape).copy()
    return origins, dirs


def _intersect_scene(origins: np.ndarray, dirs: np.ndarray):
    """Nearest hit for rays strictly inside the closed box.

    Returns (t, normal, albedo, is_sphere) — every ray hits something (the box is
    closed); `is_sphere` marks rays whose nearest hit is the sphere (layer id)."""
    n = origins.shape[0]
    t_best = np.full(n, np.inf)
    normal = np.zeros((n, 3))
    albedo = np.zeros((n, 3))

    # Box walls: for each axis, the exit plane is picked by the direction sign.
    for axis in range(3):
        d = dirs[:, axis]
        for side, bound in ((0, 0.0), (1, 1.0)):
            going = d < -EPS if side == 0 else d > EPS
            t = np.where(going, (bound - origins[:, axis]) / np.where(going, d, 1.0), np.inf)
            valid = going & (t > EPS) & (t < t_best)
            if valid.any():
                t_best = np.where(valid, t, t_best)
                nvec = np.zeros(3)
                nvec[axis] = 1.0 if side == 0 else -1.0  # inward normal
                normal[valid] = nvec
                albedo[valid] = WALL_ALBEDOS[(axis, side)]

    # Sphere.
    oc = origins - SPHERE_CENTER
    b = np.einsum("ij,ij->i", oc, dirs)
    c = np.einsum("ij,ij->i", oc, oc) - SPHERE_RADIUS**2
    disc = b * b - c
    has = disc > 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    t_s = np.where(t0 > EPS, t0, np.where(t1 > EPS, t1, np.inf))
    hit_s = has & (t_s < t_best)
    if hit_s.any():
        t_best = np.where(hit_s, t_s, t_best)
        p = origins[hit_s] + dirs[hit_s] * t_s[hit_s, None]
        nrm = p - SPHERE_CENTER
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        # Flip toward the incoming ray for interior hits (t1 root).
        flip = np.einsum("ij,ij->i", nrm, dirs[hit_s]) > 0.0
        nrm[flip] *= -1.0
        normal[hit_s] = nrm
        albedo[hit_s] = SPHERE_ALBEDO
    return t_best, normal, albedo, hit_s


def sample_free_flight(rng: np.random.Generator, sigma_t: float, n: int) -> np.ndarray:
    """Free-flight distances with pdf sigma_t * exp(-sigma_t * t) (homogeneous medium).

    Sampling distances this way makes transmittance *implicit* in the path cache: the
    probability that a recorded segment reaches distance d is exp(-sigma_t * d), so
    GATHERLIGHT needs no changes for lights inside the medium (paper §3.1).
    """
    return -np.log1p(-rng.random(n)) / sigma_t


def _isotropic_sample(rng: np.random.Generator, n: int) -> np.ndarray:
    """Uniform directions on the unit sphere (isotropic phase function)."""
    z = 1.0 - 2.0 * rng.random(n)
    phi = 2.0 * np.pi * rng.random(n)
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def _cosine_sample(normal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cosine-weighted hemisphere directions around each (N,3) normal."""
    n = normal.shape[0]
    u1 = rng.random(n)
    u2 = rng.random(n)
    r = np.sqrt(u1)
    phi = 2.0 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(np.maximum(0.0, 1.0 - u1))], axis=1)
    # Orthonormal frame around the normal.
    helper = np.where(np.abs(normal[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    tangent = np.cross(helper, normal)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    bitangent = np.cross(normal, tangent)
    return local[:, 0:1] * tangent + local[:, 1:2] * bitangent + local[:, 2:3] * normal


def trace_path_cache(
    width: int,
    height: int,
    spp: int,
    max_bounces: int,
    seed: int,
    medium: dict | None = None,
    layer: str | None = None,
) -> PathCache:
    """Trace `spp` light-agnostic paths per pixel and return the full PathCache.

    `medium`, if given, is `{"sigma_t": float, "albedo": float}`: a homogeneous
    participating medium filling the box, free-flight sampled with an isotropic phase
    function. Segments then may end at scattering vertices (recorded t_max = sampled
    flight distance); the scatter event multiplies throughput by the single-scattering
    albedo sigma_s/sigma_t and the path continues in a uniformly sampled direction.
    A bounce is consumed per event, surface or volume.

    `layer`, if given ("sphere" or "box"), records only the paths whose *first hit*
    is on that layer's geometry (§6.1 compositing). The full scene is still traced —
    same rng stream, same bounces off all geometry — so for a fixed seed the two
    layer caches partition the full cache's segments exactly, and their GATHERLIGHT
    images sum to the full-scene image *per segment* (`n_paths` stays the full spp
    so each layer keeps the full-estimator denominator). The G-buffer aux stays the
    full scene's (it describes the camera's first hit, shared by both layers).
    """
    if layer is not None and layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
    if layer is not None and medium is not None:
        raise ValueError(
            "layered export is surface-only (no medium): scatter vertices have no first-hit owner"
        )
    rng = np.random.default_rng(seed)
    n_pixels = width * height
    seg_pixel, seg_origin, seg_dir, seg_tmax, seg_throughput = [], [], [], [], []

    for _ in range(spp):
        jitter = rng.random((n_pixels, 2))
        origins, dirs = _camera_rays(width, height, jitter)
        throughput = np.ones((n_pixels, 3))
        pixel_ids = np.arange(n_pixels, dtype=np.int64)
        keep = np.ones(n_pixels, dtype=bool)
        for _bounce in range(max_bounces):
            t, normal, albedo, is_sphere = _intersect_scene(origins, dirs)
            if _bounce == 0 and layer is not None:
                keep = is_sphere if layer == "sphere" else ~is_sphere
            if medium is not None:
                d_flight = sample_free_flight(rng, float(medium["sigma_t"]), origins.shape[0])
                scatter = d_flight < t
                t = np.where(scatter, d_flight, t)
            seg_pixel.append(pixel_ids[keep].copy())
            seg_origin.append(origins[keep].copy())
            seg_dir.append(dirs[keep].copy())
            seg_tmax.append(t[keep].copy())
            seg_throughput.append(throughput[keep].copy())
            hit_p = origins + dirs * t[:, None]
            if medium is not None:
                iso = _isotropic_sample(rng, origins.shape[0])
                origins = np.where(scatter[:, None], hit_p, hit_p + normal * 1e-4)
                dirs = np.where(scatter[:, None], iso, _cosine_sample(normal, rng))
                # Volume event: throughput *= single-scattering albedo sigma_s/sigma_t
                # (isotropic phase pdf cancels); surface event: Lambertian albedo.
                throughput = throughput * np.where(
                    scatter[:, None], float(medium["albedo"]), albedo
                )
            else:
                origins = hit_p + normal * 1e-4
                dirs = _cosine_sample(normal, rng)
                throughput = throughput * albedo  # Lambertian: brdf*cos/pdf = albedo

    # Auxiliary buffers from deterministic pixel-center primary rays. These stay
    # surface-only even with a medium: albedo/depth/normal are G-buffer features of
    # the first *surface* hit (a scatter vertex has no meaningful normal or albedo).
    origins0, dirs0 = _camera_rays(width, height, None)
    t0, normal0, albedo0, _ = _intersect_scene(origins0, dirs0)
    position0 = origins0 + dirs0 * t0[:, None]

    cache = PathCache(
        width=width,
        height=height,
        n_paths=np.full(n_pixels, spp, dtype=np.int64),
        seg_pixel=np.concatenate(seg_pixel),
        seg_origin=np.concatenate(seg_origin),
        seg_dir=np.concatenate(seg_dir),
        seg_tmax=np.concatenate(seg_tmax),
        seg_throughput=np.concatenate(seg_throughput),
        albedo=albedo0.reshape(height, width, 3),
        position=position0.reshape(height, width, 3),
        depth=t0.reshape(height, width),
        normal=normal0.reshape(height, width, 3),
        medium=dict(medium) if medium is not None else None,
    )
    cache.validate()
    return cache


def layer_ownership_mask(width: int, height: int, layer: str) -> np.ndarray:
    """(H, W) bool mask of the pixels a layer owns: where the deterministic
    pixel-center primary ray's first hit lands on the layer's geometry — the same
    convention as the aux G-buffer. The two layers' masks are disjoint and cover
    every pixel (the box is closed, so every primary ray hits something)."""
    if layer not in LAYERS:
        raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
    origins0, dirs0 = _camera_rays(width, height, None)
    _, _, _, is_sphere = _intersect_scene(origins0, dirs0)
    mask = is_sphere if layer == "sphere" else ~is_sphere
    return mask.reshape(height, width)


def render_reference(
    width: int,
    height: int,
    spp: int,
    max_bounces: int,
    seed: int,
    light: SphereLight,
    medium: dict | None = None,
) -> np.ndarray:
    """Independent rendered reference: trace fresh paths and evaluate the emissive
    sphere inline (equivalent to GATHERLIGHT over an independent path set)."""
    cache = trace_path_cache(width, height, spp, max_bounces, seed, medium=medium)
    return gather_light(cache, light)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="output .npz path-cache file")
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--height", type=int, default=48)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--bounces", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--layer",
        choices=LAYERS,
        default=None,
        help="record only paths whose first hit is on this layer (compositing, §6.1)",
    )
    parser.add_argument(
        "--medium-sigma-t",
        type=float,
        default=0.0,
        help="extinction coefficient of a homogeneous medium filling the box (0 = none)",
    )
    parser.add_argument(
        "--medium-albedo",
        type=float,
        default=0.8,
        help="single-scattering albedo sigma_s/sigma_t of the medium",
    )
    args = parser.parse_args()

    medium = None
    if args.medium_sigma_t > 0.0:
        medium = {"sigma_t": args.medium_sigma_t, "albedo": args.medium_albedo}
    cache = trace_path_cache(
        args.width, args.height, args.spp, args.bounces, args.seed, medium=medium, layer=args.layer
    )
    cache.save(args.out)
    layer_note = f" (layer: {args.layer})" if args.layer else ""
    print(
        f"traced {args.spp} spp x {args.bounces} bounces at {args.width}x{args.height}: "
        f"{cache.segment_count} segments{layer_note} -> {args.out}"
    )


if __name__ == "__main__":
    main()
