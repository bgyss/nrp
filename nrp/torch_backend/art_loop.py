"""Mixed-light-type color recovery + headless slider loop (roadmap V2 art-direction
loop, `docs/production-track.md`).

`optimize_lights.py`'s §5.3 machinery jointly recovers a light's *geometry* and
*color* from a target image, one light type at a time. The art-direction loop is a
narrower, more production-shaped task: an artist has already placed and shaped a
mixed rig of lights (sphere/quad/textured_quad, each with its own trained per-light
proxy, per `rig.py`) and wants to (a) recover per-light RGB intensities that
reproduce a hand-authored target frame, geometry held fixed, and (b) interactively
nudge those colors and see render latency — the "slider" in a color-grading tool.

`RigColorReparam` is the color-only analogue of `ReparamSphereLights`/
`ReparamQuadLights`: one inverse-softplus `u_rgb` (3,) per active rig light, geometry
copied verbatim from the rig's own lights. `predicted_image` is `LightRig.render`'s
differentiable twin — same per-light `model(...) * rgb` sum (Eq. 3, generalized to N
lights per Eq. 1's linearity), but keeping the graph instead of `no_grad`.
`optimize_colors` runs Adam over `RigColorReparam.parameters` against a
Reinhard-tonemapped MSE (Eq. 6), mirroring `optimize_lights.optimize`'s loss and
report shape but scoped to color. `slider_loop` is the non-differentiable, latency-
measuring counterpart: it applies a sequence of rgb nudges to a working copy of the
rig and times `LightRig.render` for each, the way a live color-grading slider would.

`TexturedQuadLight` has no `.rgb` (`rig.py`'s `render_per_light` already documents
why: its texture bakes in the full per-texel emission, so the proxy's raw output is
used as-is). `RigColorReparam` follows the same precedent: textured-quad lights are
not given a `u_rgb` parameter at all — their contribution to `predicted_image` is the
model's raw output, unscaled, exactly as `LightRig.render_per_light` computes it, and
`to_rig()`/`constrained_rgbs()` simply omit them. This keeps the reparam consistent
with the rig's existing rendering semantics rather than inventing a new "texture
brightness scalar" the rest of the codebase doesn't have a precedent for.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..lights import QuadLight, SphereLight, TexturedQuadLight
from ..metrics import psnr, ssim, tonemap_srgb
from ..path_cache import PathCache
from .optimize_lights import inv_softplus, reinhard
from .rig import LightRig, RigLight
from .train import light_param_vector, pixel_tensors


def _with_rgb(light, rgb: np.ndarray):
    """A copy of `light` with its geometry unchanged and `.rgb` replaced (only valid
    for sphere/quad lights, which have an `.rgb` field; see module docstring for why
    `TexturedQuadLight` never reaches this function)."""
    if isinstance(light, SphereLight):
        return SphereLight(center=light.center, radius=light.radius, rgb=rgb)
    if isinstance(light, QuadLight):
        return QuadLight(
            center=light.center,
            normal=light.normal,
            width=light.width,
            height=light.height,
            rgb=rgb,
        )
    raise TypeError(f"light type {type(light).__name__} has no optimizable .rgb")


class RigColorReparam:
    """Per-light-name inverse-softplus `u_rgb` for every active, colorable rig light.

    Geometry (and, for `TexturedQuadLight`, the texture) is copied from the rig's own
    lights and held fixed — only `.rgb` is reparameterized and optimized.
    """

    def __init__(self, rig: LightRig, init_rgbs: dict[str, np.ndarray], device: torch.device):
        self.rig = rig
        self.device = device
        self.names: list[str] = []
        self.u_rgb: dict[str, torch.Tensor] = {}
        for rl in rig.active_lights():
            if isinstance(rl.light, TexturedQuadLight):
                continue
            rgb = np.asarray(init_rgbs[rl.name], dtype=np.float64).clip(min=1e-4)
            u = inv_softplus(
                torch.as_tensor(rgb, dtype=torch.float32, device=device)
            ).requires_grad_(True)
            self.names.append(rl.name)
            self.u_rgb[rl.name] = u

    @property
    def parameters(self) -> list[torch.Tensor]:
        return [self.u_rgb[name] for name in self.names]

    def constrained_rgbs(self) -> dict[str, torch.Tensor]:
        return {name: torch.nn.functional.softplus(self.u_rgb[name]) for name in self.names}

    def to_rig(self) -> LightRig:
        rgbs = self.constrained_rgbs()
        new_lights = []
        for rl in self.rig.lights:
            if rl.name in rgbs:
                rgb = rgbs[rl.name].detach().cpu().numpy().astype(np.float64)
                light = _with_rgb(rl.light, rgb)
            else:
                light = rl.light
            new_lights.append(RigLight(name=rl.name, light=light, mute=rl.mute, solo=rl.solo))
        return LightRig(new_lights, self.rig.models)


def predicted_image(
    rig: LightRig, reparam: RigColorReparam, xy: torch.Tensor, aux: torch.Tensor
) -> torch.Tensor:
    """Differentiable analogue of `LightRig.render`: sums, over `rig.active_lights()`,
    `model(xy, aux, light_param_vector(light)) * rgb` for colorable lights, or the raw
    model output for `TexturedQuadLight` (no `.rgb` to scale by — same convention as
    `LightRig.render_per_light`), with gradients flowing to `reparam.parameters`."""
    device = xy.device
    n_px = xy.shape[0]
    rgbs = reparam.constrained_rgbs()
    image = torch.zeros((n_px, 3), device=device)
    for rl in rig.active_lights():
        params = torch.as_tensor(
            light_param_vector(rl.light), dtype=torch.float32, device=device
        ).expand(n_px, -1)
        out = rig.models[rl.name](xy, aux, params)
        if rl.name in rgbs:
            out = out * rgbs[rl.name]
        image = image + out
    return image


def _constant_contribution(
    rig: LightRig, reparam: RigColorReparam, xy: torch.Tensor, aux: torch.Tensor
) -> torch.Tensor:
    """Sum of `model(xy, aux, light_param_vector(light))` over every active rig light
    that has no `reparam` parameter (i.e. `TexturedQuadLight`, which contributes its
    raw model output unscaled — see module docstring). This term has no gradient with
    respect to `reparam.parameters` — geometry and texture are held fixed throughout
    `optimize_colors` — so it is the same tensor on every optimization step; computed
    once here instead of every step (H2, hoisted out of `optimize_colors`'s loop)."""
    device = xy.device
    n_px = xy.shape[0]
    colorable = set(reparam.names)
    out = torch.zeros((n_px, 3), device=device)
    with torch.no_grad():
        for rl in rig.active_lights():
            if rl.name in colorable:
                continue
            params = torch.as_tensor(
                light_param_vector(rl.light), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            out = out + rig.models[rl.name](xy, aux, params)
    return out


def _colorable_predicted_image(
    rig: LightRig,
    reparam: RigColorReparam,
    xy: torch.Tensor,
    aux: torch.Tensor,
    constant_out: torch.Tensor,
    colorable_params: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Per-step body of `optimize_colors`'s loop: differentiable sum over only
    `reparam.names` (the lights with a trainable `u_rgb`), plus the precomputed
    `constant_out` for every other active light — equivalent to `predicted_image`
    but without re-running the constant lights' forward pass every step, and without
    rebuilding each light's (fixed) param tensor every step either."""
    rgbs = reparam.constrained_rgbs()
    image = constant_out
    for name in reparam.names:
        out = rig.models[name](xy, aux, colorable_params[name]) * rgbs[name]
        image = image + out
    return image


def optimize_colors(
    rig: LightRig, cache: PathCache, target: np.ndarray, steps: int, lr: float, seed: int = 0
) -> dict:
    """Adam over `RigColorReparam.parameters`, full-batch Reinhard-tonemapped MSE
    against `target`, mirroring `optimize_lights.optimize`'s loss/report shape but
    scoped to per-light color recovery (geometry fixed)."""
    device = torch.device("cpu")
    torch.manual_seed(seed)
    n_px = cache.height * cache.width
    xy, aux = pixel_tensors(cache, device)
    tgt = torch.as_tensor(
        np.asarray(target, dtype=np.float64).reshape(n_px, 3), dtype=torch.float32, device=device
    )

    init_rgbs = {
        rl.name: rl.light.rgb
        for rl in rig.active_lights()
        if not isinstance(rl.light, TexturedQuadLight)
    }
    reparam = RigColorReparam(rig, init_rgbs, device)

    # H2 hoist: TexturedQuadLight (and any other non-colorable active light) has no
    # `u_rgb` and its geometry/texture is fixed for the whole loop, so its forward
    # pass is identical on every step -- compute it once instead of `steps` times.
    constant_out = _constant_contribution(rig, reparam, xy, aux)
    colorable_by_name = {rl.name: rl for rl in rig.active_lights() if rl.name in reparam.names}
    colorable_params = {
        name: torch.as_tensor(
            light_param_vector(rl.light), dtype=torch.float32, device=device
        ).expand(n_px, -1)
        for name, rl in colorable_by_name.items()
    }

    opt = torch.optim.Adam(reparam.parameters, lr=lr) if reparam.parameters else None
    loss_curve: list[float] = []
    for _step in range(steps):
        pred = _colorable_predicted_image(rig, reparam, xy, aux, constant_out, colorable_params)
        diff = reinhard(pred) - reinhard(tgt)
        loss = (diff**2).mean()
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_curve.append(loss.detach().item())

    optimized_rig = reparam.to_rig()
    with torch.no_grad():
        pred_final = (
            _colorable_predicted_image(rig, reparam, xy, aux, constant_out, colorable_params)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    target_flat = np.asarray(target, dtype=np.float64).reshape(n_px, 3)
    pred_img = pred_final.reshape(cache.height, cache.width, 3)
    target_img = target_flat.reshape(cache.height, cache.width, 3)

    if not loss_curve:
        loss_curve = [0.0]
    return {
        "optimized_rig": optimized_rig,
        "steps": steps,
        "proxy_loss_first": loss_curve[0],
        "proxy_loss_last": loss_curve[-1],
        "proxy_loss_curve": loss_curve[:: max(1, steps // 50)] if steps else loss_curve,
        "proxy_vs_target_psnr_db": psnr(pred_final, target_flat),
        "proxy_vs_target_ssim": ssim(
            tonemap_srgb(pred_img), tonemap_srgb(target_img), data_range=1.0
        ),
    }


def slider_loop(
    rig: LightRig,
    cache: PathCache,
    adjustments: list[dict],
    device: torch.device | None = None,
) -> dict:
    """Apply `adjustments` (`{"light": name, "rgb": [r,g,b]}`) one at a time to a
    working copy of `rig`, timing `LightRig.render` after each nudge (a warmup render
    before the first timed one is excluded, since it pays for one-time setup a real
    interactive session wouldn't repeat)."""
    device = device or torch.device("cpu")
    working = LightRig(
        [RigLight(name=rl.name, light=rl.light, mute=rl.mute, solo=rl.solo) for rl in rig.lights],
        rig.models,
    )
    by_name = {rl.name: rl for rl in working.lights}

    working.render(cache, device)  # warmup, untimed

    latency_ms: list[float] = []
    for adj in adjustments:
        rl = by_name[adj["light"]]
        rl.light = _with_rgb(rl.light, np.asarray(adj["rgb"], dtype=np.float64))
        by_name[adj["light"]] = rl
        start = time.perf_counter()
        working.render(cache, device)
        latency_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(latency_ms, dtype=np.float64)
    return {
        "n_adjustments": len(adjustments),
        "latency_ms": latency_ms,
        "latency_ms_mean": float(arr.mean()) if arr.size else 0.0,
        "latency_ms_p95": float(np.percentile(arr, 95)) if arr.size else 0.0,
    }
