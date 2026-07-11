"""LightRig: a named collection of per-light proxies with solo/mute and N-way composite.

A production shot rarely has one light — it has a rig: a key, a fill, a rim, practicals,
each independently art-directed. The paper's decoupling (Eq. 1's linearity: transport is
additive over lights) means each rig light can get its *own* trained proxy from the same
shared path cache, and the final frame is just the sum of per-light renders (Eq. 3,
generalized to N lights) — exactly the one-proxy-per-layer pattern `composite.py` and
`relight_multiview.py` already use, but keyed by light name instead of camera view or
scene layer. `LightRig` adds the interactive-editing vocabulary on top of that sum: mute
(drop a light), solo (isolate one or more lights, silencing the rest — the same semantics
as a DAW channel strip or NLE track), and JSON (de)serialization of the rig's light
parameters and mute/solo state (models are referenced by name, not embedded in the JSON;
callers keep their own model-path manifest, as `relight_multiview`'s view manifest does
for `(model, cache)` pairs).

`train_monolithic` is the deliberately-not-relightable baseline this rig is evaluated
against: one MLP fit directly to a single fixed composite image (`light_param_dim=0`
means the light-conditioning input is dropped entirely, so the network cannot be asked
for a different light — any change to the rig requires retraining this baseline from
scratch, unlike the per-light proxies above).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from ..lights import QuadLight, SphereLight, TexturedQuadLight, light_from_dict
from ..path_cache import PathCache
from .model import TorchNRP, relative_mse_loss
from .train import light_param_vector, pixel_tensors

Light = SphereLight | QuadLight | TexturedQuadLight


def light_type_of(light: Light) -> str:
    """The rig's dispatch key for a light instance. `SphereLight.to_dict()` omits a
    "type" key (there's only one sphere variant), so this defaults to "sphere" exactly
    as `nrp.lights.light_from_dict` already does for untyped dicts."""
    return light.to_dict().get("type", "sphere")


@dataclass
class RigLight:
    """One named light in the rig, plus its interactive solo/mute state."""

    name: str
    light: Light
    mute: bool = False
    solo: bool = False

    def to_dict(self) -> dict:
        d = dict(self.light.to_dict())
        d["name"] = self.name
        d["mute"] = self.mute
        d["solo"] = self.solo
        return d

    @classmethod
    def from_dict(cls, d: dict) -> RigLight:
        light_dict = {k: v for k, v in d.items() if k not in ("name", "mute", "solo")}
        return cls(
            name=d["name"],
            light=light_from_dict(light_dict),
            mute=d.get("mute", False),
            solo=d.get("solo", False),
        )


class LightRig:
    """N rig lights, each with its own independently-trained proxy keyed by light name
    (never shared across lights, even same-typed ones — that's what makes this the
    literal "per-light" proxy the production track calls for, not "per-type")."""

    def __init__(self, lights: list[RigLight], models: dict[str, TorchNRP]):
        self.lights = lights
        self.models = models

    def active_lights(self) -> list[RigLight]:
        """Solo, if any light has it set, silences everything else (even non-muted
        lights); otherwise every non-muted light is active."""
        soloed = [rl for rl in self.lights if rl.solo]
        if soloed:
            return [rl for rl in soloed if not rl.mute]
        return [rl for rl in self.lights if not rl.mute]

    def render_per_light(
        self, cache: PathCache, device: torch.device | None = None
    ) -> dict[str, np.ndarray]:
        """One (H, W, 3) image per active light: `model(xy, aux, params) * light.rgb`."""
        device = device or torch.device("cpu")
        xy, aux = pixel_tensors(cache, device)
        images: dict[str, np.ndarray] = {}
        with torch.no_grad():
            for rl in self.active_lights():
                n_px = xy.shape[0]
                params = torch.as_tensor(
                    light_param_vector(rl.light), dtype=torch.float32, device=device
                ).expand(n_px, -1)
                rgb = torch.as_tensor(rl.light.rgb, dtype=torch.float32, device=device)
                out = self.models[rl.name](xy, aux, params) * rgb
                images[rl.name] = (
                    out.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)
                )
        return images

    def render(self, cache: PathCache, device: torch.device | None = None) -> np.ndarray:
        """The N-way composite: elementwise sum of `render_per_light` (Eq. 3, summed
        over lights per Eq. 1's linearity of transport)."""
        images = self.render_per_light(cache, device)
        composite = np.zeros((cache.height, cache.width, 3), dtype=np.float64)
        for image in images.values():
            composite += image
        return composite

    def to_dict(self) -> dict:
        return {"lights": [rl.to_dict() for rl in self.lights]}

    @classmethod
    def from_dict(cls, d: dict, models: dict[str, TorchNRP]) -> LightRig:
        return cls([RigLight.from_dict(rd) for rd in d["lights"]], models)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str, models: dict[str, TorchNRP]) -> LightRig:
        with open(path) as f:
            d = json.load(f)
        return cls.from_dict(d, models)


def train_monolithic(
    cache: PathCache,
    target_image: np.ndarray,
    hidden_width: int,
    hidden_layers: int,
    iters: int,
    lr: float,
    seed: int = 0,
) -> tuple[TorchNRP, list[float]]:
    """Fit one MLP directly to a single fixed composite image — the non-relightable
    baseline. `light_type="sphere"` is nominal (`TorchNRP` requires a valid
    `SUPPORTED_LIGHT_TYPES` string even though `light_param_dim=0` means no light
    parameters ever reach the network); plain full-batch Adam against
    `relative_mse_loss` (Eq. 4), since there is only one training target — no
    pool/sampling machinery is needed."""
    torch.manual_seed(seed)
    device = torch.device("cpu")
    xy, aux = pixel_tensors(cache, device)
    n_px = xy.shape[0]
    target = torch.as_tensor(
        np.asarray(target_image, dtype=np.float64).reshape(-1, 3),
        dtype=torch.float32,
        device=device,
    )
    model = TorchNRP(
        light_type="sphere",
        light_param_dim=0,
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        use_encoding=True,
    ).to(device)
    empty_params = torch.zeros((n_px, 0), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_curve: list[float] = []
    for _ in range(iters):
        pred = model(xy, aux, empty_params)
        loss = relative_mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_curve.append(loss.detach().item())
    model.eval()
    return model, loss_curve
