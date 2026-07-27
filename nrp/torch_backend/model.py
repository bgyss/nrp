"""Torch neural render proxy: hashgrid-encoded MLP per paper §4.3, loss per Eq. 4.

Network inputs beyond the light's shape parameters are the pixel coordinates px
(hashgrid-encoded, 2D) and the auxiliary pixel features Fpx (albedo 3 + depth 1 +
normal 3 = 7D), exactly the nine extra inputs the paper lists. Output is the
pre-emission contribution N_type(px, Fpx, v) of Eq. 2; the final pixel value is the
emission-weighted sum over lights (Eq. 3). A softplus head keeps contributions
positive (the paper does not specify its head; this is the one deviation here).
Representation-track R1 adds an opt-in 3D hashgrid over first-hit world position;
the paper-faithful 2D path remains the default.

Light shape parameters (emission E(v) is factored out, Eq. 1):
  sphere:        center (3) + radius (1) = 4
  quad:          center (3) + normal (3) + width + height = 8
  textured_quad: quad geometry (8) + flattened RGB texture, fixed per model config

Ablation switches (roadmap item 10, paper Table 2): `use_aux=False` drops the 7D
G-buffer features and `use_encoding=False` feeds the raw 2D pixel coordinates instead
of the hashgrid encoding. `forward` keeps its (spatial, aux, params) signature either
way so training/relighting/inverse code is variant-agnostic; disabled inputs are ignored.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .encoding import HashEncoding2D, HashEncoding3D, HashEncodingTriPlane

LIGHT_PARAM_DIMS = {"sphere": 4, "quad": 8}
SUPPORTED_LIGHT_TYPES = {"sphere", "quad", "textured_quad"}
SUPPORTED_SPATIAL_ENCODINGS = {"pixel2d", "world3d", "world_triplane"}


def relative_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Relative MSE of Müller et al. [MRNK21] as used in Eq. 4: the denominator is the
    stop-gradient of the *prediction*, ε = 0.01 (both paper-exact)."""
    return ((pred - target) ** 2 / (pred.detach() ** 2 + eps)).mean()


def inverse_softplus(y: float, floor: float = 1e-6) -> float:
    """softplus^-1(y), floored so y=0 doesn't hit log(0)."""
    y = max(float(y), floor)
    return math.log(math.expm1(y)) if y < 20.0 else y


class TorchNRP(nn.Module):
    def __init__(
        self,
        light_type: str = "sphere",
        hidden_width: int = 128,
        hidden_layers: int = 4,
        encoding: dict | None = None,
        spatial_encoding: str = "pixel2d",
        world_bounds: dict | None = None,
        use_encoding: bool = True,
        use_aux: bool = True,
        light_param_dim: int | None = None,
        texture_kernel: bool = False,
    ):
        super().__init__()
        if light_type not in SUPPORTED_LIGHT_TYPES:
            raise ValueError(f"light_type must be one of {sorted(SUPPORTED_LIGHT_TYPES)}")
        if spatial_encoding not in SUPPORTED_SPATIAL_ENCODINGS:
            raise ValueError(
                f"spatial_encoding must be one of {sorted(SUPPORTED_SPATIAL_ENCODINGS)}"
            )
        if spatial_encoding != "pixel2d" and not use_encoding:
            raise ValueError(f"spatial_encoding {spatial_encoding!r} requires use_encoding=true")
        if light_param_dim is None:
            if light_type not in LIGHT_PARAM_DIMS:
                raise ValueError(f"light_param_dim is required for light_type {light_type!r}")
            light_param_dim = LIGHT_PARAM_DIMS[light_type]
        if texture_kernel:
            # H3 (docs/hardening-track.md): gather_textured_quad is linear in the
            # texture -- each texel is an independent constant emitter (Eq. 1). The
            # kernel head bakes that structure in: the MLP sees only pixel features
            # + quad geometry (8) and predicts a non-negative per-texel throughput
            # kernel, contracted with the texture at the output. The texture never
            # enters the MLP, so generalization across textures is structural, not
            # learned. `forward` keeps its (xy, aux, params) signature: params is
            # still the full geometry(8)+flattened-texture vector.
            if light_type != "textured_quad":
                raise ValueError("texture_kernel requires light_type 'textured_quad'")
            if light_param_dim <= 8 or (light_param_dim - 8) % 3 != 0:
                raise ValueError(
                    f"texture_kernel needs light_param_dim = 8 + 3*n_texels, got {light_param_dim}"
                )
        self.light_type = light_type
        self.light_param_dim = int(light_param_dim)
        self.texture_kernel = bool(texture_kernel)
        self.spatial_encoding = spatial_encoding
        self.use_encoding = use_encoding
        self.use_aux = use_aux
        if spatial_encoding != "pixel2d":
            if world_bounds is None or set(world_bounds) != {"min", "max"}:
                raise ValueError(
                    f"{spatial_encoding} requires world_bounds with exactly 'min' and 'max'"
                )
            world_min = torch.as_tensor(world_bounds["min"], dtype=torch.float32)
            world_max = torch.as_tensor(world_bounds["max"], dtype=torch.float32)
            if world_min.shape != (3,) or world_max.shape != (3,):
                raise ValueError("world_bounds min and max must each contain three values")
            if not bool(torch.isfinite(world_min).all() and torch.isfinite(world_max).all()):
                raise ValueError("world_bounds must be finite")
            if not bool((world_max > world_min).all()):
                raise ValueError("world_bounds max must be greater than min on every axis")
            self.register_buffer("world_min", world_min)
            self.register_buffer("world_extent", world_max - world_min)
            normalized_bounds = {"min": world_min.tolist(), "max": world_max.tolist()}
        else:
            self.world_min = None
            self.world_extent = None
            normalized_bounds = None
        self.config = {
            "light_type": light_type,
            "light_param_dim": self.light_param_dim,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "encoding": encoding or {},
            "spatial_encoding": spatial_encoding,
            "world_bounds": normalized_bounds,
            "use_encoding": use_encoding,
            "use_aux": use_aux,
            "texture_kernel": self.texture_kernel,
        }
        if not use_encoding:
            self.encoding = None
        elif spatial_encoding == "world3d":
            self.encoding = HashEncoding3D(**(encoding or {}))
        elif spatial_encoding == "world_triplane":
            self.encoding = HashEncodingTriPlane(**(encoding or {}))
        else:
            self.encoding = HashEncoding2D(**(encoding or {}))
        spatial_dim = self.encoding.output_dim if use_encoding else 2
        light_in_dim = 8 if self.texture_kernel else self.light_param_dim
        out_dim = self.light_param_dim - 8 if self.texture_kernel else 3
        in_dim = spatial_dim + (7 if use_aux else 0) + light_in_dim
        layers: list[nn.Module] = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_width, hidden_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_width if hidden_layers else in_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def init_output_scale(
        self, target_scale: float, mean_texture_value: float | None = None
    ) -> None:
        """Re-init the output head so predictions start near `target_scale` instead
        of nn.Linear's default (softplus(~0) ~= 0.69 regardless of scene). Root cause
        (H1, docs/hardening-track.md): when true supervision targets are far dimmer
        than that default -- e.g. QuadLight pool targets on the kitchen cache, median
        ~0.001-0.005 -- the loss gradient is persistently negative-signed for most
        pool samples, and Adam's per-step displacement is ~lr regardless of gradient
        magnitude (confirmed: neither eps nor grad-norm clipping change the drift).
        Over a normal training budget this walks the pre-softplus logit past
        float32's softplus-derivative-underflow point, permanently zeroing the
        network. Starting near the true scale removes the sustained one-directional
        push before it can reach that point.

        Kernel-head models (texture_kernel=True) predict per-texel kernel weights
        whose contraction with the texture is the output, so the per-weight scale
        is target_scale / (n_texels * mean_texture_value)."""
        last = self.mlp[-1]
        if self.texture_kernel:
            n_texels = (self.light_param_dim - 8) // 3
            denom = max(n_texels * float(mean_texture_value or 0.5), 1e-6)
            target_scale = target_scale / denom
        with torch.no_grad():
            last.weight.zero_()
            last.bias.fill_(inverse_softplus(target_scale))

    def forward(
        self, spatial_coords: torch.Tensor, aux: torch.Tensor, light_params: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate pixels from 2D coordinates or first-hit world positions.

        ``spatial_coords`` is ``(N, 2)`` normalized pixel xy for ``pixel2d`` and
        ``(N, 3)`` raw first-hit world position for ``world3d``. ``aux`` retains
        the paper's seven albedo/depth/normal columns in both modes.
        """
        if self.spatial_encoding != "pixel2d":
            if spatial_coords.ndim != 2 or spatial_coords.shape[1] != 3:
                raise ValueError(
                    f"{self.spatial_encoding} spatial_coords must have shape (N, 3), "
                    f"got {tuple(spatial_coords.shape)}"
                )
            normalized = ((spatial_coords - self.world_min) / self.world_extent).clamp(0.0, 1.0)
            spatial = self.encoding(normalized)
        else:
            spatial = self.encoding(spatial_coords) if self.encoding is not None else spatial_coords
        if self.texture_kernel:
            geometry, texture = light_params[:, :8], light_params[:, 8:]
            parts = [spatial, aux, geometry] if self.use_aux else [spatial, geometry]
            kernel = F.softplus(self.mlp(torch.cat(parts, dim=1)))
            return (kernel * texture).view(texture.shape[0], -1, 3).sum(dim=1)
        parts = [spatial, aux, light_params] if self.use_aux else [spatial, light_params]
        return F.softplus(self.mlp(torch.cat(parts, dim=1)))

    def save(self, path: str) -> None:
        torch.save({"config": self.config, "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str) -> TorchNRP:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(**blob["config"])
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


def sphere_params(center: torch.Tensor, radius: torch.Tensor, n: int) -> torch.Tensor:
    """Broadcast one sphere's (center, radius) to an (N, 4) light-parameter block."""
    return torch.cat([center.reshape(1, 3).expand(n, 3), radius.reshape(1, 1).expand(n, 1)], dim=1)


def quad_params(
    center: torch.Tensor,
    normal: torch.Tensor,
    width: torch.Tensor,
    height: torch.Tensor,
    n: int,
) -> torch.Tensor:
    """Broadcast one quad's parameters to an (N, 8) block (normal is normalized here so
    gradients flow through the normalization during inverse optimization)."""
    unit = normal / torch.linalg.vector_norm(normal)
    return torch.cat(
        [
            center.reshape(1, 3).expand(n, 3),
            unit.reshape(1, 3).expand(n, 3),
            width.reshape(1, 1).expand(n, 1),
            height.reshape(1, 1).expand(n, 1),
        ],
        dim=1,
    )
