"""Torch neural render proxy: hashgrid-encoded MLP per paper §4.3, loss per Eq. 4.

Network inputs beyond the light's shape parameters are the pixel coordinates px
(hashgrid-encoded, 2D) and the auxiliary pixel features Fpx (albedo 3 + depth 1 +
normal 3 = 7D), exactly the nine extra inputs the paper lists. Output is the
pre-emission contribution N_type(px, Fpx, v) of Eq. 2; the final pixel value is the
emission-weighted sum over lights (Eq. 3). A softplus head keeps contributions
positive (the paper does not specify its head; this is the one deviation here).

Light shape parameters (emission E(v) is factored out, Eq. 1):
  sphere: center (3) + radius (1) = 4
  quad:   center (3) + normal (3) + width + height = 8
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from .encoding import HashEncoding2D

LIGHT_PARAM_DIMS = {"sphere": 4, "quad": 8}


def relative_mse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Relative MSE of Müller et al. [MRNK21] as used in Eq. 4: the denominator is the
    stop-gradient of the *prediction*, ε = 0.01 (both paper-exact)."""
    return ((pred - target) ** 2 / (pred.detach() ** 2 + eps)).mean()


class TorchNRP(nn.Module):
    def __init__(
        self,
        light_type: str = "sphere",
        hidden_width: int = 128,
        hidden_layers: int = 4,
        encoding: dict | None = None,
    ):
        super().__init__()
        if light_type not in LIGHT_PARAM_DIMS:
            raise ValueError(f"light_type must be one of {sorted(LIGHT_PARAM_DIMS)}")
        self.light_type = light_type
        self.config = {
            "light_type": light_type,
            "hidden_width": hidden_width,
            "hidden_layers": hidden_layers,
            "encoding": encoding or {},
        }
        self.encoding = HashEncoding2D(**(encoding or {}))
        in_dim = self.encoding.output_dim + 7 + LIGHT_PARAM_DIMS[light_type]
        layers: list[nn.Module] = []
        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_width, hidden_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_width if hidden_layers else in_dim, 3))
        self.mlp = nn.Sequential(*layers)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, pixel_xy: torch.Tensor, aux: torch.Tensor, light_params: torch.Tensor
    ) -> torch.Tensor:
        """pixel_xy (N,2) in [0,1]^2, aux (N,7), light_params (N, 4 or 8) -> (N,3)."""
        x = torch.cat([self.encoding(pixel_xy), aux, light_params], dim=1)
        return F.softplus(self.mlp(x))

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
