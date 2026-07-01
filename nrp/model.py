"""Compact neural proxy: numpy MLP with manual autodiff (NRP M4).

Deliberately dependency-free (this project is numpy-only): a small
fully-connected network with explicit forward/backward passes, a sinusoidal positional
encoding of pixel coordinates standing in for the paper's hashgrid encoding (documented
substitution — a hashgrid needs a learned table + trilinear interpolation and is an
optimization, not a correctness requirement at toy scale), softplus output so predicted
contributions stay positive and smooth, and Adam.

Inputs per sample (see `encode_inputs`): positional-encoded pixel xy, albedo (3),
depth (1), normal (3), light center (3), light radius (1).
Output: RGB throughput-sum contribution *before* emission (light rgb) scaling.

`backward` also returns gradients with respect to the *inputs*, which is what makes the
proxy differentiable in the light parameters for inverse optimization (M5): GATHERLIGHT
itself has zero gradient almost everywhere in center/radius (hard visibility
indicator), so the smooth proxy is the differentiable path, exactly the paper's point.
"""

from __future__ import annotations

import numpy as np

PE_FREQS = 4  # sinusoidal frequencies for pixel-xy encoding


def encode_pixel_xy(xy: np.ndarray) -> np.ndarray:
    """xy in [0,1]^2, shape (N,2) -> (N, 2 + 2*2*PE_FREQS) positional encoding."""
    feats = [xy]
    for k in range(PE_FREQS):
        w = (2.0**k) * np.pi
        feats.append(np.sin(w * xy))
        feats.append(np.cos(w * xy))
    return np.concatenate(feats, axis=1)


def encode_inputs(
    pixel_xy: np.ndarray,
    albedo: np.ndarray,
    depth: np.ndarray,
    normal: np.ndarray,
    position: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray,
) -> np.ndarray:
    """Assemble the (N, D) input matrix.

    Besides the raw light parameters, two derived geometric features are appended —
    the first-hit-to-light-center offset (center - position) and its norm — because a
    compact MLP conditions far better on relative geometry than on absolute light
    position (verified empirically in this repo: without these, inverse optimization
    collapsed to the parameter-bound corner regardless of seed). Gradients with
    respect to the raw light parameters must therefore chain through these derived
    columns; `light_param_gradient` does that.
    """
    diff = center - position
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    return np.concatenate(
        [
            encode_pixel_xy(pixel_xy),
            albedo,
            depth.reshape(-1, 1),
            normal,
            position,
            center,
            radius.reshape(-1, 1),
            diff,
            dist,
        ],
        axis=1,
    )


_PE_DIM = 2 + 4 * PE_FREQS
INPUT_DIM = _PE_DIM + 3 + 1 + 3 + 3 + 3 + 1 + 3 + 1
_CENTER_COLS = slice(_PE_DIM + 10, _PE_DIM + 13)
_RADIUS_COL = _PE_DIM + 13
_DIFF_COLS = slice(_PE_DIM + 14, _PE_DIM + 17)
_DIST_COL = _PE_DIM + 17


def light_param_gradient(
    dx: np.ndarray, position: np.ndarray, center: np.ndarray
) -> tuple[np.ndarray, float]:
    """Chain dL/d(input columns) back to (dL/d(center) (3,), dL/d(radius) scalar).

    center enters the encoding three ways: raw columns, diff = center - position, and
    dist = |diff| (d dist / d center = diff / dist)."""
    diff = center - position
    dist = np.linalg.norm(diff, axis=1, keepdims=True)
    dist = np.maximum(dist, 1e-12)
    d_center = (
        dx[:, _CENTER_COLS].sum(axis=0)
        + dx[:, _DIFF_COLS].sum(axis=0)
        + (dx[:, _DIST_COL : _DIST_COL + 1] * (diff / dist)).sum(axis=0)
    )
    d_radius = float(dx[:, _RADIUS_COL].sum())
    return d_center, d_radius


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


class ProxyMLP:
    """Small ReLU MLP with softplus output head and manual backprop."""

    def __init__(self, hidden: tuple[int, ...] = (64, 64), out_dim: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        dims = [INPUT_DIM, *hidden, out_dim]
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / dims[i]), size=(dims[i], dims[i + 1]))
            for i in range(len(dims) - 1)
        ]
        self.biases = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self._cache: list[np.ndarray] = []

    @property
    def parameter_count(self) -> int:
        return sum(w.size for w in self.weights) + sum(b.size for b in self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = [x]
        h = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = h @ w + b
            h = np.maximum(z, 0.0) if i < len(self.weights) - 1 else _softplus(z)
            self._cache.append(z)
            self._cache.append(h)
        return h

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, list, list]:
        """Given dL/d(output), return (dL/d(input), dL/dW list, dL/db list).
        Must be called immediately after forward() on the same batch."""
        d_w = [None] * len(self.weights)
        d_b = [None] * len(self.biases)
        grad = dout
        for i in reversed(range(len(self.weights))):
            z = self._cache[1 + 2 * i]
            h_in = self._cache[2 * i]  # layer input: x for i=0, previous activation otherwise
            if i == len(self.weights) - 1:
                grad = grad * (1.0 / (1.0 + np.exp(-z)))  # softplus'
            else:
                grad = grad * (z > 0.0)
            d_w[i] = h_in.T @ grad
            d_b[i] = grad.sum(axis=0)
            grad = grad @ self.weights[i].T
        return grad, d_w, d_b

    def grad_wrt_inputs(self, x: np.ndarray, dout: np.ndarray) -> np.ndarray:
        """Convenience: forward + backward, returning only dL/d(input)."""
        self.forward(x)
        dx, _, _ = self.backward(dout)
        return dx

    def save(self, path: str) -> None:
        arrays = {}
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez(path, n_layers=len(self.weights), **arrays)

    @classmethod
    def load(cls, path: str) -> ProxyMLP:
        z = np.load(path)
        n = int(z["n_layers"])
        model = cls.__new__(cls)
        model.weights = [z[f"w{i}"] for i in range(n)]
        model.biases = [z[f"b{i}"] for i in range(n)]
        model._cache = []
        return model


class Adam:
    def __init__(self, model: ProxyMLP, lr: float = 1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.model = model
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m_w = [np.zeros_like(w) for w in model.weights]
        self.v_w = [np.zeros_like(w) for w in model.weights]
        self.m_b = [np.zeros_like(b) for b in model.biases]
        self.v_b = [np.zeros_like(b) for b in model.biases]

    def step(self, d_w: list, d_b: list) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for i in range(len(self.model.weights)):
            for param, grad, m, v in (
                (self.model.weights[i], d_w[i], self.m_w[i], self.v_w[i]),
                (self.model.biases[i], d_b[i], self.m_b[i], self.v_b[i]),
            ):
                m *= self.beta1
                m += (1.0 - self.beta1) * grad
                v *= self.beta2
                v += (1.0 - self.beta2) * grad * grad
                param -= self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)


def relative_mse_loss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-2):
    """Relative MSE-style HDR loss: (pred-target)^2 / (target^2 + eps).

    Returns (loss_scalar, dL/dpred). Deviation from the paper, recorded in the NRP
    report: the paper normalizes by a stop-gradient of the *prediction*; this
    implementation normalizes by the (constant) target, which keeps the same
    relative-HDR weighting but makes the scalar loss a stable, exactly
    finite-difference-checkable objective for the hand-rolled autodiff."""
    denom = target**2 + eps
    diff = pred - target
    loss = float(np.mean(diff**2 / denom))
    dpred = (2.0 * diff / denom) / pred.size
    return loss, dpred
