"""First-class device / precision resolution for the torch backend (S6).

Every torch-backend entrypoint (train / bench / relight / optimize_lights /
streamed_train) funnels its `device` and `precision` options through this module,
so validation and availability errors are uniform — the exact seam the CUDA
bring-up (S7) needs. `cuda` (or `mps`) requested on a machine without it fails
immediately with an actionable message instead of a deep torch traceback.

Precision names are the config/CLI vocabulary for S5's autocast levers:
`fp32` (default, eager float32), `fp16`, `bf16`. `autocast(device, precision)`
returns the matching `torch.autocast` context (a no-op context for fp32), and
`synchronize(device)` is the device-dispatched barrier benchmarks need.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

VALID_DEVICES = ("cpu", "mps", "cuda")
VALID_PRECISIONS = ("fp32", "fp16", "bf16")

PRECISION_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_device(name: str | None) -> torch.device:
    """Validate a device name and confirm the backend is actually available."""
    name = name or "cpu"
    if name not in VALID_DEVICES:
        raise ValueError(f"device must be one of {'|'.join(VALID_DEVICES)}, got {name!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device 'cuda' requested but no CUDA runtime is available "
            "(torch.cuda.is_available() is False). Run on a CUDA machine — see "
            "the S7 cloud runbook — or pass device 'cpu' or 'mps'."
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "device 'mps' requested but the MPS backend is unavailable "
            "(torch.backends.mps.is_available() is False). Pass device 'cpu', "
            "or 'cuda' on a CUDA machine."
        )
    return torch.device(name)


def resolve_precision(name: str | None) -> str:
    """Validate a precision name; None means fp32."""
    name = name or "fp32"
    if name not in VALID_PRECISIONS:
        raise ValueError(f"precision must be one of {'|'.join(VALID_PRECISIONS)}, got {name!r}")
    return name


def autocast(device: torch.device, precision: str):
    """`torch.autocast` context for (device, precision); no-op context for fp32."""
    precision = resolve_precision(precision)
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=PRECISION_DTYPES[precision])


def synchronize(device: torch.device) -> None:
    """Device-dispatched barrier: wait for queued kernels before reading a clock."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
