"""Registry of spatial encoders.

Separate from `encoding` so an encoder defined in any module can register itself
without importing the module that imports it back.
"""

from __future__ import annotations

import inspect

import torch

#: name -> encoder class. `build_encoder` is the only construction path the model uses,
#: so adding an arm is one decorator rather than another if/elif branch.
SPATIAL_ENCODERS: dict[str, type] = {}

#: Set once `_ensure_loaded` has imported `encoding` (which in turn imports
#: `sparse_encoding`), so every built-in encoder is registered exactly once no
#: matter which module a caller imports first.
_loaded = False


def _ensure_loaded() -> None:
    """Guarantee the built-in encoders are registered before the registry is read.

    In a fresh interpreter, `SPATIAL_ENCODERS` is empty until something imports
    `encoding` (directly, or transitively via `model`/`train`) -- an import-order
    coincidence that made `build_encoder`/`encoder_schedule_params` correct only by
    accident. Calling this first makes the registry self-sufficient regardless of
    what the caller happened to import already. Guarded by `_loaded` rather than
    relying on Python's module-import cache alone so this is a cheap no-op on every
    call after the first, and so re-entrancy during `encoding`'s own import (it does
    not currently call back into this module, but nothing prevents it from starting
    to) can never recurse: the flag is set before the import runs.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True
    from . import encoding  # noqa: F401


def _floor_cell(pos: torch.Tensor, res: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower cell corner and interpolation fraction for scaled coordinates.

    Clamping to res-1 (not res) keeps the cell [x0, x0+1] non-degenerate and
    frac within [0, 1]. Output is identical either way for in-range inputs; the
    invariant matters to the sparse-index encoder, which indexes corners directly.

    Lives here (not in `encoding.py`) so both `encoding.py` and `sparse_encoding.py`
    can import it without either importing the other.
    """
    pos0 = torch.floor(pos).long().clamp_(0, res - 1)
    frac = pos - pos0
    return pos0, frac


def register_encoder(name: str):
    def wrap(cls):
        if name in SPATIAL_ENCODERS:
            raise ValueError(f"spatial encoder {name!r} is already registered")
        SPATIAL_ENCODERS[name] = cls
        return cls

    return wrap


def encoder_wants_occupancy(name: str, config: dict | None = None) -> bool:
    """Whether constructing `name` with `config` requires an occupancy argument.

    True either because the encoder class always needs one (e.g. `world_sparse`,
    whose exact index has no dense/hashed fallback) or because this particular
    config opts a class with a uniform default into `allocation: "occupancy"`
    (e.g. `world3d`). Shared by `build_encoder` and any caller (single-view
    `train`, camera-conditioned `train_conditioned`) that must build occupancy
    from a cache before construction.
    """
    _ensure_loaded()
    if name not in SPATIAL_ENCODERS:
        raise ValueError(
            f"unknown spatial encoding {name!r}; expected one of {sorted(SPATIAL_ENCODERS)}"
        )
    cls = SPATIAL_ENCODERS[name]
    config = config or {}
    return bool(getattr(cls, "needs_occupancy", False) or config.get("allocation") == "occupancy")


def build_encoder(name: str, config: dict | None = None, occupancy=None):
    """Construct a spatial encoder, supplying occupancy only to arms that need it."""
    _ensure_loaded()
    kwargs = dict(config or {})
    if encoder_wants_occupancy(name, kwargs):
        if occupancy is None:
            raise ValueError(f"spatial encoding {name!r} requires occupancy")
        kwargs["occupancy"] = occupancy
    return SPATIAL_ENCODERS[name](**kwargs)


def encoder_schedule_params(name: str, config: dict | None = None) -> tuple[int, int, int]:
    """Return (levels, base_resolution, finest_resolution) for an encoder.

    Explicit config values win; anything omitted falls back to the ENCODER CLASS's
    own constructor default, so an occupancy schedule built for a config can never
    diverge from the schedule the encoder will construct for that same config.
    """
    _ensure_loaded()
    if name not in SPATIAL_ENCODERS:
        raise ValueError(
            f"unknown spatial encoding {name!r}; expected one of {sorted(SPATIAL_ENCODERS)}"
        )
    cls = SPATIAL_ENCODERS[name]
    params = inspect.signature(cls.__init__).parameters
    config = config or {}
    result = []
    for key in ("levels", "base_resolution", "finest_resolution"):
        if key in config:
            result.append(int(config[key]))
            continue
        default = params[key].default if key in params else inspect.Parameter.empty
        if default is inspect.Parameter.empty:
            raise ValueError(
                f"spatial encoding {name!r} has no config value or class default for {key!r}"
            )
        result.append(int(default))
    return tuple(result)
