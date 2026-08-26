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


def build_encoder(name: str, config: dict | None = None, occupancy=None):
    """Construct a spatial encoder, supplying occupancy only to arms that need it."""
    if name not in SPATIAL_ENCODERS:
        raise ValueError(
            f"unknown spatial encoding {name!r}; expected one of {sorted(SPATIAL_ENCODERS)}"
        )
    cls = SPATIAL_ENCODERS[name]
    kwargs = dict(config or {})
    # Arm A opts into occupancy through its config rather than a class flag, because
    # allocation="uniform" must keep working with no cache available.
    wants_occupancy = (
        getattr(cls, "needs_occupancy", False) or kwargs.get("allocation") == "occupancy"
    )
    if wants_occupancy:
        if occupancy is None:
            raise ValueError(f"spatial encoding {name!r} requires occupancy")
        kwargs["occupancy"] = occupancy
    return cls(**kwargs)


def encoder_schedule_params(name: str, config: dict | None = None) -> tuple[int, int, int]:
    """Return (levels, base_resolution, finest_resolution) for an encoder.

    Explicit config values win; anything omitted falls back to the ENCODER CLASS's
    own constructor default, so an occupancy schedule built for a config can never
    diverge from the schedule the encoder will construct for that same config.
    """
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
