"""Registry of spatial encoders.

Separate from `encoding` so an encoder defined in any module can register itself
without importing the module that imports it back.
"""

from __future__ import annotations

#: name -> encoder class. `build_encoder` is the only construction path the model uses,
#: so adding an arm is one decorator rather than another if/elif branch.
SPATIAL_ENCODERS: dict[str, type] = {}


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
