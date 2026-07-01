"""Build the M6/M7 art-directed inputs for NRP inverse optimization.

Produces, in --out-dir:
  base_light.json   the scene's current light
  base_image.npy    GATHERLIGHT image under that light (the "current" appearance)
  art_target.npy    hand-painted objective: base image with a warm brightening
                    painted over a right-wall region (an *objective*, not a
                    physically realizable render)
  art_mask.npy      objective weights: painted region emphasized 5x
  protect_mask.npy  AGGR-style protected region (bottom strip, HUD-like) for M7

Run from the project root:
  uv run python examples/make_art_target.py \
      --cache out/toy/path_cache.npz --out-dir out/toy/art
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.gather_light import gather_light  # noqa: E402
from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402

BASE_LIGHT = {"center": [0.3, 0.7, 0.4], "radius": 0.15, "rgb": [5.0, 5.0, 5.5]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    cache = PathCache.load(args.cache)
    h, w = cache.height, cache.width
    base = gather_light(cache, SphereLight.from_dict(BASE_LIGHT))

    # "Paint" a warm brightening over a right-wall region (rows/cols in pixel space).
    target = base.copy()
    r0, r1 = h // 5, h // 2
    c0, c1 = int(w * 0.62), int(w * 0.95)
    target[r0:r1, c0:c1] *= np.array([2.6, 1.9, 1.2])

    mask = np.ones((h, w))
    mask[r0:r1, c0:c1] = 5.0

    protect = np.zeros((h, w))
    protect[int(h * 0.85) :, :] = 1.0  # bottom strip, AGGR HUD-style protected region

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "base_light.json"), "w") as f:
        json.dump(BASE_LIGHT, f, indent=2)
    np.save(os.path.join(args.out_dir, "base_image.npy"), base)
    np.save(os.path.join(args.out_dir, "art_target.npy"), target)
    np.save(os.path.join(args.out_dir, "art_mask.npy"), mask)
    np.save(os.path.join(args.out_dir, "protect_mask.npy"), protect)
    print(
        f"wrote art-direction inputs to {args.out_dir} "
        f"(painted region rows {r0}:{r1}, cols {c0}:{c1}; "
        f"protected rows {int(h * 0.85)}:{h})"
    )


if __name__ == "__main__":
    main()
