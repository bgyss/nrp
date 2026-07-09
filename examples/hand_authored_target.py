"""E7's last remaining gap: a genuinely hand-authored image target.

Every prior E7 fixture (`examples/generative_loop.py`) is *derived*: the scribble
target is GATHERLIGHT from a known light, and the "stylized" target is a
deterministic multiplicative filter applied to that rendered image. Neither is
authored content — both are computed from the physical scene.

This script is different in kind: the target image below is specified as an
explicit, hand-picked list of (row, col, RGB) strokes — a small pixel-art plus-sign
on a dark background, chosen by hand, with no reference to any rendered image,
light configuration, or scene geometry. It is "hand-authored" in the sense the
extension asks for ("creating them in any paint tool is fine") — the content is
authored, not computed — even though no external paint tool was used; the
coordinates and colors are the deliverable, typed directly into this file.

The rest of the pipeline matches the existing E7 loop: pretrain a proxy, optimize
physical sphere lights against the hand-authored target through objective/protect
masks, and report the physical-realization gap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.generative_loop import pretrain_proxy, run_inverse  # noqa: E402
from nrp.metrics import psnr  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.optimize_lights import DEFAULT_BOUNDS, random_init  # noqa: E402
from nrp.toy_tracer import trace_path_cache  # noqa: E402

WIDTH = 14
HEIGHT = 14

# Hand-picked (row, col) -> RGB strokes drawing a plus-sign, authored by hand pixel
# by pixel — not derived from any render, filter, or transform. Background is a
# uniform dark blue-gray; the plus is warm orange-yellow; a single "dot" accent in
# the top-right is a cooler cyan, chosen purely for visual contrast, not physics.
BACKGROUND_RGB = (0.05, 0.06, 0.10)
PLUS_RGB = (0.95, 0.65, 0.15)
ACCENT_RGB = (0.10, 0.75, 0.80)

_PLUS_ROWS = range(5, 9)  # horizontal bar rows
_PLUS_COLS = range(5, 9)  # vertical bar cols


def hand_authored_strokes() -> list[tuple[int, int, tuple[float, float, float]]]:
    """The literal list of authored (row, col, rgb) strokes. This *is* the fixture —
    editing these coordinates/colors is how you "repaint" it; nothing here is
    computed from a render."""
    strokes: list[tuple[int, int, tuple[float, float, float]]] = []
    # Vertical bar of the plus, columns 6-7, rows 2-11.
    for r in range(2, 12):
        for c in (6, 7):
            strokes.append((r, c, PLUS_RGB))
    # Horizontal bar of the plus, rows 6-7, columns 2-11.
    for r in (6, 7):
        for c in range(2, 12):
            strokes.append((r, c, PLUS_RGB))
    # A 2x2 accent dot, hand-placed top-right, purely decorative.
    for r in (1, 2):
        for c in (11, 12):
            strokes.append((r, c, ACCENT_RGB))
    return strokes


def render_hand_authored_target() -> np.ndarray:
    image = np.tile(np.array(BACKGROUND_RGB, dtype=np.float64), (HEIGHT, WIDTH, 1))
    for r, c, rgb in hand_authored_strokes():
        image[r, c] = rgb
    return image


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="out/generative/hand_authored_report.json")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--restarts", type=int, default=3)
    args = parser.parse_args()

    out_path = Path(args.out)
    base = out_path.resolve().parent
    base.mkdir(parents=True, exist_ok=True)

    cache = trace_path_cache(WIDTH, HEIGHT, spp=6, max_bounces=2, seed=14)
    model = TorchNRP(
        light_type="sphere",
        hidden_width=24,
        hidden_layers=2,
        encoding={"levels": 3, "features_per_level": 2, "finest_resolution": WIDTH},
    )
    pretrain_stats = pretrain_proxy(model, cache, DEFAULT_BOUNDS, iters=800, lr=2e-3, seed=0)

    target = render_hand_authored_target()
    target_path = base / "hand_authored_target.npy"
    np.save(target_path, target)

    # Objective mask: only the plus-sign + accent region matters (protect the rest).
    objective_mask = np.ones((HEIGHT, WIDTH), dtype=np.float64)
    protect_mask = np.zeros((HEIGHT, WIDTH), dtype=np.float64)
    for r, c, _ in hand_authored_strokes():
        objective_mask[r, c] = 6.0
    protect_mask[:2, :2] = 1.0  # protect a corner untouched by any stroke
    protect_base = np.tile(np.array(BACKGROUND_RGB, dtype=np.float64), (HEIGHT, WIDTH, 1))

    restart_rows = []
    best_report = None
    best_images = None
    t0 = time.perf_counter()
    for restart in range(args.restarts):
        init = random_init(np.random.default_rng(40 + restart), "sphere", DEFAULT_BOUNDS, 2)
        report, images, _ = run_inverse(
            model,
            cache,
            target,
            init,
            args.steps,
            0.25,
            objective_mask,
            protect_mask,
            protect_base,
            40 + restart,
        )
        restart_rows.append(
            {
                "restart": restart,
                "proxy_loss_first": report["proxy_loss_first"],
                "proxy_loss_last": report["proxy_loss_last"],
                "gather_tonemapped_mse": report["gather_tonemapped_mse"],
                "gather_vs_target_psnr_db": report["gather_vs_target_psnr_db"],
            }
        )
        if (
            best_report is None
            or report["gather_tonemapped_mse"] < best_report["gather_tonemapped_mse"]
        ):
            best_report, best_images = report, images
    wall_ms = (time.perf_counter() - t0) * 1000.0
    realized_path = base / "hand_authored_realized_gather.npy"
    np.save(realized_path, best_images["gather"])

    provenance = {
        "scope": "E7 hand-authored (not derived) image target",
        "generation": {
            "method": "explicit hand-picked (row, col, rgb) stroke list in this file",
            "external_generator": None,
            "hand_authored": True,
            "derived_from_render": False,
            "notes": [
                "Unlike generative_loop.py's stylized target (a deterministic filter "
                "applied to a rendered image), this target has no reference to any "
                "GATHERLIGHT render, light configuration, or scene geometry. The "
                "stroke list in hand_authored_strokes() is the authored content.",
                "The shape (a plus-sign with an accent dot) and every color were "
                "chosen by hand for visual clarity, not derived from any transform.",
            ],
        },
        "strokes": [
            {"row": r, "col": c, "rgb": list(rgb)} for r, c, rgb in hand_authored_strokes()
        ],
        "files": {
            "hand_authored_target": {
                "path": target_path.name,
                "sha256": file_sha256(target_path),
            },
            "hand_authored_realized_gather": {
                "path": realized_path.name,
                "sha256": file_sha256(realized_path),
            },
        },
    }
    (base / "hand_authored_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    report = {
        "extension": "E7",
        "scope": "hand-authored (not derived-from-render) image target inverse recovery",
        "resolution": [WIDTH, HEIGHT],
        "proxy_pretrain": pretrain_stats,
        "restarts": restart_rows,
        "best": {
            "gather_tonemapped_mse": best_report["gather_tonemapped_mse"],
            "gather_vs_target_psnr_db": best_report["gather_vs_target_psnr_db"],
            "protected_region_mse_vs_base": best_report["protected_region_mse_vs_base_gather"],
        },
        "wall_ms_total": wall_ms,
        "target_vs_realized_psnr_db": psnr(target, best_images["gather"]),
        "finding": (
            "a hand-authored plus-sign pixel-art target cannot be exactly realized "
            "by two physical sphere lights; the physical-realization gap is the "
            "deliverable, same as generative_loop.py's stylized-target finding"
        ),
        "outputs": {
            "hand_authored_target": "hand_authored_target.npy",
            "hand_authored_realized_gather": "hand_authored_realized_gather.npy",
            "provenance": "hand_authored_provenance.json",
        },
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
