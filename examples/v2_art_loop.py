"""V2 report: art-direction loop on the V1 rig (docs/production-track.md).

The V1 rig (`examples/v1_rig.py`) authored and trained 8 per-light proxies (3
`SphereLight`, 3 `QuadLight`, 2 `TexturedQuadLight`) over the T1 kitchen cache. This
script plays the artist's side of `nrp.torch_backend.art_loop`: hand-author a "graded"
target frame by picking new RGB intensities for the 6 colorable (sphere/quad) rig
lights, reset those same 6 lights to a neutral white starting guess, and run
`optimize_colors` to recover intensities that reproduce the target through the
*already-trained* proxies (geometry held fixed throughout -- this is a color grade,
not a re-placement). `TexturedQuadLight` lights (`neon_sign`, `tv_glow`) have no
`.rgb` field (`rig.py`/`art_loop.py` document why: their texture already bakes in
per-texel emission), so they are excluded from both the target's color changes and
the optimization -- they render identically in the initial guess and the target and
contribute zero loss either way.

After recovery, the script verifies the recovered rig survives a save/reload round
trip bit-for-bit (JSON has no lossy encoding for floats here, so this should be an
exact match, not just close), gates predicted-vs-target convergence with
`nrp.quality.gate.evaluate_gate` (`draft` tier first; if that tier is not met, this
falls back to reporting against `preview` explicitly rather than silently passing a
failing gate), then runs `slider_loop` with a scripted sequence of per-light nudges
to measure headless interactive-grading latency.

Documented finding, not a bug in this script: on the real T1 kitchen cache, three of
the V1 rig's six colorable lights (`window`, `ceiling_panel`, `practical`) have
proxies whose raw output is ~0 everywhere (pre-existing V1/Task-2 issue -- their own
`out/v1-rig/report.json["additivity_gate"]` already fails). Since the render is
`model(...) * rgb`, an all-zero model output means `.rgb` has no gradient signal for
those lights regardless of the target color authored for them, so the convergence
gate can pass at very high PSNR (dominated by the lights that do contribute) while
those three lights' "recovered" color is really just the untouched neutral guess.
`_raw_proxy_magnitude` measures this directly and `recovery_caveats` in the report
flags any colorable light this affects, so the report is honest about which colors
were actually verified as recovered.

Usage:
  uv run python examples/v2_art_loop.py --rig out/v1-rig/rig.json \
      --models-dir out/v1-rig/models --cache out/kitchen-512/path_cache.npz \
      --out-dir out/v2-artloop
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nrp.path_cache import PathCache  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.torch_backend.art_loop import _with_rgb, optimize_colors, slider_loop  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.rig import LightRig, RigLight  # noqa: E402
from nrp.torch_backend.train import light_param_vector, pixel_tensors  # noqa: E402

# A colorable light's raw (unscaled, pre-`.rgb`-multiply) proxy output below this
# peak radiance is treated as "no signal": `predicted_image`/`LightRig.render` both
# compute `model(...) * rgb`, so if the model output is ~0 everywhere, the loss has
# no gradient with respect to that light's `u_rgb` no matter what target color was
# authored for it -- recovery is undetermined, not merely slow to converge.
_RAW_OUTPUT_EPS = 1e-6

# ---------------------------------------------------------------------------
# The hand-authored "art direction" target: new RGB intensities for the 6
# colorable (sphere/quad) V1 rig lights, chosen to read as a distinct grade from
# the V1 defaults (out/v1-rig/rig.json) -- warmer key, cooler fill/rim, a warmer
# window bounce, a dimmed ceiling panel, and a hotter practical. neon_sign/tv_glow
# (TexturedQuadLight, no .rgb) are deliberately absent -- see module docstring.
# ---------------------------------------------------------------------------
ART_DIRECTION_TARGET_RGB = {
    "key": [4.5, 3.5, 2.5],  # brighter and warmer than V1's flat white 3.0
    "fill": [0.6, 0.8, 1.8],  # dimmer and cooler/bluer
    "rim": [0.8, 1.2, 2.5],  # swapped from warm amber to a cool rim
    "window": [2.5, 2.0, 1.2],  # warmed up from V1's cool blue daylight
    "ceiling_panel": [1.0, 1.0, 1.0],  # pulled way down from V1's bright neutral
    "practical": [3.5, 2.0, 0.5],  # pushed hotter/more saturated amber
}

NEUTRAL_GUESS_RGB = [1.0, 1.0, 1.0]  # all-white unit intensity, per light


def _apply_rgbs(rig: LightRig, rgbs: dict) -> LightRig:
    """A copy of `rig` with each named light's `.rgb` replaced from `rgbs` (only
    valid for sphere/quad lights; lights not in `rgbs` -- including any
    `TexturedQuadLight` -- are copied unchanged)."""
    new_lights = []
    for rl in rig.lights:
        if rl.name in rgbs:
            light = _with_rgb(rl.light, np.asarray(rgbs[rl.name], dtype=np.float64))
        else:
            light = rl.light
        new_lights.append(RigLight(name=rl.name, light=light, mute=rl.mute, solo=rl.solo))
    return LightRig(new_lights, rig.models)


def default_adjustments(colorable_names: list[str]) -> list[dict]:
    """~10 scripted per-light color nudges, alternating across the colorable rig
    lights (round-robin), for `slider_loop`'s headless interactive-grading pass."""
    rng = np.random.default_rng(0)
    adjustments = []
    n = 10
    for i in range(n):
        name = colorable_names[i % len(colorable_names)]
        rgb = rng.uniform(0.3, 4.0, size=3).tolist()
        adjustments.append({"light": name, "rgb": rgb})
    return adjustments


def _raw_proxy_magnitude(rig: LightRig, cache: PathCache, names: list[str]) -> dict:
    """Per-named-light mean/max of `model(xy, aux, params)` *before* the `.rgb`
    multiply, for every light in `names` -- independent of any rgb value, so it
    isolates whether a colorable light's proxy has any signal to optimize at all
    on this cache, versus its rendered contribution merely being scaled down by a
    small `.rgb`."""
    device = torch.device("cpu")
    xy, aux = pixel_tensors(cache, device)
    n_px = xy.shape[0]
    by_name = {rl.name: rl for rl in rig.lights}
    magnitudes = {}
    with torch.no_grad():
        for name in names:
            rl = by_name[name]
            params = torch.as_tensor(
                light_param_vector(rl.light), dtype=torch.float32, device=device
            ).expand(n_px, -1)
            out = rig.models[name](xy, aux, params).cpu().numpy()
            magnitudes[name] = {"mean": float(out.mean()), "max": float(out.max())}
    return magnitudes


def run_art_loop(
    rig: LightRig,
    cache: PathCache,
    target_rgbs: dict,
    steps: int,
    lr: float,
    adjustments: list[dict],
    seed: int = 0,
    gate_tier: str = "draft",
    fallback_tier: str = "preview",
) -> dict:
    """Core report logic (no argparse/IO): builds the hand-authored target from
    `target_rgbs`, resets those same lights to an all-white neutral guess, runs
    `optimize_colors`, gates predicted-vs-target convergence, verifies a save/reload
    round trip renders identically, and runs `slider_loop`. Returns a JSON-ready
    report dict; `report["optimized_rig"]` and `report["target"]` are attached for
    the caller to save/inspect (removed before JSON serialization by the caller)."""
    target_rig = _apply_rgbs(rig, target_rgbs)
    target = target_rig.render(cache)

    neutral_rgbs = {name: NEUTRAL_GUESS_RGB for name in target_rgbs}
    guess_rig = _apply_rgbs(rig, neutral_rgbs)

    t0 = time.perf_counter()
    opt_report = optimize_colors(guess_rig, cache, target, steps=steps, lr=lr, seed=seed)
    wall_clock_seconds = time.perf_counter() - t0

    optimized_rig = opt_report["optimized_rig"]
    pred_final = optimized_rig.render(cache)

    gate = evaluate_gate(pred_final, target, tier=gate_tier)
    gate["tier_requested"] = gate_tier
    used_fallback = False
    if not gate["passed"] and fallback_tier != gate_tier:
        fallback_gate = evaluate_gate(pred_final, target, tier=fallback_tier)
        fallback_gate["tier_requested"] = fallback_tier
        used_fallback = True
    else:
        fallback_gate = None

    # Save/reload round trip: the recovered rig must export and re-load to an
    # identical render (the rung's "recovered rig is exported and re-loadable"
    # check), verified here rather than assumed.
    with tempfile.TemporaryDirectory() as tmp:
        roundtrip_path = os.path.join(tmp, "roundtrip_rig.json")
        optimized_rig.save(roundtrip_path)
        reloaded_rig = LightRig.load(roundtrip_path, optimized_rig.models)
        reloaded_render = reloaded_rig.render(cache)
    reload_identical = bool(np.array_equal(pred_final, reloaded_render))

    slider_report = slider_loop(optimized_rig, cache, adjustments)

    # Diagnostic: flag colorable lights whose proxy has ~zero raw output on this
    # cache (see _raw_proxy_magnitude docstring) -- for those lights, u_rgb has no
    # gradient and "recovered_rgbs" below reflects the untouched neutral guess, not
    # an actual recovery, even though the overall image gate can still pass (their
    # contribution to both predicted and target is ~0 regardless of rgb).
    colorable_names = list(target_rgbs.keys())
    raw_output = _raw_proxy_magnitude(rig, cache, colorable_names)
    recovery_caveats = [
        {
            "light": name,
            "reason": (
                f"proxy raw output on this cache is ~0 (max={mag['max']:.3e}); "
                "u_rgb has no gradient, so this light's recovered_rgb below is the "
                "untouched neutral guess, not a verified recovery"
            ),
        }
        for name, mag in raw_output.items()
        if mag["max"] < _RAW_OUTPUT_EPS
    ]

    report = {
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "art_direction_target_rgb": target_rgbs,
        "neutral_guess_rgb": NEUTRAL_GUESS_RGB,
        "proxy_loss_first": opt_report["proxy_loss_first"],
        "proxy_loss_last": opt_report["proxy_loss_last"],
        "proxy_loss_curve": opt_report["proxy_loss_curve"],
        "proxy_vs_target_psnr_db": opt_report["proxy_vs_target_psnr_db"],
        "proxy_vs_target_ssim": opt_report["proxy_vs_target_ssim"],
        "convergence_gate": gate,
        "convergence_gate_fallback": fallback_gate,
        "convergence_gate_used_fallback": used_fallback,
        "wall_clock_seconds": wall_clock_seconds,
        "reload_identical": reload_identical,
        "slider_loop": slider_report,
        "recovered_rgbs": {
            rl.name: rl.light.rgb.tolist()
            for rl in optimized_rig.lights
            if hasattr(rl.light, "rgb") and rl.name in target_rgbs
        },
        "colorable_light_raw_output_magnitude": raw_output,
        "recovery_caveats": recovery_caveats,
        "optimized_rig": optimized_rig,
        "target": target,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rig", required=True, help="V1 rig.json path")
    parser.add_argument("--models-dir", required=True, help="dir containing <light>.pt files")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--gate-tier", default="draft", choices=("preview", "draft", "final"))
    parser.add_argument(
        "--fallback-tier",
        default="preview",
        choices=("preview", "draft", "final"),
        help="tier reported if --gate-tier is missed (honesty fallback, not a silent pass)",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.rig) as f:
        rig_dict = json.load(f)
    models_manifest = rig_dict.get("models")
    if models_manifest:
        models = {
            name: TorchNRP.load(os.path.join(args.models_dir, os.path.basename(rel)))
            for name, rel in models_manifest.items()
        }
    else:
        models = {
            rl["name"]: TorchNRP.load(os.path.join(args.models_dir, f"{rl['name']}.pt"))
            for rl in rig_dict["lights"]
        }
    rig = LightRig.from_dict(rig_dict, models)

    cache = PathCache.load(args.cache)

    rig_names = {rl.name for rl in rig.lights}
    missing = set(ART_DIRECTION_TARGET_RGB) - rig_names
    if missing:
        raise SystemExit(f"target names {sorted(missing)} not found in rig {args.rig}")
    target_rgbs = dict(ART_DIRECTION_TARGET_RGB)
    colorable_names = list(target_rgbs.keys())
    adjustments = default_adjustments(colorable_names)

    report = run_art_loop(
        rig,
        cache,
        target_rgbs,
        steps=args.steps,
        lr=args.lr,
        adjustments=adjustments,
        gate_tier=args.gate_tier,
        fallback_tier=args.fallback_tier,
    )

    optimized_rig = report.pop("optimized_rig")
    target = report.pop("target")

    recovered_rig_path = os.path.join(args.out_dir, "recovered_rig.json")
    optimized_rig.save(recovered_rig_path)

    target_path = os.path.join(args.out_dir, "target.npy")
    np.save(target_path, target)

    report["hardware"] = {
        "platform": platform.platform(),
        "torch_num_threads": torch.get_num_threads(),
    }
    report["files"] = {
        "recovered_rig": os.path.relpath(recovered_rig_path, args.out_dir),
        "target": os.path.relpath(target_path, args.out_dir),
    }

    if report["convergence_gate_used_fallback"]:
        print(
            f"WARNING: convergence missed {args.gate_tier} tier; reporting against "
            f"{args.fallback_tier} tier instead (see report['convergence_gate_fallback'])"
        )
    if report["recovery_caveats"]:
        names = [c["light"] for c in report["recovery_caveats"]]
        print(
            f"WARNING: {names} have ~zero proxy output on this cache -- their "
            "recovered_rgbs are not a verified color recovery (see "
            "report['recovery_caveats'])"
        )

    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
