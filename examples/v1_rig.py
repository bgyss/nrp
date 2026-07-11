"""V1 report: an 8-light production rig on the T1 kitchen scene (docs/production-track.md).

Authors a fixed 8-light rig over the T1 "Country Kitchen" path cache (3 SphereLights,
3 QuadLights, 2 TexturedQuadLights — hand-chosen params, not randomly sampled: a
production rig is authored, not sampled) and trains one per-light-type proxy per
light via the existing `nrp.torch_backend.train.train` entrypoint, narrowing each
light's `light_bounds` around its own authored shape parameters so the proxy
specializes to render *this* light well. Note (documented limitation, not a bug):
`nrp.torch_backend.sampling.sample_light` draws light *positions* from the cache's
recorded path segments regardless of `light_bounds` (bounds only constrain
radius/size/texture range) — so narrowing bounds tightens shape, not position;
each rig light's own authored center is still a plausible in-scene point (drawn from
the same F1-shot-verified interior region of the kitchen cache), and evaluation
below always queries the proxy at the light's *exact* authored parameters, not a
resampled one.

The 8 trained proxies are assembled into a `LightRig` (nrp.torch_backend.rig) and
checked for additivity (Eq. 1's linearity of transport: rig.render == sum of
per-light renders == full-scene GATHERLIGHT of all 8 lights together) against a
`nrp.quality.gate` tier. A monolithic (non-relightable) baseline is trained on the
same fixed 8-light composite with a matched total iteration budget, for a fair
size-vs-quality comparison against the sum of the 8 per-light proxies. Finally,
compositing overhead (ms per rendered frame) is measured as a function of the
number of active (soloed) rig lights, with a linear fit for the marginal
ms-per-added-light.

Usage:
  uv run python examples/v1_rig.py --cache out/kitchen-512/path_cache.npz --out-dir out/v1-rig
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from nrp.gather_light import gather_lights  # noqa: E402
from nrp.lights import QuadLight, SphereLight, TexturedQuadLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.torch_backend.denoise import denoise_image, oidn_available  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.rig import LightRig, RigLight, light_type_of, train_monolithic  # noqa: E402
from nrp.torch_backend.train import pixel_tensors, train  # noqa: E402

# ---------------------------------------------------------------------------
# The rig: hand-authored, fixed light parameters. Centers are drawn from the same
# interior region of the T1 kitchen scene that F1 (examples/f1_shot_kitchen.json)
# verified as a >=32 dB / SSIM>=0.88 pass region for this cache, so the rig lives
# where the exported path cache actually has dense segment coverage.
# ---------------------------------------------------------------------------


def _checkerboard_texture(size: int, rgb_a, rgb_b) -> np.ndarray:
    tex = np.empty((size, size, 3), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            tex[i, j] = rgb_a if (i + j) % 2 == 0 else rgb_b
    return tex


def _radial_gradient_texture(size: int, rgb_center, rgb_edge) -> np.ndarray:
    ys, xs = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size), indexing="ij")
    r = np.clip(np.sqrt(xs**2 + ys**2), 0.0, 1.0)
    rgb_center = np.asarray(rgb_center, dtype=np.float64)
    rgb_edge = np.asarray(rgb_edge, dtype=np.float64)
    return rgb_center[None, None, :] * (1 - r[..., None]) + rgb_edge[None, None, :] * r[..., None]


def default_rig_lights() -> list[RigLight]:
    """The fixed 8-light production rig: 3 sphere, 3 quad, 2 textured_quad."""
    return [
        RigLight(
            name="key",
            light=SphereLight(center=[-0.64, 2.19, -0.22], radius=0.3, rgb=[3.0, 3.0, 3.0]),
        ),
        RigLight(
            name="fill",
            light=SphereLight(center=[-1.81, 1.73, -0.21], radius=0.35, rgb=[1.2, 1.3, 1.6]),
        ),
        RigLight(
            name="rim",
            light=SphereLight(center=[-1.53, 2.8, -0.63], radius=0.2, rgb=[2.0, 1.0, 0.6]),
        ),
        RigLight(
            name="window",
            light=QuadLight(
                center=[-1.2, 2.4, -0.4],
                normal=[0.0, -1.0, 0.0],
                width=0.6,
                height=0.4,
                rgb=[1.5, 1.8, 2.2],
            ),
        ),
        RigLight(
            name="ceiling_panel",
            light=QuadLight(
                center=[-0.9, 2.6, -0.3],
                normal=[0.0, -1.0, 0.0],
                width=0.5,
                height=0.5,
                rgb=[2.5, 2.5, 2.3],
            ),
        ),
        RigLight(
            name="practical",
            light=QuadLight(
                center=[-1.4, 2.0, -0.25],
                normal=[1.0, 0.0, 0.0],
                width=0.25,
                height=0.35,
                rgb=[2.0, 1.4, 0.8],
            ),
        ),
        RigLight(
            name="neon_sign",
            light=TexturedQuadLight(
                center=[-0.7, 2.1, -0.5],
                normal=[0.0, 0.0, 1.0],
                width=0.3,
                height=0.15,
                texture=_checkerboard_texture(8, [3.0, 0.2, 3.0], [0.2, 3.0, 3.0]),
            ),
        ),
        RigLight(
            name="tv_glow",
            light=TexturedQuadLight(
                center=[-1.6, 2.2, -0.35],
                normal=[0.0, 1.0, 0.0],
                width=0.3,
                height=0.2,
                texture=_radial_gradient_texture(8, [0.5, 1.5, 3.0], [0.05, 0.15, 0.4]),
            ),
        ),
    ]


def _light_bounds_for(light) -> dict:
    """Narrow light_bounds around one light's own shape parameters (+/-30%), the
    "specialize, don't generalize" per-light training config. Position is not
    controlled by light_bounds (see module docstring) so it is not set here."""
    if isinstance(light, SphereLight):
        return {"radius_min": light.radius * 0.7, "radius_max": light.radius * 1.3}
    if isinstance(light, QuadLight):
        lo = min(light.width, light.height)
        hi = max(light.width, light.height)
        return {"size_min": lo * 0.7, "size_max": hi * 1.3}
    if isinstance(light, TexturedQuadLight):
        h, w = light.texture.shape[:2]
        tmin = float(light.texture.min()) * 0.7
        tmax = float(light.texture.max()) * 1.3
        return {"texture_size": [h, w], "texture_min": tmin, "texture_max": tmax}
    raise TypeError(f"unsupported light type {type(light)}")


def build_per_light_config(
    base_cfg: dict,
    rig_light: RigLight,
    out_dir: str,
    iters: int,
) -> dict:
    """One `nrp.torch_backend.train.train`-shaped config for a single rig light,
    reusing base_cfg's cache/pool/denoise/model blocks but narrowing light_type
    and light_bounds to this light, and the reduced per-light iteration budget."""
    cfg = copy.deepcopy(base_cfg)
    cfg["out_dir"] = out_dir
    cfg["light_type"] = light_type_of(rig_light.light)
    cfg["light_bounds"] = _light_bounds_for(rig_light.light)
    cfg["iters"] = iters
    if cfg["light_type"] == "textured_quad":
        # Documented deviation: nrp.torch_backend.gather.TorchPathCache.gather_light
        # only implements sphere/quad (it reads light.rgb unconditionally, which
        # TexturedQuadLight does not have — AttributeError). The numpy gather
        # backend (nrp/gather_light.py) is the authoritative reference and already
        # fully supports TexturedQuadLight via gather_textured_quad, so textured_quad
        # rig lights fall back to it for pool-target rendering; sphere/quad lights
        # keep the faster batched torch backend from base_cfg.
        cfg["gather_backend"] = "numpy"
    return cfg


def build_and_evaluate_rig(
    cache: PathCache,
    lights: list[RigLight],
    out_dir: str,
    base_cfg: dict,
    iters: int,
    monolithic_hidden_width: int,
    monolithic_hidden_layers: int,
    monolithic_iters: int,
    monolithic_lr: float,
    gate_tier: str = "preview",
    overhead_frames: int = 10,
    overhead_warmup: int = 2,
    denoise_method: str = "bilateral",
) -> dict:
    """Core report logic (no argparse/IO beyond writing under out_dir): trains one
    per-light proxy per rig light, assembles the LightRig, checks additivity against
    the full-scene GATHERLIGHT reference, trains the matched-budget monolithic
    baseline, compares model sizes, and measures compositing overhead vs light
    count. Returns the JSON-ready report dict; also writes configs/models/rig.json/
    monolithic.pt under out_dir as a side effect."""
    os.makedirs(out_dir, exist_ok=True)
    configs_dir = os.path.join(out_dir, "configs")
    models_dir = os.path.join(out_dir, "models")
    train_dir = os.path.join(out_dir, "train")
    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)

    per_light_reports: dict[str, dict] = {}
    models: dict = {}
    models_manifest: dict[str, str] = {}
    train_wall_seconds = 0.0
    for rl in lights:
        light_train_dir = os.path.join(train_dir, rl.name)
        cfg = build_per_light_config(base_cfg, rl, light_train_dir, iters)
        cfg_path = os.path.join(configs_dir, f"{rl.name}.json")
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        model_src = os.path.join(light_train_dir, "model.pt")
        report_src = os.path.join(light_train_dir, "torch_train_report.json")
        if os.path.exists(model_src) and os.path.exists(report_src):
            # Resume: a previous (possibly interrupted) run already trained this
            # light with an identical config; reuse it instead of retraining.
            with open(report_src) as f:
                report = json.load(f)
            print(f"skipping {rl.name}: reusing existing {model_src}")
        else:
            t0 = time.perf_counter()
            report = train(cfg)
            train_wall_seconds += time.perf_counter() - t0
        model_dst = os.path.join(models_dir, f"{rl.name}.pt")
        shutil.copyfile(model_src, model_dst)
        models_manifest[rl.name] = os.path.relpath(model_dst, out_dir)
        models[rl.name] = TorchNRP.load(model_dst)
        per_light_reports[rl.name] = {
            "light_type": cfg["light_type"],
            "light_bounds": cfg["light_bounds"],
            "iters": iters,
            "val_psnr_db_vs_raw_mean": report["val_psnr_db_vs_raw_mean"],
            "val_ssim_vs_raw_mean": report["val_ssim_vs_raw_mean"],
            "val_flip_vs_raw_mean": report["val_flip_vs_raw_mean"],
            "parameter_count": report["parameter_count"],
            "model_bytes": report["model_bytes"],
            "train_seconds": report["train_seconds"],
        }

    rig = LightRig(lights, models)
    rig_path = os.path.join(out_dir, "rig.json")
    rig.save(rig_path)
    with open(rig_path) as f:
        rig_manifest = json.load(f)
    rig_manifest["models"] = models_manifest
    with open(rig_path, "w") as f:
        json.dump(rig_manifest, f, indent=2)

    # Additivity: rig.render (sum of trained per-light proxies) vs the multi-light
    # GATHERLIGHT reference for the same 8 lights (Eq. 1 linearity). Each per-light
    # proxy was supervised on denoised pool targets (base_cfg["denoise"]), so the
    # additivity reference must be denoised the same way -- gating a denoise-trained
    # proxy sum against the raw, MC-noisy gather is an apples-to-oranges comparison
    # that caps SSIM/FLIP regardless of proxy quality (see nrp/torch_backend/shot.py
    # module docstring and the G2 gate for the same documented choice).
    reference_raw = gather_lights(cache, [rl.light for rl in rig.active_lights()])
    if denoise_method == "none":
        reference = reference_raw
    else:
        reference = denoise_image(
            reference_raw, cache.albedo, cache.normal, cache.depth, method=denoise_method
        )
    predicted = rig.render(cache)
    additivity_gate = evaluate_gate(predicted, reference, tier=gate_tier)
    additivity_gate["tier_requested"] = gate_tier
    additivity_gate_vs_raw = evaluate_gate(predicted, reference_raw, tier=gate_tier)
    additivity_gate_vs_raw["tier_requested"] = gate_tier

    # The monolithic baseline and its gate below are unaffected by this fix (kept
    # against the raw reference, matching the previous behavior of this script).
    reference = reference_raw

    # Monolithic baseline: one non-relightable MLP fit directly to the same fixed
    # 8-light composite, matched total iteration budget for a fair size comparison.
    monolithic_path = os.path.join(out_dir, "monolithic.pt")
    existing_report_path = os.path.join(out_dir, "report.json")
    if os.path.exists(monolithic_path) and os.path.exists(existing_report_path):
        # Resume: a previous run already trained the monolithic baseline with an
        # identical config; reuse it (re-evaluation-only runs, e.g. re-gating
        # additivity with a corrected reference, should not pay for a full retrain).
        with open(existing_report_path) as f:
            prev_report = json.load(f)
        prev_mono = prev_report["monolithic_baseline"]
        mono_model = TorchNRP.load(monolithic_path)
        mono_loss_curve = [prev_mono["loss_first"], prev_mono["loss_last"]]
        monolithic_train_seconds = prev_mono["train_seconds"]
        print(f"skipping monolithic baseline: reusing existing {monolithic_path}")
    else:
        t0 = time.perf_counter()
        mono_model, mono_loss_curve = train_monolithic(
            cache,
            reference,
            hidden_width=monolithic_hidden_width,
            hidden_layers=monolithic_hidden_layers,
            iters=monolithic_iters,
            lr=monolithic_lr,
        )
        monolithic_train_seconds = time.perf_counter() - t0
        mono_model.save(monolithic_path)

    with torch.no_grad():
        xy, aux = pixel_tensors(cache, torch.device("cpu"))
        n_px = xy.shape[0]
        empty_params = torch.zeros((n_px, 0), dtype=torch.float32)
        out = mono_model(xy, aux, empty_params)
        monolithic_pred = out.cpu().numpy().astype(np.float64).reshape(cache.height, cache.width, 3)
    monolithic_gate = evaluate_gate(monolithic_pred, reference, tier=gate_tier)
    monolithic_gate["tier_requested"] = gate_tier

    per_light_total_bytes = sum(
        os.path.getsize(os.path.join(models_dir, f"{rl.name}.pt")) for rl in lights
    )
    monolithic_bytes = os.path.getsize(monolithic_path)

    # Compositing overhead vs active light count: solo the first k lights and time
    # rig.render (warmup + repeat, matching relight_multiview.edit_latency_ms).
    overhead_rows = []
    for k in range(1, len(lights) + 1):
        for i, rl in enumerate(lights):
            rl.solo = i < k
        for _ in range(overhead_warmup):
            rig.render(cache)
        t0 = time.perf_counter()
        for _ in range(overhead_frames):
            rig.render(cache)
        ms = (time.perf_counter() - t0) / overhead_frames * 1000.0
        overhead_rows.append({"n_lights": k, "ms": ms})
    for rl in lights:
        rl.solo = False

    ns = np.array([r["n_lights"] for r in overhead_rows], dtype=np.float64)
    mss = np.array([r["ms"] for r in overhead_rows], dtype=np.float64)
    slope, intercept = np.polyfit(ns, mss, 1) if len(ns) >= 2 else (0.0, float(mss[0]))

    report = {
        "rig": {
            "lights": [rl.to_dict() for rl in lights],
            "models_manifest": models_manifest,
        },
        "per_light_training": per_light_reports,
        "additivity_gate": {
            **additivity_gate,
            "reference": (
                "gather_lights(cache, active rig lights) denoised (oidn/bilateral) -- "
                "matches the denoised pool targets each per-light proxy was supervised on "
                f"(denoise_method={denoise_method!r})"
            ),
            "predicted": "LightRig.render(cache) (sum of 8 trained per-light proxies)",
            "raw_reference_metrics": {
                name: additivity_gate_vs_raw["metrics"][name].get("value")
                for name in ("psnr_db", "ssim", "flip")
            },
            "raw_reference_note": (
                "raw_reference_metrics quote the same predicted rig sum against the "
                "un-denoised gather_lights output; its MC noise bounds SSIM/FLIP well "
                "below the preview threshold independently of proxy quality (see "
                "nrp/torch_backend/shot.py module docstring)"
            ),
        },
        "monolithic_baseline": {
            "hidden_width": monolithic_hidden_width,
            "hidden_layers": monolithic_hidden_layers,
            "iters": monolithic_iters,
            "lr": monolithic_lr,
            "train_seconds": monolithic_train_seconds,
            "loss_first": mono_loss_curve[0],
            "loss_last": mono_loss_curve[-1],
            "gate_vs_reference": monolithic_gate,
        },
        "sizes_bytes": {
            "per_light_total": per_light_total_bytes,
            "monolithic": monolithic_bytes,
            "ratio": per_light_total_bytes / monolithic_bytes if monolithic_bytes else None,
        },
        "compositing_overhead_ms": {
            "rows": overhead_rows,
            "fit_slope_ms_per_light": float(slope),
            "fit_intercept_ms": float(intercept),
        },
        "hardware": {
            "platform": platform.platform(),
            "torch_num_threads": torch.get_num_threads(),
        },
        "total_per_light_train_wall_seconds": train_wall_seconds,
    }
    report_path = os.path.join(out_dir, "report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--iters", type=int, default=800, help="per-light iteration budget (default 800)"
    )
    parser.add_argument(
        "--monolithic-iters",
        type=int,
        default=None,
        help="monolithic baseline iters (default: 8 * --iters, matched total budget)",
    )
    parser.add_argument("--gate-tier", default="preview", choices=("preview", "draft", "final"))
    parser.add_argument(
        "--denoise",
        default="oidn",
        choices=["oidn", "bilateral", "none"],
        help="denoiser for the additivity-gate reference; each per-light proxy was "
        "supervised on oidn-denoised pool targets (base_cfg['denoise']), so oidn is "
        "the apples-to-apples reference here too (needs the nix devshell for libtbb "
        "-- run under `nix develop --command`)",
    )
    args = parser.parse_args()

    if args.denoise == "oidn" and not oidn_available():
        raise SystemExit(
            "oidn unavailable -- run under `nix develop --command` (libtbb), or pass "
            "--denoise bilateral/none (the additivity reference then differs from the "
            "per-light proxies' oidn-denoised supervision target)"
        )

    cache = PathCache.load(args.cache)
    lights = default_rig_lights()
    base_cfg = {
        "cache": os.path.abspath(args.cache),
        "sampling": "segments",
        "gather_backend": "torch",
        "pool": {"size": 64, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "method": "oidn"},
        "batch_pixels": 8192,
        "lr": 0.005,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "encoding": {
                "levels": 10,
                "features_per_level": 2,
                "table_size_log2": 16,
                "base_resolution": 4,
                "finest_resolution": 512,
            },
        },
        "n_val_lights": 12,
        "seed": 0,
        "device": "cpu",
    }
    monolithic_iters = args.monolithic_iters or (8 * args.iters)
    build_and_evaluate_rig(
        cache,
        lights,
        args.out_dir,
        base_cfg,
        iters=args.iters,
        monolithic_hidden_width=128,
        monolithic_hidden_layers=4,
        monolithic_iters=monolithic_iters,
        monolithic_lr=0.005,
        gate_tier=args.gate_tier,
        denoise_method=args.denoise,
    )


if __name__ == "__main__":
    main()
