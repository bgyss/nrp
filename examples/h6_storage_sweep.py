"""H6 report: sweep F2's two storage levers (approval-frame gating, residual
precision) against the F1 kitchen shot, and find the crossover point (if any)
where proxy + residuals beats raw frames.

F2's baseline (`out/f2-shot/report.json`): fp16 residuals, every frame an approval
frame, cost 1.17x the raw frame bytes -- fp16 residuals are noise-dominated, so
storing one per frame plus the shared model loses to just storing the frames. This
sweeps `examples.f2_final_shot.render_final_shot`'s `residual_precision` (fp16/int8)
and `approval_frames` (all frames, or a sparse subset -- everything else falls back
to proxy-only, gated at `--non-approval-gate-tier`) across the same F1 shot and
records `storage.proxy_plus_residuals_over_raw` per configuration.

Usage:
  uv run python examples/h6_storage_sweep.py \
      --model out/kitchen-512-torch-g2/model.pt --cache out/kitchen-512/path_cache.npz \
      --keyframes examples/f1_shot_kitchen.json --out-dir out/h6-storage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from examples.f2_final_shot import render_final_shot  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.denoise import oidn_available  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402

# (label, residual_precision, approval_fraction) -- approval_fraction=1.0 keeps
# F2's original "every frame is an approval frame" behavior; smaller fractions
# approve an evenly-spaced subset (always including frame 0 and the last frame,
# so the visible ends of the shot are always exactly reconstructible) and fall
# back to proxy-only (non_approval_gate_tier) elsewhere.
SWEEP = [
    ("fp16_all_frames_approval (F2 baseline)", "fp16", 1.0),
    ("int8_all_frames_approval", "int8", 1.0),
    ("fp16_half_approval", "fp16", 0.5),
    ("int8_half_approval", "int8", 0.5),
    ("fp16_quarter_approval", "fp16", 0.25),
    ("int8_quarter_approval", "int8", 0.25),
    ("fp16_tenth_approval", "fp16", 0.1),
    ("int8_tenth_approval", "int8", 0.1),
    ("int8_sparse_approval", "int8", 0.05),
]


def _approval_set(n_frames: int, fraction: float) -> set[int]:
    if fraction >= 1.0:
        return set(range(n_frames))
    n_approve = max(2, round(n_frames * fraction))
    idx = {round(i * (n_frames - 1) / (n_approve - 1)) for i in range(n_approve)}
    return idx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--keyframes", required=True)
    parser.add_argument("--out-dir", default="out/h6-storage")
    parser.add_argument("--denoise", default="oidn", choices=["oidn", "bilateral", "none"])
    parser.add_argument("--non-approval-gate-tier", default="preview")
    args = parser.parse_args()

    if args.denoise == "oidn" and not oidn_available():
        raise SystemExit(
            "oidn unavailable -- run under `nix develop --command`, or pass "
            "--denoise bilateral/none (won't match F2's original oidn-denoised numbers)"
        )

    model = TorchNRP.load(args.model)
    cache = PathCache.load(args.cache)
    with open(args.keyframes) as f:
        spec = json.load(f)
    n_frames = int(spec["frames"])

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for label, precision, fraction in SWEEP:
        approval = _approval_set(n_frames, fraction) if fraction < 1.0 else None
        run_dir = Path(args.out_dir) / label.split(" ")[0]
        report = render_final_shot(
            model,
            cache,
            spec,
            run_dir,
            denoise_method=args.denoise,
            encode=False,
            model_path=args.model,
            residual_precision=precision,
            approval_frames=approval,
            non_approval_gate_tier=args.non_approval_gate_tier,
        )
        storage = report["storage"]
        rows.append(
            {
                "label": label,
                "residual_precision": precision,
                "approval_fraction_requested": fraction,
                "n_approval_frames": report["n_approval_frames"],
                "n_proxy_only_frames": report["n_proxy_only_frames"],
                "all_frames_pass_declared_gate": report["all_frames_pass_declared_gate"],
                "flagged_frames": report["flagged_frames"],
                "proxy_plus_residuals_bytes": storage["proxy_plus_residuals_bytes"],
                "raw_frames_bytes_total": storage["raw_frames_bytes_total"],
                "proxy_plus_residuals_over_raw": storage["proxy_plus_residuals_over_raw"],
                "beats_raw": storage["proxy_plus_residuals_over_raw"] < 1.0,
            }
        )
        print(
            f"{label}: {storage['proxy_plus_residuals_over_raw']:.3f}x raw "
            f"({'PASS' if report['all_frames_pass_declared_gate'] else 'FAIL'} declared gate)"
        )

    crossing = next(
        (r for r in rows if r["beats_raw"] and r["all_frames_pass_declared_gate"]), None
    )
    report = {
        "rung": "H6",
        "scope": "storage-vs-quality sweep of F2's two levers on the F1 kitchen shot",
        "n_frames": n_frames,
        "denoise": args.denoise,
        "non_approval_gate_tier": args.non_approval_gate_tier,
        "sweep": rows,
        "crossover": crossing,
        "crossover_found": crossing is not None,
        "note": (
            "crossover is the first swept configuration (in SWEEP order) that both "
            "beats raw-frame storage and keeps every frame passing its declared "
            "tier gate; None means no swept configuration cleared both bars -- an "
            "honest floor, not a bug, per this rung's own 'or a documented floor' "
            "acceptance criterion"
        ),
    }
    report_path = Path(args.out_dir) / "sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path}")
    if crossing:
        ratio = crossing["proxy_plus_residuals_over_raw"]
        print(f"crossover: {crossing['label']} at {ratio:.3f}x raw")
    else:
        print("no crossover found within the swept configurations")


if __name__ == "__main__":
    main()
