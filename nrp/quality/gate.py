"""Perceptual quality gates (production-track rung T3).

Promotes the repo's SSIM/FLIP/PSNR metrics from ablation tooling to pass/fail
gates, generalizing the E9 supervisor-trust-verdict ladder: a named threshold set
per quality tier (preview / draft / final), a small API any report script can call,
and a CLI that either gates a pred/ref image pair or re-emits an existing report
JSON with a ``quality_gate`` verdict attached next to every metric block it finds.

Metric conventions match the rest of the repo (`nrp.metrics`, the E9/E10 reports):
PSNR on linear HDR radiance, SSIM and FLIP on Reinhard-tonemapped sRGB
(``tonemap_srgb``). A gate passes only if every thresholded metric passes; missing
metrics are reported as ``"skipped"`` and do not fail the gate (report re-emission
must work on reports that only carry a subset of the metrics).

The default tier thresholds are *this repo's* named conventions — chosen against
our toy/cornell/kitchen reports, not taken from the paper — and every gate result
embeds the thresholds it was evaluated against so a report is self-describing.

CLI examples::

    # gate a rendered image against a reference (arrays saved with np.save)
    python -m nrp.quality.gate images pred.npy ref.npy --tier draft

    # re-emit an existing report through the gate (E10 ablation key names)
    python -m nrp.quality.gate report out/ablation/report.json --tier draft \
        --psnr-key psnr_db_mean --ssim-key ssim_mean --flip-key flip_mean \
        --out out/ablation/report_gated.json

Exit status: 0 if every evaluated gate passed, 1 otherwise (so mise tasks and CI
can consume it directly).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..metrics import flip, psnr, ssim, tonemap_srgb

TIERS = ("preview", "draft", "final")


@dataclass(frozen=True)
class TierThresholds:
    """Pass/fail bounds for one tier: PSNR in dB on linear radiance (min),
    SSIM on tonemapped sRGB (min), FLIP on tonemapped sRGB (max)."""

    psnr_db_min: float
    ssim_min: float
    flip_max: float


# Named per-tier thresholds (repo convention, see module docstring). "preview" is
# the interactive proxy tier, "draft" the cached-GATHERLIGHT working tier, "final"
# the approval-frame tier where residual identity should make metrics near-exact.
DEFAULT_THRESHOLDS: dict[str, TierThresholds] = {
    "preview": TierThresholds(psnr_db_min=20.0, ssim_min=0.80, flip_max=0.15),
    "draft": TierThresholds(psnr_db_min=30.0, ssim_min=0.90, flip_max=0.08),
    "final": TierThresholds(psnr_db_min=40.0, ssim_min=0.98, flip_max=0.02),
}


def load_thresholds(path: str) -> dict[str, TierThresholds]:
    """Load a thresholds table from JSON: {tier: {psnr_db_min, ssim_min, flip_max}}.
    Tiers absent from the file keep their defaults."""
    table = dict(DEFAULT_THRESHOLDS)
    for tier, spec in json.loads(Path(path).read_text()).items():
        table[tier] = TierThresholds(**spec)
    return table


def _as_float(value) -> float | None:
    """Metric values in this repo's reports are floats or the string "inf"
    (JSON has no Infinity); anything else is treated as missing."""
    if value == "inf":
        return math.inf
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def gate_metrics(
    metrics: dict,
    tier: str,
    thresholds: dict[str, TierThresholds] | None = None,
) -> dict:
    """Gate already-computed metrics against a named tier.

    `metrics` may contain `psnr_db`, `ssim`, `flip` (missing ones are skipped,
    not failed). Returns a JSON-ready dict with per-metric verdicts, the overall
    pass/fail, and the thresholds used — the generalized E9 trust verdict.
    """
    table = thresholds or DEFAULT_THRESHOLDS
    if tier not in table:
        raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(table)}")
    t = table[tier]
    checks = {
        "psnr_db": (metrics.get("psnr_db"), "min", t.psnr_db_min),
        "ssim": (metrics.get("ssim"), "min", t.ssim_min),
        "flip": (metrics.get("flip"), "max", t.flip_max),
    }
    per_metric = {}
    evaluated = 0
    failed = []
    for name, (raw, sense, bound) in checks.items():
        value = _as_float(raw)
        if value is None:
            per_metric[name] = {"verdict": "skipped", "reason": "metric not provided"}
            continue
        evaluated += 1
        ok = value >= bound if sense == "min" else value <= bound
        per_metric[name] = {
            "value": raw,
            "threshold": bound,
            "sense": sense,
            "verdict": "pass" if ok else "fail",
        }
        if not ok:
            failed.append(name)
    passed = evaluated > 0 and not failed
    return {
        "tier": tier,
        "thresholds": asdict(t),
        "metrics": per_metric,
        "metrics_evaluated": evaluated,
        "passed": passed,
        "verdict": (
            "no metrics evaluated"
            if evaluated == 0
            else f"pass at {tier} tier"
            if passed
            else f"fail at {tier} tier: {', '.join(failed)}"
        ),
    }


def evaluate_gate(
    pred_hdr: np.ndarray,
    ref_hdr: np.ndarray,
    tier: str,
    thresholds: dict[str, TierThresholds] | None = None,
) -> dict:
    """Compute the gate's three metrics from a linear-HDR image pair and gate them.

    PSNR on linear radiance, SSIM/FLIP on Reinhard-tonemapped sRGB — the same
    preprocessing as every existing report. The result additionally records the
    measured metric values and the gate's own evaluation time
    (``evaluation_seconds``), so report scripts can quote gate overhead.
    """
    t0 = time.perf_counter()
    pred_ldr = tonemap_srgb(pred_hdr)
    ref_ldr = tonemap_srgb(ref_hdr)
    metrics = {
        "psnr_db": psnr(pred_hdr, ref_hdr),
        "ssim": ssim(pred_ldr, ref_ldr, data_range=1.0),
        "flip": flip(pred_ldr, ref_ldr),
    }
    result = gate_metrics(metrics, tier, thresholds)
    result["evaluation_seconds"] = time.perf_counter() - t0
    result["display_metric_preprocess"] = "Reinhard tonemap + sRGB before SSIM/FLIP"
    return result


def re_emit_report(
    report: dict,
    tier: str,
    psnr_key: str = "psnr_db",
    ssim_key: str = "ssim",
    flip_key: str = "flip",
    thresholds: dict[str, TierThresholds] | None = None,
) -> tuple[dict, list[dict]]:
    """Re-emit an existing report through the gate.

    Walks the report JSON; every dict that carries at least one of the three
    metric keys gets a sibling ``"quality_gate"`` entry (existing content is never
    modified, so the original conclusions are unchanged by construction). Returns
    the augmented copy plus a flat list of the attached gates with their JSON
    paths, ordered by discovery.
    """
    out = copy.deepcopy(report)
    attached: list[dict] = []

    def walk(node, path):
        if isinstance(node, dict):
            metrics = {
                "psnr_db": node.get(psnr_key),
                "ssim": node.get(ssim_key),
                "flip": node.get(flip_key),
            }
            if any(_as_float(v) is not None for v in metrics.values()):
                gate = gate_metrics(metrics, tier, thresholds)
                node["quality_gate"] = gate
                attached.append({"path": path, **gate})
            for key, child in node.items():
                if key != "quality_gate":
                    walk(child, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, child in enumerate(node):
                walk(child, f"{path}[{i}]")

    walk(out, "")
    return out, attached


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nrp.quality.gate", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--tier", choices=TIERS, required=True)
    common.add_argument("--thresholds", help="JSON file overriding the default per-tier thresholds")
    common.add_argument("--out", help="write the gate result / re-emitted report here")

    p_img = sub.add_parser(
        "images", parents=[common], help="gate a rendered .npy image against a reference .npy"
    )
    p_img.add_argument("pred")
    p_img.add_argument("ref")

    p_rep = sub.add_parser(
        "report",
        parents=[common],
        help="re-emit an existing report JSON with quality_gate verdicts attached",
    )
    p_rep.add_argument("report")
    p_rep.add_argument("--psnr-key", default="psnr_db")
    p_rep.add_argument("--ssim-key", default="ssim")
    p_rep.add_argument("--flip-key", default="flip")

    args = parser.parse_args(argv)
    thresholds = load_thresholds(args.thresholds) if args.thresholds else None

    if args.mode == "images":
        result = evaluate_gate(np.load(args.pred), np.load(args.ref), args.tier, thresholds)
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
        return 0 if result["passed"] else 1

    report = json.loads(Path(args.report).read_text())
    gated, attached = re_emit_report(
        report,
        args.tier,
        psnr_key=args.psnr_key,
        ssim_key=args.ssim_key,
        flip_key=args.flip_key,
        thresholds=thresholds,
    )
    if not attached:
        print(f"no metric blocks found in {args.report} with the given keys", file=sys.stderr)
        return 1
    for gate in attached:
        print(f"{gate['path'] or '<root>'}: {gate['verdict']}")
    if args.out:
        Path(args.out).write_text(json.dumps(gated, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0 if all(g["passed"] for g in attached) else 1


if __name__ == "__main__":
    raise SystemExit(main())
