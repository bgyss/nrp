"""T3 measure: quality-gate evaluation overhead vs render time at 512x512.

Times `nrp.quality.gate.evaluate_gate` (PSNR + SSIM + FLIP + tonemap) against the
E9 quality tiers rendered from a real 512x512 cache (the T1 kitchen scene by
default) and reports gate_seconds / render_seconds per tier.

Accounting: a gate consumes a rendered *pair* — the image being gated plus the
reference it is judged against — so its overhead is gate_seconds divided by the
time spent producing that pair. The <5% claim is measured for the E9 approval
flow (draft gated against a final-tier reference). The strict single-render
ratios are also reported, including the proxy preview tier, which is *faster*
than the gate itself (that is the point of the proxy) — nobody should quote the
<5% number for per-frame preview gating.

  uv run python examples/gate_overhead.py \
      --cache out/kitchen-512/path_cache.npz \
      --model out/kitchen-512-torch/model.pt \
      --out out/quality/gate_overhead.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.lights import SphereLight  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.quality.gate import evaluate_gate  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402
from nrp.torch_backend.relight import render_quality_tier  # noqa: E402


def timed(fn, repeats: int = 1):
    best = None
    value = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        value = fn()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return value, best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default="out/kitchen-512/path_cache.npz")
    parser.add_argument("--model", default="out/kitchen-512-torch/model.pt")
    parser.add_argument("--out", default="out/quality/gate_overhead.json")
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats (min taken)")
    args = parser.parse_args()

    cache_path = Path(args.cache)
    model_path = Path(args.model)
    if not cache_path.exists() or not model_path.exists():
        raise SystemExit(
            f"expected {cache_path} and {model_path}; export/train the T1 scene first "
            f"(see docs/production-track.md)."
        )

    cache = PathCache.load(str(cache_path))
    model = TorchNRP.load(str(model_path))
    light = [SphereLight(center=[1.0, 1.8, 3.6], radius=0.38, rgb=[1.0, 1.0, 1.0])]

    renders = {}
    times = {}
    # "final" here is GATHERLIGHT from the same cache (no fresh high-spp cache in
    # this measurement), which *understates* the true final-tier render time — the
    # overhead ratio below is therefore conservative.
    for tier in ("preview", "draft", "final"):
        (image, _meta), seconds = timed(
            lambda t=tier: render_quality_tier(model, cache, light, quality=t),
            repeats=args.repeats,
        )
        renders[tier], times[tier] = image, seconds

    reference = renders["final"]
    rows = {}
    for tier in ("preview", "draft"):
        gate, gate_seconds = timed(
            lambda t=tier: evaluate_gate(renders[t], reference, t), repeats=args.repeats
        )
        pair_seconds = times[tier] + times["final"]
        rows[tier] = {
            "render_seconds": times[tier],
            "reference_render_seconds": times["final"],
            "gate_seconds": gate_seconds,
            "gate_over_render": gate_seconds / times[tier],
            "gate_over_render_pair": gate_seconds / pair_seconds,
            "gate_verdict": gate["verdict"],
        }

    report = {
        "rung": "T3",
        "scope": "gate evaluation overhead vs render time at 512x512",
        "resolution": [cache.width, cache.height],
        "cache": str(cache_path),
        "cache_segments": cache.segment_count,
        "model": str(model_path),
        "timing": "min over repeats",
        "repeats": args.repeats,
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "tiers": rows,
        "claim_basis": (
            "the <5% overhead claim is gate_seconds / (gated render + reference "
            "render) for the draft tier gated against a final-tier reference — the "
            "E9 approval flow; a gate cannot run without both renders. The reference "
            "here is a same-cache gather, which understates true final-tier cost, so "
            "the ratio is conservative. The strict single-render ratios are also "
            "reported; the preview proxy render is faster than the gate itself."
        ),
        "draft_pair_overhead_below_5pct": rows["draft"]["gate_over_render_pair"] < 0.05,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
