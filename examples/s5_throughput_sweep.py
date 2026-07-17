"""S5: paper-scale training-throughput sweep — batch size / autocast / torch.compile.

Runs `examples/mitsuba_cornell_128_torch.json` (the committed paper-scale config:
8x256 + 2^14 hashgrid, 50k iters, MPS 37.2 iters/s -> 22.4 min, 35.19 dB) with each
lever applied separately, then combined, and reports per-arm iters/s, wall-clock to
the 35 dB committed quality bar (from the every-1k-iterations checkpoint metrics),
and final held-out PSNR. Batch-size arms run at the *equal effective sample budget*
(iters scaled by 4096/batch, sqrt-scaled LR), so their final PSNR is directly
comparable to the fp32 eager baseline under the roadmap-item-3 criterion
(within 0.5 dB at equal samples).

An arm that crashes (e.g. torch.compile immaturity on MPS) is recorded as an
honest negative with the exception text, not silently dropped.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.train import load_config, train  # noqa: E402

BASELINE_ITERS = 50_000
BASELINE_BATCH = 4096
QUALITY_BAR_DB = 35.0


def arm_overrides(name: str) -> dict:
    if name == "baseline":
        return {}
    if name.startswith("batch"):
        batch = int(name.removeprefix("batch"))
        scale = batch / BASELINE_BATCH
        return {
            "batch_pixels": batch,
            "iters": int(round(BASELINE_ITERS / scale)),
            # Adam: sqrt LR scaling with batch (linear over-shoots at these sizes)
            "lr": 0.005 * math.sqrt(scale),
            "lr_min": 5e-5 * math.sqrt(scale),
        }
    if name in ("bf16", "fp16"):
        return {"precision": name}
    if name == "compile":
        return {"compile": True}
    raise SystemExit(f"unknown arm {name!r} (use baseline|batchN|bf16|fp16|compile or a+b)")


def run_arm(base_cfg: dict, name: str, device: str, out_root: Path) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for part in name.split("+"):
        if part != "baseline":
            cfg.update(arm_overrides(part))
    cfg["device"] = device
    cfg["gather_backend"] = "torch"
    cfg["out_dir"] = str(out_root / f"{device}_{name.replace('+', '_')}")
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    row = {
        "arm": name,
        "device": device,
        "iters": cfg["iters"],
        "batch_pixels": cfg["batch_pixels"],
        "lr": cfg["lr"],
        "precision": cfg.get("precision", "fp32"),
        "compile": bool(cfg.get("compile")),
    }
    t0 = time.perf_counter()
    report_path = Path(cfg["out_dir"]) / "torch_train_report.json"
    try:
        if report_path.exists():
            # a previous sweep invocation already trained this arm to completion
            report = json.loads(report_path.read_text())
            print(f"reusing completed run {report_path}", flush=True)
        else:
            report = train(cfg)
    except Exception as exc:  # honest negative, not a crash of the sweep
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback_tail"] = traceback.format_exc().splitlines()[-3:]
        row["wall_seconds"] = time.perf_counter() - t0
        return row
    row["status"] = "ok"
    row["train_seconds"] = report["train_seconds"]
    row["pool_seconds"] = report["pool_build_seconds"]
    row["iters_per_second"] = cfg["iters"] / report["train_seconds"]
    row["samples_per_second"] = cfg["iters"] * cfg["batch_pixels"] / report["train_seconds"]
    row["final_psnr_db"] = report["val_psnr_db_vs_raw_mean"]
    ckpts = report.get("checkpoint_metrics", [])
    row["seconds_to_35db"] = next(
        (c["train_seconds"] for c in ckpts if c["val_psnr_db_vs_raw_mean"] >= QUALITY_BAR_DB),
        None,
    )
    row["checkpoint_psnr"] = [
        {
            "iteration": c["iteration"],
            "psnr_db": c["val_psnr_db_vs_raw_mean"],
            "train_seconds": c["train_seconds"],
        }
        for c in ckpts
    ]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="examples/mitsuba_cornell_128_torch.json")
    parser.add_argument("--out", default="out/s5-throughput/report.json")
    parser.add_argument("--out-root", default="out/s5-throughput/runs")
    parser.add_argument(
        "--arms",
        default="baseline,batch16384,batch65536,bf16,fp16,compile",
        help="comma-separated arm names; combine levers with '+' (e.g. batch16384+bf16)",
    )
    parser.add_argument("--devices", default="mps,cpu")
    parser.add_argument("--iters", type=int, default=None, help="override baseline iters")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    if args.iters is not None:
        base_cfg["iters"] = args.iters
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root)

    report = {
        "rung": "S5",
        "scope": "paper-scale training throughput levers (batch / autocast / compile)",
        "config": args.config,
        "baseline_reference": {
            "mps_iters_per_second": 37.2,
            "mps_wall_seconds_50k": 1344,
            "cpu_iters_per_second": 23.2,
            "final_psnr_db": 35.19,
            "source": "docs/performance.md paper-scale section",
        },
        "quality_bar_db": QUALITY_BAR_DB,
        "equal_quality_criterion": "final PSNR within 0.5 dB of fp32 eager baseline at "
        "equal effective sample budget (iters scaled by 4096/batch)",
        "rows": [],
    }
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        report["rows"] = prev.get("rows", [])
        done = {(r["arm"], r["device"]) for r in report["rows"]}
    else:
        done = set()

    for device in args.devices.split(","):
        for name in args.arms.split(","):
            name, device = name.strip(), device.strip()
            if (name, device) in done:
                print(f"skipping {device}/{name}: already in report", flush=True)
                continue
            print(f"== arm {name} on {device} ==", flush=True)
            row = run_arm(base_cfg, name, device, out_root)
            report["rows"].append(row)
            out_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps({k: v for k, v in row.items() if k != "checkpoint_psnr"}, indent=2),
                flush=True,
            )
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
