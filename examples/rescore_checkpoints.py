"""Re-score a completed r1_parity-schema run at a different held-out light count.

Trains nothing. A run directory already holds one `model.pt` per (arm, seed); the
number of held-out lights those checkpoints were SCORED against is an evaluation
choice, not a training one, so it can be revisited after the fact for the cost of a
forward pass. On Kitchen 128 that is ~16 s per seed for all four arms, against ~548 s
of training per seed -- which is why every committed verdict on this track can be
re-read for free. See docs/superpowers/plans/2026-08-29-heldout-light-estimator.md.

Because `build_val_set` draws lights one at a time from `default_rng([seed, 0x5EED])`,
a larger set extends the committed one rather than replacing it: re-scoring at the
original count must reproduce the original numbers exactly, and the test asserts it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nrp.experiment_gate import EquivalenceGate  # noqa: E402
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.train import (  # noqa: E402
    build_val_set,
    evaluate,
    load_trained_model,
    model_tensors,
)


def rescore(
    run_dir: Path,
    cache: PathCache,
    *,
    seeds,
    arms,
    control_arm: str,
    base_cfg: dict,
    n_gate_lights: int,
) -> dict:
    device = torch.device("cpu")
    psnr: dict[tuple[str, int], np.ndarray] = {}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        val_set = build_val_set(cache, cfg, n_gate_lights)
        for arm in arms:
            model_path = run_dir / "train" / arm / f"seed{seed}" / "model.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"no checkpoint for {arm} seed {seed}: {model_path}")
            # load_trained_model, not TorchNRP.load: occupancy-allocated arms
            # (world_sparse, occupancy world3d) cannot round-trip without it.
            model = load_trained_model(str(model_path), cache)
            spatial, aux = model_tensors(cache, model, device)
            rows = evaluate(model, val_set, spatial, aux, device)
            psnr[(arm, seed)] = np.asarray([r["psnr_db_vs_raw"] for r in rows], dtype=np.float64)
            del model

    gate = EquivalenceGate()
    out = {"n_gate_lights": int(n_gate_lights), "seeds": list(seeds), "arms": {}}
    for arm in arms:
        if arm == control_arm:
            continue
        per_light = [psnr[(arm, s)] - psnr[(control_arm, s)] for s in seeds]
        means = [float(d.mean()) for d in per_light]
        within = float(np.mean([d.var(ddof=1) for d in per_light]))
        out["arms"][arm] = {
            "per_seed_mean_delta_db": means,
            "mean_db": float(st.mean(means)),
            "between_seed_sd_db": float(st.pstdev(means)),
            "light_sem_db": math.sqrt(within / n_gate_lights),
            "gate": gate.evaluate(means),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--gate-lights", type=int, default=96)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    report = json.loads((args.run_dir / "report.json").read_text())
    cache = PathCache.load(str(args.cache))
    result = rescore(
        args.run_dir,
        cache,
        seeds=report["seeds"],
        arms=[report["control_arm"], *report["world_arms"]],
        control_arm=report["control_arm"],
        base_cfg=report["training_config"],
        n_gate_lights=args.gate_lights,
    )
    result["source_report"] = str(args.run_dir / "report.json")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    for arm, row in result["arms"].items():
        print(
            f"{arm:24s} mean={row['mean_db']:+.3f} sd={row['between_seed_sd_db']:.3f} "
            f"light_sem={row['light_sem_db']:.3f} verdict={row['gate']['verdict']} "
            f"seeds_needed={row['gate'].get('seeds_needed')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
