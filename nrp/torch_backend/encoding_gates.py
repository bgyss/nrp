"""Gates G1-G5 for the world-anchored encoding redesign.

Kept free of training and I/O so the promotion logic is unit-testable in milliseconds,
following the split examples/r1_promotion.py already uses. Per-seed passes are required
throughout: a good mean never rescues a failing seed.
"""

from __future__ import annotations

import statistics

REQUIRED_ROTATIONS = (0.0, 90.0, 180.0)


def _deltas(rows: list[dict]) -> list[float]:
    return [float(row["delta_db"]) for row in rows]


def g1_generalization(
    rows: list[dict],
    threshold_db: float = 1.0,
    expected_seeds: set | None = None,
    expected_cameras: set | None = None,
) -> dict:
    """Held-out-camera promotion gate: every seed at every camera must clear the bar.

    The baseline is the nearest trained view's pixel2d proxy reused at the held-out
    camera -- the only thing a screen-space proxy can do at a novel camera.

    ``expected_seeds``/``expected_cameras`` are optional coverage requirements: when
    supplied, every expected seed must appear, and every expected camera must appear
    for every expected seed, or the gate fails regardless of the deltas. When both are
    omitted, coverage is not checked (today's behaviour) and ``coverage_complete`` is
    reported as ``None`` so a reader can tell coverage was never verified, rather than
    conflating "not checked" with "checked and complete".
    """
    if not rows:
        coverage_requested = expected_seeds is not None or expected_cameras is not None
        return {
            "passed": False,
            "reason": "no rows",
            "failures": [],
            "threshold_db": threshold_db,
            "coverage_complete": False if coverage_requested else None,
        }
    failures = [
        {
            "arm": row["arm"],
            "seed": row["seed"],
            "camera": row["camera"],
            "delta_db": float(row["delta_db"]),
        }
        for row in rows
        if float(row["delta_db"]) < threshold_db
    ]
    deltas = _deltas(rows)

    coverage_complete: bool | None
    if expected_seeds is None and expected_cameras is None:
        coverage_complete = None
    else:
        by_seed_cameras: dict = {}
        for row in rows:
            by_seed_cameras.setdefault(row["seed"], set()).add(row["camera"])
        want_seeds = expected_seeds or set()
        want_cameras = expected_cameras or set()
        coverage_complete = all(
            seed in by_seed_cameras and want_cameras <= by_seed_cameras[seed] for seed in want_seeds
        )

    return {
        "passed": not failures and (coverage_complete is not False),
        "threshold_db": threshold_db,
        "failures": failures,
        "n_rows": len(rows),
        "mean_delta_db": statistics.fmean(deltas),
        "worst_delta_db": min(deltas),
        "coverage_complete": coverage_complete,
    }


def g2_capacity_context(rows: list[dict]) -> dict:
    """Reported, never gated. pixel2d is a per-pixel lookup table at these settings,
    so single-view parity measures memorization, not representation quality."""
    return {
        "gated": False,
        "note": (
            "pixel2d is fully dense below its finest level and one vertex per pixel at it, "
            "so single-view parity scores a memorizer at memorization. Reported for audit."
        ),
        "rows": rows,
    }


def g3_stability(
    rows: list[dict], collision_fractions: dict | None = None, threshold_db: float = 1.0
) -> dict:
    """Per-seed pass required; mean/std reported as context only.

    Also asserts the sparse arm's key-collision fraction is exactly zero -- but only
    when ``world_sparse`` rows were actually produced. The assertion is checked
    against the arms that were *measured* (derived from ``rows``), not against
    whatever happens to be present in ``collision_fractions``: a missing entry for a
    measured arm is a failure, never a silent pass. ``collision_assertions_checked``
    tells a reader whether "collision_assertions_passed: True" means "verified zero"
    (checked=True) or "world_sparse was never measured" (checked=False) -- the two
    must never be conflated.
    """
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row["delta_db"]))
    seeds_passing = sum(1 for deltas in by_seed.values() if min(deltas) >= threshold_db)
    deltas = _deltas(rows)
    collisions = collision_fractions or {}

    measured_arms = {row["arm"] for row in rows}
    sparse_was_measured = "world_sparse" in measured_arms
    if sparse_was_measured:
        sparse_collision = collisions.get("world_sparse")
        collision_assertions_checked = True
        collision_assertions_passed = sparse_collision == 0.0
    else:
        collision_assertions_checked = False
        collision_assertions_passed = False

    passed = (
        bool(by_seed)
        and seeds_passing == len(by_seed)
        and (not sparse_was_measured or collision_assertions_passed)
    )
    return {
        "seeds_total": len(by_seed),
        "seeds_passing": seeds_passing,
        "passed": passed,
        "mean_delta_db": statistics.fmean(deltas) if deltas else 0.0,
        "std_delta_db": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "collision_fractions": collisions,
        "collision_assertions_checked": collision_assertions_checked,
        "collision_assertions_passed": collision_assertions_passed,
    }


def g4_frame_robustness(rows: list[dict], threshold_db: float = 1.0) -> dict:
    """Worst orientation governs, and the full rotation matrix must be present.

    Coverage requires every measured (seed, camera) pair to have been tested at all
    required rotations -- a superset check over the pooled rotation values alone is
    satisfiable by a single row at each rotation while every other seed/camera pair
    was only ever tested at 0 degrees.
    """
    seen = {float(row["rotation_degrees"]) for row in rows}
    by_pair: dict = {}
    for row in rows:
        pair = (row["seed"], row["camera"])
        by_pair.setdefault(pair, set()).add(float(row["rotation_degrees"]))
    required = set(REQUIRED_ROTATIONS)
    coverage_complete = bool(by_pair) and all(
        required <= rotations for rotations in by_pair.values()
    )
    deltas = _deltas(rows)
    worst = min(deltas) if deltas else float("-inf")
    return {
        "passed": bool(coverage_complete and deltas and worst >= threshold_db),
        "coverage_complete": coverage_complete,
        "rotations_seen": sorted(seen),
        "required_rotations": list(REQUIRED_ROTATIONS),
        "worst_delta_db": worst,
        "threshold_db": threshold_db,
    }


def g5_fallback_decomposition(rows: list[dict]) -> dict:
    """Mandatory decomposition so a good G1 cannot hide behind a lucky fallback."""
    fractions = [float(row["out_of_occupancy_fraction"]) for row in rows]
    inside = [
        row["in_occupancy_psnr_db"] for row in rows if row.get("in_occupancy_psnr_db") is not None
    ]
    outside = [
        row["out_occupancy_psnr_db"] for row in rows if row.get("out_occupancy_psnr_db") is not None
    ]
    incomplete = [
        row
        for row in rows
        if float(row["out_of_occupancy_fraction"]) > 0.0
        and row.get("out_occupancy_psnr_db") is None
    ]
    return {
        "gated": False,
        "complete": bool(rows) and not incomplete,
        "incomplete_rows": incomplete,
        "mean_out_of_occupancy_fraction": statistics.fmean(fractions) if fractions else 0.0,
        "max_out_of_occupancy_fraction": max(fractions) if fractions else 0.0,
        "mean_in_occupancy_psnr_db": statistics.fmean(inside) if inside else None,
        "mean_out_occupancy_psnr_db": statistics.fmean(outside) if outside else None,
    }


def stop_reason(gates: dict) -> str | None:
    """The spec's stop condition: no arm passing G1 across the full matrix closes the
    track as a characterized negative. No further tuning rounds."""
    arms = gates.get("arms", {})
    if not arms:
        return "no arms were measured"
    for arm in arms.values():
        if (
            arm.get("g1", {}).get("passed")
            and arm.get("g3", {}).get("passed")
            and arm.get("g4", {}).get("passed")
        ):
            return None
    failing = ", ".join(sorted(arms))
    return (
        f"no arm passed G1 across all seeds, cameras, and orientations ({failing}); "
        "close as a characterized negative with the G5 decomposition"
    )
