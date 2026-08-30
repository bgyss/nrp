# Held-Out-Light Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the representation track's promotion gate decidable by measuring each
seed's paired delta over enough held-out lights, instead of buying precision with
seeds — and re-read every existing K/R verdict under the fixed estimator without
retraining anything.

**Architecture:** The gate's unit of observation is a seed, whose value is the mean
paired PSNR delta over the held-out light set. That set is 12 lights, while the
per-light delta on Country Kitchen has sd ≈ 3.3 dB — so a seed's value carries
±0.94 dB of pure light-sampling error at a −0.5 dB threshold. This plan splits the
gate's held-out light count (`n_gate_lights`) from the training-time checkpoint
validation count (`n_val_lights`), adds an evaluation-only re-scoring path so
committed checkpoints can be re-read at any light count, and re-reads the three
existing campaigns. No training-loop, encoder, or gate-rule change.

**Tech Stack:** Python 3.12, numpy, PyTorch 2.12 (CPU), OIDN (single-threaded),
`unittest`, `uv`, `nix develop` for the OIDN/TBB shell.

## Global Constraints

- The gate threshold stays **−0.5 dB**. This plan does not weaken it.
- The gate rule stays `nrp/experiment_gate.py`'s `EquivalenceGate` with looks
  `(8, 16, 24, 32, 40, 48)`, cap 48, α = 0.05 split six ways. No new looks, no
  off-schedule evaluation.
- The **estimand is unchanged**: mean paired PSNR delta versus the same-seed
  `pixel2d` control over lights drawn from the existing light prior
  (`sample_light(..., "sphere", {"radius_min": 0.05, "radius_max": 0.25}, "segments")`).
  Only its *precision* changes. No trimming, no median, no degenerate-light
  rejection — those change the estimand and are explicitly out of scope (see
  "Rejected" below).
- `n_gate_lights` must be **pre-registered before any new training**, and every
  report must record it.
- Held-out light sets remain per-seed, built by the dedicated validation RNG
  `np.random.default_rng([seed, 0x5EED])`, and shared by all arms at that seed.
  Because `build_val_set` draws lights in a loop from that stream, a larger count
  is a strict **superset** of a smaller one — this is what makes re-reads
  comparable to committed runs.
- Every re-read must reproduce the committed 12-light numbers exactly when run at
  `n_gate_lights=12`. That equality is the harness's correctness test.
- Run anything that touches OIDN under `nix develop --command …` (the venv's
  `libOpenImageDenoise` needs the devshell's `libtbb.12.dylib`).
- `uv run ruff check .` and `uv run python -m unittest discover -s tests` must pass
  at every commit.

---

## Measured evidence this plan is built on

Re-scoring the 32 committed checkpoints in `out/r1-parity-kitchen-eq/train/` at 96
held-out lights instead of 12 — **same checkpoints, no retraining, 16 s per seed
versus ~548 s of training per seed** — moves every published number:

| arm | mean Δ @12 | mean Δ @96 | between-seed sd @12 | @96 | light-sem @12 | @96 | seeds needed @12 | @96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `world_sparse` | −0.493 | **−0.009** | 0.961 | 0.631 | 0.942 | 0.298 | 34 | **17** |
| `world_normal_triplane` | −0.446 | −0.416 | 0.651 | 0.556 | 1.140 | 0.319 | 18 | **14** |
| `world3d` | −1.368 | −0.674 | 1.468 | 0.803 | 1.072 | 0.320 | 73 | **25** |

Three consequences:

1. At 12 lights the light-sampling standard error (0.94–1.14 dB) is **as large as or
   larger than the entire between-seed spread** the gate is fighting. For
   `world_normal_triplane` the implied training-stochasticity variance is
   indistinguishable from zero.
2. `world_sparse`'s published −0.493 dB Kitchen deficit is almost entirely
   estimator noise: the same checkpoints read at 96 lights sit at −0.009 dB, i.e.
   dead on parity. [Correction, applied in the fixes pass that merged this plan:
   the result docs deliberately do not draw this inference — the 96-light
   verdict is still `continue`, and 17 more seeds are needed to say `world_sparse`
   is at parity with `pixel2d`. See
   `docs/performance.md#r1-kitchen-parity-re-read-at-96-held-out-lights-2026-08-29`.
   This plan is preserved as written; the claim above is superseded.]
3. All three arms move from "needs 18–73 seeds" to "needs 14–25", i.e. from
   partly-past the 48-seed cap to comfortably inside it. `world_sparse` becomes
   decidable at the n = 24 look — 16 more seeds, ≈ 2.4 CPU-hours.

Degenerate held-out lights are **not** the driver and are not addressed here: 4 of
96 lights have a control PSNR below 10 dB (one at −2.4 dB) and carry mean |Δ| of
4.03 dB versus 2.02 dB for the rest, but dropping them moves the per-light sd only
from 3.27 to 3.06 dB. Recorded as an observation, not a remediation.

**Rejected alternatives.** (a) *Run more seeds at 12 lights* — this is what the
track has been doing; it costs ~9 CPU-minutes per seed to buy the variance
reduction that 84 extra lights buy for 12 CPU-seconds. (b) *Trim or reject
degenerate lights* — changes the estimand, is not the dominant term, and would make
new numbers incomparable with committed ones. (c) *Loosen the threshold or the
looks schedule* — the gate design is sound; the estimator feeding it was not.

---

## File Structure

- `nrp/torch_backend/train.py` — `build_val_set` gains an explicit count argument so
  callers can build a gate set independently of the training-time validation set.
  No training-loop behavior change.
- `examples/r1a_variance.py` — `build_frozen_validation_sets` gains an
  `n_gate_lights` argument; this is the shared helper `r1_parity.py` and
  `r1_kitchen_k1.py` both import, so it is the single seam for all three campaigns.
- `examples/r1_parity.py` — `BASE_TRAIN_CONFIG` gains `n_gate_lights`, a
  `--gate-lights` CLI flag, the strict-key list gains the new key, and the report
  records it.
- `examples/r1_kitchen_k1.py` — same flag, and its control-compatibility check
  learns the new key so a sweep cannot be read against a control scored at a
  different light count.
- `examples/rescore_checkpoints.py` — **new.** Evaluation-only re-read of a
  committed `r1_parity`-schema run directory at an arbitrary gate-light count.
  This is what makes Tasks 5–7 free.
- `tests/test_r1_parity.py`, `tests/test_r1_kitchen_k1.py` — extended.
- `tests/test_rescore_checkpoints.py` — **new.**
- `docs/performance.md`, `docs/representation-track.md`,
  `docs/plans/2026-08-27-kitchen-parity-next-steps.md`, `docs/tracks.md` — updated
  with the re-read verdicts once measured.

---

### Task 1: `build_val_set` takes an explicit light count

**Files:**
- Modify: `nrp/torch_backend/train.py:307-330`
- Test: `tests/test_torch_train.py`

**Interfaces:**
- Produces: `build_val_set(cache, cfg, n_lights: int | None = None) -> list[dict]`.
  When `n_lights` is `None` the behavior is exactly today's
  (`cfg.get("n_val_lights", 12)`). The RNG stream is unchanged, so
  `build_val_set(c, cfg, 96)[:12] == build_val_set(c, cfg, 12)` light-for-light.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_torch_train.py`:

```python
def test_build_val_set_larger_count_is_a_superset(self):
    """A bigger gate set must extend the committed one, not resample it --
    this is what makes a re-read comparable to the run it re-reads."""
    cache = _tiny_cache()
    cfg = {"seed": 3, "light_type": "sphere",
           "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
           "sampling": "segments", "denoise": {"enabled": False},
           "n_val_lights": 4}
    small = build_val_set(cache, cfg, 4)
    large = build_val_set(cache, cfg, 10)
    self.assertEqual(len(small), 4)
    self.assertEqual(len(large), 10)
    for a, b in zip(small, large):
        self.assertEqual(a["light"], b["light"])

def test_build_val_set_defaults_to_config_count(self):
    cache = _tiny_cache()
    cfg = {"seed": 3, "light_type": "sphere",
           "light_bounds": {"radius_min": 0.05, "radius_max": 0.25},
           "sampling": "segments", "denoise": {"enabled": False},
           "n_val_lights": 4}
    self.assertEqual(len(build_val_set(cache, cfg)), 4)
```

Use whatever tiny-cache helper the module already defines; if there is none, build
one with `PathCache` the way the neighbouring tests in that file do.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_torch_train -v`
Expected: FAIL — `build_val_set() takes 2 positional arguments but 3 were given`

- [ ] **Step 3: Write minimal implementation**

In `nrp/torch_backend/train.py`, change the signature and the loop bound only:

```python
def build_val_set(cache: PathCache, cfg: dict, n_lights: int | None = None) -> list[dict]:
    """Fixed held-out validation set: fresh light configurations from a dedicated RNG
    (never the training RNG, so evaluating cannot perturb pool sampling), each with
    its raw GATHERLIGHT reference (physically grounded) and the denoised one (what
    the network is supervised with), computed once and reused at every checkpoint.

    `n_lights` overrides `cfg["n_val_lights"]` so a caller can build a LARGER
    held-out set for a promotion gate than the one training validates against,
    without paying for the larger set at every checkpoint. Lights are drawn one at a
    time from the same seeded stream, so a larger set is a strict superset of a
    smaller one -- that is what lets a committed run be re-read at a higher count.
    """
    val_rng = np.random.default_rng([cfg.get("seed", 0), 0x5EED])
    val_set = []
    count = cfg.get("n_val_lights", 12) if n_lights is None else int(n_lights)
    for _ in range(count):
```

Leave the body of the loop untouched.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_torch_train -v`
Expected: PASS, and no other test in the module changes status.

- [ ] **Step 5: Commit**

```bash
git add nrp/torch_backend/train.py tests/test_torch_train.py
git commit -m "feat: build_val_set takes an explicit held-out light count"
```

---

### Task 2: Frozen gate sets are sized independently of training validation

**Files:**
- Modify: `examples/r1a_variance.py:241-254`
- Test: `tests/test_r1_parity.py`

**Interfaces:**
- Consumes: `build_val_set(cache, cfg, n_lights)` from Task 1.
- Produces: `build_frozen_validation_sets(cache, base_cfg, seeds, n_gate_lights=None)`
  returning the same `(validation_sets, specs_by_seed)` pair as today. `None` keeps
  today's behavior.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_r1_parity.py`:

```python
def test_frozen_sets_honor_an_explicit_gate_light_count(self):
    """The gate's held-out set is sized by n_gate_lights, not by the
    training-time n_val_lights -- raising it must not slow training."""
    cache = _small_cache()
    base = dict(r1_parity.BASE_TRAIN_CONFIG)
    base["n_val_lights"] = 2
    base["denoise"] = {"enabled": False}
    sets, specs = r1a_variance.build_frozen_validation_sets(cache, base, (0,), n_gate_lights=6)
    self.assertEqual(len(sets[0]), 6)
    self.assertEqual(len(specs["0"]), 6)
    small, _ = r1a_variance.build_frozen_validation_sets(cache, base, (0,))
    self.assertEqual(len(small[0]), 2)
    for a, b in zip(small[0], sets[0]):
        self.assertEqual(a["light"], b["light"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_r1_parity -v`
Expected: FAIL — `build_frozen_validation_sets() got an unexpected keyword argument 'n_gate_lights'`

- [ ] **Step 3: Write minimal implementation**

In `examples/r1a_variance.py`:

```python
def build_frozen_validation_sets(
    cache: PathCache,
    base_cfg: dict,
    seeds: tuple[int, ...] | list[int],
    n_gate_lights: int | None = None,
) -> tuple[dict[int, list[dict]], dict[str, list[dict]]]:
    """Build exactly one held-out set per seed for all matrix arms to share.

    `n_gate_lights` sizes the GATE's held-out set independently of the training-time
    `n_val_lights`. They are separate because they buy different things: the
    training set is re-scored at every checkpoint (so it must stay cheap), while the
    gate set is scored once per (arm, seed) and its size is what sets the precision
    of the per-seed delta the gate consumes. At 12 lights on Country Kitchen that
    precision is +/-0.94 dB against a -0.5 dB threshold; see
    docs/superpowers/plans/2026-08-29-heldout-light-estimator.md.
    """
    validation_sets = {}
    specs_by_seed = {}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        val_set = build_val_set(cache, cfg, n_gate_lights)
        validation_sets[seed] = val_set
        specs = validation_light_specs(val_set, cfg["light_type"])
        specs_by_seed[str(seed)] = specs
    return validation_sets, specs_by_seed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_r1_parity tests.test_r1a_variance -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/r1a_variance.py tests/test_r1_parity.py
git commit -m "feat: gate held-out light count is independent of training validation"
```

---

### Task 3: `r1_parity` pre-registers and records `n_gate_lights`

**Files:**
- Modify: `examples/r1_parity.py:168-179` (`BASE_TRAIN_CONFIG`), the
  `build_frozen_validation_sets` call at `:455`, `reproduce_command` at `:301`, the
  argument parser, and the report assembly
- Test: `tests/test_r1_parity.py`

**Interfaces:**
- Consumes: `build_frozen_validation_sets(..., n_gate_lights=...)` from Task 2.
- Produces: `BASE_TRAIN_CONFIG["n_gate_lights"] = 96`; a `--gate-lights` int flag
  defaulting to 96; `report["gate_lights"]` and `report["training_config"]["n_gate_lights"]`.

**Why 96.** It is the smallest power-of-two-ish count measured to push the
light-sampling standard error (0.30 dB) well below the residual between-seed spread
(0.56–0.80 dB), so the estimator stops being the binding term. Going higher has
sharply diminishing returns — the remaining variance is genuine training
stochasticity, which only seeds reduce. Pre-registered here, before any new run.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_r1_parity.py`:

```python
def test_gate_lights_default_is_pre_registered_at_96(self):
    self.assertEqual(r1_parity.BASE_TRAIN_CONFIG["n_gate_lights"], 96)

def test_reproduce_command_records_gate_lights(self):
    args = argparse.Namespace(
        cache="c.npz", out_dir="o", iters=3000, base_resolution=4,
        finest_resolution=128, denoise_method="oidn", gate="equivalence",
        max_seeds=8, bootstrap_seed=0, bootstrap_resamples=2000, gate_lights=96,
    )
    cmd = r1_parity.reproduce_command(args, (0, 1))
    self.assertIn("--gate-lights 96", cmd)
```

Match `reproduce_command`'s real `Namespace` fields as they exist in the file; add
only `gate_lights`.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_r1_parity -v`
Expected: FAIL — `KeyError: 'n_gate_lights'`

- [ ] **Step 3: Write minimal implementation**

In `examples/r1_parity.py`, add to `BASE_TRAIN_CONFIG` after `"n_val_lights": 12,`:

```python
    # The GATE's held-out set. Separate from n_val_lights (the training-time
    # checkpoint set) because only this one sets the precision of the per-seed delta
    # the equivalence gate consumes. At 12 lights the light-sampling standard error
    # on Kitchen 128 is +/-0.94 dB against a -0.5 dB threshold -- larger than the
    # between-seed spread the gate is trying to resolve. 96 puts it at 0.30 dB.
    "n_gate_lights": 96,
```

Add the flag to the parser:

```python
    parser.add_argument(
        "--gate-lights",
        type=int,
        default=BASE_TRAIN_CONFIG["n_gate_lights"],
        help="held-out lights per seed for the promotion gate (default 96)",
    )
```

Thread it into the base config where the parser's other overrides are applied:

```python
    base_cfg["n_gate_lights"] = args.gate_lights
```

Pass it at the call site (`:455`):

```python
        batch_sets, batch_specs = build_frozen_validation_sets(
            cache, base_cfg, batch, n_gate_lights=base_cfg.get("n_gate_lights")
        )
```

Add `--gate-lights {args.gate_lights} ` to the f-string in `reproduce_command`, and
add `"gate_lights": args.gate_lights,` to the report dict next to
`"bootstrap_seed"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_r1_parity -v && uv run ruff check .`
Expected: PASS, no lint findings.

- [ ] **Step 5: Commit**

```bash
git add examples/r1_parity.py tests/test_r1_parity.py
git commit -m "feat: r1_parity pre-registers a 96-light gate set"
```

---

### Task 4: `r1_kitchen_k1` inherits the flag and guards control compatibility

**Files:**
- Modify: `examples/r1_kitchen_k1.py:110-125` (`_STRICT_TRAINING_CONFIG_KEYS`), its
  parser, and its `build_frozen_validation_sets` call site
- Test: `tests/test_r1_kitchen_k1.py`

**Interfaces:**
- Consumes: `BASE_TRAIN_CONFIG["n_gate_lights"]` and `--gate-lights` from Task 3.
- Produces: a `ValueError` when a sweep's gate-light count differs from the control
  report's.

A sweep read against a control scored on a different number of held-out lights would
attribute an estimator difference to the swept resolution — exactly the class of
confound `_STRICT_TRAINING_CONFIG_KEYS` already exists to catch.

**Legacy controls (decided by the user, 2026-08-29).** Every committed control
report predates this key, and the existing check reads `control_training_config.get(key)`
— so a bare `None != 96` would reject every historical control and break both the
documented K1 command and Task 7. **A missing `n_gate_lights` normalizes to 12, its
historical value, on both sides before comparison.** A legacy control therefore still
pairs with a 12-light run and correctly raises against a 96-light run — where the
numbers genuinely are incomparable. This normalization applies to `n_gate_lights`
only; every other strict key keeps its exact `.get(key)` comparison.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_r1_kitchen_k1.py`:

```python
def test_control_with_a_different_gate_light_count_is_rejected(self):
    control = self._control_report(n_gate_lights=12)
    run_cfg = dict(control["training_config"])
    run_cfg["n_gate_lights"] = 96
    with self.assertRaisesRegex(ValueError, "n_gate_lights"):
        r1_kitchen_k1.validate_control_report(
            control, cache_path="out/kitchen/path_cache.npz",
            base_resolution=4, run_training_config=run_cfg,
        )

def test_a_control_predating_the_key_reads_as_twelve_lights(self):
    """Committed controls have no n_gate_lights. Absence is 12, not None --
    otherwise every historical control is rejected by a 12-light run."""
    control = self._control_report()
    control["training_config"].pop("n_gate_lights", None)
    run_cfg = dict(control["training_config"])
    run_cfg["n_gate_lights"] = 12
    r1_kitchen_k1.validate_control_report(
        control, cache_path="out/kitchen/path_cache.npz",
        base_resolution=4, run_training_config=run_cfg,
    )

def test_a_control_predating_the_key_is_rejected_by_a_96_light_run(self):
    control = self._control_report()
    control["training_config"].pop("n_gate_lights", None)
    run_cfg = dict(control["training_config"])
    run_cfg["n_gate_lights"] = 96
    with self.assertRaisesRegex(ValueError, "n_gate_lights"):
        r1_kitchen_k1.validate_control_report(
            control, cache_path="out/kitchen/path_cache.npz",
            base_resolution=4, run_training_config=run_cfg,
        )
```

Build `self._control_report` on whatever fixture helper the module already uses for
`validate_control_report` tests, adding `n_gate_lights` to its `training_config`.
Match the real function name and argument names in the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m unittest tests.test_r1_kitchen_k1 -v`
Expected: FAIL — no `ValueError` raised (the key is not checked)

- [ ] **Step 3: Write minimal implementation**

Add `"n_gate_lights"` to `_STRICT_TRAINING_CONFIG_KEYS` in
`examples/r1_kitchen_k1.py`:

```python
    "n_val_lights",
    # The gate's held-out light count sets the precision of every per-seed delta in
    # the report. A sweep read against a control scored at a different count would
    # silently attribute an estimator difference to the swept resolution.
    "n_gate_lights",
)

#: Controls written before `n_gate_lights` existed were all scored at 12 lights, so
#: absence means 12 rather than "unknown". Comparing a bare `.get()` would make every
#: committed control incompatible with every run, including the 12-light re-reads
#: whose whole purpose is to reproduce those controls exactly.
_LEGACY_TRAINING_CONFIG_DEFAULTS = {"n_gate_lights": 12}
```

and normalize both sides inside the strict-key loop:

```python
        for key in _STRICT_TRAINING_CONFIG_KEYS:
            missing = _LEGACY_TRAINING_CONFIG_DEFAULTS.get(key)
            control_value = control_training_config.get(key, missing)
            run_value = run_training_config.get(key, missing)
            if control_value != run_value:
```

Keys absent from `_LEGACY_TRAINING_CONFIG_DEFAULTS` fall back to `None`, preserving
today's exact comparison for every other strict key.

Add the same `--gate-lights` argument as Task 3 (default
`r1_parity.BASE_TRAIN_CONFIG["n_gate_lights"]`), write it into the run config, and
pass `n_gate_lights=` at this module's `build_frozen_validation_sets` call site.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m unittest tests.test_r1_kitchen_k1 tests.test_r1_parity -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/r1_kitchen_k1.py tests/test_r1_kitchen_k1.py
git commit -m "feat: K1 sweep binds its gate-light count to the control's"
```

---

### Task 5: Evaluation-only re-scoring of committed checkpoints

**Files:**
- Create: `examples/rescore_checkpoints.py`
- Test: `tests/test_rescore_checkpoints.py`

**Interfaces:**
- Consumes: `load_trained_model`, `model_tensors`, `evaluate`, `build_val_set` from
  `nrp.torch_backend.train`; `EquivalenceGate` from `nrp.experiment_gate`;
  `pair_validation_metrics` from `examples/r1a_variance.py`.
- Produces:
  `rescore(run_dir: Path, cache: PathCache, *, seeds, arms, control_arm, base_cfg, n_gate_lights) -> dict`
  returning `{"n_gate_lights": int, "seeds": list[int], "arms": {arm: {"per_seed_mean_delta_db": [...], "mean_db": float, "between_seed_sd_db": float, "light_sem_db": float, "gate": {...}}}}`.
- CLI: `--run-dir`, `--cache`, `--gate-lights`, `--out`.

This trains nothing. It reloads the `model.pt` files a completed run already wrote
and re-scores them against a larger held-out set. `load_trained_model` (not
`TorchNRP.load`) is mandatory — it rebuilds the occupancy that `world_sparse` and
occupancy-allocated `world3d` need.

- [ ] **Step 1: Write the failing test**

Create `tests/test_rescore_checkpoints.py`:

```python
"""Tests for the evaluation-only re-scorer (examples/rescore_checkpoints.py)."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    """Same file-location import the other examples/ runners' tests use."""
    spec = importlib.util.spec_from_file_location(
        "rescore_checkpoints", ROOT / "examples" / "rescore_checkpoints.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestRescore(unittest.TestCase):
    def test_rescore_at_the_original_count_reproduces_the_committed_deltas(self):
        """The correctness test for the whole re-read: at n_gate_lights=12 the
        re-scorer must return the committed report's numbers exactly, because it is
        reading the same checkpoints against the same lights."""
        run_dir = ROOT / "out/r1-parity-kitchen-eq"
        if not (run_dir / "report.json").exists():
            self.skipTest("committed 8-seed Kitchen parity run not present")

        from nrp.path_cache import PathCache

        runner = load_runner()
        report = json.loads((run_dir / "report.json").read_text())
        cache = PathCache.load(str(ROOT / "out/kitchen/path_cache.npz"))
        got = runner.rescore(
            run_dir, cache,
            seeds=report["seeds"],
            arms=["pixel2d", *report["world_arms"]],
            control_arm=report["control_arm"],
            base_cfg=report["training_config"],
            n_gate_lights=12,
        )
        for arm in report["world_arms"]:
            expected = [
                float(sum(r["delta_db"] for r in row["per_light_deltas"])
                      / len(row["per_light_deltas"]))
                for row in report["comparisons"][arm]
            ]
            for a, b in zip(got["arms"][arm]["per_seed_mean_delta_db"], expected):
                self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nix develop --command uv run python -m unittest tests.test_rescore_checkpoints -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rescore_checkpoints'`

- [ ] **Step 3: Write minimal implementation**

Create `examples/rescore_checkpoints.py`:

```python
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
            psnr[(arm, seed)] = np.asarray(
                [r["psnr_db_vs_raw"] for r in rows], dtype=np.float64
            )
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `nix develop --command uv run python -m unittest tests.test_rescore_checkpoints -v`
Expected: PASS — the re-scored per-seed deltas equal the committed report's to 6
decimal places. (If `out/r1-parity-kitchen-eq/` is absent the test skips; that is
intended for CI, but this task is not done until it has been seen to PASS locally.)

- [ ] **Step 5: Commit**

```bash
git add examples/rescore_checkpoints.py tests/test_rescore_checkpoints.py
git commit -m "feat: re-score committed checkpoints at a different gate-light count"
```

---

### Task 6: Re-read R1 Kitchen parity at 96 lights

**Files:**
- Create: `out/r1-parity-kitchen-eq/rescore-96.json`
- Modify: `docs/performance.md` (new subsection under "K1 re-run under the
  equivalence gate"), `docs/representation-track.md` (R1 row + "R1 parity
  re-measurement" section)

No training. This is the first campaign re-read.

- [ ] **Step 1: Run the re-score**

```bash
nix develop --command uv run python examples/rescore_checkpoints.py \
  --run-dir out/r1-parity-kitchen-eq \
  --cache out/kitchen/path_cache.npz \
  --gate-lights 96 \
  --out out/r1-parity-kitchen-eq/rescore-96.json
```

Expected (measured 2026-08-29, must reproduce):

```
world_sparse             mean=-0.009 sd=0.631 light_sem=0.298 verdict=continue seeds_needed=17
world_normal_triplane    mean=-0.416 sd=0.556 light_sem=0.319 verdict=continue seeds_needed=14
world3d                  mean=-0.674 sd=0.803 light_sem=0.320 verdict=continue seeds_needed=25
```

- [ ] **Step 2: Confirm the 12-light control reproduces**

```bash
nix develop --command uv run python examples/rescore_checkpoints.py \
  --run-dir out/r1-parity-kitchen-eq \
  --cache out/kitchen/path_cache.npz \
  --gate-lights 12 \
  --out out/r1-parity-kitchen-eq/rescore-12.json
```

Expected: `-0.493 / -0.446 / -1.368`, `seeds_needed` `34 / 18 / 73` — bit-for-bit
the committed `report.json`. If this does not match, **stop**: the re-scorer is
wrong and every 96-light number is void.

- [ ] **Step 3: Write the docs section**

Add to `docs/performance.md` a subsection
`### Re-read at 96 held-out lights (2026-08-29)` under the K1 equivalence-gate
section, carrying the comparison table from this plan's "Measured evidence"
section verbatim, plus:

- that no checkpoint was retrained and the 12-light re-read reproduces the
  committed report exactly;
- that `world_sparse`'s published −0.493 dB is retracted **as a measurement of the
  arm** and stands only as a measurement made with a 12-light estimator; the same
  checkpoints read at 96 lights give −0.009 dB;
- that all three verdicts remain `continue` — **nothing is promoted by this
  re-read**, the arms simply moved from needing 18–73 seeds to needing 14–25;
- the cost asymmetry: 16 s/seed of extra evaluation versus ~548 s/seed of training.

Update the R1 row and the "R1 parity re-measurement" narrative in
`docs/representation-track.md` to cite `rescore-96.json` and to state that the
per-seed numbers in the pre-2026-08-29 tables were computed with a 12-light
estimator whose standard error exceeded the gate threshold.

- [ ] **Step 4: Verify**

Run: `uv run python -m unittest discover -s tests && uv run ruff check . && mise run pipeline-audit`
Expected: tests and lint pass. `pipeline-audit` has a known pre-existing failure
from purged E1–E9 report dirs — confirm the failure set is unchanged, do not fix it
here.

- [ ] **Step 5: Commit**

```bash
git add out/r1-parity-kitchen-eq/rescore-96.json out/r1-parity-kitchen-eq/rescore-12.json \
        docs/performance.md docs/representation-track.md
git commit -m "docs: re-read R1 Kitchen parity at 96 held-out lights"
```

---

### Task 7: Re-read the K1 sweep at 96 lights

**Files:**
- Create: `out/r1-kitchen-parity-k1-eq/rescore-96.json`
- Modify: `docs/performance.md`,
  `docs/plans/2026-08-27-kitchen-parity-next-steps.md` (the outcome banner)

The sweep's 40 checkpoints (5 resolutions × 8 seeds) are all on disk. Its per-setting
deltas are measured against the **fixed** `pixel2d` control from
`out/r1-parity-kitchen-eq`, so the control must be re-scored at the same 96 lights —
Task 6 already produced exactly that.

- [ ] **Step 1: Extend the re-scorer for the sweep's directory layout**

The sweep stores checkpoints as `train/finest<N>/seed<S>/model.pt` with no
per-directory `pixel2d` arm. Add to `examples/rescore_checkpoints.py`:

```python
def rescore_sweep(
    sweep_dir: Path,
    control_dir: Path,
    cache: PathCache,
    *,
    seeds,
    resolutions,
    control_arm: str,
    base_cfg: dict,
    n_gate_lights: int,
) -> dict:
    """Re-score a K1-style resolution sweep against a fixed control run.

    The sweep trains only the swept arm; its baseline is the `pixel2d` arm of a
    separate control run. Both are scored against the same per-seed held-out set, so
    the control's checkpoints are reloaded here rather than its numbers reused --
    reusing numbers scored at a different light count is the confound this whole
    change exists to remove.
    """
    device = torch.device("cpu")
    gate = EquivalenceGate()
    out = {"n_gate_lights": int(n_gate_lights), "seeds": list(seeds), "resolutions": {}}
    for seed in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = seed
        val_set = build_val_set(cache, cfg, n_gate_lights)
        control_path = control_dir / "train" / control_arm / f"seed{seed}" / "model.pt"
        model = load_trained_model(str(control_path), cache)
        spatial, aux = model_tensors(cache, model, device)
        control_psnr = np.asarray(
            [r["psnr_db_vs_raw"] for r in evaluate(model, val_set, spatial, aux, device)],
            dtype=np.float64,
        )
        del model
        for res in resolutions:
            path = sweep_dir / "train" / f"finest{res}" / f"seed{seed}" / "model.pt"
            model = load_trained_model(str(path), cache)
            spatial, aux = model_tensors(cache, model, device)
            arm_psnr = np.asarray(
                [r["psnr_db_vs_raw"] for r in evaluate(model, val_set, spatial, aux, device)],
                dtype=np.float64,
            )
            del model
            row = out["resolutions"].setdefault(str(res), {"per_seed": [], "per_light_var": []})
            delta = arm_psnr - control_psnr
            row["per_seed"].append(float(delta.mean()))
            row["per_light_var"].append(float(delta.var(ddof=1)))
    for res, row in out["resolutions"].items():
        row["mean_db"] = float(st.mean(row["per_seed"]))
        row["between_seed_sd_db"] = float(st.pstdev(row["per_seed"]))
        row["light_sem_db"] = math.sqrt(st.mean(row["per_light_var"]) / n_gate_lights)
        row["gate"] = gate.evaluate(row["per_seed"])
    return out
```

Add a `--sweep-dir` / `--control-dir` / `--resolutions` mode to `main` that calls it.

- [ ] **Step 2: Test it reproduces the committed sweep at 12 lights**

Add to `tests/test_rescore_checkpoints.py` a test mirroring Task 5's, comparing
`rescore_sweep(..., n_gate_lights=12)`'s `per_seed` values against the committed
`out/r1-kitchen-parity-k1-eq/report.json` per-resolution per-seed deltas
(`+0.24, −2.43, −0.44, +0.09, +0.11, +0.54, −0.41, −1.64` for resolution 128),
`assertAlmostEqual(..., places=6)`, skipping if the run directory is absent.

Run: `nix develop --command uv run python -m unittest tests.test_rescore_checkpoints -v`
Expected: PASS.

- [ ] **Step 3: Run the sweep re-score**

```bash
nix develop --command uv run python examples/rescore_checkpoints.py \
  --sweep-dir out/r1-kitchen-parity-k1-eq \
  --control-dir out/r1-parity-kitchen-eq \
  --cache out/kitchen/path_cache.npz \
  --resolutions 32 48 64 96 128 \
  --gate-lights 96 \
  --out out/r1-kitchen-parity-k1-eq/rescore-96.json
```

Record every resolution's mean, between-seed sd, light-sem, verdict, and
`seeds_needed`. Recompute the resolution-vs-mean-delta Spearman ρ and its two-sided
permutation p over the 120 orderings of 5 points, exactly as the committed report
does.

- [ ] **Step 4: Read the result against K1's own stop condition**

The K1 plan's stop condition is unchanged and must be applied as written: K2–K4 are
conditional on K1 **confirming** the predicted negative correlation between
`finest_resolution` and the parity delta.

- If the re-read shows a significant negative correlation → the vertex-support
  hypothesis is supported; K2 becomes runnable.
- If it shows a significant non-negative correlation → the hypothesis is falsified
  on evidence rather than on noise, and K2–K4 stay cancelled *for a reason*.
- If it is still undecided → report the new `seeds_needed` per setting and hand the
  seed-count decision to the user. **Do not** widen the threshold, drop a seed, or
  add an arm.

Whichever it is, write it down. Do not select the reading that unblocks the track.

- [ ] **Step 5: Update the docs and commit**

Replace the outcome banner at the top of
`docs/plans/2026-08-27-kitchen-parity-next-steps.md` with a third dated entry
recording the 96-light re-read, keeping the two existing entries intact — this repo
preserves superseded results rather than editing them away. Add the matching
`docs/performance.md` subsection with the full per-resolution table.

```bash
git add examples/rescore_checkpoints.py tests/test_rescore_checkpoints.py \
        out/r1-kitchen-parity-k1-eq/rescore-96.json \
        docs/performance.md docs/plans/2026-08-27-kitchen-parity-next-steps.md
git commit -m "docs: re-read the K1 sweep at 96 held-out lights"
```

---

### Task 8: Re-read the held-out-camera redesign campaign

**Files:**
- Create: `out/r1-encoding-redesign/rescore-96.json`
- Modify: `docs/performance.md`, `docs/representation-track.md`, `docs/tracks.md`

`examples/r1_encoding_redesign.py` scores its rows with `n_val_lights` of **4** in one
place and 12 in another, then reads a 1.0 dB comparative margin against ~1.9 dB of
row-to-row noise. That campaign closed R1 as "a characterized negative" and is what
blocks R2–R6. It is read with the same defective estimator and must be re-read
before that closure stands.

- [ ] **Step 1: Confirm its checkpoints are on disk**

```bash
find out/r1-encoding-redesign -name model.pt | wc -l
```

If zero, this task is **blocked, not skipped**: record that in the docs, state that
the redesign campaign's closure is unverified under the fixed estimator, and stop.
Do not retrain the campaign as part of this plan — that is a separate, much larger
decision for the user.

- [ ] **Step 2: Re-score at 96 lights**

Adapt the Task 5 entry point to the redesign run's directory layout (per-arm,
per-seed, per-rotation). Re-score every row, recompute G1 (1.0 dB comparative
margin), G3 (all 5 seeds), and the 15 dB absolute floor, and report how many of the
83 previously-failing rows still fail.

- [ ] **Step 3: Verify at the original count**

Re-score at the campaign's original light count and confirm the committed per-row
deltas reproduce. Same rule as Task 6 Step 2: if they do not, stop.

- [ ] **Step 4: Write the verdict**

State plainly whether the redesign's negative survives. If it does, R1 stays closed
and the finding is now robust to the estimator. If it does not, say so — and note
that this does **not** promote R1: it means the campaign must be re-read at the
gate's own schedule, and the user decides whether to re-run it.

- [ ] **Step 5: Commit**

```bash
git add out/r1-encoding-redesign/rescore-96.json examples/rescore_checkpoints.py \
        docs/performance.md docs/representation-track.md docs/tracks.md
git commit -m "docs: re-read the R1 encoding-redesign campaign at 96 held-out lights"
```

---

### Task 9: Decision point — hand the seed budget to the user

**Files:**
- Create: `docs/status/2026-08-29.md`

**Do not start new training without an explicit decision.** Tasks 1–8 add zero
training cost; everything past here spends CPU-hours.

- [ ] **Step 1: Write the status report**

`docs/status/2026-08-29.md` states, for each of R1 parity, the K1 sweep, and the
redesign campaign: the verdict at 96 lights, the `seeds_needed` figure, and the
CPU-hour cost of reaching the next scheduled look. Use the measured rate: one seed
is 4 arms × 137 s ≈ 9.2 CPU-minutes for a parity run.

From the measured R1 parity re-read, the concrete options are:

| target | next look | extra seeds | est. cost |
|---|---:|---:|---:|
| `world_sparse` decidable (needs 17) | n = 24 | 16 | ≈ 2.5 CPU-hours |
| `world_normal_triplane` (needs 14) | n = 16 | 8 | ≈ 1.2 CPU-hours |
| `world3d` (needs 25) | n = 32 | 24 | ≈ 3.7 CPU-hours |
| all three decidable | n = 32 | 24 | ≈ 3.7 CPU-hours |

- [ ] **Step 2: Ask the user which to fund, then stop**

Present the table and stop. The run at the next look is a separate piece of work
with its own pre-registration.

- [ ] **Step 3: Commit**

```bash
git add docs/status/2026-08-29.md
git commit -m "docs: 2026-08-29 status -- the gate is decidable at 14-25 seeds"
```

---

## Out of scope

- **`TorchNRP.load` cannot reload occupancy-allocated arms** outside the R1 runner,
  which independently blocks R5 (WebGPU export) and R6. It is an engineering defect,
  not a statistical one, and is unaffected by this plan. `load_trained_model` in
  `nrp/torch_backend/train.py:186` already contains the reconstruction recipe a
  fix would generalize. Worth a separate plan.
- **K2, K3, K4** stay cancelled until Task 7 reports. Their gating condition is
  K1 confirming the predicted direction, and this plan does not presume the answer.
- **Trimming, median, or robust estimators** for the per-light delta. Recorded as a
  measurement (4/96 lights below 10 dB control PSNR), not remediated, because it
  changes the estimand and is not the dominant variance term.
- **Any change to the gate rule, threshold, look schedule, or arm set.**
