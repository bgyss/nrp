# SDD progress — world-anchored encoding redesign

Plan: docs/superpowers/plans/2026-08-26-world-anchored-encoding-redesign.md
Spec: docs/superpowers/specs/2026-08-26-world-anchored-encoding-redesign-design.md
Branch: representation-encoding-redesign (base: codex/goal-implement-r2)

Pre-flight fixes applied before dispatch (approved by user):
- Task 4: replaced a tautological test assertion with a real one
- Task 3: shared _grid_capacity_report helper instead of mandated duplication
- Task 3/4: registry moved to nrp/torch_backend/encoder_registry.py to break the import cycle

## Tasks
Task 0: complete (commits 6dc31b6..ebc8b15, review clean)
Task 1: complete (commits ebc8b15..55ca5e5, review clean after 2 fix rounds)
  - authorized deviation: clamp change is a no-op, kept as invariant; spec corrected in c1f38a8
  - authorized deviation: wired spatial_encoding/world_bounds through train_streamed
  - _floor_cell extracted in encoding.py; both forwards call it; tests import it
Task 2: complete (commit d7f6a82, review clean)
  - kitchen founding measurement independently reproduced: 78,084 verts / 4096 slots = 0.0525
  - MINOR (deferred to final review): zero-vertex edge case in capacity_report is handled but untested
Task 3: complete (commit 967e1f7, review clean)
  - side-effect import verified load-bearing in a fresh interpreter
  - SUPPORTED_SPATIAL_ENCODINGS fully migrated (model.py + train.py), no stale set
  - MINOR (plan-mandated): test_occupancy_encoder_without_occupancy_raises passes vacuously
    until an encoder sets needs_occupancy=True. MUST be shown red-then-green in Task 4.
NOTE: `ruff format .` reformats ~10 files inherited non-conformant from main/codex branch.
  Dispatches now instruct formatting ONLY touched paths. stash@{0} holds one such drift
  capture from the Task 4 fix round (pre-existing churn, safe to drop; left for the user).
Task 4: complete (commits 6fa062c..22ddfcc, review clean after 1 fix round)
  - zero-collision claim verified from code: key packing injective over [0,res], side=res+1
  - capacity_report() now MEASURES collisions from keys buffer (was hard-coded 0.0 literal;
    the G3 gate in Task 8 asserts that field, so it would have verified a constant)
  - needs_occupancy flag shown red-then-green; buffer round-trip now discriminating
  - base_resolution/finest_resolution now validated against supplied occupancy
Task 5: complete (commit a31120c, review clean, no fix round)
  - implementer self-caught that brief's constant-filled tables couldn't detect a wrong
    AXIS_TO_PLANE coordinate pair; added a discriminating test, shown red-then-green
  - world_transform interaction verified: normals rotated by world_basis (model.py:264-265)
  - MINOR (deferred): gradient test doesn't assert ZERO grad on non-selected planes
Task 6: complete (commits 3a5e1cc..18ab347, review clean after 1 fix round)
  - NOTE: Task 6 first review was done inline by the controller (subagent hit a session limit)
  - arm A validates occupancy schedule BEFORE truncation; now covered by a small-budget test
  - schedule formula de-duplicated: both grid constructors call occupancy.level_resolutions;
    verified character-for-character identical to the removed inline copies
Task 7: complete (commit 0bcc818, review clean, no fix round)
  - BRIEF ERROR caught by implementer: my cross-product order gave right=[-1,0,0], a
    MIRRORED basis. Would have produced a mirrored camera arc and a false negative for
    the whole experiment. Corrected to right=cross(UP,fwd), up=cross(fwd,right).
  - fixed pre-existing layer_ownership_mask bug (never accepted camera args)
  - MINOR (deferred): test_default_is_bit_identical_... name promises more than its own
    assertions; real guarantee is in the adjacent reproduces-default test

CARRY-FORWARD REQUIREMENT FOR TASK 11 (runner), raised by the Task 8 reviewer:
  The runner must NOT treat g3["passed"] or g5["complete"] as standalone gating signals.
  It must also surface g3["collision_assertions_checked"], g4["coverage_complete"],
  g1["coverage_complete"], and the row counts, and must pass expected_seeds/expected_cameras
  to g1 so coverage is actually verified rather than assumed.
Task 8: complete (commits 0bcc818..f5564b5, review clean after THREE fix rounds)
  - the unearned-pass defect was introduced 3x, twice BY fixes for it:
    r1 g3 collision all() vacuous; r2 g3 passed not wired + g5 empty + g4 pooled coverage;
    r3 g1 coverage all() vacuous on empty/partial args. Now fail-closed w/ coverage_status.
  - final review built its own empty-collection audit table from source; no 4th instance
  - MINOR (deferred): stop_reason raises AttributeError on a non-dict arm value (fail-loud,
    not an unearned pass) -- one-line isinstance guard suggested
Task 9: complete (commits f5564b5..d458ee0, review clean after 1 fix round)
  - all three hard-coded "world3d" sites generalized; occupancy spans UNION of all views
  - FOUND: occupancy builder's default finest_resolution=128 vs HashEncoding3D's 256 meant
    arm A could not train with a default config. Root cause: nothing exercised
    train_conditioned end-to-end with an occupancy arm. Fixed via encoder_schedule_params
    (reads class __init__ defaults); literal fallbacks removed; e2e regression added.
Task 10: complete (commit c0e455f, review clean, no fix round)
  - each of 5 tests proven capable of failing via deliberate breakage
  - implementer strengthened the checkpoint test against the REAL structure
    (dict with exactly config+state_dict) instead of the brief's guess
  - KNOWN LIMITATION: corner-exactness uses enc._index() as its own oracle, so a bug
    inside _index's formula is masked there. CORRECTED CLAIM (final whole-branch
    review, fix 10): test_hash_matches_reference_at_high_resolution does NOT check
    _index against the instant-ngp reference independently -- it re-types the
    production expression and imports _PRIMES from production, so a wrong prime or
    a changed hash design would pass it unchanged. It DOES genuinely discriminate
    the '&'/'^' precedence defect, which is why it earns its place.

DECISION (2026-08-26): G1 gains an absolute 15 dB PSNR floor alongside the 1.0 dB margin.
  Rationale + provenance in the spec (commit 7a58e98). 15 dB is INVENTED for this
  campaign; the 1.0 dB margin is inherited from the approved R2 ladder. Task 11 must
  pass absolute_floor_db=15.0 to g1_generalization.
G1 absolute floor: complete (commit 5e9a242, review clean, no fix round)
  - reviewer explicitly hunted a 4th unearned-pass instance and found none, by own
    adversarial execution. Conditions conjunctive; both reasons recorded when both fail;
    absolute_floor_db=None byte-identical to pre-change.
Task 11: complete (commits 6775528, ef545d4, review clean, no fix round)
  - all 5 binding requirements verified in calling code; smoke report showed
    g1.absolute_floor_db=15.0 and g1.coverage_status="complete" (coverage path live)
  - rotated_camera rotates camera AND geometry with the same convention as
    transform_cache -> tests frame-change, not viewpoint-change. Correct for G4.
  - PRODUCTION LIMITATION (recorded, not fixed): TorchNRP.load cannot reload
    occupancy-allocated arms; occupancy is not in the checkpoint and is needed to size
    tables before load_state_dict. The runner rebuilds it from config world_bounds +
    encoding dict. Other callers (relight, bench, webgpu export) still cannot. Affects
    later rungs R5/R6.
  - MINOR (deferred): _predict duplicates relight's loop and drops its TexturedQuadLight
    case; suggested fix is adding view_dir to relight and deleting _predict.

PROBE (seed 0, rotation 0, all 3 arms) -- out/r1-encoding-redesign/probe.json
  timing: 17.5 min per (seed,rotation) combo -> full 15-combo campaign ~= 4.4 h
  gate wiring VERIFIED live: G4 coverage_complete=False on a single rotation;
    G1 coverage_status="complete"; G3 collision_assertions_checked=True with
    world_sparse measured at exactly 0.0; stop_reason non-null; report written
    before the nonzero exit.
  first real signal (ONE seed -- not conclusive):
    world_sparse (arm B, primary): worst delta -0.52 dB  FAIL, out-of-occupancy 6.1%
    world_normal_triplane (arm C): worst delta +3.09 dB  pass, oof 0%
    world3d occupancy-alloc (A):   worst delta +1.73 dB  pass, oof 0%
    held-out PSNR ~20.4 dB, so the 15 dB floor is not the binding constraint.
  NOTE: arm B -- the hypothesis I argued was strongest -- is the worst here, and G5
    immediately shows a plausible mechanism (6.1% of held-out queries fall outside the
    occupied set and get zero features at the finest level). DO NOT tune arm B in
    response to one probe seed; run the campaign as specified and let the 5-seed
    matrix decide. Occupancy dilation is future work, not a mid-campaign fix.
  MINOR (reporting): collision_by_arm dict is passed whole to every arm's g3, so a
    non-sparse arm's report echoes {"world_sparse": 0.0} while correctly reporting
    collision_assertions_checked=False. Noisy, not wrong.

CAMPAIGN RUN 1 ABORTED at 18:10 (~40 min in) -- CONFOUND, not a crash.
  ARM_ENCODING_CONFIG used empty dicts, so each arm fell back to its own encoder
  CLASS defaults, which differ:
    world_sparse          levels 8, finest 128, 108,057 params
    world_normal_triplane levels 3, finest 255, 158,717 params
    world3d               levels 8, finest 255, 330,567 params
  Arm B (the probe's loser) was running at HALF the finest resolution and a THIRD the
  parameters of arm A, so "sparse loses to hash" was inseparable from schedule and
  capacity. Probe result is therefore NOT interpretable as a representation finding.
  Cause: implementer replaced the plan's explicit per-arm configs with {} (good DRY
  instinct, silently changed the experiment); review approved it as avoiding duplicated
  literals. Neither was wrong locally -- the design intent lived only in my plan text.
  RESOLUTION: the spec forbids matching PARAMETER budgets (that starved the original
  world3d). It does NOT license differing RESOLUTION SCHEDULES. Corrected to a common
  ladder (base 4, finest 64 == the 64^2 render resolution, ~one vertex per pixel, the
  same relationship pixel2d has), capacity left unequal and reported by G2.

CAMPAIGN RUN 2 COMPLETE but INVALID -- evaluation lights were never rotated.
  Report preserved as out/r1-encoding-redesign/report-INVALID-unrotated-lights.json
  main() computes eval_lights ONCE per seed from the UNROTATED cache (~line 461) and
  reuses it for every rotation (~line 543). transform_cache rotates geometry and
  rotated_camera rotates the camera, but the light stays at its original world position
  -> at 90/180 deg the light sits wrong relative to the rotated scene. That is a
  DIFFERENT PHYSICAL SETUP, not a change of coordinate frame, so G4 measured nothing
  meaningful and G1 was contaminated (it pools all rotations).
  Signature: delta_db by rotation -- 0 deg mean +2.00 min -0.57 (healthy);
    90 deg min -7.14; 180 deg mean -1.31 min -18.21 max +33.59.
    Worst rows show psnr=52 dB vs baseline=70 dB, i.e. near-BLACK references where
    trivial differences produce huge dB swings.
  The 0-degree rows are the only valid data in run 2 and they look encouraging:
    world_sparse at 0 deg = mean +2.00 dB, worst -0.57 dB across 20 rows.
  SAME CLASS as the mirrored camera basis (task 7): an incomplete transform producing a
  plausible-looking experiment that measures the wrong thing. Neither crashed.
Light-rotation fix: commit 80d07a6, review clean.
  Reviewer built an independent world-coordinate enumeration: seg_origin, seg_dir,
  position, normal all rotated; seg_tmax/albedo/depth/radius/rgb provably invariant;
  world_bounds + occupancy rebuilt per rotation from already-rotated caches; light
  center rotated (the fix); non-SphereLight raises TypeError instead of passing through.
  campaign_peak proven bit-identical under rotation (gather_lights uses only relative
  geometry), so computing it once at rotation 0 is correct, not a shortcut.
  The buggy case is pinned as a permanent regression test.
Task 12: complete (commit 7a69096) -- CHARACTERIZED NEGATIVE, no arm promoted.
  Run 3 valid. All arms beat baseline on average (+1.88/+1.40/+1.41 dB) but G1 requires
  >=1.0 dB on every one of 60 rows; 24-30 fell short. std ~1.9 dB vs a 1.0 dB per-row bar.
  world_sparse strongest and uniquely frame-robust (+2.00/+1.83/+1.80 across rotations);
  zero collisions measured; G5 shows fallback is NOT the limiter (out-occ 24.44 dB >
  in-occ 23.46 dB), so the shortfall is uniform.
  Zero rows failed the 15 dB floor -> outcome driven purely by the comparative margin.
  pipeline-audit: only the pre-existing bedroom_cache.npz failure (verified by stashing).

DEFERRED MINORS for final review triage:
  1. capacity_report zero-vertex edge case untested (task 2)
  2. arm C gradient test doesn't assert ZERO grad on non-selected planes (task 5)
  3. test_default_is_bit_identical_... name overclaims vs its own assertions (task 7)
  4. stop_reason raises AttributeError on a non-dict arm value (task 8)
  5. corner-exactness uses _index as its own oracle; closed by task 1's reference test (task 10)
  6. _predict duplicates relight's loop and drops TexturedQuadLight (task 11)
  7. TorchNRP.load cannot reload occupancy-allocated arms -- PRODUCTION LIMITATION,
     affects R5/R6 (task 11)
  8. collision_by_arm echoes world_sparse into non-sparse arms' reports (cosmetic)
  9. parameter_count absent from report.json; taken from run.log (task 12)

FINAL WHOLE-BRANCH REVIEW (opus): "ready with fixes". Found a 14th unearned-pass
  instance -- _sparse_collision_fraction returned 0.0 (the "verified zero collisions"
  value) for an empty/malformed capacity report, feeding the one field the spec requires
  arm B to ASSERT. It was the only untested function in the runner.
  Also: registry broken in a cold interpreter (import side-effect only); zero-collision
  contract keyed off the literal "world_sparse" in two files (fail-open by rename);
  docs labelled a linear radiance peak as "peak PSNR ... dB"; arm C's aux[:,4:7] slice
  untested despite 60 published rows resting on it.
  All 10 fixes applied (39dd9c6, af6cacd, 107969e). 528 tests.
FINAL RE-REVIEW: merge READY. No 15th instance; every new default checked for
  direction, guarantees_zero_collisions defaults to the safe value. Doc figures
  (83/180 failing rows, peaks 6.82-12.27) independently recomputed from report.json.
  One Minor left: test_every_encoder_declares_the_interface doesn't assert the new
  guarantees_zero_collisions flag is declared (safe default, completeness only).

# SDD progress — equivalence gate (2026-08-28)

Plan: docs/superpowers/plans/2026-08-28-equivalence-gate.md
Spec: docs/superpowers/specs/2026-08-28-equivalence-gate-design.md
Branch: k1-kitchen-vertex-support (base for this plan: 8827a1c)

Context: K1 falsified the Kitchen vertex-support hypothesis; chasing its ~1.5 dB
run-to-run noise found OIDN nondeterminism (fixed, d715f74); the deterministic
Kitchen re-measurement (b426f26) then exposed that the -0.5 dB per-seed gate
rejects an at-parity arm 76-91% of the time and gets WORSE with more seeds.
This plan replaces the rule's structure.

Pre-flight fixes applied before dispatch (commit 8827a1c):
- test constructed EquivalenceGate(looks=(1,8)) which the constructor rejects;
  construction sat outside assertRaises so the test would ERROR. Split in two.
- two simulation thresholds sat within 3 Monte Carlo se of measured rates at 600
  trials (pass>=0.75 @ std 1.00; fail>=0.90 @ std 1.67). Loosened to 0.70/0.85.

## Tasks
Task 1: dispatched (base 8827a1c) — Student-t quantile, no scipy
Task 1: complete (commits 8827a1c..5c7447c, review clean after 1 fix round)
  - reviewer verified numerics independently (Simpson integration + closed-form
    Cauchy/df=2 CDFs), not just by re-running the implementer's tests
  - FIXED (Important): t_ppf bisected on a hard-coded [-1e3, 1e3] bracket and
    SILENTLY returned the clamped endpoint outside it -- t_ppf(0.999999, 1) gave
    1000.0 instead of 318310. Now expands geometrically and raises ValueError
    naming p/df; can never return a clamped endpoint. Pinned by a Cauchy
    closed-form test shown red (1000.0 != 318309.886) then green.
  - not a live defect (gate uses p~0.9958, df 7-47) but the module underpins a
    gate whose principle is refusing unsupported verdicts; silent wrong numbers
    are the exact failure mode it exists to prevent
  - re-reviewer confirmed the 200-doubling cap is unreachable for valid (p, df):
    p collapsing to 0.0/1.0 is caught earlier by input validation
  - note: subagents run outside direnv, so oidn tests skip -> suite shows 7 skips
    where the direnv shell shows 4. Not a regression.
Task 2: dispatched (base 5c7447c) — gate rule, verdicts, look schedule
Task 2: complete (commits 5c7447c..95f1513, review clean after 1 fix round)
  - reviewer re-derived the interval arithmetic independently for pass/fail/
    continue/cap cases; hunted unearned passes explicitly and found NONE
    (NaN/Inf/empty/single-seed all raise or degrade to continue, never pass)
  - FIXED (Important, PLAN-MANDATED defect -- the brief's own code): evaluate()
    called seeds_needed unconditionally, and seeds_needed was a linear n+=1 scan
    recomputing t_ppf each step. On a realistic outlier ([0.0]*7+[1000.0],
    std~354) it stalled >2 min and then RAISED at the n>100000 guard, turning a
    valid verdict into an exception. Now exponential bracket + binary search:
    same integer as the old scan (verified over 27 (std, half_width) pairs by an
    independent brute force), 0.15 s, returns 3480205 as a number.
  - monotonicity of half_width_at(n) -- the binary search's invariant -- checked
    numerically at small n where the t quantile moves fastest
  - ACCEPTED DEVIATION: evaluate() returns an extra alpha_overall key beyond the
    brief's 15. Kept deliberately: it makes the Bonferroni correction auditable
    from a report alone. Do not re-flag at final review.
Task 3: dispatched (base 95f1513) — power simulations
Task 3: complete (commit 29fd169, review clean, no fix round)
  - implementer's underpowered->pass mutation caught only 2 of 6 tests, so the
    reviewer ran its OWN mutations: dropped Bonferroni (caught), ddof=0 (caught),
    cap-branch pass (caught), boundary >= to > (uncaught, measure-zero on
    continuous data -- expected). Every behavior-changing mutation is caught.
  - the underpowered-at-cap test does the real discriminating work; the other
    tests cover distinct properties (parity promotion, worse-arm rejection,
    legacy-rule regression) so their silence is correct, not laxity
  - all four numeric bounds sit 4-8 standard errors from their measured values
    at 600 trials; not flake-prone
  - MINOR (deferred to final review): second half of
    test_legacy_rule_gets_worse_with_more_seeds_and_the_gate_does_not is
    decorative -- both arms saturate to pass=1.0 under mutation, so the
    non-strict >= holds trivially. Property is owned by the refuses-to-certify
    test. Consider deleting that half.
Task 4: dispatched (base 29fd169) — r1_parity integration
Task 4: complete (commit e1afe81, review clean, no fix round)
  - BRIEF ERROR caught by implementer: arm_gate_verdict called gate.evaluate()
    unconditionally, which raises on off-schedule seed counts -- crashing the
    pre-existing 3- and 5-seed tests under binding="per_seed". Fixed by computing
    the equivalence sub-verdict only at scheduled looks.
  - reviewer traced that resolution for unearned passes: equivalence is None only
    under binding="per_seed", where decisive comes from legacy["pass"] alone, so
    a missing verdict can never read as a pass. Off-schedule under
    binding="equivalence" raises loudly. No 15th unearned-pass instance.
  - early stop gated on len(trained_seeds) in gate.looks -- cannot fire mid-look;
    seeds_run (not args.seeds) drives the report and the reproduction command
  - pre-existing tests relocated to verdict["per_seed"][...] with nothing weakened
  - MINOR (deferred): plan_seed_batches silently caps above 48 rather than erroring
  - MINOR (deferred): GATE_DELTA_DB now dead in r1_parity except for two tests
Task 5: dispatched (base e1afe81) — docs
Task 5: complete (commits e1afe81..27d195d, docs only)
  - CONTROLLER ERROR: my verification step told the implementer to grep for the
    literal anchor string, so it rewrote the heading as
    "## The equivalence-gate-from-2026-08-28" to satisfy the grep. Markdown
    already slugifies "The equivalence gate (from 2026-08-28)" to that anchor,
    so the readable heading was correct all along. Restored in 27d195d.
    Lesson: verify a LINK RESOLVES, don't grep for a slug in prose.

FINAL WHOLE-BRANCH REVIEW (opus): NOT READY -> fixed -> READY WITH FIXES -> closed.
  CRITICAL 1: changing arm_gate_verdict's default binding broke r1_kitchen_k1's
    run_sweep (5 seeds is not a scheduled look) -- the documented K1 command would
    have crashed AFTER a ~2 h sweep. Bound K1 to per_seed; verified by replay that
    no published K1 number or verdict changed.
  CRITICAL 2: r1_parity's recorded reproduce command omitted --denoise-method, so
    replaying the deterministic Kitchen measurement silently used bilateral.
    Now records all 11 args, with a test asserting it.
  IMPORTANT: explicit --seeds under the equivalence default now fails fast before
    training; check_control_compatibility now raises on pool/lr/batch_pixels/model/
    sampling/light_type/light_bounds/n_val_lights mismatch (iters stays a warning).
  IMPORTANT (my over-claim): docs said K1 "falsified" the hypothesis on a Spearman
    over 5 points (rho=+0.20, p~0.75) against ~1 dB noise vs a 0.5 dB effect.
    Restated as "not supported, and underpowered"; n=5 bootstrap CI labelled
    descriptive. Factual results and the K2-K4 cancellation kept.
  RE-REVIEW found the CRITICAL 1 regression test was a hand-copied mirror of the
    call site: reverting the buggy line left the suite green 32/32. Replaced with a
    run_sweep test proven to fail (33 passed/1 error) when binding="per_seed" is
    removed and pass (34/34) when restored.
  out/ artifacts deliberately NOT edited -- their incomplete command field is
    historical fact; the correction lives in docs naming the authoritative command.
  Final: 662 tests OK (7 skips), ruff clean, 21 commits.

## K1 re-run under the equivalence gate (2026-08-28, follow-on)

User request: "re-run K1 under the new gate". Decisions taken with the user:
extend the control to the gate's first look rather than splice a control-only
extension; stop at that first look (n=8) rather than follow the schedule upward.

- examples/r1_kitchen_k1.py: added --gate {equivalence,per-seed}; default is
  equivalence, DEFAULT_SEEDS is now the first look (0-7), DEFAULT_CONTROL_REPORT
  points at the 8-seed deterministic control. run_sweep takes binding/gate_rule
  and calls check_seed_binding_compatibility BEFORE the first training (the old
  code would have raised after the last one). Report gained gate_rule,
  gate_schedule, and a per-resolution gate_verdict word so continue/underpowered
  cannot be read as a fail; console label prints the verdict, not PASS/FAIL.
- tests: the run_sweep fixture became a module-level helper shared by the
  per-seed and equivalence binding tests; new tests cover the 8-seed equivalence
  binding, the pre-training refusal at an off-schedule seed count (asserting no
  training ran), and that the default seed count is a scheduled look. 665 OK.
- control: out/r1-parity-kitchen-eq (4 arms x 8 seeds, oidn, ~1.6 h). All three
  world arms `continue`; means -0.45/-0.49/-1.37 dB.
- sweep: out/r1-kitchen-parity-k1-eq (5 resolutions x 8 seeds, ~3 h). Every
  resolution `continue`; rho = -0.30 (permutation p = 0.68), not monotonic.
  K2-K4 stay unrun -- the stop condition needs a CONFIRMED direction.
- free determinism check: the sweep's finest128 row is bit-identical per seed to
  the control run's world_sparse arm from a separate process, so the earlier
  ~1.5 dB run-to-run variation is resolved by the single-threaded OIDN fix.
- docs/performance.md: new section at the end; the old K1 section is marked
  superseded with its per-seed numbers retracted as measurements (its conclusion
  survives). docs/plans/2026-08-27-kitchen-parity-next-steps.md banner updated.

# SDD progress — held-out-light estimator (2026-08-29)

Plan: docs/superpowers/plans/2026-08-29-heldout-light-estimator.md
Branch: heldout-light-estimator (base: b043423, off main ddec973)

Context: K1 is "undecided at 8 seeds" because the gate's per-seed value is a mean
over only 12 held-out lights, whose per-light delta sd on Kitchen is ~3.3 dB ->
+/-0.94 dB standard error against a -0.5 dB threshold. Re-scoring the SAME committed
checkpoints at 96 lights (16 s/seed vs ~548 s/seed to train one) moves world_sparse
from -0.493 to -0.009 dB and drops seeds_needed 34->17 / 18->14 / 73->25. Tasks 1-8
add zero training cost.

Pre-flight fix (decided with the user before dispatch):
- Task 4 would have added n_gate_lights to _STRICT_TRAINING_CONFIG_KEYS with a bare
  .get(), making every committed control (which predates the key) read None vs 96 and
  raise -- breaking the documented K1 command and Task 7. Resolution: absence
  normalizes to 12 on both sides, via _LEGACY_TRAINING_CONFIG_DEFAULTS. Plan amended
  in b043423 with two extra tests pinning both directions.

## Tasks
Task 1: complete (commit a6393c0, review clean, no fix round)
  - build_val_set(cache, cfg, n_lights=None); None path byte-identical to before
  - superset property mutation-tested BY THE REVIEWER: reseeding the RNG per count
    makes the new test fail immediately, so it genuinely discriminates
  - BRIEF ERROR caught by implementer: assertEqual(a["light"], b["light"]) can never
    pass -- SphereLight is a plain @dataclass with ndarray fields, so its generated
    __eq__ raises "ambiguous truth value" for ANY two instances, equal or not
    (pre-existing defect in nrp/lights.py, unrelated). Substituted .to_dict() compare.
  - MINOR (deferred, plan-mandated): brief named tests/test_torch_train.py as the
    target, but tests/test_torch_backend.py is the established home for train.py unit
    tests and already contains the identical trace_path_cache(5,4,2,1,seed=9) fixture
    the new _tiny_cache() duplicates. Splits train.py coverage across two modules.
    Consolidation candidate for final review.
Task 2: complete (commit 5ca15aa, review clean, no fix round)
  - build_frozen_validation_sets(..., n_gate_lights=None) threads into build_val_set's
    3rd arg; all three call sites (r1_parity:455, r1a_variance:385, r1_kitchen_k1:580)
    pass 3 positional args and are unaffected
  - reviewer mutation-tested: ignoring n_gate_lights FAILS the test (the real
    regression risk -- a runner wiring the key and it silently doing nothing)
  - MINOR (observation only): a behaviorally-equivalent implementation that sets
    cfg["n_val_lights"] before calling build_val_set also passes. Same observable
    contract, so not a gap.
  - load_r1a_variance mirrors the established load_runner() importlib pattern;
    _small_cache() reuses trace_path_cache(...) rather than inventing a fixture style
Task 3: complete (commits dd93470..67b3e26, review clean after 1 fix round)
  - BASE_TRAIN_CONFIG["n_gate_lights"]=96 pre-registered; --gate-lights flag; recorded
    in reproduce_command AND report["gate_lights"] + training_config
  - FIXED (Important): the actual wiring line was UNTESTED -- reviewer's own mutation
    (drop n_gate_lights= from the call site in run_experiment) left all 43 tests green.
    Same defect class as the prior campaign's hand-copied mirror test.
    Fix 67b3e26 adds RunExperimentGateLightsTest, which drives the REAL run_experiment
    with train/load_trained_model/evaluate_model/build_frozen_validation_sets patched on
    the module namespace, so the production call site executes.
  - re-reviewer reproduced red/green ITSELF: mutation -> 1 failure ([None] != [20]);
    restore -> 44 OK; git diff clean. Assertion discriminating: n_val_lights=12 vs
    n_gate_lights=20 vs default 96 all distinct, so a wrong-key swap yields [12] and
    still fails. Second assertion checks the value survives into report["validation_specs"].
  - production code unchanged by the fix (git diff dd93470..67b3e26 -- ':!tests/' empty)
Task 4: complete (commit 6d37eef, review clean, no fix round)
  - n_gate_lights added to _STRICT_TRAINING_CONFIG_KEYS + _LEGACY_TRAINING_CONFIG_DEFAULTS
    ({"n_gate_lights": 12}); every OTHER strict key still falls back to None (verified)
  - --gate-lights flag, threaded into base_cfg, the call site, and the reproduce command
  - DEVIATION 1 (justified, verified by reviewer against commit 67b3e26): main() passed
    the STATIC BASE_TRAIN_CONFIG to check_control_compatibility, so the new guard was a
    no-op for any non-default --gate-lights -- it would have falsely rejected a legacy
    12-light control run with --gate-lights 12. Now built from args.gate_lights, matching
    how --iters and --denoise-method are already checked.
  - DEVIATION 2 (justified, verified): run_sweep_with_fakes structurally CANNOT reach this
    module's call site -- build_frozen_validation_sets is called in main() (line 606), not
    in run_sweep (which receives validation_sets already built). r1_parity differs. Test
    drives main() directly instead.
  - reviewer ran 3 mutations itself, all caught: (a) drop n_gate_lights= at call site ->
    [None] != [20]; (b) revert legacy normalization -> legacy-control test fails; (c) drop
    the key from _STRICT_TRAINING_CONFIG_KEYS -> both mismatch tests fail.
  - 676 tests OK (7 expected optional-dep skips), ruff clean
  - MINOR (deferred): main() builds run_training_config (shallow) and base_cfg (deep) from
    BASE_TRAIN_CONFIG separately, each overriding n_gate_lights. Deliberate -- the
    compatibility check must fail fast before the expensive cache load.
Task 5: complete (commit 99f6613, review clean, no fix round)
  - examples/rescore_checkpoints.py: evaluation-only re-read of a completed
    r1_parity-schema run at any gate-light count. Trains nothing.
  - REPRODUCTION TEST RAN FOR REAL (not skipped), 27.6-29.1s over 3 runs, reviewer
    re-ran it itself: 32 checkpoints re-scored at n_gate_lights=12 match the committed
    out/r1-parity-kitchen-eq/report.json to 6 decimal places. This is the correctness
    basis for every downstream re-read.
  - reviewer mutations, all caught: (a) sign swap control-arm -> deltas negated;
    (b) TorchNRP.load instead of load_trained_model -> ValueError "world_sparse requires
    occupancy" (confirms the occupancy claim is real, not decorative); (c) n_gate_lights+1
    -> deltas shift ~0.028 dB.
  - delta orientation verified against r1_parity's pair_validation_metrics (candidate -
    control), so a world arm beating pixel2d is POSITIVE
  - variance axes verified NOT transposed: light_sem_db = sqrt(mean_over_seeds(
    var_ddof1(per-light deltas)) / n_gate_lights); between_seed_sd_db = pstdev(per-seed
    means). EquivalenceGate.evaluate called with per-seed MEANS.
  - no rescore_sweep (correctly deferred to Task 7)
  - MINOR (brief inconsistency, fix for Task 7's brief): brief's prose Interfaces list
    mentions pair_validation_metrics but its own code block inlines the equivalent.
Task 6: complete (commit 4238f6a, review clean, no fix round)
  - out/r1-parity-kitchen-eq/rescore-96.json + rescore-12.json; NO training, no committed
    report.json modified (reviewer checked --name-only)
  - reviewer INDEPENDENTLY RE-RAN both re-scores: 96-light -0.009/-0.416/-0.674 with
    seeds_needed 17/14/25, byte-for-byte equal to the committed rescore-96.json; 12-light
    -0.493/-0.446/-1.368 with 34/18/73, bit-for-bit against the historical report.json's
    own gate fields
  - every doc number recomputed from the JSON by the reviewer, not just read
  - scoping verified: both doc sections state all three verdicts remain `continue`,
    nothing is promoted, and the -0.493 -> -0.009 move is scoped as "retracted as a
    measurement of the arm", NOT as world_sparse being vindicated
  - additive-only diffs on both markdown files; new anchor resolves
  - pipeline-audit failure set unchanged (pre-existing bedroom_cache.npz)
  - MINOR (deferred, controller-caused): the brief sent the new paragraph to the
    "R1 parity re-measurement" section, which is about the older per-seed-gate toy/Kitchen
    comparison rather than the equivalence-gate material. Placement worth a second look at
    final review.
Task 7: complete (commits 51bc02a..de8cca6, review clean after 2 fix rounds)
  THE HEADLINE RESULT. 12-light reproduction PASSED at 6dp (reviewer re-ran it: 2 tests,
  82.6s, both ok not skipped). At 96 lights every K1 mean collapses toward zero:
    res:  32      48      64      96      128
    @12: -0.409  -0.248  -0.580  -0.381  -0.493
    @96: -0.025  +0.036  +0.021  +0.138  -0.009   seeds_needed 13/16/9/19/17 (was 46/22/19/47/34)
  Spearman flips +0.30 (p=0.683) from -0.30 at 12 lights ON THE IDENTICAL CHECKPOINTS --
  evidence the correlation is 8-seed noise, not a trend. K1 REMAINS UNDECIDED; K2-K4 stay
  cancelled per the unchanged stop condition. finest128 row bit-identical to Task 6's
  world_sparse re-read (independent-path cross-check).
  - CRITICAL (fixed, 0941e93): finest64 newly reads `pass` and the docs presented it as a
    result. Reviewer established it passes on the SMALLEST SEED SD (0.402 vs 0.560-0.735),
    NOT the best mean -- its mean (+0.021) is the MEDIAN of five; finest96 (+0.138) is best.
    Under a pooled sd (0.613) NO setting passes and finest64 isn't even closest (finest96 is).
    Margin 0.0042 dB; flips to `continue` on a 0.82% sd rise; sd's own spread at n=8 is 27%.
    Both docs now open with a bolded "not evidence that finest_resolution=64 is at parity",
    matching the register Task 6 set, and name the multiplicity explicitly.
  - IMPORTANT (fixed): seeds_needed was described as "more seeds"; it is a TOTAL (finest64=9
    is ONE more than the 8 run) and measures each setting's own interval, not the seeds to
    resolve a rank correlation.
  - IMPORTANT (fixed, de8cca6 -- MY error, propagated from my fix prompt): family-wise rate
    mixed framings, citing 1-(1-0.0042)^30 = 11-12% while saying "at one look each"
    (exponent 5 = 2.07%). All five settings have looks_taken=1, so the REALIZED rate is
    ~2.1%; ~11-12% is the design ceiling if every arm ran all six looks. Both now stated
    separately and labelled.
  - 3 MINORs fixed: "same magnitude" compared 0.20 to 0.30; table sd column is ddof=0 while
    the CI uses ddof=1 std_db (footnoted); tension with the pre-registered falsifier string
    (unqualified by significance) now acknowledged with the conservative reading justified.
  - fix authors twice corrected numbers I handed them rather than copying (0.524 was
    finest32's POPULATION sd, not a sample sd; flip threshold 0.82% not 0.9%)
Task 8: complete (commit 1af76c9, review clean, no fix round)
  RECOVERY: the Opus implementer hit an API session rate limit and was terminated
  mid-task -- no commit, no task-8-report.md. Its uncommitted work was found intact
  (rescore-96.json 332KB + code + tests + 3 docs). Controller verified the reproduction
  test, full suite (680 OK, 4 skips) and ruff, then committed it. The task reviewer was
  therefore the FIRST real verification; briefed accordingly.
  - BRIEF ERROR corrected by the implementer: I said the campaign used n_val_lights 4 and
    12. The real per-row gate count is N_EVAL_LIGHTS = 8; the 4 and 12 are training-time
    validation sizes in unrelated config dicts that never reach a gate row. Reviewer
    confirmed the correction.
  - 96 lights built as TWELVE independent draws of the campaign's own 8-light
    configuration, so the estimand is unchanged; group 0 is bit-identical to the committed
    draw (frozen_lights draws one at a time from default_rng([seed, 0xE1C0DE])).
  - rescore_encoding_redesign IMPORTS the campaign's own camera arc, cache/camera/light
    rotation, peak, model reload and gate functions rather than reimplementing them;
    reviewer checked function by function. Only the outer loop and grouping are new.
  - RUN-2 BUG GUARD PROVEN: reviewer edited rotated_lights to return lights unrotated
    (reproducing the exact bug that invalidated campaign run 2) and the reproduction test
    FAILED HARD (PSNR 47.67 vs expected 25.56, 22 dB off). Restored -> passes.
  - reviewer independently RE-RAN the re-scorer at --gate-lights 8 and diffed all 180 rows
    against the committed report.json: 0 mismatches on psnr_db, baseline_psnr_db, delta_db,
    both occupancy-split PSNRs, out_of_occupancy_fraction, baseline_camera; stop_reason
    byte-identical.
  RESULT: the negative SURVIVES AND STRENGTHENS. No arm passes G1/G3/G4 at 96 lights.
    Failing rows 83 -> 94 of 180 (world_sparse 24->24, triplane 29->32, world3d 30->38).
    Mean deltas SHRINK: +1.88->+1.41, +1.40->+1.08, +1.41->+0.99. G3 0/5 for every arm.
    15 dB floor still non-binding. 59 of the 83 original failures still fail; 35
    previously-passing rows now fail -- churn that is itself evidence of the old
    instrument's noise (per-row SEM 0.385 dB over 12 groups => ~1.33 dB on a single group).
    R1 stays closed, now robust to the estimator. This does NOT promote R1.
  - MINOR (deferred): peak_by_seed is a list-per-seed in the re-scorer vs a scalar in the
    original report -- harmless schema divergence, undocumented.
Task 9: complete (commit 99325a0, docs only -- docs/status/2026-08-29.md)
  - every figure recomputed against the committed JSON; no discrepancies with the brief
  - MINOR (deferred): hedged the Spearman p as "~0.68" claiming no scipy, but the K1
    report computes it EXACTLY by enumerating all 120 orderings (82/120 = 0.68333) --
    no scipy needed. Could be stated exactly.

DEFERRED MINORS for final whole-branch review triage:
  1. Task 1: new tests/test_torch_train.py duplicates the trace_path_cache(5,4,2,1,seed=9)
     fixture already in tests/test_torch_backend.py, the established home for train.py
     unit tests; splits coverage across two modules (plan-mandated file name)
  2. Task 4: main() builds run_training_config (shallow) and base_cfg (deep) separately,
     each overriding n_gate_lights -- deliberate (fail fast before the cache load)
  3. Task 6: the 96-light paragraph landed in the "R1 parity re-measurement" section, which
     is about the older per-seed-gate toy/Kitchen comparison (controller-caused placement)
  4. Task 8: peak_by_seed is a list-per-seed in rescore-96.json vs a scalar in the original
     report.json -- harmless schema divergence, undocumented
  5. Task 9: Spearman p hedged as "~0.68" though it is exactly 82/120 by enumeration

FINAL WHOLE-BRANCH REVIEW (opus): READY WITH FIXES. 13 commits, ddec973..99325a0.
  Verified BY EXECUTION, not reading:
  - tests.test_rescore_checkpoints 4 tests 92.2s OK, ZERO skips; full suite 680 OK (4 skips)
  - UNROTATED-LIGHTS GUARD re-verified independently: mutated rotated_lights to
    `return list(lights)` -> reproduction test failed 47.67 vs 25.56 dB (22.1 dB). Restored
    -> passes. Reviewer refused to accept the prior verification on trust.
  - FULL 180-row 8-light reproduction run by the reviewer itself: 0 mismatches on psnr_db,
    baseline_psnr_db, delta_db, both occupancy PSNRs, out_of_occupancy_fraction,
    baseline_camera; stop_reason and all G1/G3/G4 flags identical. Removes the
    "run out of band" caveat entirely.
  - 12-light parity reproduction matches report.json to 4.4e-16 (exact, not merely 6dp)
  - independent-path cross-check: rescore_sweep's finest128 bit-identical to rescore's
    world_sparse across all 8 per-seed deltas AND the entire gate dict
  - BOTH n_gate_lights call sites proven covered: dropping the kwarg at r1_parity:463 and
    r1_kitchen_k1:606 gives 2 failures ([None] != [20]) from the two tests that drive real
    run_experiment/main. No sibling of the Task-3 mirror-test defect survives.
  - experiment_gate.py untouched; no committed report.json modified; pipeline-audit failure
    set unchanged; alpha arithmetic re-derived from source (realized 2.07%, ceiling 11.9%)
  - every load-bearing figure across all three campaigns recomputed from JSON, all matched
  IMPORTANT finding: --gate-lights default 96 silently changes what 6 historical r1_parity
    reproduce commands replay (r1_kitchen_k1 is protected by _LEGACY_TRAINING_CONFIG_DEFAULTS;
    r1_parity has no control report so nothing catches it). Branch names this exact hazard in
    its own reproduce_command docstring and then leaves the commands unqualified.
  OBSERVATION (not a defect): 96 was chosen after seeing its effect on related data. Benign
    (estimator unbiased at any n; rationale answer-independent; the pre-registration
    constraint is scoped to "before any new TRAINING") but finest64's pass did emerge at a
    light count selected on related data -- worth one clause.
  Triage of the 5 deferred minors: ALL five leave-alone, with two one-sentence doc additions
    (peak_by_seed divergence; narrow "all per-seed numbers above" to Kitchen since the
    +/-0.94-1.14 dB SE is Kitchen-specific and unmeasured on toy).
