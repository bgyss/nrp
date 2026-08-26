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
    inside _index's formula is masked there. Closed by Task 1's
    test_hash_matches_reference_at_high_resolution, which checks _index against the
    instant-ngp reference independently.

DECISION (2026-08-26): G1 gains an absolute 15 dB PSNR floor alongside the 1.0 dB margin.
  Rationale + provenance in the spec (commit 7a58e98). 15 dB is INVENTED for this
  campaign; the 1.0 dB margin is inherited from the approved R2 ladder. Task 11 must
  pass absolute_floor_db=15.0 to g1_generalization.
G1 absolute floor: complete (commit 5e9a242, review clean, no fix round)
  - reviewer explicitly hunted a 4th unearned-pass instance and found none, by own
    adversarial execution. Conditions conjunctive; both reasons recorded when both fail;
    absolute_floor_db=None byte-identical to pre-change.
