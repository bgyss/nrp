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
