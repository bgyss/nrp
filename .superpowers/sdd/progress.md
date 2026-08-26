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
