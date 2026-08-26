# World-Anchored Encoding Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace representation-track R1's collision-tuned ambient 3D hash grid and its
mis-specified single-view parity gate with three capacity-honest encoding arms measured
against a held-out-camera generalization gate.

**Architecture:** A new `occupancy.py` computes exact per-level queried-vertex sets from a
`PathCache`. Encoders become interchangeable behind a registry with a uniform
`output_dim` / `forward` / `capacity_report` interface plus explicit `needs_occupancy` and
`needs_normals` flags. Three arms are added: a zero-collision sparse occupancy index
(primary), a normal-aware tri-plane, and an occupancy-allocated hash control.

**Tech Stack:** Python 3.12, PyTorch 2.12, NumPy, `unittest`, uv, mise, ruff.

**Spec:** `docs/superpowers/specs/2026-08-26-world-anchored-encoding-redesign-design.md`

## Global Constraints

- Base branch is `representation-encoding-redesign`, which sits on `codex/goal-implement-r2`. Do not rebase onto `main`.
- Run tests with `uv run python -m unittest discover -s tests` (or `mise run test`). Never invoke pytest.
- Lint with `uv run ruff check .`; format ONLY the paths you touched
  (`uv run ruff format <your files>`). Do NOT run `uv run ruff format .` — several
  files inherited from `main` and `codex/goal-implement-r2` are already non-conformant,
  and a repo-wide format drags unrelated churn into every commit.
- The repo is not an installed package (`[tool.uv] package = false`); tests import modules through the existing `sys.path` shim.
- The committed `pixel2d` path must stay behaviourally unchanged, and existing 2D checkpoints must load unchanged.
- Statistical assertions compare windowed means, never single minibatch losses.
- Optional-dependency tests skip via `@unittest.skipUnless(...)`; never fail when an extra is absent.
- Single Apple Silicon laptop: CPU/MPS only. CUDA stays parked.
- All measured claims land in `out/r1-encoding-redesign/report.json` **and** `docs/performance.md` with hardware context.
- Honest negative results are deliverables. Never tune past a stop condition to manufacture a pass.
- Commit after every task. Use `git add <exact paths>`, never `git add -A`.

---

### Task 0: Audit and pin the inherited camera and rotation machinery

The plan reuses camera/rotation code from `codex/goal-implement-r2`, whose R2 pilot is a
recorded negative. That negative is attributable to encoding allocation, not to this code,
but nothing downstream should depend on it untested.

**Files:**
- Test: `tests/test_camera_machinery_audit.py` (create)

**Interfaces:**
- Consumes: `nrp.torch_backend.conditioned_multiview.camera_direction`, `.global_world_bounds`; `examples.r1_promotion.rotation_matrix_y`, `.transform_cache`, `.out_of_bounds_fraction`
- Produces: nothing importable. This task is a safety net that later tasks rely on for confidence only.

- [ ] **Step 1: Write the failing characterization tests**

Create `tests/test_camera_machinery_audit.py`:

```python
"""Characterization tests pinning inherited R2 camera/rotation behaviour.

These functions arrive from codex/goal-implement-r2 and are reused by the
encoding redesign. They were not covered by dedicated unit tests before.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.r1_promotion import (  # noqa: E402
    out_of_bounds_fraction,
    rotation_matrix_y,
    transform_cache,
)
from nrp.path_cache import PathCache  # noqa: E402
from nrp.torch_backend.conditioned_multiview import (  # noqa: E402
    camera_direction,
    global_world_bounds,
)


def _tiny_cache() -> PathCache:
    return PathCache(
        width=2,
        height=1,
        seg_pixel=np.array([0, 1], dtype=np.int64),
        seg_origin=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        seg_dir=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        seg_tmax=np.array([1.0, 1.0]),
        seg_throughput=np.ones((2, 3)),
        albedo=np.full((1, 2, 3), 0.5),
        depth=np.ones((1, 2)),
        normal=np.tile(np.array([0.0, 1.0, 0.0]), (1, 2, 1)),
        position=np.array([[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]]),
    )


class TestRotationMatrix(unittest.TestCase):
    def test_is_orthonormal_and_right_handed(self):
        for degrees in (0.0, 90.0, 180.0, 37.5):
            r = rotation_matrix_y(degrees)
            np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(r)), 1.0, places=12)

    def test_ninety_degrees_maps_x_axis_into_z(self):
        r = rotation_matrix_y(90.0)
        rotated = np.array([1.0, 0.0, 0.0]) @ r.T
        self.assertAlmostEqual(abs(float(rotated[2])), 1.0, places=12)
        self.assertAlmostEqual(float(rotated[1]), 0.0, places=12)


class TestTransformCache(unittest.TestCase):
    def test_identity_rotation_preserves_geometry(self):
        cache = _tiny_cache()
        out = transform_cache(cache, np.eye(3))
        np.testing.assert_allclose(out.position, cache.position)
        np.testing.assert_allclose(out.normal, cache.normal)
        np.testing.assert_allclose(out.seg_dir, cache.seg_dir)

    def test_rotation_preserves_segment_lengths_and_normal_unit_norm(self):
        cache = _tiny_cache()
        out = transform_cache(cache, rotation_matrix_y(90.0))
        np.testing.assert_allclose(out.seg_tmax, cache.seg_tmax)
        norms = np.linalg.norm(out.normal.reshape(-1, 3), axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-12)
        out.validate()

    def test_rejects_non_orthonormal_rotation(self):
        with self.assertRaises(ValueError):
            transform_cache(_tiny_cache(), np.full((3, 3), 2.0))


class TestOutOfBoundsFraction(unittest.TestCase):
    def test_all_inside_is_zero(self):
        positions = np.array([[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        self.assertEqual(out_of_bounds_fraction(positions, bounds), 0.0)

    def test_half_outside_is_one_half(self):
        positions = np.array([[0.5, 0.5, 0.5], [9.0, 0.1, 0.1]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        self.assertAlmostEqual(out_of_bounds_fraction(positions, bounds), 0.5)


class TestCameraDirection(unittest.TestCase):
    def test_returns_unit_vector(self):
        d = camera_direction({"origin": [0.0, 0.0, 0.0], "target": [0.0, 0.0, 4.0]})
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=12)

    def test_rejects_degenerate_direction(self):
        with self.assertRaises(ValueError):
            camera_direction({"origin": [1.0, 1.0, 1.0], "target": [1.0, 1.0, 1.0]})


class TestGlobalWorldBounds(unittest.TestCase):
    def test_bounds_enclose_every_cache(self):
        a = _tiny_cache()
        b = _tiny_cache()
        b.position = b.position + 3.0
        bounds = global_world_bounds([a, b])
        lo = np.asarray(bounds["min"])
        hi = np.asarray(bounds["max"])
        for cache in (a, b):
            p = cache.position.reshape(-1, 3)
            self.assertTrue(np.all(p >= lo - 1e-9))
            self.assertTrue(np.all(p <= hi + 1e-9))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and record actual behaviour**

Run: `uv run python -m unittest tests.test_camera_machinery_audit -v`

Expected: most tests PASS (this is characterization, not new behaviour). If any test
fails, the inherited function does not behave as the redesign assumes. **Do not change
the test to match the code.** Stop, record the exact failure in the task notes, and fix
the inherited function in a separate commit before proceeding — later tasks depend on
these semantics.

Two signatures are assumed and may differ in the inherited code: `camera_direction` may
key on `origin`/`target` or on `origin`/`direction`, and `transform_cache` may require a
`translation` argument. If a test fails on a signature mismatch (`TypeError` or `KeyError`,
not an assertion), read the function and correct the *test* to the real signature — that
is a test-authoring error, not a code defect. Assertion failures are code defects.

- [ ] **Step 3: Run the full suite to confirm no regressions**

Run: `uv run python -m unittest discover -s tests 2>&1 | tail -5`
Expected: `OK` (skips permitted).

- [ ] **Step 4: Lint and commit**

```bash
uv run ruff format tests/test_camera_machinery_audit.py
uv run ruff check tests/test_camera_machinery_audit.py
git add tests/test_camera_machinery_audit.py
git commit -m "test: pin inherited R2 camera and rotation machinery"
```

---

### Task 1: Fix the three inherited encoding defects

Landed before any arm so no defect can confound a measured result.

**Files:**
- Modify: `nrp/torch_backend/encoding.py:56-60` (2D `_index`), `:67` and `:158` (boundary clamp)
- Modify: `nrp/torch_backend/streamed_train.py:373` (`_pixel_tensors`)
- Test: `tests/test_encoding_defects.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `nrp.torch_backend.streamed_train.spatial_tensors_for(cache, model, device) -> tuple[torch.Tensor, torch.Tensor]` — used by Task 8's streamed arms.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoding_defects.py`:

```python
"""Regression tests for three defects in the inherited hash-grid code.

The precedence defect is numerically inert in every committed config because
_PRIMES[0] == 1 and ix <= res < table_size, so masking commutes with the XOR.
It only diverges once res >= table_size, which is what test_hash_matches_reference
_at_high_resolution exercises. A test using a committed config would pass on the bug.
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import _PRIMES, HashEncoding2D, HashEncoding3D  # noqa: E402


class TestHashPrecedence(unittest.TestCase):
    def test_hash_matches_reference_at_high_resolution(self):
        """res >= table_size is where the '&' / '^' precedence defect diverges."""
        enc = HashEncoding2D(
            levels=1, features_per_level=2, table_size_log2=4, base_resolution=64,
            finest_resolution=64,
        )
        self.assertFalse(enc._dense[0], "level must be hashed for this test to mean anything")
        ix = torch.arange(0, 64, dtype=torch.long)
        iy = torch.full_like(ix, 33)
        got = enc._index(ix, iy, 0)
        want = ((ix * _PRIMES[0]) ^ (iy * _PRIMES[1])) & (enc.table_size - 1)
        torch.testing.assert_close(got, want)

    def test_index_never_exceeds_table(self):
        enc = HashEncoding2D(
            levels=1, features_per_level=2, table_size_log2=4, base_resolution=64,
            finest_resolution=64,
        )
        grid = torch.arange(0, 65, dtype=torch.long)
        ix, iy = torch.meshgrid(grid, grid, indexing="ij")
        idx = enc._index(ix.reshape(-1), iy.reshape(-1), 0)
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), enc.table_size)


class TestBoundaryClamp(unittest.TestCase):
    """floor(pos).clamp(0, res) makes x0 == res a degenerate constant cell."""

    def test_2d_interpolates_at_upper_boundary(self):
        enc = HashEncoding2D(
            levels=1, features_per_level=2, table_size_log2=14, base_resolution=4,
            finest_resolution=4,
        )
        with torch.no_grad():
            enc.tables[0].uniform_(-1.0, 1.0)
        just_inside = enc(torch.tensor([[0.999, 0.999]]))
        at_corner = enc(torch.tensor([[1.0, 1.0]]))
        torch.testing.assert_close(just_inside, at_corner, atol=2e-2, rtol=0)

    def test_3d_interpolates_at_upper_boundary(self):
        enc = HashEncoding3D(
            levels=1, features_per_level=2, table_size_log2=14, base_resolution=4,
            finest_resolution=4,
        )
        with torch.no_grad():
            enc.tables[0].uniform_(-1.0, 1.0)
        just_inside = enc(torch.tensor([[0.999, 0.999, 0.999]]))
        at_corner = enc(torch.tensor([[1.0, 1.0, 1.0]]))
        torch.testing.assert_close(just_inside, at_corner, atol=2e-2, rtol=0)


class TestStreamedWorldSupport(unittest.TestCase):
    def test_streamed_module_exposes_spatial_tensors_for(self):
        from nrp.torch_backend import streamed_train

        self.assertTrue(hasattr(streamed_train, "spatial_tensors_for"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_encoding_defects -v`

Expected: `test_hash_matches_reference_at_high_resolution` FAILS (tensor mismatch);
both boundary tests FAIL (the corner returns a single table entry, not an interpolation);
`test_streamed_module_exposes_spatial_tensors_for` FAILS with `AssertionError`.

- [ ] **Step 3: Fix the precedence defect**

In `nrp/torch_backend/encoding.py`, replace line 60:

```python
        return (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) & (self.table_size - 1)
```

with:

```python
        return ((ix * _PRIMES[0]) ^ (iy * _PRIMES[1])) & (self.table_size - 1)
```

- [ ] **Step 4: Fix the boundary clamp in both grids**

In `HashEncoding2D.forward` (line 67) and `HashEncoding3D.forward` (line 158), replace:

```python
            pos0 = torch.floor(pos).long().clamp_(0, res)
```

with:

```python
            # Clamp to res-1 so the cell [x0, x0+1] is always non-degenerate; x0 == res
            # would make all corners identical and the level output constant.
            pos0 = torch.floor(pos).long().clamp_(0, res - 1)
```

The existing `x1 = (x0 + 1).clamp(max=res)` lines then need no change: `x0 <= res - 1`
already guarantees `x1 <= res`.

- [ ] **Step 5: Add world-position support to the streamed trainer**

In `nrp/torch_backend/streamed_train.py`, immediately after the `_pixel_tensors`
definition (line 373), add:

```python
def spatial_tensors_for(cache: PathCache, model, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Spatial coordinates matching the model's encoding, plus the 7-column aux block.

    The streamed path previously assumed pixel2d, which silently blocked every
    world-anchored encoding from the S1 accelerated trainer.
    """
    xy, aux = _pixel_tensors(cache, device)
    if model.spatial_encoding == "pixel2d":
        return xy, aux
    position = torch.as_tensor(
        cache.position.reshape(-1, 3), dtype=torch.float32, device=device
    )
    return position, aux
```

Then in `train_streamed`, replace the `xy, aux = _pixel_tensors(cache, device)` call with
`xy, aux = spatial_tensors_for(gbuffer_cache, model, device)`, moving it to after the
model is constructed if it currently precedes it. `n_px = xy.shape[0]` is unchanged
because both branches return one row per pixel.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_encoding_defects -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Run the full suite — the 2D path must be unchanged**

Run: `uv run python -m unittest discover -s tests 2>&1 | tail -5`
Expected: `OK`. Any 2D failure means the clamp change altered committed behaviour;
investigate before continuing rather than adjusting the expectation.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format nrp/torch_backend/encoding.py nrp/torch_backend/streamed_train.py tests/test_encoding_defects.py
uv run ruff check nrp/torch_backend/encoding.py nrp/torch_backend/streamed_train.py tests/test_encoding_defects.py
git add nrp/torch_backend/encoding.py nrp/torch_backend/streamed_train.py tests/test_encoding_defects.py
git commit -m "fix: hash precedence, boundary clamp, and streamed world-position support"
```

---

### Task 2: Per-level grid occupancy and capacity reporting

**Files:**
- Create: `nrp/torch_backend/occupancy.py`
- Test: `tests/test_occupancy.py` (create)

**Interfaces:**
- Consumes: `nrp.path_cache.PathCache`
- Produces:
  - `LevelOccupancy` dataclass with fields `level: int`, `resolution: int`, `vertices: np.ndarray` (shape `(V, ndim)`, dtype `int64`, lexicographically sorted, unique) and property `count -> int`
  - `level_resolutions(levels: int, base_resolution: int, finest_resolution: int) -> list[int]`
  - `normalize_positions(positions: np.ndarray, bounds: dict) -> np.ndarray`
  - `grid_occupancy(normalized: np.ndarray, resolutions: list[int]) -> list[LevelOccupancy]`
  - `cache_occupancy(cache: PathCache, bounds: dict, levels: int, base_resolution: int, finest_resolution: int) -> list[LevelOccupancy]`
  - `capacity_report(occupancy: list[LevelOccupancy], index_fn, slots: list[int]) -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_occupancy.py`:

```python
"""Exact per-level queried-vertex sets for world-anchored encodings."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.occupancy import (  # noqa: E402
    capacity_report,
    grid_occupancy,
    level_resolutions,
    normalize_positions,
)


class TestLevelResolutions(unittest.TestCase):
    def test_matches_hashgrid_geometric_growth(self):
        # Must reproduce HashEncoding2D/3D's schedule exactly, or occupancy would
        # describe a different grid than the encoder queries.
        self.assertEqual(level_resolutions(8, 4, 128), [4, 6, 10, 17, 28, 47, 78, 128])

    def test_single_level_is_base_resolution(self):
        self.assertEqual(level_resolutions(1, 7, 64), [7])


class TestNormalizePositions(unittest.TestCase):
    def test_maps_bounds_to_unit_cube(self):
        positions = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 8.0]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [2.0, 4.0, 8.0]}
        np.testing.assert_allclose(
            normalize_positions(positions, bounds),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        )

    def test_clamps_out_of_bounds_positions(self):
        positions = np.array([[-1.0, 5.0, 0.5]])
        bounds = {"min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 1.0]}
        out = normalize_positions(positions, bounds)
        self.assertTrue(np.all(out >= 0.0) and np.all(out <= 1.0))


class TestGridOccupancy(unittest.TestCase):
    def test_single_point_touches_eight_corners(self):
        normalized = np.array([[0.5, 0.5, 0.5]])
        occ = grid_occupancy(normalized, [4])
        self.assertEqual(occ[0].count, 8)
        self.assertEqual(occ[0].resolution, 4)

    def test_point_on_a_vertex_still_touches_eight_corners(self):
        # The clamp-to-res-1 fix means an exact vertex hit still spans a full cell.
        occ = grid_occupancy(np.array([[0.0, 0.0, 0.0]]), [4])
        self.assertEqual(occ[0].count, 8)

    def test_vertices_are_unique_and_sorted(self):
        rng = np.random.default_rng(0)
        normalized = rng.random((200, 3))
        occ = grid_occupancy(normalized, [8])
        v = occ[0].vertices
        self.assertEqual(len(np.unique(v, axis=0)), len(v))
        order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
        np.testing.assert_array_equal(v, v[order])

    def test_vertices_stay_within_resolution(self):
        rng = np.random.default_rng(1)
        occ = grid_occupancy(rng.random((200, 3)), [8])
        self.assertGreaterEqual(int(occ[0].vertices.min()), 0)
        self.assertLessEqual(int(occ[0].vertices.max()), 8)

    def test_coarse_level_has_no_more_vertices_than_fine_level(self):
        rng = np.random.default_rng(2)
        occ = grid_occupancy(rng.random((500, 3)), [4, 32])
        self.assertLessEqual(occ[0].count, occ[1].count)


class TestCapacityReport(unittest.TestCase):
    def test_perfect_index_reports_zero_collisions(self):
        occ = grid_occupancy(np.array([[0.5, 0.5, 0.5]]), [4])

        def perfect(vertices, level):
            return np.arange(len(vertices))

        report = capacity_report(occ, perfect, slots=[8])
        level = report["levels"][0]
        self.assertEqual(level["distinct_vertices"], 8)
        self.assertEqual(level["collision_fraction"], 0.0)
        self.assertEqual(level["max_slot_load"], 1)

    def test_constant_index_reports_total_collapse(self):
        occ = grid_occupancy(np.array([[0.5, 0.5, 0.5]]), [4])

        def constant(vertices, level):
            return np.zeros(len(vertices), dtype=np.int64)

        report = capacity_report(occ, constant, slots=[8])
        level = report["levels"][0]
        self.assertEqual(level["used_slots"], 1)
        self.assertAlmostEqual(level["collision_fraction"], 7 / 8)
        self.assertEqual(level["max_slot_load"], 8)

    def test_reports_slots_per_distinct_vertex(self):
        occ = grid_occupancy(np.array([[0.5, 0.5, 0.5]]), [4])

        def perfect(vertices, level):
            return np.arange(len(vertices))

        report = capacity_report(occ, perfect, slots=[16])
        self.assertAlmostEqual(report["levels"][0]["slots_per_distinct_vertex"], 2.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_occupancy -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp.torch_backend.occupancy'`.

- [ ] **Step 3: Implement the module**

Create `nrp/torch_backend/occupancy.py`:

```python
"""Exact per-level queried-vertex sets for world-anchored spatial encodings.

Rung R1's negative was caused by allocating table slots against a parameter budget
rather than against the number of grid vertices the cache actually queries. This
module makes that number a measured, tested quantity instead of an invisible one.
It knows about caches; encoders do not.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from ..path_cache import PathCache


@dataclass(frozen=True)
class LevelOccupancy:
    """The unique grid vertices one level of an encoding actually reads."""

    level: int
    resolution: int
    vertices: np.ndarray  # (V, ndim) int64, unique, lexicographically sorted

    @property
    def count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def ndim(self) -> int:
        return int(self.vertices.shape[1])


def level_resolutions(levels: int, base_resolution: int, finest_resolution: int) -> list[int]:
    """Reproduce the hashgrid geometric schedule exactly.

    Any divergence here would make the occupancy describe a different grid than the
    encoder queries, which is precisely the class of error this module exists to
    prevent.
    """
    if levels <= 0:
        raise ValueError("levels must be positive")
    if base_resolution <= 0 or finest_resolution <= 0:
        raise ValueError("resolutions must be positive")
    if finest_resolution < base_resolution:
        raise ValueError("finest_resolution must be >= base_resolution")
    growth = (
        math.exp(math.log(finest_resolution / base_resolution) / max(levels - 1, 1))
        if levels > 1
        else 1.0
    )
    return [max(int(math.floor(base_resolution * growth**level)), 1) for level in range(levels)]


def normalize_positions(positions: np.ndarray, bounds: dict) -> np.ndarray:
    """Map world positions into the unit cube using the model's stored bounds."""
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    lo = np.asarray(bounds["min"], dtype=np.float64)
    hi = np.asarray(bounds["max"], dtype=np.float64)
    if lo.shape != (3,) or hi.shape != (3,):
        raise ValueError("bounds min and max must each contain three values")
    if np.any(hi <= lo):
        raise ValueError("bounds max must exceed min on every axis")
    return np.clip((positions - lo) / (hi - lo), 0.0, 1.0)


def grid_occupancy(normalized: np.ndarray, resolutions: list[int]) -> list[LevelOccupancy]:
    """Unique vertices touched per level, matching the encoders' corner enumeration."""
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.ndim != 2:
        raise ValueError("normalized coordinates must be 2-D")
    ndim = normalized.shape[1]
    out: list[LevelOccupancy] = []
    for level, res in enumerate(resolutions):
        base = np.floor(normalized * res).astype(np.int64).clip(0, res - 1)
        corners = [base + np.asarray(offset, dtype=np.int64) for offset in itertools.product((0, 1), repeat=ndim)]
        stacked = np.concatenate(corners, axis=0)
        unique = np.unique(stacked, axis=0)
        out.append(LevelOccupancy(level=level, resolution=res, vertices=unique))
    return out


def cache_occupancy(
    cache: PathCache,
    bounds: dict,
    levels: int,
    base_resolution: int,
    finest_resolution: int,
) -> list[LevelOccupancy]:
    """Occupancy of a cache's first-hit world positions under a level schedule."""
    normalized = normalize_positions(cache.position, bounds)
    return grid_occupancy(normalized, level_resolutions(levels, base_resolution, finest_resolution))


def capacity_report(occupancy: list[LevelOccupancy], index_fn, slots: list[int]) -> dict:
    """Per-level capacity: distinct vertices, used slots, collision fraction, max load.

    `index_fn(vertices, level) -> np.ndarray` maps vertex coordinates to slot indices,
    so this stays agnostic to whether the encoder hashes, indexes densely, or uses an
    exact sparse map.
    """
    if len(slots) != len(occupancy):
        raise ValueError("slots must supply one entry per level")
    levels = []
    for occ, n_slots in zip(occupancy, slots, strict=True):
        idx = np.asarray(index_fn(occ.vertices, occ.level), dtype=np.int64)
        if idx.shape != (occ.count,):
            raise ValueError("index_fn must return one index per vertex")
        counts = np.bincount(idx, minlength=int(n_slots))
        used = int((counts > 0).sum())
        levels.append(
            {
                "level": occ.level,
                "resolution": occ.resolution,
                "distinct_vertices": occ.count,
                "slots": int(n_slots),
                "used_slots": used,
                "collision_fraction": float(1.0 - used / occ.count) if occ.count else 0.0,
                "max_slot_load": int(counts.max()) if counts.size else 0,
                "slots_per_distinct_vertex": float(n_slots / occ.count) if occ.count else 0.0,
            }
        )
    finest = levels[-1] if levels else {}
    return {
        "levels": levels,
        "total_distinct_vertices": int(sum(level["distinct_vertices"] for level in levels)),
        "total_slots": int(sum(level["slots"] for level in levels)),
        "finest_collision_fraction": finest.get("collision_fraction", 0.0),
        "finest_slots_per_distinct_vertex": finest.get("slots_per_distinct_vertex", 0.0),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_occupancy -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Reproduce the spec's diagnostic table as a guard**

Append to `tests/test_occupancy.py`:

```python
class TestKitchenDiagnosticIsReproducible(unittest.TestCase):
    """Pins the measurement the redesign is founded on (spec: Diagnosis)."""

    def test_kitchen_world3d_finest_level_is_capacity_starved(self):
        cache_path = Path(__file__).resolve().parent.parent / "out" / "kitchen" / "path_cache.npz"
        if not cache_path.exists():
            self.skipTest("kitchen cache not present; run the Mitsuba export first")
        from nrp.path_cache import PathCache
        from nrp.torch_backend.occupancy import cache_occupancy

        cache = PathCache.load(str(cache_path))
        position = np.asarray(cache.position, dtype=np.float64).reshape(-1, 3)
        bounds = {"min": position.min(axis=0).tolist(), "max": position.max(axis=0).tolist()}
        occ = cache_occupancy(cache, bounds, levels=8, base_resolution=7, finest_resolution=128)
        # Measured 78,080 against a 4096-slot table: a ~19x handicap versus pixel2d.
        self.assertGreater(occ[-1].count, 70_000)
        self.assertLess(4096 / occ[-1].count, 0.1)
```

- [ ] **Step 6: Run and commit**

```bash
uv run python -m unittest tests.test_occupancy -v
uv run ruff format nrp/torch_backend/occupancy.py tests/test_occupancy.py
uv run ruff check nrp/torch_backend/occupancy.py tests/test_occupancy.py
git add nrp/torch_backend/occupancy.py tests/test_occupancy.py
git commit -m "feat: per-level grid occupancy and capacity reporting"
```

Expected: PASS (13 tests, or 12 + 1 skip if the kitchen cache is absent).

---

### Task 3: Encoder registry and uniform interface

**Files:**
- Create: `nrp/torch_backend/encoder_registry.py`
- Modify: `nrp/torch_backend/encoding.py` (decorate the three encoders, add flags and `capacity_report`)
- Modify: `nrp/torch_backend/model.py:167-174` (replace the if/elif chain)
- Test: `tests/test_encoder_registry.py` (create)

**Interfaces:**
- Consumes: `nrp.torch_backend.occupancy.capacity_report`, `LevelOccupancy`
- Produces:
  - `SPATIAL_ENCODERS: dict[str, type]` in `nrp.torch_backend.encoder_registry`
  - `register_encoder(name: str)` decorator, in the same module
  - `build_encoder(name: str, config: dict, occupancy: list[LevelOccupancy] | None = None) -> nn.Module`, in the same module
  - `nrp.torch_backend.encoding` re-exports all three for convenience
  - Every encoder class exposes `needs_occupancy: bool`, `needs_normals: bool`, `output_dim: int`, `capacity_report() -> dict`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoder_registry.py`:

```python
"""Uniform interface and registry for spatial encoders."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoder_registry import SPATIAL_ENCODERS, build_encoder  # noqa: E402
from nrp.torch_backend.encoding import HashEncoding2D  # noqa: E402

CONFIG = {
    "levels": 3,
    "features_per_level": 2,
    "table_size_log2": 8,
    "base_resolution": 4,
    "finest_resolution": 16,
}


class TestRegistry(unittest.TestCase):
    def test_registry_contains_the_committed_encoders(self):
        self.assertLessEqual({"pixel2d", "world3d", "world_triplane"}, set(SPATIAL_ENCODERS))

    def test_build_encoder_returns_the_registered_class(self):
        enc = build_encoder("pixel2d", CONFIG)
        self.assertIsInstance(enc, HashEncoding2D)

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            build_encoder("does_not_exist", CONFIG)

    def test_occupancy_encoder_without_occupancy_raises(self):
        for name, cls in SPATIAL_ENCODERS.items():
            if getattr(cls, "needs_occupancy", False):
                with self.assertRaises(ValueError, msg=name):
                    build_encoder(name, CONFIG, occupancy=None)


class TestUniformInterface(unittest.TestCase):
    def test_every_encoder_declares_the_interface(self):
        for name, cls in SPATIAL_ENCODERS.items():
            self.assertIsInstance(getattr(cls, "needs_occupancy", None), bool, name)
            self.assertIsInstance(getattr(cls, "needs_normals", None), bool, name)
            self.assertTrue(hasattr(cls, "capacity_report"), name)

    def test_capacity_report_shape(self):
        enc = build_encoder("pixel2d", CONFIG)
        report = enc.capacity_report()
        self.assertIn("levels", report)
        self.assertEqual(len(report["levels"]), CONFIG["levels"])
        for level in report["levels"]:
            self.assertIn("slots", level)
            self.assertIn("resolution", level)


class TestModelUsesRegistry(unittest.TestCase):
    def test_pixel2d_model_is_unchanged(self):
        from nrp.torch_backend.model import TorchNRP

        model = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=CONFIG)
        out = model(torch.rand(5, 2), torch.rand(5, 7), torch.rand(5, 4))
        self.assertEqual(tuple(out.shape), (5, 3))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_encoder_registry -v`
Expected: FAIL with `ImportError: cannot import name 'SPATIAL_ENCODERS'`.

- [ ] **Step 3: Create the registry module**

The registry lives in its own module so encoders can import `register_encoder` without
importing `encoding`, which would otherwise create a cycle once arm B lands in a separate
file. Create `nrp/torch_backend/encoder_registry.py`:

```python
"""Registry of spatial encoders.

Separate from `encoding` so an encoder defined in any module can register itself
without importing the module that imports it back.
"""

from __future__ import annotations


#: name -> encoder class. `build_encoder` is the only construction path the model uses,
#: so adding an arm is one decorator rather than another if/elif branch.
SPATIAL_ENCODERS: dict[str, type] = {}


def register_encoder(name: str):
    def wrap(cls):
        if name in SPATIAL_ENCODERS:
            raise ValueError(f"spatial encoder {name!r} is already registered")
        SPATIAL_ENCODERS[name] = cls
        return cls

    return wrap


def build_encoder(name: str, config: dict | None = None, occupancy=None):
    """Construct a spatial encoder, supplying occupancy only to arms that need it."""
    if name not in SPATIAL_ENCODERS:
        raise ValueError(f"unknown spatial encoding {name!r}; expected one of {sorted(SPATIAL_ENCODERS)}")
    cls = SPATIAL_ENCODERS[name]
    kwargs = dict(config or {})
    # Arm A opts into occupancy through its config rather than a class flag, because
    # allocation="uniform" must keep working with no cache available.
    wants_occupancy = getattr(cls, "needs_occupancy", False) or kwargs.get("allocation") == "occupancy"
    if wants_occupancy:
        if occupancy is None:
            raise ValueError(f"spatial encoding {name!r} requires occupancy")
        kwargs["occupancy"] = occupancy
    return cls(**kwargs)
```

- [ ] **Step 4: Re-export from `encoding.py` and decorate the three encoders**

Add near the top of `nrp/torch_backend/encoding.py`, replacing the registry code that
would otherwise live there:

```python
from .encoder_registry import SPATIAL_ENCODERS, build_encoder, register_encoder  # noqa: F401
```

Then decorate the encoders and give them the interface.

Add `@register_encoder("pixel2d")` above `class HashEncoding2D`, `@register_encoder("world3d")`
above `class HashEncoding3D`, and `@register_encoder("world_triplane")` above
`class HashEncodingTriPlane`.

Add these two class attributes to `HashEncoding2D` and `HashEncoding3D`, immediately
below the class docstring:

```python
    needs_occupancy = False
    needs_normals = False
```

Add the same two attributes to `HashEncodingTriPlane`.

Both grid classes report identically, so define the body once as a module-level helper
rather than copying it into each class:

```python
def _grid_capacity_report(encoder) -> dict:
    """Static slot budget per level, shared by the 2D and 3D grids.

    Occupancy-aware numbers come from `nrp.torch_backend.occupancy.capacity_report`,
    which needs a cache; this is the cache-free view.
    """
    return {
        "encoding": type(encoder).__name__,
        "levels": [
            {
                "level": level,
                "resolution": res,
                "dense": bool(encoder._dense[level]),
                "slots": int(encoder.tables[level].shape[0]),
            }
            for level, res in enumerate(encoder.resolutions)
        ],
        "total_slots": int(sum(t.shape[0] for t in encoder.tables)),
    }
```

Then add this one-line method to **both** `HashEncoding2D` and `HashEncoding3D`:

```python
    def capacity_report(self) -> dict:
        return _grid_capacity_report(self)
```

For `HashEncodingTriPlane`:

```python
    def capacity_report(self) -> dict:
        per_plane = [plane.capacity_report() for plane in self.planes]
        return {
            "encoding": type(self).__name__,
            "levels": per_plane[0]["levels"],
            "planes": per_plane,
            "total_slots": int(sum(p["total_slots"] for p in per_plane)),
        }
```

- [ ] **Step 5: Route the model through the registry**

In `nrp/torch_backend/model.py`, replace the if/elif chain (lines 167-174):

```python
        if not use_encoding:
            self.encoding = None
        elif spatial_encoding == "world3d":
            self.encoding = HashEncoding3D(**(encoding or {}))
        elif spatial_encoding == "world_triplane":
            self.encoding = HashEncodingTriPlane(**(encoding or {}))
        else:
            self.encoding = HashEncoding2D(**(encoding or {}))
```

with:

```python
        if not use_encoding:
            self.encoding = None
        else:
            self.encoding = build_encoder(spatial_encoding, encoding or {}, occupancy=occupancy)
```

Add `occupancy=None` as a keyword parameter to `TorchNRP.__init__` (after `world_bounds`),
and change the import at the top of `model.py` from

```python
from .encoding import HashEncoding2D, HashEncoding3D, HashEncodingTriPlane
```

to

```python
from . import encoding as _encoding  # noqa: F401  # registers the built-in encoders
from .encoder_registry import SPATIAL_ENCODERS, build_encoder
```

Replace the module-level `SUPPORTED_SPATIAL_ENCODINGS = {...}` literal with a lookup so
the two can never drift:

```python
def _supported_spatial_encodings() -> set[str]:
    return set(SPATIAL_ENCODERS)
```

and change every `spatial_encoding not in SUPPORTED_SPATIAL_ENCODINGS` check to
`spatial_encoding not in _supported_spatial_encodings()`. Grep for remaining references:
`grep -rn "SUPPORTED_SPATIAL_ENCODINGS" nrp/ tests/ examples/` and update each.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_encoder_registry -v`
Expected: PASS (7 tests).

- [ ] **Step 7: Run the full suite**

Run: `uv run python -m unittest discover -s tests 2>&1 | tail -5`
Expected: `OK`. The registry refactor touches the model constructor, so a failure here
is a real regression.

- [ ] **Step 8: Lint and commit**

```bash
uv run ruff format nrp/torch_backend/encoder_registry.py nrp/torch_backend/encoding.py nrp/torch_backend/model.py tests/test_encoder_registry.py
uv run ruff check nrp/torch_backend/encoder_registry.py nrp/torch_backend/encoding.py nrp/torch_backend/model.py tests/test_encoder_registry.py
git add nrp/torch_backend/encoder_registry.py nrp/torch_backend/encoding.py nrp/torch_backend/model.py tests/test_encoder_registry.py
git commit -m "refactor: spatial encoder registry with a uniform interface"
```

---

### Task 4: Arm B — `SparseVoxelEncoding` (zero collisions by construction)

**Files:**
- Create: `nrp/torch_backend/sparse_encoding.py`
- Modify: `nrp/torch_backend/encoding.py` (import so the decorator registers)
- Test: `tests/test_sparse_encoding.py` (create)

**Interfaces:**
- Consumes: `nrp.torch_backend.occupancy.LevelOccupancy`, `nrp.torch_backend.encoder_registry.register_encoder`
- Produces: `SparseVoxelEncoding` registered as `"world_sparse"`, with `needs_occupancy = True`, `needs_normals = False`, `output_dim`, `capacity_report()`, and `out_of_occupancy_fraction(xyz: torch.Tensor) -> float`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sparse_encoding.py`:

```python
"""Arm B: exact sparse occupancy index, no hashing, no collisions."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.occupancy import grid_occupancy  # noqa: E402
from nrp.torch_backend.sparse_encoding import SparseVoxelEncoding  # noqa: E402

CONFIG = {"levels": 2, "features_per_level": 2, "base_resolution": 4, "finest_resolution": 8}


def _encoder(points: np.ndarray) -> SparseVoxelEncoding:
    from nrp.torch_backend.occupancy import level_resolutions

    res = level_resolutions(CONFIG["levels"], CONFIG["base_resolution"], CONFIG["finest_resolution"])
    return SparseVoxelEncoding(occupancy=grid_occupancy(points, res), **CONFIG)


class TestZeroCollisions(unittest.TestCase):
    def test_table_is_sized_exactly_to_occupancy(self):
        rng = np.random.default_rng(0)
        points = rng.random((100, 3))
        enc = _encoder(points)
        for level, occ in enumerate(enc.occupancy):
            self.assertEqual(enc.tables[level].shape[0], occ.count)

    def test_every_occupied_key_resolves_uniquely(self):
        rng = np.random.default_rng(1)
        enc = _encoder(rng.random((100, 3)))
        for level, occ in enumerate(enc.occupancy):
            keys = getattr(enc, f"keys_{level}")
            self.assertEqual(len(torch.unique(keys)), len(keys))
            self.assertTrue(bool((keys[1:] > keys[:-1]).all()), "keys must be sorted")

    def test_capacity_report_asserts_zero_collisions(self):
        rng = np.random.default_rng(2)
        enc = _encoder(rng.random((100, 3)))
        report = enc.capacity_report()
        for level in report["levels"]:
            self.assertEqual(level["collision_fraction"], 0.0)
            self.assertEqual(level["max_slot_load"], 1)


class TestLookup(unittest.TestCase):
    def test_corner_query_returns_that_table_entry(self):
        points = np.array([[0.5, 0.5, 0.5]])
        enc = _encoder(points)
        with torch.no_grad():
            for table in enc.tables:
                table.uniform_(-1.0, 1.0)
        # A query exactly on an occupied vertex has trilinear weight 1 on that corner,
        # so the level's output must equal that vertex's table row. Row order follows
        # sorted key order, so vertices[0] owns tables[0][0].
        occ0 = enc.occupancy[0]
        vertex = occ0.vertices[0]
        xyz = torch.tensor([vertex / occ0.resolution], dtype=torch.float32)
        out = enc(xyz)
        torch.testing.assert_close(
            out[0, : enc.features_per_level], enc.tables[0][0], atol=1e-5, rtol=0
        )

    def test_gradients_flow_to_the_tables(self):
        rng = np.random.default_rng(3)
        enc = _encoder(rng.random((50, 3)))
        out = enc(torch.rand(10, 3))
        out.sum().backward()
        self.assertTrue(any(t.grad is not None and bool(t.grad.abs().sum() > 0) for t in enc.tables))

    def test_output_dim_matches_forward_width(self):
        rng = np.random.default_rng(4)
        enc = _encoder(rng.random((50, 3)))
        self.assertEqual(enc(torch.rand(7, 3)).shape[1], enc.output_dim)


class TestFallback(unittest.TestCase):
    def test_unoccupied_query_contributes_zero_and_stays_differentiable(self):
        # Occupancy built from one corner of the cube; query the opposite corner.
        enc = _encoder(np.array([[0.01, 0.01, 0.01]]))
        far = torch.tensor([[0.99, 0.99, 0.99]], requires_grad=False)
        out = enc(far)
        self.assertTrue(torch.isfinite(out).all())
        # Differentiability: a table that received no gradient must still not crash.
        out.sum().backward()

    def test_out_of_occupancy_fraction_is_one_for_unseen_region(self):
        enc = _encoder(np.array([[0.01, 0.01, 0.01]]))
        frac = enc.out_of_occupancy_fraction(torch.tensor([[0.99, 0.99, 0.99]]))
        self.assertAlmostEqual(frac, 1.0)

    def test_out_of_occupancy_fraction_is_zero_for_training_points(self):
        rng = np.random.default_rng(5)
        points = rng.random((80, 3))
        enc = _encoder(points)
        frac = enc.out_of_occupancy_fraction(torch.tensor(points, dtype=torch.float32))
        self.assertAlmostEqual(frac, 0.0)


class TestCheckpointRoundTrip(unittest.TestCase):
    def test_keys_survive_state_dict_round_trip(self):
        rng = np.random.default_rng(6)
        points = rng.random((60, 3))
        enc = _encoder(points)
        clone = _encoder(points)
        clone.load_state_dict(enc.state_dict())
        query = torch.rand(12, 3)
        torch.testing.assert_close(enc(query), clone(query))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_sparse_encoding -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp.torch_backend.sparse_encoding'`.

- [ ] **Step 3: Implement the encoder**

Create `nrp/torch_backend/sparse_encoding.py`:

```python
"""Arm B of the encoding redesign: an exact sparse occupancy index.

The path cache is static and fully known before training, so the set of grid
vertices any level will ever read can be enumerated up front. Storing exactly those
vertices removes hashing, and therefore collisions, by construction rather than by
tuning -- which is what R1A's +-1.74 dB seed spread was mostly measuring.

A vertex outside the occupied set (only a novel camera produces these) contributes
zero features at that level and is carried by the coarser levels. The fallback is
differentiable and its per-camera frequency is reported, never hidden.
"""

from __future__ import annotations

import itertools

import torch
from torch import nn

from .encoder_registry import register_encoder


def _vertex_codes(vertices: torch.Tensor, resolution: int) -> torch.Tensor:
    """Fold integer vertex coordinates into one int64 key. Unique for res <= ~1e6."""
    side = resolution + 1
    return (vertices[:, 2] * side + vertices[:, 1]) * side + vertices[:, 0]


@register_encoder("world_sparse")
class SparseVoxelEncoding(nn.Module):
    needs_occupancy = True
    needs_normals = False

    def __init__(
        self,
        occupancy,
        levels: int = 8,
        features_per_level: int = 2,
        base_resolution: int = 4,
        finest_resolution: int = 128,
        **_ignored,
    ):
        super().__init__()
        if len(occupancy) != levels:
            raise ValueError(f"occupancy has {len(occupancy)} levels, expected {levels}")
        if features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        self.levels = levels
        self.features_per_level = features_per_level
        self.occupancy = list(occupancy)
        self.resolutions = [occ.resolution for occ in self.occupancy]
        self.tables = nn.ParameterList()
        for level, occ in enumerate(self.occupancy):
            if occ.count == 0:
                raise ValueError(f"level {level} has empty occupancy")
            vertices = torch.as_tensor(occ.vertices, dtype=torch.long)
            keys = _vertex_codes(vertices, occ.resolution)
            order = torch.argsort(keys)
            self.register_buffer(f"keys_{level}", keys[order].contiguous())
            self.tables.append(
                nn.Parameter(torch.empty(occ.count, features_per_level).uniform_(-1e-4, 1e-4))
            )
            # Table rows follow sorted key order so searchsorted indexes them directly.
            self.occupancy[level] = type(occ)(
                level=occ.level, resolution=occ.resolution, vertices=occ.vertices[order.numpy()]
            )

    @property
    def output_dim(self) -> int:
        return self.levels * self.features_per_level

    def _lookup(self, vertices: torch.Tensor, level: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (row indices, hit mask) for integer vertex coordinates."""
        keys = getattr(self, f"keys_{level}")
        codes = _vertex_codes(vertices, self.resolutions[level])
        idx = torch.searchsorted(keys, codes).clamp(max=keys.numel() - 1)
        hit = keys[idx] == codes
        return idx, hit

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        outputs = []
        for level, res in enumerate(self.resolutions):
            pos = xyz * res
            pos0 = torch.floor(pos).long().clamp_(0, res - 1)
            frac = pos - pos0
            table = self.tables[level]
            out = torch.zeros(
                (xyz.shape[0], self.features_per_level), dtype=table.dtype, device=table.device
            )
            for offset in itertools.product((0, 1), repeat=3):
                corner = pos0 + torch.tensor(offset, device=pos0.device, dtype=pos0.dtype)
                weight = torch.ones((xyz.shape[0], 1), dtype=table.dtype, device=table.device)
                for axis, o in enumerate(offset):
                    w = frac[:, axis : axis + 1]
                    weight = weight * (w if o else (1.0 - w))
                idx, hit = self._lookup(corner, level)
                out = out + table[idx] * weight * hit.unsqueeze(1).to(table.dtype)
            outputs.append(out)
        return torch.cat(outputs, dim=1)

    def out_of_occupancy_fraction(self, xyz: torch.Tensor) -> float:
        """Fraction of query points whose finest-level base cell is unoccupied.

        Reported per held-out camera so a good gate result cannot hide behind a
        favourable fallback (spec: G5).
        """
        level = self.levels - 1
        res = self.resolutions[level]
        pos0 = torch.floor(xyz * res).long().clamp_(0, res - 1)
        _, hit = self._lookup(pos0, level)
        return float((~hit).to(torch.float64).mean())

    def capacity_report(self) -> dict:
        levels = []
        for level, occ in enumerate(self.occupancy):
            levels.append(
                {
                    "level": level,
                    "resolution": occ.resolution,
                    "dense": False,
                    "sparse": True,
                    "distinct_vertices": occ.count,
                    "slots": int(self.tables[level].shape[0]),
                    "used_slots": occ.count,
                    "collision_fraction": 0.0,
                    "max_slot_load": 1,
                    "slots_per_distinct_vertex": 1.0,
                    "key_bytes": int(getattr(self, f"keys_{level}").numel() * 8),
                }
            )
        return {
            "encoding": type(self).__name__,
            "levels": levels,
            "total_slots": int(sum(level["slots"] for level in levels)),
            "total_key_bytes": int(sum(level["key_bytes"] for level in levels)),
        }
```

- [ ] **Step 4: Import the module so the decorator runs**

`sparse_encoding` imports only `encoder_registry`, so there is no cycle and the import
goes at the top of `nrp/torch_backend/encoding.py` beside the others:

```python
from . import sparse_encoding as _sparse_encoding  # noqa: F401  # registers arm B
```

Place it *after* the `encoder_registry` import so `register_encoder` is bound first.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_sparse_encoding -v`
Expected: PASS (10 tests).

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run python -m unittest discover -s tests 2>&1 | tail -5
uv run ruff format nrp/torch_backend/sparse_encoding.py nrp/torch_backend/encoding.py tests/test_sparse_encoding.py
uv run ruff check nrp/torch_backend/sparse_encoding.py nrp/torch_backend/encoding.py tests/test_sparse_encoding.py
git add nrp/torch_backend/sparse_encoding.py nrp/torch_backend/encoding.py tests/test_sparse_encoding.py
git commit -m "feat: arm B sparse occupancy encoding with zero collisions"
```

Expected: `OK`.

---

### Task 5: Arm C — `NormalAwareTriPlane`

**Files:**
- Modify: `nrp/torch_backend/encoding.py` (add the class beside `HashEncodingTriPlane`)
- Modify: `nrp/torch_backend/model.py` (pass normals when `needs_normals`)
- Test: `tests/test_normal_aware_triplane.py` (create)

**Interfaces:**
- Consumes: `HashEncoding2D`, `nrp.torch_backend.encoder_registry.register_encoder`
- Produces: `NormalAwareTriPlane` registered as `"world_normal_triplane"`, `needs_normals = True`, `forward(xyz: torch.Tensor, normals: torch.Tensor) -> torch.Tensor`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_normal_aware_triplane.py`:

```python
"""Arm C: each point reads only the plane aligned with its surface normal."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import NormalAwareTriPlane  # noqa: E402

CONFIG = {
    "levels": 2,
    "features_per_level": 2,
    "table_size_log2": 8,
    "base_resolution": 4,
    "finest_resolution": 16,
}


class TestPlaneSelection(unittest.TestCase):
    def test_output_dim_is_one_plane_not_three(self):
        enc = NormalAwareTriPlane(**CONFIG)
        self.assertEqual(enc.output_dim, CONFIG["levels"] * CONFIG["features_per_level"])

    def test_axis_aligned_normal_selects_the_expected_plane(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with torch.no_grad():
            for i, plane in enumerate(enc.planes):
                for table in plane.tables:
                    table.fill_(float(i + 1))
        xyz = torch.tensor([[0.5, 0.5, 0.5]])
        for axis in range(3):
            normal = torch.zeros(1, 3)
            normal[0, axis] = 1.0
            out = enc(xyz, normal)
            self.assertAlmostEqual(float(out[0, 0]), float(axis + 1), places=5)

    def test_sign_of_normal_does_not_change_plane_choice(self):
        enc = NormalAwareTriPlane(**CONFIG)
        xyz = torch.tensor([[0.3, 0.4, 0.5]])
        up = enc(xyz, torch.tensor([[0.0, 1.0, 0.0]]))
        down = enc(xyz, torch.tensor([[0.0, -1.0, 0.0]]))
        torch.testing.assert_close(up, down)

    def test_batch_with_mixed_normals_matches_per_row_evaluation(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with torch.no_grad():
            for plane in enc.planes:
                for table in plane.tables:
                    table.uniform_(-1.0, 1.0)
        xyz = torch.rand(6, 3)
        normals = torch.eye(3).repeat(2, 1)
        batched = enc(xyz, normals)
        for i in range(6):
            single = enc(xyz[i : i + 1], normals[i : i + 1])
            torch.testing.assert_close(batched[i : i + 1], single)


class TestGradients(unittest.TestCase):
    def test_gradients_flow_to_the_selected_plane(self):
        enc = NormalAwareTriPlane(**CONFIG)
        xyz = torch.rand(8, 3)
        normals = torch.zeros(8, 3)
        normals[:, 1] = 1.0
        enc(xyz, normals).sum().backward()
        selected = [t.grad for t in enc.planes[1].tables if t.grad is not None]
        self.assertTrue(any(bool(g.abs().sum() > 0) for g in selected))


class TestValidation(unittest.TestCase):
    def test_missing_normals_raises(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with self.assertRaises(ValueError):
            enc(torch.rand(4, 3), None)

    def test_zero_normal_raises(self):
        enc = NormalAwareTriPlane(**CONFIG)
        with self.assertRaises(ValueError):
            enc(torch.rand(1, 3), torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_normal_aware_triplane -v`
Expected: FAIL with `ImportError: cannot import name 'NormalAwareTriPlane'`.

- [ ] **Step 3: Implement the encoder**

Append to `nrp/torch_backend/encoding.py`, after `HashEncodingTriPlane`:

```python
@register_encoder("world_normal_triplane")
class NormalAwareTriPlane(nn.Module):
    """Tri-plane where each point reads only the plane aligned with its normal.

    The concatenating tri-plane reads all three planes, tripling capacity pressure and
    tying quality to the world frame -- the corrected R1C matrix measured a -3.651 dB
    worst-orientation failure. Selecting the plane by surface normal makes the choice
    follow geometry instead of the frame, and reads one plane instead of three.

    The argmax is discontinuous across normal boundaries; gradients flow through the
    features, not the selection. Expected artifacts are seams at sharp surface-orientation
    discontinuities, which the per-camera error maps will show if they matter.
    """

    needs_occupancy = False
    needs_normals = True

    #: dominant normal axis -> the two coordinate axes spanning the plane it reads
    AXIS_TO_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def __init__(
        self,
        levels: int = 3,
        features_per_level: int = 2,
        table_size_log2: int = 13,
        base_resolution: int = 4,
        finest_resolution: int = 256,
    ):
        super().__init__()
        config = {
            "levels": levels,
            "features_per_level": features_per_level,
            "table_size_log2": table_size_log2,
            "base_resolution": base_resolution,
            "finest_resolution": finest_resolution,
        }
        self.planes = nn.ModuleList([HashEncoding2D(**config) for _ in range(3)])
        self.levels = levels
        self.features_per_level = features_per_level
        self.resolutions = self.planes[0].resolutions

    @property
    def output_dim(self) -> int:
        return self.planes[0].output_dim

    def forward(self, xyz: torch.Tensor, normals: torch.Tensor | None = None) -> torch.Tensor:
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must have shape (N, 3), got {tuple(xyz.shape)}")
        if normals is None:
            raise ValueError("NormalAwareTriPlane requires normals")
        if normals.shape != xyz.shape:
            raise ValueError(f"normals must match xyz shape, got {tuple(normals.shape)}")
        magnitude = normals.abs()
        if bool((magnitude.sum(dim=1) <= 1e-8).any()):
            raise ValueError("normals must be non-zero")
        axis = magnitude.argmax(dim=1)
        out = torch.zeros(
            (xyz.shape[0], self.output_dim), dtype=xyz.dtype, device=xyz.device
        )
        for a in range(3):
            mask = axis == a
            if not bool(mask.any()):
                continue
            u, v = self.AXIS_TO_PLANE[a]
            out[mask] = self.planes[a](xyz[mask][:, (u, v)])
        return out

    def capacity_report(self) -> dict:
        per_plane = [plane.capacity_report() for plane in self.planes]
        return {
            "encoding": type(self).__name__,
            "levels": per_plane[0]["levels"],
            "planes": per_plane,
            "total_slots": int(sum(p["total_slots"] for p in per_plane)),
        }
```

- [ ] **Step 4: Pass normals from the model**

In `nrp/torch_backend/model.py`, inside `forward`, replace:

```python
            normalized = ((world_coords - self.world_min) / self.world_extent).clamp(0.0, 1.0)
            spatial = self.encoding(normalized)
```

with:

```python
            normalized = ((world_coords - self.world_min) / self.world_extent).clamp(0.0, 1.0)
            if getattr(self.encoding, "needs_normals", False):
                # aux columns are albedo(3) + depth(1) + normal(3); normals are the last 3.
                normals = aux[:, 4:7]
                if self.world_origin is not None:
                    normals = normals @ self.world_basis
                spatial = self.encoding(normalized, normals)
            else:
                spatial = self.encoding(normalized)
```

Verify the aux column order before relying on the `4:7` slice:
`grep -n "albedo\|depth\|normal" nrp/torch_backend/train.py | grep -n "cat\|stack"`.
If the order differs, correct the slice — do not correct the comment.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_normal_aware_triplane -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run python -m unittest discover -s tests 2>&1 | tail -5
uv run ruff format nrp/torch_backend/encoding.py nrp/torch_backend/model.py tests/test_normal_aware_triplane.py
uv run ruff check nrp/torch_backend/encoding.py nrp/torch_backend/model.py tests/test_normal_aware_triplane.py
git add nrp/torch_backend/encoding.py nrp/torch_backend/model.py tests/test_normal_aware_triplane.py
git commit -m "feat: arm C normal-aware tri-plane encoding"
```

Expected: `OK`.

---

### Task 6: Arm A — occupancy-allocated hash control

**Files:**
- Modify: `nrp/torch_backend/encoding.py` (`HashEncoding3D.__init__`)
- Test: `tests/test_occupancy_allocation.py` (create)

**Interfaces:**
- Consumes: `nrp.torch_backend.occupancy.LevelOccupancy`
- Produces: `HashEncoding3D(..., allocation: str = "uniform", occupancy=None, slot_budget: int | None = None)`; `needs_occupancy` stays `False` because `allocation="uniform"` must keep working without a cache.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_occupancy_allocation.py`:

```python
"""Arm A control: size hash tables from measured occupancy, not a parameter budget."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import HashEncoding3D  # noqa: E402
from nrp.torch_backend.occupancy import grid_occupancy, level_resolutions  # noqa: E402

BASE = {"levels": 4, "features_per_level": 2, "base_resolution": 4, "finest_resolution": 32}


def _occ(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    res = level_resolutions(BASE["levels"], BASE["base_resolution"], BASE["finest_resolution"])
    return grid_occupancy(rng.random((n, 3)), res)


class TestUniformAllocationUnchanged(unittest.TestCase):
    def test_default_matches_committed_behaviour(self):
        enc = HashEncoding3D(table_size_log2=8, **BASE)
        for level, res in enumerate(enc.resolutions):
            expected = (res + 1) ** 3 if (res + 1) ** 3 <= 256 else 256
            self.assertEqual(enc.tables[level].shape[0], expected)


class TestOccupancyAllocation(unittest.TestCase):
    def test_slots_cover_measured_occupancy_when_budget_allows(self):
        occ = _occ()
        budget = sum(o.count for o in occ) * 2
        enc = HashEncoding3D(
            allocation="occupancy", occupancy=occ, slot_budget=budget, table_size_log2=8, **BASE
        )
        for level, o in enumerate(occ):
            self.assertGreaterEqual(enc.tables[level].shape[0], o.count)

    def test_levels_the_budget_cannot_serve_are_dropped_not_crushed(self):
        occ = _occ()
        # Budget large enough for the two coarsest levels only.
        budget = occ[0].count + occ[1].count
        enc = HashEncoding3D(
            allocation="occupancy", occupancy=occ, slot_budget=budget, table_size_log2=8, **BASE
        )
        self.assertLess(enc.levels, BASE["levels"])
        self.assertEqual(enc.output_dim, enc.levels * BASE["features_per_level"])

    def test_rejects_occupancy_allocation_without_occupancy(self):
        with self.assertRaises(ValueError):
            HashEncoding3D(allocation="occupancy", table_size_log2=8, **BASE)

    def test_rejects_unknown_allocation(self):
        with self.assertRaises(ValueError):
            HashEncoding3D(allocation="nonsense", table_size_log2=8, **BASE)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_occupancy_allocation -v`
Expected: the four `TestOccupancyAllocation` tests FAIL with `TypeError: __init__() got an
unexpected keyword argument 'allocation'`; `TestUniformAllocationUnchanged` PASSES.

- [ ] **Step 3: Implement occupancy allocation**

In `HashEncoding3D.__init__`, add these parameters after `finest_resolution`:

```python
        allocation: str = "uniform",
        occupancy=None,
        slot_budget: int | None = None,
```

and replace the table-construction loop (currently lines 126-135) with:

```python
        if allocation not in {"uniform", "occupancy"}:
            raise ValueError("allocation must be 'uniform' or 'occupancy'")
        if allocation == "occupancy" and occupancy is None:
            raise ValueError("allocation='occupancy' requires occupancy")
        self.allocation = allocation
        self.tables = nn.ParameterList()
        self._dense = []
        if allocation == "uniform":
            sizes = []
            for res in self.resolutions:
                n_vertices = (res + 1) ** 3
                dense = n_vertices <= self.table_size
                self._dense.append(dense)
                sizes.append(n_vertices if dense else self.table_size)
        else:
            # Size each level from the vertices the cache actually reads, coarsest
            # first, and drop levels the budget cannot serve rather than crushing them
            # to a few percent of their occupancy -- the failure R1 measured.
            if len(occupancy) != levels:
                raise ValueError(f"occupancy has {len(occupancy)} levels, expected {levels}")
            budget = int(slot_budget) if slot_budget is not None else self.table_size * levels
            sizes = []
            remaining = budget
            for occ in occupancy:
                want = int(occ.count)
                if want > remaining:
                    break
                sizes.append(want)
                self._dense.append(want == (occ.resolution + 1) ** 3)
                remaining -= want
            if not sizes:
                raise ValueError("slot_budget is too small to serve even the coarsest level")
            self.levels = len(sizes)
            self.resolutions = self.resolutions[: self.levels]
        for size in sizes:
            self.tables.append(
                nn.Parameter(torch.empty(size, features_per_level).uniform_(-1e-4, 1e-4))
            )
```

For `allocation="occupancy"`, `_index` must address the compacted table. Add at the top
of `_index`, before the dense branch:

```python
        if getattr(self, "allocation", "uniform") == "occupancy" and not self._dense[level]:
            hashed = (ix * _PRIMES[0]) ^ (iy * _PRIMES[1]) ^ (iz * _PRIMES[2])
            return hashed % self.tables[level].shape[0]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_occupancy_allocation -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the full suite and commit**

```bash
uv run python -m unittest discover -s tests 2>&1 | tail -5
uv run ruff format nrp/torch_backend/encoding.py tests/test_occupancy_allocation.py
uv run ruff check nrp/torch_backend/encoding.py tests/test_occupancy_allocation.py
git add nrp/torch_backend/encoding.py tests/test_occupancy_allocation.py
git commit -m "feat: arm A occupancy-allocated hash tables"
```

Expected: `OK`.

---

### Task 7: Look-at camera for the toy tracer

`nrp/toy_tracer.py:59` renders pinhole rays looking down `+z` with only a `--camera-pos`
origin override, so today it can produce a translation arc but not a rotating one. R3 needs
this regardless.

**Files:**
- Modify: `nrp/toy_tracer.py:53-74` (`_camera_rays`), `:286`, `:401`, `:461-486` (CLI)
- Test: `tests/test_toy_camera_lookat.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `_camera_rays(width, height, jitter, camera_pos=None, camera_target=None)`; `trace(..., camera_target: np.ndarray | None = None)`; CLI flag `--camera-target X Y Z`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_toy_camera_lookat.py`:

```python
"""Look-at camera orientation for the toy tracer."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.toy_tracer import CAM_POS, _camera_rays  # noqa: E402


class TestBackwardCompatibility(unittest.TestCase):
    def test_default_is_bit_identical_to_the_committed_plus_z_camera(self):
        # Committed toy caches and their tests depend on this exact ray set.
        origins, dirs = _camera_rays(8, 8, None)
        self.assertTrue(np.allclose(origins, CAM_POS))
        centre = dirs[len(dirs) // 2 + 4]
        self.assertGreater(float(centre[2]), 0.9)

    def test_explicit_target_along_plus_z_reproduces_the_default(self):
        a_o, a_d = _camera_rays(8, 8, None)
        b_o, b_d = _camera_rays(8, 8, None, camera_pos=CAM_POS, camera_target=CAM_POS + np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(a_o, b_o, atol=1e-12)
        np.testing.assert_allclose(a_d, b_d, atol=1e-12)


class TestLookAt(unittest.TestCase):
    def test_centre_ray_points_at_the_target(self):
        pos = np.array([0.5, 0.5, 0.1])
        target = np.array([0.9, 0.5, 0.9])
        _, dirs = _camera_rays(9, 9, None, camera_pos=pos, camera_target=target)
        centre = dirs[9 * 4 + 4]
        want = (target - pos) / np.linalg.norm(target - pos)
        np.testing.assert_allclose(centre, want, atol=2e-2)

    def test_all_directions_are_unit_length(self):
        _, dirs = _camera_rays(6, 6, None, camera_pos=np.array([0.2, 0.5, 0.2]), camera_target=np.array([0.8, 0.5, 0.8]))
        np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-12)

    def test_rejects_degenerate_target(self):
        pos = np.array([0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            _camera_rays(4, 4, None, camera_pos=pos, camera_target=pos)

    def test_rejects_target_parallel_to_up_axis(self):
        with self.assertRaises(ValueError):
            _camera_rays(4, 4, None, camera_pos=np.array([0.5, 0.1, 0.5]), camera_target=np.array([0.5, 0.9, 0.5]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_toy_camera_lookat -v`
Expected: the `TestLookAt` tests and `test_explicit_target_...` FAIL with
`TypeError: _camera_rays() got an unexpected keyword argument 'camera_target'`.

- [ ] **Step 3: Implement the look-at basis**

Replace `_camera_rays` in `nrp/toy_tracer.py`:

```python
#: World up axis for the look-at basis. The toy box is y-up.
CAM_UP = np.array([0.0, 1.0, 0.0])


def _camera_rays(
    width: int,
    height: int,
    jitter: np.ndarray | None,
    camera_pos: np.ndarray | None = None,
    camera_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole rays. Default looks down +z; camera_target orients the camera instead.

    jitter is (N,2) in [0,1) or None for pixel centers.
    """
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    px = xs.reshape(-1).astype(np.float64)
    py = ys.reshape(-1).astype(np.float64)
    if jitter is None:
        jx = jy = 0.5
    else:
        jx, jy = jitter[:, 0], jitter[:, 1]
    half = np.tan(np.radians(CAM_FOV_DEG) / 2.0)
    u = ((px + jx) / width * 2.0 - 1.0) * half
    v = -((py + jy) / height * 2.0 - 1.0) * half * (height / width)
    cam_pos = CAM_POS if camera_pos is None else np.asarray(camera_pos, dtype=np.float64)
    if camera_target is None:
        dirs = np.stack([u, v, np.ones_like(u)], axis=1)
    else:
        target = np.asarray(camera_target, dtype=np.float64)
        forward = target - cam_pos
        norm = np.linalg.norm(forward)
        if norm < EPS:
            raise ValueError("camera_target must differ from the camera position")
        forward = forward / norm
        if abs(float(forward @ CAM_UP)) > 1.0 - 1e-6:
            raise ValueError("camera_target must not be parallel to the up axis")
        right = np.cross(forward, CAM_UP)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        dirs = u[:, None] * right + v[:, None] * up + forward
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.broadcast_to(cam_pos, dirs.shape).copy()
    return origins, dirs
```

The default branch is byte-identical to the committed code path, which is what
`test_default_is_bit_identical_to_the_committed_plus_z_camera` guards.

- [ ] **Step 4: Thread the parameter through `trace` and the CLI**

Add `camera_target: np.ndarray | None = None` to the `trace` signature (line ~286) and to
the render helper at line ~401, and forward it to **all three** `_camera_rays` call sites
(lines ~318, ~355, ~385). Locate them with `grep -n "_camera_rays(" nrp/toy_tracer.py` and
update each; missing one produces a G-buffer that disagrees with the segments.

Add the CLI flag beside `--camera-pos`:

```python
    parser.add_argument(
        "--camera-target",
        type=float,
        nargs=3,
        default=None,
        help="aim the toy camera at this world point instead of looking down +z",
    )
```

and pass it through:

```python
        camera_target=np.asarray(args.camera_target, dtype=np.float64) if args.camera_target else None,
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_toy_camera_lookat -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Run the full suite — committed toy caches must be unaffected**

Run: `uv run python -m unittest discover -s tests 2>&1 | tail -5`
Expected: `OK`. A failure in `test_gather_light`, `test_layers`, or `test_training_smoke`
means the default camera path changed; the default branch must stay bit-identical.

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format nrp/toy_tracer.py tests/test_toy_camera_lookat.py
uv run ruff check nrp/toy_tracer.py tests/test_toy_camera_lookat.py
git add nrp/toy_tracer.py tests/test_toy_camera_lookat.py
git commit -m "feat: look-at camera orientation for the toy tracer"
```

---

### Task 8: Gate functions G1–G5

Pure functions first, separated from the expensive runner so the gate logic is testable in
milliseconds. This is the same split `examples/r1_promotion.py` uses (`promotion_gate` /
`promotion_stop_reason`).

**Files:**
- Create: `nrp/torch_backend/encoding_gates.py`
- Test: `tests/test_encoding_gates.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `g1_generalization(rows: list[dict], threshold_db: float = 1.0) -> dict`
  - `g2_capacity_context(rows: list[dict]) -> dict`
  - `g3_stability(rows: list[dict]) -> dict`
  - `g4_frame_robustness(rows: list[dict], threshold_db: float = 1.0) -> dict`
  - `g5_fallback_decomposition(rows: list[dict]) -> dict`
  - `stop_reason(gates: dict) -> str | None`
  - Every row is a dict with keys `arm`, `seed`, `camera`, `rotation_degrees`, `delta_db`, `psnr_db`, `baseline_psnr_db`, `out_of_occupancy_fraction`, `in_occupancy_psnr_db`, `out_occupancy_psnr_db`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_encoding_gates.py`:

```python
"""Gate logic for the encoding redesign, separated from the expensive runner."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding_gates import (  # noqa: E402
    g1_generalization,
    g3_stability,
    g4_frame_robustness,
    g5_fallback_decomposition,
    stop_reason,
)


def _row(**kw):
    base = {
        "arm": "world_sparse",
        "seed": 0,
        "camera": "held0",
        "rotation_degrees": 0.0,
        "delta_db": 2.0,
        "psnr_db": 22.0,
        "baseline_psnr_db": 20.0,
        "out_of_occupancy_fraction": 0.0,
        "in_occupancy_psnr_db": 22.0,
        "out_occupancy_psnr_db": None,
    }
    base.update(kw)
    return base


class TestG1(unittest.TestCase):
    def test_passes_when_every_row_clears_the_threshold(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        gate = g1_generalization(rows)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["failures"], [])

    def test_one_failing_camera_fails_the_whole_gate(self):
        rows = [_row(seed=s, camera=f"held{c}") for s in range(5) for c in range(4)]
        rows[7]["delta_db"] = 0.4
        gate = g1_generalization(rows)
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["failures"]), 1)

    def test_a_good_mean_does_not_rescue_a_failing_seed(self):
        rows = [_row(seed=0, delta_db=-5.0), _row(seed=1, delta_db=9.0)]
        gate = g1_generalization(rows)
        self.assertFalse(gate["passed"])
        self.assertGreater(gate["mean_delta_db"], 1.0)

    def test_threshold_is_inclusive(self):
        gate = g1_generalization([_row(delta_db=1.0)], threshold_db=1.0)
        self.assertTrue(gate["passed"])


class TestG3(unittest.TestCase):
    def test_reports_per_seed_pass_and_spread(self):
        rows = [_row(seed=s, delta_db=float(s)) for s in range(5)]
        gate = g3_stability(rows)
        self.assertEqual(gate["seeds_passing"], 4)  # seeds 1..4 clear 1.0 dB
        self.assertEqual(gate["seeds_total"], 5)
        self.assertIn("std_delta_db", gate)

    def test_asserts_zero_collision_for_sparse_arm(self):
        rows = [_row(seed=0)]
        gate = g3_stability(rows, collision_fractions={"world_sparse": 0.0})
        self.assertTrue(gate["collision_assertions_passed"])

    def test_nonzero_collision_for_sparse_arm_is_a_failure(self):
        rows = [_row(seed=0)]
        gate = g3_stability(rows, collision_fractions={"world_sparse": 0.01})
        self.assertFalse(gate["collision_assertions_passed"])


class TestG4(unittest.TestCase):
    def test_worst_orientation_governs(self):
        rows = [
            _row(rotation_degrees=0.0, delta_db=3.0),
            _row(rotation_degrees=90.0, delta_db=3.0),
            _row(rotation_degrees=180.0, delta_db=0.2),
        ]
        gate = g4_frame_robustness(rows)
        self.assertFalse(gate["passed"])
        self.assertAlmostEqual(gate["worst_delta_db"], 0.2)

    def test_incomplete_rotation_matrix_is_not_a_pass(self):
        rows = [_row(rotation_degrees=0.0, delta_db=3.0)]
        gate = g4_frame_robustness(rows)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["coverage_complete"])


class TestG5(unittest.TestCase):
    def test_decomposes_error_by_occupancy(self):
        rows = [
            _row(out_of_occupancy_fraction=0.1, in_occupancy_psnr_db=24.0, out_occupancy_psnr_db=15.0)
        ]
        gate = g5_fallback_decomposition(rows)
        self.assertAlmostEqual(gate["mean_out_of_occupancy_fraction"], 0.1)
        self.assertAlmostEqual(gate["mean_in_occupancy_psnr_db"], 24.0)
        self.assertAlmostEqual(gate["mean_out_occupancy_psnr_db"], 15.0)

    def test_missing_decomposition_is_flagged(self):
        rows = [_row(out_of_occupancy_fraction=0.3, out_occupancy_psnr_db=None)]
        gate = g5_fallback_decomposition(rows)
        self.assertFalse(gate["complete"])


class TestStopReason(unittest.TestCase):
    def test_no_arm_passing_g1_stops_the_track(self):
        gates = {"arms": {"world_sparse": {"g1": {"passed": False}, "g4": {"passed": False}}}}
        self.assertIsNotNone(stop_reason(gates))

    def test_a_passing_arm_clears_the_stop(self):
        gates = {"arms": {"world_sparse": {"g1": {"passed": True}, "g4": {"passed": True}}}}
        self.assertIsNone(stop_reason(gates))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_encoding_gates -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nrp.torch_backend.encoding_gates'`.

- [ ] **Step 3: Implement the gates**

Create `nrp/torch_backend/encoding_gates.py`:

```python
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


def g1_generalization(rows: list[dict], threshold_db: float = 1.0) -> dict:
    """Held-out-camera promotion gate: every seed at every camera must clear the bar.

    The baseline is the nearest trained view's pixel2d proxy reused at the held-out
    camera -- the only thing a screen-space proxy can do at a novel camera.
    """
    if not rows:
        return {"passed": False, "reason": "no rows", "failures": [], "threshold_db": threshold_db}
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
    return {
        "passed": not failures,
        "threshold_db": threshold_db,
        "failures": failures,
        "n_rows": len(rows),
        "mean_delta_db": statistics.fmean(deltas),
        "worst_delta_db": min(deltas),
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


def g3_stability(rows: list[dict], collision_fractions: dict | None = None, threshold_db: float = 1.0) -> dict:
    """Per-seed pass required; mean/std reported as context only."""
    by_seed: dict[int, list[float]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(float(row["delta_db"]))
    seeds_passing = sum(1 for deltas in by_seed.values() if min(deltas) >= threshold_db)
    deltas = _deltas(rows)
    collisions = collision_fractions or {}
    sparse_ok = all(
        value == 0.0 for arm, value in collisions.items() if arm == "world_sparse"
    )
    return {
        "seeds_total": len(by_seed),
        "seeds_passing": seeds_passing,
        "passed": bool(by_seed) and seeds_passing == len(by_seed),
        "mean_delta_db": statistics.fmean(deltas) if deltas else 0.0,
        "std_delta_db": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
        "collision_fractions": collisions,
        "collision_assertions_passed": sparse_ok,
    }


def g4_frame_robustness(rows: list[dict], threshold_db: float = 1.0) -> dict:
    """Worst orientation governs, and the full rotation matrix must be present."""
    seen = {float(row["rotation_degrees"]) for row in rows}
    coverage_complete = seen >= set(REQUIRED_ROTATIONS)
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
    inside = [row["in_occupancy_psnr_db"] for row in rows if row.get("in_occupancy_psnr_db") is not None]
    outside = [row["out_occupancy_psnr_db"] for row in rows if row.get("out_occupancy_psnr_db") is not None]
    incomplete = [
        row for row in rows
        if float(row["out_of_occupancy_fraction"]) > 0.0 and row.get("out_occupancy_psnr_db") is None
    ]
    return {
        "gated": False,
        "complete": not incomplete,
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
    for name, arm in arms.items():
        if arm.get("g1", {}).get("passed") and arm.get("g4", {}).get("passed"):
            return None
    failing = ", ".join(sorted(arms))
    return (
        f"no arm passed G1 across all seeds, cameras, and orientations ({failing}); "
        "close as a characterized negative with the G5 decomposition"
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_encoding_gates -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format nrp/torch_backend/encoding_gates.py tests/test_encoding_gates.py
uv run ruff check nrp/torch_backend/encoding_gates.py tests/test_encoding_gates.py
git add nrp/torch_backend/encoding_gates.py tests/test_encoding_gates.py
git commit -m "feat: G1-G5 gate logic for the encoding redesign"
```

---

### Task 9: Generalize `train_conditioned` to any world-anchored encoding

`train_conditioned` hard-codes `"world3d"` in three places: it raises unless
`model.spatial_encoding == "world3d"`, passes `spatial_encoding="world3d"` to `TorchNRP`,
and calls `spatial_tensors(cache, device, "world3d")`. None of the new arms can train
through it until this is lifted. It also never builds occupancy, which arms A and B need.

**Files:**
- Modify: `nrp/torch_backend/conditioned_multiview.py:293-330`
- Modify: `nrp/torch_backend/train.py:88-100` (`spatial_tensors` encoding allow-list)
- Test: `tests/test_conditioned_encodings.py` (create)

**Interfaces:**
- Consumes: `nrp.torch_backend.occupancy.cache_occupancy`, `nrp.torch_backend.encoder_registry.SPATIAL_ENCODERS`
- Produces: `train_conditioned(cfg, resume=False)` accepting any registered world-anchored `model.spatial_encoding`; `WORLD_ENCODINGS: frozenset[str]` in `nrp.torch_backend.train`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_conditioned_encodings.py`:

```python
"""Camera-conditioned training must accept every world-anchored encoding."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.conditioned_multiview import train_conditioned  # noqa: E402
from nrp.torch_backend.train import WORLD_ENCODINGS, spatial_tensors  # noqa: E402


class TestWorldEncodingSet(unittest.TestCase):
    def test_contains_every_world_arm(self):
        self.assertLessEqual(
            {"world3d", "world_triplane", "world_sparse", "world_normal_triplane"},
            set(WORLD_ENCODINGS),
        )

    def test_excludes_pixel2d(self):
        self.assertNotIn("pixel2d", WORLD_ENCODINGS)


class TestSpatialTensors(unittest.TestCase):
    def test_returns_positions_for_every_world_encoding(self):
        from tests.test_camera_machinery_audit import _tiny_cache

        import torch

        cache = _tiny_cache()
        for name in WORLD_ENCODINGS:
            spatial, aux = spatial_tensors(cache, torch.device("cpu"), name)
            self.assertEqual(spatial.shape[1], 3, name)
            self.assertEqual(aux.shape[1], 7, name)


class TestGuardRejectsPixel2d(unittest.TestCase):
    def test_pixel2d_is_still_rejected(self):
        cfg = {"model": {"camera_conditioned": True, "spatial_encoding": "pixel2d"}}
        with self.assertRaises(ValueError):
            train_conditioned(cfg)

    def test_unregistered_encoding_is_rejected(self):
        cfg = {"model": {"camera_conditioned": True, "spatial_encoding": "not_an_arm"}}
        with self.assertRaises(ValueError):
            train_conditioned(cfg)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_conditioned_encodings -v`
Expected: FAIL with `ImportError: cannot import name 'WORLD_ENCODINGS'`.

- [ ] **Step 3: Define the world-encoding set in `train.py`**

Add near the top of `nrp/torch_backend/train.py`:

```python
from .encoder_registry import SPATIAL_ENCODERS

#: Encodings that consume first-hit world positions rather than pixel coordinates.
WORLD_ENCODINGS = frozenset(name for name in SPATIAL_ENCODERS if name != "pixel2d")
```

Then in `spatial_tensors`, replace:

```python
    if spatial_encoding in {"world3d", "world_triplane"}:
```

with:

```python
    if spatial_encoding in WORLD_ENCODINGS:
```

- [ ] **Step 4: Lift the hard-coded encoding in `train_conditioned`**

In `nrp/torch_backend/conditioned_multiview.py`, replace the guard:

```python
    if model_cfg.get("spatial_encoding", "pixel2d") != "world3d":
        raise ValueError("R2 training requires model.spatial_encoding='world3d'")
```

with:

```python
    encoding_name = model_cfg.get("spatial_encoding", "pixel2d")
    if encoding_name not in WORLD_ENCODINGS:
        raise ValueError(
            "camera-conditioned training requires a world-anchored spatial encoding; "
            f"got {encoding_name!r}, expected one of {sorted(WORLD_ENCODINGS)}"
        )
```

Build occupancy for the arms that need it, immediately after `bounds = global_world_bounds(caches)`:

```python
    encoding_cfg = model_cfg.get("encoding") or {}
    encoder_cls = SPATIAL_ENCODERS[encoding_name]
    needs_occupancy = (
        getattr(encoder_cls, "needs_occupancy", False)
        or encoding_cfg.get("allocation") == "occupancy"
    )
    occupancy = None
    if needs_occupancy:
        # Occupancy spans the union of every training view, so a held-out camera
        # looking at the same surfaces lands inside the occupied set.
        stacked = np.concatenate([cache.position.reshape(-1, 3) for cache in caches], axis=0)
        occupancy = grid_occupancy(
            normalize_positions(stacked, bounds),
            level_resolutions(
                int(encoding_cfg.get("levels", 8)),
                int(encoding_cfg.get("base_resolution", 4)),
                int(encoding_cfg.get("finest_resolution", 128)),
            ),
        )
```

Change the model construction to use the selected encoding and pass occupancy:

```python
        spatial_encoding=encoding_name,
        world_bounds=bounds,
        occupancy=occupancy,
```

Change the tensor build:

```python
    spatial_rows, aux_rows = zip(
        *(spatial_tensors(cache, device, encoding_name) for cache in caches), strict=True
    )
```

Add the imports at the top of the module:

```python
from .encoder_registry import SPATIAL_ENCODERS
from .occupancy import grid_occupancy, level_resolutions, normalize_positions
from .train import WORLD_ENCODINGS
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run python -m unittest tests.test_conditioned_encodings -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Run the full suite and commit**

```bash
uv run python -m unittest discover -s tests 2>&1 | tail -5
uv run ruff format nrp/torch_backend/conditioned_multiview.py nrp/torch_backend/train.py tests/test_conditioned_encodings.py
uv run ruff check nrp/torch_backend/conditioned_multiview.py nrp/torch_backend/train.py tests/test_conditioned_encodings.py
git add nrp/torch_backend/conditioned_multiview.py nrp/torch_backend/train.py tests/test_conditioned_encodings.py
git commit -m "feat: camera-conditioned training accepts any world-anchored encoding"
```

Expected: `OK`. `tests/test_conditioned_multiview.py` must stay green — the world3d path
is unchanged, only generalized.

---

### Task 10: Cross-cutting verification

The three checks the spec requires that no single arm owns.

**Files:**
- Test: `tests/test_encoding_crosscutting.py` (create)

**Interfaces:**
- Consumes: every encoder, `nrp.torch_backend.streamed_train.spatial_tensors_for`
- Produces: nothing importable.

- [ ] **Step 1: Write the tests**

Create `tests/test_encoding_crosscutting.py`:

```python
"""Cross-cutting guarantees: corner exactness, checkpoint back-compat, streamed parity."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nrp.torch_backend.encoding import HashEncoding2D, HashEncoding3D  # noqa: E402
from nrp.torch_backend.model import TorchNRP  # noqa: E402

SMALL = {
    "levels": 2,
    "features_per_level": 2,
    "table_size_log2": 14,
    "base_resolution": 4,
    "finest_resolution": 8,
}


class TestCornerExactness(unittest.TestCase):
    """A query exactly on a grid vertex must return that vertex's feature."""

    def test_2d_corner_returns_table_entry(self):
        enc = HashEncoding2D(**SMALL)
        with torch.no_grad():
            for table in enc.tables:
                table.uniform_(-1.0, 1.0)
        res = enc.resolutions[0]
        xy = torch.tensor([[2.0 / res, 3.0 / res]], dtype=torch.float32)
        want = enc.tables[0][enc._index(torch.tensor([2]), torch.tensor([3]), 0) % enc.tables[0].shape[0]]
        got = enc(xy)[:, : enc.features_per_level]
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)

    def test_3d_corner_returns_table_entry(self):
        enc = HashEncoding3D(**SMALL)
        with torch.no_grad():
            for table in enc.tables:
                table.uniform_(-1.0, 1.0)
        res = enc.resolutions[0]
        xyz = torch.tensor([[2.0 / res, 3.0 / res, 1.0 / res]], dtype=torch.float32)
        idx = enc._index(torch.tensor([2]), torch.tensor([3]), torch.tensor([1]), 0)
        want = enc.tables[0][idx % enc.tables[0].shape[0]]
        got = enc(xyz)[:, : enc.features_per_level]
        torch.testing.assert_close(got, want, atol=1e-5, rtol=0)


class TestCheckpointBackCompat(unittest.TestCase):
    def test_committed_pixel2d_checkpoint_still_loads(self):
        path = Path(__file__).resolve().parent.parent / "out" / "toy-torch" / "model.pt"
        if not path.exists():
            self.skipTest("committed toy torch checkpoint not present")
        state = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = state["config"] if isinstance(state, dict) and "config" in state else None
        self.assertIsNotNone(cfg, "checkpoint must carry its model config")
        self.assertEqual(cfg.get("spatial_encoding", "pixel2d"), "pixel2d")

    def test_round_trip_preserves_predictions(self):
        model = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=SMALL)
        clone = TorchNRP(light_type="sphere", hidden_width=8, hidden_layers=1, encoding=SMALL)
        clone.load_state_dict(model.state_dict())
        xy, aux, params = torch.rand(4, 2), torch.rand(4, 7), torch.rand(4, 4)
        torch.testing.assert_close(model(xy, aux, params), clone(xy, aux, params))


class TestStreamedParity(unittest.TestCase):
    """S1 convention: the streamed path must agree with the in-memory path."""

    def test_spatial_tensors_for_matches_train_spatial_tensors(self):
        from tests.test_camera_machinery_audit import _tiny_cache
        from nrp.torch_backend.streamed_train import spatial_tensors_for
        from nrp.torch_backend.train import spatial_tensors

        cache = _tiny_cache()
        device = torch.device("cpu")
        for name in ("pixel2d", "world3d"):
            model = TorchNRP(
                light_type="sphere",
                hidden_width=8,
                hidden_layers=1,
                encoding=SMALL,
                spatial_encoding=name,
                world_bounds=None if name == "pixel2d" else {"min": [-1.0, -1.0, -1.0], "max": [2.0, 2.0, 2.0]},
            )
            streamed, streamed_aux = spatial_tensors_for(cache, model, device)
            direct, direct_aux = spatial_tensors(cache, device, name)
            torch.testing.assert_close(streamed, direct)
            torch.testing.assert_close(streamed_aux, direct_aux)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests**

Run: `uv run python -m unittest tests.test_encoding_crosscutting -v`
Expected: PASS (5 tests, or 4 + 1 skip if the toy checkpoint is absent).

If `test_committed_pixel2d_checkpoint_still_loads` fails on the checkpoint's key layout
rather than skipping, inspect the real structure with
`uv run python -c "import torch; print(list(torch.load('out/toy-torch/model.pt', map_location='cpu', weights_only=False)))"`
and correct the test to the real key names. That is a test-authoring correction, not a
code defect.

- [ ] **Step 3: Lint and commit**

```bash
uv run ruff format tests/test_encoding_crosscutting.py
uv run ruff check tests/test_encoding_crosscutting.py
git add tests/test_encoding_crosscutting.py
git commit -m "test: corner exactness, checkpoint back-compat, and streamed parity"
```

---

### Task 11: Experiment runner

**Files:**
- Create: `examples/r1_encoding_redesign.py`
- Modify: `mise.toml` (add the task)
- Test: `tests/test_r1_encoding_redesign.py` (create)

**Interfaces:**
- Consumes: `nrp.torch_backend.encoding_gates` (all G-functions and `stop_reason`), `nrp.toy_tracer.trace`, `nrp.torch_backend.conditioned_multiview.train_conditioned`, `examples.r1_promotion.rotation_matrix_y` and `.transform_cache`
- Produces:
  - `camera_arc(n_train: int, n_held_out: int) -> tuple[list[dict], list[dict]]` — each camera is `{"name": str, "origin": [3], "target": [3]}`
  - `nearest_trained_camera(held_out: dict, trained: list[dict]) -> dict`
  - `main(argv) -> int` writing `out/r1-encoding-redesign/report.json`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_r1_encoding_redesign.py`:

```python
"""Camera-arc construction and report shape for the encoding redesign runner."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples.r1_encoding_redesign import (  # noqa: E402
    camera_arc,
    nearest_trained_camera,
)


class TestCameraArc(unittest.TestCase):
    def test_returns_the_requested_counts(self):
        trained, held_out = camera_arc(8, 4)
        self.assertEqual(len(trained), 8)
        self.assertEqual(len(held_out), 4)

    def test_held_out_cameras_are_never_trained_cameras(self):
        trained, held_out = camera_arc(8, 4)
        trained_origins = {tuple(np.round(c["origin"], 9)) for c in trained}
        for camera in held_out:
            self.assertNotIn(tuple(np.round(camera["origin"], 9)), trained_origins)

    def test_held_out_cameras_are_interpolated_not_extrapolated(self):
        trained, held_out = camera_arc(8, 4)
        # Every held-out camera must lie inside the convex hull of the trained arc
        # along each axis; extrapolation is R3's question, not this gate's.
        lo = np.min([c["origin"] for c in trained], axis=0)
        hi = np.max([c["origin"] for c in trained], axis=0)
        for camera in held_out:
            origin = np.asarray(camera["origin"])
            self.assertTrue(np.all(origin >= lo - 1e-9))
            self.assertTrue(np.all(origin <= hi + 1e-9))

    def test_every_camera_has_a_distinct_name(self):
        trained, held_out = camera_arc(8, 4)
        names = [c["name"] for c in trained + held_out]
        self.assertEqual(len(names), len(set(names)))

    def test_targets_are_inside_the_toy_box(self):
        trained, held_out = camera_arc(8, 4)
        for camera in trained + held_out:
            target = np.asarray(camera["target"])
            self.assertTrue(np.all(target > 0.0) and np.all(target < 1.0))


class TestNearestTrainedCamera(unittest.TestCase):
    def test_picks_the_closest_origin(self):
        trained = [
            {"name": "a", "origin": [0.0, 0.0, 0.0], "target": [0.5, 0.5, 0.5]},
            {"name": "b", "origin": [1.0, 0.0, 0.0], "target": [0.5, 0.5, 0.5]},
        ]
        held = {"name": "h", "origin": [0.9, 0.0, 0.0], "target": [0.5, 0.5, 0.5]}
        self.assertEqual(nearest_trained_camera(held, trained)["name"], "b")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify the tests fail**

Run: `uv run python -m unittest tests.test_r1_encoding_redesign -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'examples.r1_encoding_redesign'`.

- [ ] **Step 3: Implement the runner**

Create `examples/r1_encoding_redesign.py`. Implement in this order, committing after the
pure functions pass their tests:

```python
"""Representation-track R1 redesign: three encoding arms against a held-out-camera gate.

Supersedes the experimental design of docs/plans/2026-07-27-r1-next-experiments.md.
The previous gate compared world-anchored encodings against a pixel2d control that is
fully dense below its finest level and one vertex per pixel at it -- a per-pixel lookup
table, optimal at single-view reconstruction and unable to render any other camera. This
runner instead measures what world anchoring is for: quality at a camera never trained on.

Writes out/r1-encoding-redesign/report.json and exits nonzero when the binding gate
fails, after writing all evidence. That nonzero exit is expected for a recorded negative.

Usage:
  uv run python examples/r1_encoding_redesign.py --out out/r1-encoding-redesign/report.json
  uv run python examples/r1_encoding_redesign.py --seeds 0 --arms world_sparse  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

from nrp.torch_backend.encoding_gates import (  # noqa: E402
    g1_generalization,
    g2_capacity_context,
    g3_stability,
    g4_frame_robustness,
    g5_fallback_decomposition,
    stop_reason,
)

#: The toy box is the unit cube; cameras sit inside it on a shallow arc around the
#: centre, all aimed at the same interior point so the sphere stays in frame.
ARC_CENTRE = np.array([0.5, 0.5, 0.55])
ARC_RADIUS = 0.42
ARC_SPAN_DEG = 70.0
ARC_HEIGHT = 0.5
ARM_NAMES = ("world_sparse", "world_normal_triplane", "world3d")


def _camera_at(angle_deg: float, name: str) -> dict:
    theta = np.radians(angle_deg)
    origin = np.array(
        [
            ARC_CENTRE[0] + ARC_RADIUS * np.sin(theta),
            ARC_HEIGHT,
            ARC_CENTRE[2] - ARC_RADIUS * np.cos(theta),
        ]
    )
    return {"name": name, "origin": origin.tolist(), "target": ARC_CENTRE.tolist()}


def camera_arc(n_train: int, n_held_out: int) -> tuple[list[dict], list[dict]]:
    """Trained cameras evenly spaced on the arc; held-out cameras strictly between them.

    Held-out cameras interpolate rather than extrapolate: extrapolation beyond the arc
    is R3's question, and mixing the two would confound this gate.
    """
    if n_train < 2:
        raise ValueError("need at least two trained cameras")
    if n_held_out < 1 or n_held_out > n_train - 1:
        raise ValueError("held-out cameras must fit strictly between trained cameras")
    angles = np.linspace(-ARC_SPAN_DEG / 2.0, ARC_SPAN_DEG / 2.0, n_train)
    trained = [_camera_at(float(a), f"train{i}") for i, a in enumerate(angles)]
    gaps = np.linspace(0, n_train - 2, n_held_out).round().astype(int)
    held_out = [
        _camera_at(float((angles[g] + angles[g + 1]) / 2.0), f"held{i}")
        for i, g in enumerate(gaps)
    ]
    return trained, held_out


def nearest_trained_camera(held_out: dict, trained: list[dict]) -> dict:
    """G1's baseline: the trained view whose pixel2d proxy gets reused at this camera."""
    origin = np.asarray(held_out["origin"], dtype=np.float64)
    return min(trained, key=lambda c: float(np.linalg.norm(np.asarray(c["origin"]) - origin)))
```

Then add the stage functions below the pure ones. `train_conditioned` accepts any
world-anchored encoding after Task 9, so the arm config is a plain dict:

```python
from nrp.gather_light import gather_lights
from nrp.metrics import psnr
from nrp.path_cache import PathCache
from nrp.toy_tracer import trace
from nrp.torch_backend.conditioned_multiview import train_conditioned
from nrp.torch_backend.relight import relight
from nrp.torch_backend.train import train as train_single

from examples.r1_promotion import rotation_matrix_y, transform_cache


def export_arc(cameras: list[dict], seed_dir: Path, args) -> dict[str, Path]:
    """Trace one cache per camera. Held-out caches are exported for evaluation only."""
    paths = {}
    seed_dir.mkdir(parents=True, exist_ok=True)
    for camera in cameras:
        out_path = seed_dir / f"{camera['name']}.npz"
        if not out_path.exists() or not args.skip_export:
            cache = trace(
                width=args.width,
                height=args.width,
                spp=args.spp,
                bounces=args.bounces,
                seed=args.trace_seed,
                camera_pos=np.asarray(camera["origin"], dtype=np.float64),
                camera_target=np.asarray(camera["target"], dtype=np.float64),
            )
            cache.validate()
            cache.save(str(out_path))
        paths[camera["name"]] = out_path
    return paths


def arm_config(arm: str, seed: int, manifest: Path, out_dir: Path, args) -> dict:
    """Config for one camera-conditioned arm. Matches train_conditioned's schema."""
    return {
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "light_type": "sphere",
        "light_bounds": {"radius_min": 0.1, "radius_max": 0.4},
        "sampling": "segments",
        "pool": {"size": 32, "replace_every": 5, "replace_count": 2},
        "denoise": {"enabled": True, "method": args.denoise_method},
        "iters": args.iters,
        "batch_pixels": 4096,
        "lr": 0.005,
        "n_val_lights": 12,
        "seed": seed,
        "device": args.device,
        "model": {
            "hidden_width": 128,
            "hidden_layers": 4,
            "camera_conditioned": True,
            "spatial_encoding": arm,
            "encoding": ARM_ENCODINGS[arm],
        },
    }


#: Per-arm encoding config. Capacity is reported by G2, never matched by parameter count --
#: parameter matching is what handed world3d a 19x slot deficit in the original R1.
ARM_ENCODINGS = {
    "world_sparse": {"levels": 8, "features_per_level": 2, "base_resolution": 4, "finest_resolution": 64},
    "world_normal_triplane": {
        "levels": 4, "features_per_level": 2, "table_size_log2": 14,
        "base_resolution": 4, "finest_resolution": 64,
    },
    "world3d": {
        "levels": 8, "features_per_level": 2, "table_size_log2": 14,
        "base_resolution": 4, "finest_resolution": 64, "allocation": "occupancy",
    },
}


def rotated_caches(paths: dict[str, Path], rotation_deg: float) -> dict[str, PathCache]:
    rotation = rotation_matrix_y(rotation_deg)
    out = {}
    for name, path in paths.items():
        cache = PathCache.load(str(path))
        out[name] = cache if rotation_deg == 0.0 else transform_cache(cache, rotation)
    return out


def evaluate_camera(
    model, arm: str, camera: dict, cache: PathCache, baseline_model, lights: list
) -> dict:
    """One gate row: conditioned proxy vs the nearest trained view's pixel2d proxy."""
    reference = gather_lights(cache, lights)
    predicted = relight(model, cache, lights)
    baseline = relight(baseline_model, cache, lights)
    row = {
        "arm": arm,
        "camera": camera["name"],
        "psnr_db": psnr(predicted, reference),
        "baseline_psnr_db": psnr(baseline, reference),
        "out_of_occupancy_fraction": 0.0,
        "in_occupancy_psnr_db": None,
        "out_occupancy_psnr_db": None,
    }
    row["delta_db"] = row["psnr_db"] - row["baseline_psnr_db"]
    encoder = model.encoding
    if hasattr(encoder, "out_of_occupancy_fraction"):
        positions = torch.as_tensor(cache.position.reshape(-1, 3), dtype=torch.float32)
        normalized = ((positions - model.world_min) / model.world_extent).clamp(0.0, 1.0)
        row["out_of_occupancy_fraction"] = encoder.out_of_occupancy_fraction(normalized)
        level = encoder.levels - 1
        res = encoder.resolutions[level]
        pos0 = torch.floor(normalized * res).long().clamp_(0, res - 1)
        _, hit = encoder._lookup(pos0, level)
        mask = hit.numpy().reshape(cache.height, cache.width)
        if mask.any():
            row["in_occupancy_psnr_db"] = psnr(predicted[mask], reference[mask])
        if (~mask).any():
            row["out_occupancy_psnr_db"] = psnr(predicted[~mask], reference[~mask])
    return row
```

`main(argv)` then loops seeds × arms × rotations × held-out cameras, collecting rows.
For each `(seed, rotation)` it exports or reuses the arc, writes a `views.json` manifest
of the **trained** cameras only, trains one conditioned proxy per arm through
`train_conditioned(arm_config(...))`, trains one `pixel2d` baseline per trained camera
through `train_single`, then calls `evaluate_camera` for each held-out camera against
`nearest_trained_camera(...)`'s baseline. Every row gets `seed` and `rotation_degrees`
stamped on it before it reaches the gates. Validation lights are frozen once per seed and
shared across all arms, matching the R1A convention.

The report structure:

```python
    report = {
        "scene": "toy_box",
        "width": args.width,
        "height": args.width,
        "spp": args.spp,
        "iters": args.iters,
        "seeds": list(args.seeds),
        "cameras": {"trained": trained, "held_out": held_out},
        "hardware": {"platform": sys.platform, "device": args.device},
        "arms": {
            arm: {
                "rows": rows_by_arm[arm],
                "capacity_report": capacity_by_arm[arm],
                "g1": g1_generalization(rows_by_arm[arm], args.threshold_db),
                "g3": g3_stability(rows_by_arm[arm], collision_by_arm, args.threshold_db),
                "g4": g4_frame_robustness(rows_by_arm[arm], args.threshold_db),
                "g5": g5_fallback_decomposition(rows_by_arm[arm]),
            }
            for arm in args.arms
        },
        "g2_capacity_context": g2_capacity_context(capacity_rows),
        "promoted": False,
    }
    report["stop_reason"] = stop_reason(report)
    report["promoted"] = report["stop_reason"] is None
```

CLI flags: `--out`, `--seeds` (nargs, default `0 1 2 3 4`), `--arms` (nargs, default
`ARM_NAMES`), `--rotations` (nargs, default `0 90 180`), `--width` (default 64), `--spp`
(default 16), `--iters` (default 3000), `--bounces` (default 4), `--trace-seed` (default 0),
`--denoise-method` (default `bilateral`), `--device` (default `cpu`),
`--threshold-db` (default 1.0), `--skip-export`.

- [ ] **Step 4: Run the runner's unit tests**

Run: `uv run python -m unittest tests.test_r1_encoding_redesign -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Smoke-run the full pipeline at tiny scale**

Run:
```bash
uv run python examples/r1_encoding_redesign.py \
  --out out/r1-encoding-redesign/smoke.json \
  --seeds 0 --arms world_sparse --rotations 0 \
  --width 16 --spp 2 --iters 50
```
Expected: completes without traceback and writes `out/r1-encoding-redesign/smoke.json`.
A nonzero exit is acceptable and expected here — 50 iterations will not clear G1, and G4
will report `coverage_complete: false` because only one rotation ran. What must not
happen is a crash, a missing report, or an empty `rows` list.

- [ ] **Step 6: Add the mise task**

In `mise.toml`, beside the existing `r1a-variance` task:

```toml
[tasks.r1-encoding-redesign]
description = "R1 redesign: three encoding arms against the held-out-camera gate"
run = "uv run python examples/r1_encoding_redesign.py --out out/r1-encoding-redesign/report.json"
```

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff format examples/r1_encoding_redesign.py tests/test_r1_encoding_redesign.py
uv run ruff check examples/r1_encoding_redesign.py tests/test_r1_encoding_redesign.py
git add examples/r1_encoding_redesign.py tests/test_r1_encoding_redesign.py mise.toml
git commit -m "feat: encoding redesign experiment runner and mise task"
```

---

### Task 12: Run the campaign and record the result

This is the only expensive task. Follow the long-job convention: `nohup` plus a pid, then
poll — a backgrounded Bash call dies at ten minutes.

**Files:**
- Create: `out/r1-encoding-redesign/report.json` (generated)
- Modify: `docs/performance.md`, `docs/representation-track.md`, `docs/tracks.md`

- [ ] **Step 1: Launch the full campaign**

```bash
mkdir -p out/r1-encoding-redesign
nohup uv run python examples/r1_encoding_redesign.py \
  --out out/r1-encoding-redesign/report.json \
  > out/r1-encoding-redesign/run.log 2>&1 &
echo $! > out/r1-encoding-redesign/run.pid
```

Poll with `tail -20 out/r1-encoding-redesign/run.log` until the process exits. Do not
use a foreground `sleep`.

- [ ] **Step 2: Read the gate outcome without editing it**

```bash
uv run python -c "
import json
r = json.load(open('out/r1-encoding-redesign/report.json'))
print('promoted:', r['promoted'])
print('stop_reason:', r['stop_reason'])
for arm, data in r['arms'].items():
    print(arm, 'G1', data['g1']['passed'], 'worst', round(data['g1']['worst_delta_db'], 3),
          '| G4', data['g4']['passed'], '| seeds', f\"{data['g3']['seeds_passing']}/{data['g3']['seeds_total']}\")
"
```

**If no arm passes:** that is the spec's stop condition. Record the negative with the G5
decomposition. Do not add a fourth arm, widen the threshold, drop a seed, or re-run with a
different rotation set. Those are the moves this redesign exists to stop.

- [ ] **Step 3: Write the results into the docs**

Add a `## World-anchored encoding redesign (representation-track rung R1)` section to
`docs/performance.md` containing the per-arm gate table, the measured collision fractions
(0.0% for `world_sparse`, asserted), the G2 capacity table including `world_sparse`'s key
bytes, the G5 out-of-occupancy fractions, and the hardware context (macOS 27 arm64,
Python 3.12.11, PyTorch 2.12.1, CPU).

In `docs/representation-track.md`, update the R1 status row to the measured outcome and add
a `## R1 redesign` section that states plainly:

1. the previous negative is now explained by the measured ~19× allocation handicap
   (78,080 distinct vertices against 4,096 slots, versus 16,641 against 16,384);
2. the gate changed from single-view parity to held-out-camera generalization, and why —
   `pixel2d` at these settings is a per-pixel lookup table;
3. the outcome of each arm against G1/G3/G4, per seed, with no averaging.

In `docs/tracks.md`, update row 7's status.

- [ ] **Step 4: Verify referenced paths exist**

Run: `mise run pipeline-audit`
Expected: passes for the new paths. A pre-existing unrelated failure is documented in the
project memory; confirm any failure is that one and not a path this task introduced.

- [ ] **Step 5: Run the full suite, lint, and commit**

```bash
uv run python -m unittest discover -s tests 2>&1 | tail -5
uv run ruff check .
git add out/r1-encoding-redesign/report.json docs/performance.md docs/representation-track.md docs/tracks.md
git commit -m "R1 redesign: record encoding-arm results against the held-out-camera gate"
```

Expected: `OK`.

---

## Notes for the implementer

- **The stop condition is real.** If Task 12 produces no passing arm, the deliverable is
  the characterized negative. Three prior campaigns were spent tuning past exactly this
  point; that is the failure mode this plan is designed around.
- **Never weaken a gate to make it pass.** Per-seed passes are required throughout, and a
  favourable mean is not evidence.
- **Task 0 is not optional.** It pins inherited behaviour that Tasks 9, 11, and 12 depend on directly.
