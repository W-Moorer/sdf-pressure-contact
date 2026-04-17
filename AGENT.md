# AGENT.md

This repository is a **research prototype** for an **SDF-native pressure-field contact dynamics workflow**. Your job is to evolve it into a more general and better-structured system **without breaking the existing paper benchmarks, tests, and reproducible outputs**.

## 1. Mission

Promote the repository from a **benchmark-driven validation scaffold** into a **general SDF-based contact engine** with the following long-term target:

- mesh ingestion
- mesh-to-SDF geometry support
- pairwise body-vs-body narrow-band contact evaluation
- 6-DoF rigid stepping on generic geometry pairs
- preserved paper benchmark reproducibility

This is **not** yet a universal arbitrary-mesh contact solver. Treat the current code as a strong prototype with reusable core ideas, not as a finished engine.

## 2. Current reality of the codebase

### What already exists

- A substantial SDF-native contact pipeline centered on:
  - `src/pfcsdf/contact/native_band.py`
  - `src/pfcsdf/contact/active_set.py`
  - `src/pfcsdf/contact/local_normal.py`
  - `src/pfcsdf/contact/wrench.py`
- 6-DoF rigid-state and integration building blocks:
  - `src/pfcsdf/dynamics/rotation.py`
  - `src/pfcsdf/dynamics/rigid_state.py`
  - `src/pfcsdf/dynamics/rigid_integrators.py`
- Reproducible paper/benchmark structure:
  - `experiments/run_main_tables.py`
  - `experiments/run_efficiency_tables.py`
  - `experiments/run_plot_suite.py`
  - `experiments/run_complex_case.py`
  - `experiments/run_complex_case_6dof.py`
- Good regression coverage under `tests/`

### What the repository does **not** yet provide

- No generic mesh loader / importer / repair pipeline
- No true mesh-to-SDF backend
- No general body-vs-body pair contact API
- No broad phase or scene manager
- No general frictional solver
- No robust arbitrary mesh-vs-mesh contact workflow

### Important boundary

The current 6-DoF complex benchmark is still **specialized**. It uses a sampled support cloud and plane-overlap logic, rather than a full general geometry-pair contact formulation.

Relevant files:

- `src/pfcsdf/geometry/complex_bodies.py`
- `src/pfcsdf/dynamics/benchmarks_complex_6dof.py`

## 3. Product definition to optimize for

Target the following product definition:

> A general SDF-based contact engine that supports analytic primitives first, then mesh-backed SDF geometry, then generic pairwise contact evaluation, while preserving the benchmark and paper workflows.

This means:

- **do not** optimize only for short-term benchmark hacking
- **do not** throw away paper-facing scripts
- **do not** replace validated code paths unless tests remain green
- **do** isolate special-case benchmark logic from reusable engine logic

## 4. Work priorities

Follow this order unless explicitly told otherwise.

### Priority 1 — Geometry API unification

Introduce a common geometry abstraction so that existing primitives and future mesh-backed SDFs share the same interface.

Desired direction:

- create a geometry protocol/base abstraction
- standardize `signed_distance(...)`
- support `gradient(...)` where available
- support transforms/wrappers
- keep existing analytic geometry working

This is the **first task** because it is the safest structural step and unlocks later work.

### Priority 2 — Mesh ingestion and mesh-backed SDF

After the geometry API exists, add a minimal mesh pipeline:

- import OBJ/STL/PLY
- compute or store a grid SDF representation
- expose it through the same geometry interface
- add tests against sphere/box reference cases

Prefer **stable and testable** over fast and clever.

### Priority 3 — Generic pairwise contact model

Move from special-case support-profile/support-cloud logic toward a reusable narrow-phase API:

- geometry A vs geometry B
- static environment support first
- dynamic-vs-static before dynamic-vs-dynamic

### Priority 4 — Upgrade 6-DoF benchmark path

Replace the `SupportCloud3D`-specific pathway with generic geometry-pair evaluation while preserving the existing benchmark outputs as much as possible.

### Priority 5 — Scene and broad phase

Only after the narrow-phase path is stable.

### Priority 6 — Friction

Only after normal-only generic geometry contact is working and benchmarked.

## 5. Hard constraints

### Do not break these unless the task explicitly requires it

- `experiments/run_main_tables.py`
- `experiments/run_efficiency_tables.py`
- `experiments/run_plot_suite.py`
- `experiments/run_complex_case.py`
- `experiments/run_complex_case_6dof.py`
- the LaTeX paper integration files
- existing public result table formats unless intentionally versioned

### Keep the following green whenever practical

- all existing `tests/test_step_*.py`
- especially:
  - `test_step_21_native_band_flat.py`
  - `test_step_22_native_band_sphere.py`
  - `test_step_32_active_set_continuity.py`
  - `test_step_33_native_band_continuity_update.py`
  - `test_step_35_predictor_corrector_controller.py`
  - `test_step_50_complex_body_benchmark.py`
  - `test_step_56_complex_6dof_benchmark.py`

## 6. Development rules

### Rule A — Prefer additive refactors

When in doubt:

- add a new module
- add an adapter layer
- migrate call sites gradually
- avoid large rewrites in one patch

### Rule B — Preserve benchmark semantics

A benchmark-specific implementation is acceptable **only if it is isolated** and marked as such.
Do not contaminate reusable engine modules with benchmark-only shortcuts.

### Rule C — Improve types and docstrings as you go

Use dataclasses, protocols, and small focused APIs.
Document assumptions and coordinate conventions.

### Rule D — Every meaningful refactor needs tests

For any new abstraction, add or update tests.
At minimum add:

- unit tests for the new API
- at least one regression test showing old behavior still works

### Rule E — Prefer explicit acceptance criteria

Every task should state what “done” means before implementation starts.

## 7. Near-term roadmap

### Phase 1: geometry abstraction

Deliverables:

- `src/pfcsdf/geometry/base.py` or equivalent
- a `SignedDistanceGeometry` protocol or abstract base class
- adapters for existing primitives
- optional transformed geometry wrapper
- tests that verify compatibility with current signed-distance usage

Acceptance criteria:

- existing primitive-based tests still pass
- `native_band.py` can rely on the common geometry contract instead of ad hoc assumptions
- no paper scripts regress

### Phase 2: mesh-backed SDF

Deliverables:

- `src/pfcsdf/geometry/mesh.py`
- mesh load support
- grid-SDF-backed query object
- tests comparing mesh SDF against an analytic sphere or box

Acceptance criteria:

- a user can load a watertight mesh and query signed distance through the common geometry interface
- at least one mesh-vs-plane test exists

### Phase 3: pairwise narrow phase

Deliverables:

- `src/pfcsdf/contact/pairwise_native_band.py` or equivalent
- generic evaluation for geometry A vs geometry B
- migration path from special-case plane/support logic

Acceptance criteria:

- at least one current benchmark can run through the new pairwise path
- numeric behavior remains in-family with existing baselines

## 8. Non-goals for the next few tasks

Do **not** start with:

- a full scene graph
- GPU acceleration
- high-performance mesh repair
- complete Coulomb friction
- arbitrary CAD robustness
- rewriting the paper

These are later-stage concerns.

## 9. Expected working style

For each task:

1. inspect current code paths
2. restate the goal in concrete engineering terms
3. identify the smallest safe refactor
4. implement it
5. run targeted tests first
6. run broader tests if touched scope warrants it
7. summarize changed files, invariants preserved, and next recommended step

## 10. Definition of success

A successful sequence of contributions should make the answer to this question progressively more true:

> Can this repository ingest a generic mesh-backed geometry and evaluate contact through the same reusable SDF-based pipeline used by the analytic cases?

Right now the truthful answer is “not yet.”
The roadmap should move it toward “yes” without losing the reproducible evidence already present in the repository.
