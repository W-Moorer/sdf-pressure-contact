# Restructure plan

## Goal

Keep only the pieces needed for:

1. loading arbitrary watertight triangle meshes,
2. evaluating reasonably accurate normal contact,
3. benchmarking evaluator accuracy,
4. optionally embedding the evaluator into simple dynamics.

## File responsibilities

### `sdf_contact/core.py`
Pure math, poses, rigid bodies, source abstraction, and contact result dataclasses.

### `sdf_contact/geometry.py`
Mesh and primitive geometry backends plus world / body factory helpers.

### `sdf_contact/evaluators.py`
Two local evaluators:
- `BaselineGridLocalEvaluator` for regression and comparison,
- `PolygonHighAccuracyLocalEvaluator` as the default main path.

### `sdf_contact/pipeline.py`
Pair traversal, wrench aggregation, optional implicit stepper.

### `sdf_contact/benchmarks.py`
Local evaluator comparison helpers and benchmark metrics.

## What to benchmark before touching the solver

- sphere–plane shallow indentation,
- eccentric sphere–plane force and moment,
- curvature scans,
- stiffness scans,
- convergence under raster / quadrature refinement,
- arbitrary mesh against a high-resolution reference evaluator.

## Definition of “more accurate”

A new evaluator should be considered better only if it improves all of the following:

- total normal force,
- moment / first moment,
- convergence with refinement,
- robustness to pose perturbations and mesh resolution,
- repeatability across equivalent formulations.

## Current limitations

The included mesh SDF backend is self-contained but brute-force and intended as a reference implementation. It is good for cleanup, correctness work, and small / medium benchmarks, but not yet for large production scenes.
