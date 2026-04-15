# Formal Endpoint Final Package

This package is the most complete endpoint-style implementation produced in this session.

## What changed relative to the earlier formal-pressure package

- Added a first-class **band mechanics / local-normal accumulator** layer.
- Added a separate **sheet representation** layer recovered by measure-preserving clustering of band cells.
- Kept the shared high-accuracy outer geometry pipeline:
  - initial pair frame
  - support polygon extraction
  - triangulation
  - quadrature
  - per-column local solve
- Preserved compatibility with the existing `ContactManager` / `GlobalImplicitSystemSolver6D` interface by exposing `compute_source_pair(...)` while also providing `compute_source_pair_bundle(...)` for richer output.

## New endpoint-style objects

- `BandCellSample`
- `BandMechanicsResult`
- `SheetPatchGeometry`
- `SheetRepresentation`
- `FormalEndpointBandSheetEvaluator`

## What is now explicit in code

For each projected quadrature sample, the evaluator now computes:

- a formal local closure `delta`
- a zero-thickness sheet location `s*`
- the pressure-difference normal `grad(h) / ||grad(h)||`
- local one-dimensional accumulator scalars `I_force`, `I_area`, `I_s`
- per-cell band mechanics contributions
- a separate sheet patch reconstruction

## Remaining approximations

This package is much closer to the formal endpoint, but it still contains practical approximations:

- the outer support footprint is still driven by projected-cell activity rather than a true sparse VDB/NanoVDB narrow-band field;
- the mesh SDF backend is still a correctness-oriented brute-force backend;
- sheet clustering is geometric and local, not yet a full topology-aware patch tracker across time;
- the global implicit solver still uses finite-difference Jacobians.

## Validation

Run:

```bash
PYTHONPATH=. python benchmarks/run_endpoint_final_validation.py
```

Outputs are written to:

- `results/formal_endpoint_final_validation/`
