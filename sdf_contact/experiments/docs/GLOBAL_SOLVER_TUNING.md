# Global solver tuning after free-body contact/release benchmarks

This note documents the first solver-focused debugging pass performed after the endpoint-style contact evaluator had already passed static / quasi-static checks.

## What was changed

The contact evaluator was left unchanged. Only the global solver layer was modified.

### Legacy solver
- backward Euler position/orientation update
- force evaluated at end-of-step trial state
- forward-difference finite-difference Jacobian
- current-velocity predictor

### Tuned solver
- implicit-midpoint style evaluation state
- final pose updated from midpoint kinematics
- force evaluated at midpoint trial state
- central-difference finite-difference Jacobian
- explicit-force predictor from the current state
- slightly stronger Newton iteration budget and line search

## Why

The free-body release benchmarks showed that the old solver was strongly dissipative:
- release speeds were much smaller than the conservative reference
- energy at first release had large negative drift
- local contact force law stayed fairly reasonable, so the dominant issue was global time integration rather than the contact evaluator itself

## Result summary

See `results/solver_tuning_diagnostics/summary.md`.

In short:
- sphere centered release-speed error dropped from about 41.6% to about 9.4%
- flat punch centered release-speed error dropped from about 80.0% to about 2.6%
- normalized release-energy drift improved from about -0.90 to about +0.03 for the sphere and from about -0.92 to about -0.02 for the flat punch

This indicates that the first bottleneck in the free-body tests was indeed the global solver / time discretization, not the endpoint contact evaluator.
