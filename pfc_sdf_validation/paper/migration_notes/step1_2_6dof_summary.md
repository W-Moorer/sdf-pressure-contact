# 6-DoF foundation (Step 1 + Step 2)

This drop adds the first 6-DoF foundation layer without changing the existing SDF-native contact pipeline.

## Added modules

- `src/pfcsdf/dynamics/rotation.py`
- `src/pfcsdf/dynamics/rigid_state.py`
- `src/pfcsdf/dynamics/rigid_integrators.py`
- `src/pfcsdf/contact/wrench.py` (reference-shift helpers)

## Added tests

- `tests/test_step_53_rigid_state_6dof.py`
- `tests/test_step_54_rigid_wrench_integration.py`

## What is now possible

- represent a rigid body on `SE(3)` using position + rotation matrix,
- compute world inertia, angular momentum, and kinetic energy,
- advance rigid-body translation and rotation from a distributed `PairWrench`,
- compare midpoint vs semi-implicit rigid-body stepping,
- shift a wrench consistently between reference points.

## What is intentionally not included yet

- full wrench-aware continuity/controller,
- 6-DoF complex benchmark,
- rigid-body reference comparison and long-horizon plots.
