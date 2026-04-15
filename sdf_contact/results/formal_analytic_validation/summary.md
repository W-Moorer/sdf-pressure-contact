# Formal endpoint analytic validation

## What was changed in code
- Added `FormalPressureFieldLocalEvaluator` on the same patch/quadrature architecture.
- Replaced legacy spring-gap traction with the formal linear pressure law `p_i = k_i d_i`.
- Per column, solved the 1D equilibrium split `k_A d_A = k_B d_B`, `d_A + d_B = delta`.
- Computed sheet normal from `grad h = -k_A grad(phi_A) + k_B grad(phi_B)` rather than `n_A - n_B` alone.
- Built formal patch support from column-overlap intervals on each projected cell, not from a single overlap slice at the center plane.

## Centered sphere-plane, equal stiffness
- legacy polygon mean relative force error: 24.0393%
- formal evaluator mean relative force error: 0.2652%
- formal evaluator max relative force error: 0.7478%
- formal evaluator max body moment norm: 3.598952e-04

## Off-axis sphere-plane, equal stiffness
- max relative force error: 0.2725%
- max |body Mx|: 3.346628e-04
- max |body Mz|: 1.323835e-04
- max relative error of ground Mz vs -x*Fy_ideal: 0.8974%

## Centered sphere-plane, unequal stiffness
- formal evaluator mean relative force error: 0.2652%
- formal evaluator max relative force error: 0.7478%
- formal evaluator max body moment norm: 6.477491e-04

## Reading of the result
- Under a single untuned evaluator configuration, centered and off-axis analytic sphere-plane cases are now within about 1% force error.
- The same code path also matches the unequal-stiffness analytic force law `keq * cap_volume` without any case-specific branching.
- This does **not** mean the full formal endpoint is finished; it shows that the central constitutive and sheet-location corrections are working inside the shared architecture.

## Still missing before the full formal endpoint
- True band mechanics with `w_eta = delta_eta(h) |grad h| chi_A chi_B` as a first-class layer, not just a column-wise sheet evaluator.
- A separate sheet-representation output layer, so force computation and patch geometry are no longer mixed.
- A full local-normal accumulator with explicit `I^(F), I^(A), I^(s)` one-dimensional integrals per sheet element.
- Better narrow-band SDF backends and BVH/block traversal for large meshes.
- Consistent derivatives / Jacobians for the global implicit solve once the local evaluator is frozen.