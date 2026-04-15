# Benchmark Acceptance Standard

This document turns the current benchmark suite into an **admission standard**.
It answers one specific question:

> Under which benchmark cases do we call an evaluator "accurate enough"?

The standard is intentionally split into two profiles:

- **Research profile**: strong enough for algorithm iteration, ablation, and comparative studies.
- **Production profile**: strong enough that the remaining error is small enough to justify treating the local evaluator as a near-stable backend.

The current benchmark suite contains both **analytic-truth cases** and **reference-consistency cases**.
That distinction matters:

- If an analytic solution exists, accuracy is judged against analytic truth.
- If no analytic solution exists, accuracy is judged against a higher-resolution numerical reference and geometric sanity checks.

## Benchmark families and what they measure

### A. `centered_sphere_plane_scan`
Analytic primitive case. Measures normal force against the cap-volume reference law.

What it tells us:
- whether the force scale is physically calibrated;
- whether the evaluator preserves symmetry;
- whether the local evaluator is stable over a range of shallow penetrations.

### B. `centered_sphere_plane_convergence`
Analytic primitive case at one fixed penetration depth.

What it tells us:
- whether increasing local patch resolution actually converges;
- whether remaining error is dominated by quadrature resolution or by the force model / geometry model itself.

### C. `mesh_tilted_box_plane_scan`
General mesh case without closed-form truth.

What it tells us:
- whether the evaluator stays consistent with a higher-resolution polygon reference;
- whether both force and first moment remain stable on nontrivial geometry.

### D. `offaxis_reference_center_sensitivity`
Geometric sanity case.

What it tells us:
- whether the result is polluted by arbitrary reference-center choices;
- whether a nominally symmetric off-axis setup creates spurious horizontal forces.

## Hard rule: mandatory sanity before accuracy

An evaluator is **not accurate** if it fails geometric sanity, even if some force values look good.
Therefore, the following are always mandatory:

1. **Reference-center invariance** must pass.
2. **Centered symmetry** must pass.
3. **Resolution stability** must pass.

Only after these pass do we look at force and moment accuracy.

## Metrics

The admission script computes the following metrics.

### 1. Centered primitive force metrics
From `centered_sphere_plane_scan.csv` and `centered_sphere_plane_calibration_summary.csv`:

- `mean_rel_err_raw`: mean relative force error vs analytic cap-volume law.
- `mean_rel_err_calibrated`: mean relative error after one scalar best-fit calibration.
- `max_centered_moment_ratio`: `max(|Mz| / max(|Fy|, eps))` over centered primitive states.

Interpretation:
- raw error measures **true physical calibration**;
- calibrated error measures whether the evaluator shape is right up to a scalar stiffness rescale;
- centered moment ratio measures symmetry leakage.

### 2. Convergence metrics
From `centered_sphere_plane_convergence.csv`:

- `tail_spread_rel`: relative spread of the highest 3 resolutions.
- `last_step_rel_change`: relative change from the final two resolutions.

Interpretation:
- small values mean the local evaluator has essentially stabilized.

### 3. Mesh reference metrics
From `mesh_tilted_box_plane_scan.csv`:

- `mean_rel_err_vs_ref_Fy`
- `max_rel_err_vs_ref_Fy`
- `mean_rel_err_vs_ref_Mz`
- `max_rel_err_vs_ref_Mz`

Interpretation:
- these are not analytic-truth metrics;
- they quantify self-consistency relative to a higher-resolution polygon reference.

### 4. Off-axis invariance metrics
From `offaxis_reference_center_sensitivity.csv`:

- `max_horizontal_to_vertical_ratio`
- `fy_reference_mode_rel_diff`

Interpretation:
- the first catches spurious sideways force;
- the second catches dependence on the arbitrary plane reference center.

## Admission profiles

## Research profile

An evaluator is **Research-accurate** if it passes all of the following:

- `max_centered_moment_ratio <= 1e-3`
- `tail_spread_rel <= 2e-2`
- `last_step_rel_change <= 2e-2`
- `max_horizontal_to_vertical_ratio <= 1e-4`
- `fy_reference_mode_rel_diff <= 1e-4`
- `mean_rel_err_calibrated <= 2e-2`
- `mean_rel_err_vs_ref_Fy <= 3e-2`
- `max_rel_err_vs_ref_Fy <= 8e-2`
- `mean_rel_err_vs_ref_Mz <= 3e-2`
- `max_rel_err_vs_ref_Mz <= 8e-2`

This profile means:
- the evaluator is good enough for algorithmic comparison,
- geometry-driven failure modes are under control,
- and remaining error is small enough to support further model development.

## Production profile

An evaluator is **Production-accurate** only if it passes the stricter set:

- `max_centered_moment_ratio <= 1e-4`
- `tail_spread_rel <= 5e-3`
- `last_step_rel_change <= 5e-3`
- `max_horizontal_to_vertical_ratio <= 1e-5`
- `fy_reference_mode_rel_diff <= 1e-5`
- `mean_rel_err_raw <= 1e-1`
- `mean_rel_err_calibrated <= 1e-2`
- `mean_rel_err_vs_ref_Fy <= 1e-2`
- `max_rel_err_vs_ref_Fy <= 3e-2`
- `mean_rel_err_vs_ref_Mz <= 1e-2`
- `max_rel_err_vs_ref_Mz <= 3e-2`

This profile means:
- the evaluator is no longer only shape-correct after calibration,
- but also close enough in absolute scale and mesh consistency to be treated as a stable default backend.

## How to interpret failures

### Fails sanity, passes force error
Not acceptable. The evaluator is geometrically untrustworthy.

### Passes calibrated primitive force, fails raw primitive force
The local evaluator shape is probably reasonable, but the force law or global scale is not calibrated.

### Passes primitive cases, fails mesh reference cases
The evaluator still has geometry-generalization issues.

### Passes research, fails production
This is the expected middle state for an active research codebase.
It means:
- keep the evaluator,
- do not rewrite the global solver yet,
- improve force-law calibration and geometry handling first.

## Current intended usage

Run the benchmark suite first, then run the admission script:

```bash
python benchmarks/run_local_evaluator_benchmarks.py
python benchmarks/run_acceptance_gates.py
```

The admission report is written to:

- `results/local_benchmarks/acceptance_report.md`
- `results/local_benchmarks/acceptance_metrics.csv`
- `results/local_benchmarks/acceptance_overview.csv`
- `results/local_benchmarks/acceptance_status.json`

## Current strategic rule

Do **not** refactor the global solver until at least one evaluator clears the **Research profile** with comfortable margin.
Only consider deeper global-layer work after the evaluator is stable under this admission standard.
