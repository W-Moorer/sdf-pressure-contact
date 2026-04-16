# Release Mainline

This cleaned package keeps only the currently best-supported mainline in the repository root.

## Mainline kept in root

- `sdf_contact/`
  - current formal-endpoint-style contact evaluator
  - flat-support area prior for large flat contacts
  - tuned default midpoint solver with adaptive substepping
- `benchmarks/`
  - local accuracy benchmarks
  - acceptance gates
  - formal analytic validation
  - endpoint final validation
  - quasi-static dynamics benchmarks
  - free-body dynamics benchmarks
  - flat-support-area-prior regression
- `results/`
  - current mainline benchmark results and acceptance outputs

## Moved to `experiments/`

These branches were useful diagnostics but are **not** the recommended default path because they did not consistently beat the current `step_end` reporting strategy or the current default solver:

- solver tuning diagnostics artifacts
- onset-event localized force reporting
- onset-aligned force reporting
- onset micro-step force reconstruction
- early-active interval average-force reporting
- equivalent contact state reporting
- impulse-matched contact state reporting

They remain preserved under `experiments/` for future research or reference.

## Recommended default strategy

Use the repository root as the authoritative implementation:

1. formal endpoint evaluator
2. flat support area prior enabled for large flat contacts
3. tuned default solver (midpoint + central FD + adaptive)
4. `step_end` as the current best first-contact reporting quantity

## What is still research-grade

The repository remains research-grade rather than production-complete. The most likely next improvements are:

- better early-active / first-contact comparison metrics
- faster mesh SDF backend
- larger-scale mesh regressions
- more rigorous dynamic theory-to-simulation comparison for first-contact events
