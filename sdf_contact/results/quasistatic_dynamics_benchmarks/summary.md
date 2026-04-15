# Quasi-static dynamics benchmarks

These are **time-domain driven quasi-static benchmarks** for the endpoint evaluator.
The body pose follows a smooth press/release trajectory, and the same endpoint evaluator configuration is used in all cases.
This benchmark layer is intentionally used before free rigid-body impact tests so that we can verify that the endpoint contact law still matches the static analytic baseline inside a time-domain workflow.

## Shared evaluator configuration
- patch raster_cells = 18
- patch support_radius_floor_scale = 0.9
- sheet bisection_steps = 18
- equal body/ground stiffness = 12000.0
- damping_gamma = 0.0
- no case-specific evaluator branching was used

## Centered sphere press/release
- active-step mean relative Fy error vs static analytic baseline: 0.2567%
- active-step max relative Fy error vs static analytic baseline: 0.3478%
- max penetration depth: 0.035000
- max |vy|: 0.125664

## Centered flat punch press/release
- active-step mean relative Fy error vs static analytic baseline: 0.9167%
- active-step max relative Fy error vs static analytic baseline: 0.9168%
- max penetration depth: 0.012000
- max |vy|: 0.050265

## Off-axis sphere press/release
- active-step mean relative Fy error vs static analytic baseline: 0.2567%
- active-step max relative Fy error vs static analytic baseline: 0.3478%
- active-step mean relative ground Mz error vs static analytic baseline: 0.2933%
- active-step max relative ground Mz error vs static analytic baseline: 0.3062%
- max penetration depth: 0.035000
- max |vy|: 0.125664

## Reading of the result
- If these time-domain driven cases stay close to the static analytic baseline, it shows that the endpoint evaluator keeps its static correctness when moved into a benchmark workflow with trajectories and velocities.
- This is the right first dynamics gate before debugging free-body impacts, rebounds, and friction.
- These are not yet free rigid-body impact benchmarks; they are deliberate quasi-static dynamic regressions used to isolate contact-law correctness from global time-integration instability.