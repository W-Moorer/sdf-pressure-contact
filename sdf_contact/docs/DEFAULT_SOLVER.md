# Default solver policy

The repository now uses the **tuned solver** as the default global implicit solver.

## Default integrator settings

`IntegratorConfig()` now defaults to:

- `scheme = implicit_midpoint`
- `jacobian_mode = central`
- `predictor_mode = explicit_force`
- stronger backtracking line search
- `adaptive_substepping = True`

The external API did not change. Existing code that constructs `GlobalImplicitSystemSolver6D(contact_manager)` without a custom config now gets the tuned midpoint solver automatically.

## Why this is now the default

Free-body contact/release benchmarks showed that the old backward-Euler-style default was the main source of artificial energy loss during release, even when the local endpoint contact evaluator was already accurate.

With the tuned midpoint default:

- release-speed error on centered sphere free release dropped substantially,
- normalized energy drift at first release moved close to zero,
- the same free-body benchmarks were solved without requiring case-specific solver branches.

## Adaptive substepping policy

Adaptive substepping is enabled by default, but it is **internal** to a single outer time step.

That means:

- the public time step remains `cfg.dt`,
- the solver may internally split a hard step into two half-steps,
- benchmark scripts can still use one fixed nominal `dt` while getting a more robust default solver.

In the current free-body regression set, the improved midpoint/Newton default already keeps residuals small enough that substepping is rarely triggered. This is still valuable because it means the default path is both more accurate **and** not relying on aggressive step splitting to appear stable.
