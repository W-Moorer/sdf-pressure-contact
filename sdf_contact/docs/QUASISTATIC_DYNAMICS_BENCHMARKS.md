# Quasi-static dynamics benchmarks

This repository now includes a first time-domain dynamics gate for the endpoint evaluator:

- centered sphere-plane press/release
- centered flat-punch-plane press/release
- off-axis sphere-plane press/release

These are **driven quasi-static** benchmarks, not free rigid-body impact tests.
The body trajectory is prescribed with a smooth press/release profile and the same endpoint evaluator configuration is used in all cases.

Why this layer exists:

1. it checks that the endpoint contact law keeps the same force/moment quality once embedded in a time-domain workflow;
2. it avoids mixing contact-law error with global free-body integration instability too early;
3. it gives a clean regression target before moving on to impact, rebound, and friction.

Run:

```bash
PYTHONPATH=. python benchmarks/run_quasistatic_dynamics_benchmarks.py
```

Outputs are written to:

```text
results/quasistatic_dynamics_benchmarks/
```

The summary file reports active-contact mean/max relative error against the matching static analytic baseline.
