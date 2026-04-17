from __future__ import annotations

from pfcsdf.dynamics.benchmarks import FlatImpactSetup, benchmark_flat_impact_error, run_flat_impact_benchmark


def test_substep_improves_onset_timing_and_state_error() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.18,
    )
    dt = 0.04
    mid = run_flat_impact_benchmark(setup, dt=dt, scheme="midpoint")
    sub = run_flat_impact_benchmark(setup, dt=dt, scheme="midpoint_substep")

    mid_err = benchmark_flat_impact_error(setup, mid)
    sub_err = benchmark_flat_impact_error(setup, sub)

    assert sub_err.state_error < mid_err.state_error
    assert sub_err.impulse_error < mid_err.impulse_error
    assert sub.used_substeps.max() > 1
