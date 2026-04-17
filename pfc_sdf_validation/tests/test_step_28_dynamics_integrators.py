from __future__ import annotations

from pfcsdf.dynamics.benchmarks import FlatImpactSetup, benchmark_flat_impact_error, run_flat_impact_benchmark


def test_midpoint_beats_semi_implicit_on_flat_impact() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.22,
    )
    dt = 0.03
    semi = run_flat_impact_benchmark(setup, dt=dt, scheme="semi_implicit")
    mid = run_flat_impact_benchmark(setup, dt=dt, scheme="midpoint")

    semi_err = benchmark_flat_impact_error(setup, semi)
    mid_err = benchmark_flat_impact_error(setup, mid)

    assert mid_err.state_error < semi_err.state_error
    assert mid_err.impulse_error <= semi_err.impulse_error + 1e-12
