from __future__ import annotations

from pfcsdf.dynamics.benchmarks import FlatImpactSetup, benchmark_flat_impact_error, run_flat_impact_benchmark


def test_event_aware_midpoint_is_no_worse_than_midpoint_substep() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.18,
    )
    dt = 0.04
    legacy = run_flat_impact_benchmark(setup, dt=dt, scheme="midpoint_substep")
    unified = run_flat_impact_benchmark(setup, dt=dt, scheme="event_aware_midpoint")

    legacy_err = benchmark_flat_impact_error(setup, legacy)
    unified_err = benchmark_flat_impact_error(setup, unified)

    assert unified.used_substeps.max() > 1
    assert unified_err.state_error <= legacy_err.state_error + 1e-12
    assert unified_err.impulse_error <= legacy_err.impulse_error + 1e-12


def test_event_aware_midpoint_beats_plain_midpoint_on_onset_sensitive_case() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.18,
    )
    dt = 0.04
    plain = run_flat_impact_benchmark(setup, dt=dt, scheme="midpoint")
    unified = run_flat_impact_benchmark(setup, dt=dt, scheme="event_aware_midpoint")

    plain_err = benchmark_flat_impact_error(setup, plain)
    unified_err = benchmark_flat_impact_error(setup, unified)

    assert unified_err.state_error < plain_err.state_error
    assert unified_err.impulse_error < plain_err.impulse_error
