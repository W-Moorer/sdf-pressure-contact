from __future__ import annotations

import numpy as np

from pfcsdf.dynamics.benchmarks import (
    SphereImpactSetup,
    benchmark_sphere_impact_error,
    run_sphere_impact_benchmark,
    sphere_reference_history,
)
from pfcsdf.dynamics.events import EventAwareControllerConfig


def test_sphere_long_horizon_crosses_release_and_work_consistent_is_more_stable() -> None:
    setup = SphereImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        sphere_radius=1.0,
        sphere_stiffness=12.0,
        plane_stiffness=18.0,
        t_final=1.50,
    )
    reference = sphere_reference_history(setup)
    assert reference.release_time is not None
    assert setup.t_final > reference.release_time

    controller = EventAwareControllerConfig(max_depth=4, work_mismatch_relative_tol=0.02)
    hist_mid = run_sphere_impact_benchmark(setup, dt=0.05, scheme="event_aware_midpoint", controller=controller)
    hist_work = run_sphere_impact_benchmark(setup, dt=0.05, scheme="event_aware_midpoint_work_consistent", controller=controller)

    err_mid = benchmark_sphere_impact_error(setup, hist_mid)
    err_work = benchmark_sphere_impact_error(setup, hist_work)

    assert err_work.peak_force_error < err_mid.peak_force_error
    assert abs(hist_work.energy_drift) < abs(hist_mid.energy_drift)
    assert err_work.release_timing_error < err_mid.release_timing_error
    assert np.isfinite(err_work.rebound_velocity_error)
