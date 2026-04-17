from __future__ import annotations

import numpy as np

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import FlatImpactSetup, NativeBandFlatContactModel, benchmark_flat_impact_error, run_flat_impact_benchmark
from pfcsdf.dynamics.events import EventAwareControllerConfig
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def _build_native_model() -> NativeBandFlatContactModel:
    area_side_stiffness = 200.0
    return NativeBandFlatContactModel(
        mass=1.0,
        grid=UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(4, 4, 20)),
        footprint=BoxFootprint(0.8, 0.8),
        law_a=LinearPressureLaw(area_side_stiffness),
        law_b=LinearPressureLaw(area_side_stiffness),
        config=NativeBandAccumulatorConfig(eta=8.0, band_half_width=12.0),
        max_depth_a=0.2,
        max_depth_b=0.2,
        use_continuity_warm_start=True,
        boundary_only_update=True,
        continuity_dilation_radius=1,
    )



def test_work_consistent_scheme_improves_energy_and_state_on_analytic_flat_impact() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.18,
    )
    dt = 0.04
    impulse_corrected = run_flat_impact_benchmark(setup, dt=dt, scheme="event_aware_midpoint_impulse_corrected")
    work_consistent = run_flat_impact_benchmark(setup, dt=dt, scheme="event_aware_midpoint_work_consistent")

    impulse_err = benchmark_flat_impact_error(setup, impulse_corrected)
    work_err = benchmark_flat_impact_error(setup, work_consistent)

    assert abs(work_consistent.energy_drift) < abs(impulse_corrected.energy_drift)
    assert work_err.state_error < impulse_err.state_error
    assert work_err.force_error < impulse_err.force_error
    assert np.isfinite(np.nanmax(work_consistent.work_mismatch))



def test_work_mismatch_controller_can_trigger_additional_substeps() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.18,
    )
    relaxed = run_flat_impact_benchmark(
        setup,
        dt=0.04,
        scheme="event_aware_midpoint_work_consistent",
        controller=EventAwareControllerConfig(max_depth=10, work_mismatch_relative_tol=0.05),
    )
    strict = run_flat_impact_benchmark(
        setup,
        dt=0.04,
        scheme="event_aware_midpoint_work_consistent",
        controller=EventAwareControllerConfig(max_depth=10, work_mismatch_relative_tol=1e-3),
    )

    assert np.max(strict.used_substeps) >= np.max(relaxed.used_substeps)
    assert abs(strict.energy_drift) <= abs(relaxed.energy_drift)



def test_work_consistent_scheme_preserves_native_band_response_and_reports_work_metric() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=64.0,
        t_final=0.08,
    )
    controller = EventAwareControllerConfig(max_depth=6, work_mismatch_relative_tol=0.01)
    impulse_corrected = run_flat_impact_benchmark(
        setup,
        dt=0.01,
        scheme="event_aware_midpoint_impulse_corrected",
        model=_build_native_model(),
        controller=controller,
    )
    work_consistent = run_flat_impact_benchmark(
        setup,
        dt=0.01,
        scheme="event_aware_midpoint_work_consistent",
        model=_build_native_model(),
        controller=controller,
    )

    rel_force_diff = abs(work_consistent.forces[-1] - impulse_corrected.forces[-1]) / max(abs(impulse_corrected.forces[-1]), 1e-12)
    impulse_err = benchmark_flat_impact_error(setup, impulse_corrected)
    work_err = benchmark_flat_impact_error(setup, work_consistent)

    assert rel_force_diff < 1e-4
    assert work_err.force_error <= impulse_err.force_error + 1e-8
    assert work_err.state_error <= impulse_err.state_error + 1e-4
    assert np.isfinite(np.nanmax(work_consistent.work_mismatch))
