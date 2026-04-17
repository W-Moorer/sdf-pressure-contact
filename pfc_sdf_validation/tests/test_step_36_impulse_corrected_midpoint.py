from __future__ import annotations

import numpy as np

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import FlatImpactSetup, NativeBandFlatContactModel, benchmark_flat_impact_error, run_flat_impact_benchmark
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw



def _build_native_model() -> NativeBandFlatContactModel:
    area_side_stiffness = 200.0
    return NativeBandFlatContactModel(
        mass=1.0,
        grid=UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(5, 5, 21)),
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



def test_impulse_corrected_event_aware_midpoint_improves_analytic_flat_impact() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.18,
    )
    dt = 0.04
    baseline = run_flat_impact_benchmark(setup, dt=dt, scheme="event_aware_midpoint")
    corrected = run_flat_impact_benchmark(setup, dt=dt, scheme="event_aware_midpoint_impulse_corrected")

    baseline_err = benchmark_flat_impact_error(setup, baseline)
    corrected_err = benchmark_flat_impact_error(setup, corrected)

    assert corrected_err.impulse_error < baseline_err.impulse_error
    assert corrected_err.force_error < baseline_err.force_error
    assert corrected_err.state_error < baseline_err.state_error



def test_impulse_corrected_scheme_preserves_native_band_dynamic_response() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=64.0,
        t_final=0.08,
    )
    baseline = run_flat_impact_benchmark(setup, dt=0.01, scheme="event_aware_midpoint", model=_build_native_model())
    corrected = run_flat_impact_benchmark(
        setup,
        dt=0.01,
        scheme="event_aware_midpoint_impulse_corrected",
        model=_build_native_model(),
    )

    rel_force_diff = abs(corrected.forces[-1] - baseline.forces[-1]) / max(abs(baseline.forces[-1]), 1e-12)
    rel_impulse_diff = abs(corrected.impulse - baseline.impulse) / max(abs(baseline.impulse), 1e-12)

    assert rel_force_diff < 1e-12
    assert rel_impulse_diff < 1e-12
