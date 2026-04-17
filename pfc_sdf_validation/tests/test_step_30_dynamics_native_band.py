from __future__ import annotations

import numpy as np

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import (
    AnalyticLinearFlatContactModel,
    FlatImpactSetup,
    NativeBandFlatContactModel,
    benchmark_flat_impact_error,
    run_flat_impact_benchmark,
)
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
    )


def test_native_band_contact_model_matches_analytic_force_scale() -> None:
    model_native = _build_native_model()
    k_eq = 100.0
    model_exact = AnalyticLinearFlatContactModel(mass=1.0, contact_stiffness=k_eq * model_native.effective_area)

    gap = -0.05
    f_native = model_native.force(gap)
    f_exact = model_exact.force(gap)
    rel = abs(f_native - f_exact) / f_exact
    assert rel < 0.1


def test_native_band_midpoint_substep_runs_dynamic_flat_benchmark() -> None:
    native_model = _build_native_model()
    k_eq = 100.0
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=k_eq * native_model.effective_area,
        t_final=0.14,
    )
    hist = run_flat_impact_benchmark(setup, dt=0.02, scheme="midpoint_substep", model=native_model)
    err = benchmark_flat_impact_error(setup, hist)

    assert hist.onset_time is not None
    assert err.onset_timing_error < 0.03
    assert err.force_error < 0.2
