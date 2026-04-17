from __future__ import annotations

import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    build_higher_order_sparse_active_traversal,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.dynamics.benchmarks import (
    FlatImpactSetup,
    NativeBandFlatContactModel,
    benchmark_flat_impact_error,
    run_flat_impact_benchmark,
)
from pfcsdf.geometry.primitives import BoxFootprint, PlaneSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def _build_flat_fields(*, overlap: float, k_a: float, k_b: float):
    dx = dy = 0.08
    dz = 0.02
    nx, ny, nz = 14, 10, 21
    grid = UniformGrid3D(
        origin=np.array([-0.5 * nx * dx, -0.5 * ny * dy, -0.5 * nz * dz]),
        spacing=np.array([dx, dy, dz]),
        shape=(nx, ny, nz),
    )
    sdf_a = PlaneSDF(point=np.array([0.0, 0.0, -0.5 * overlap]), normal=np.array([0.0, 0.0, -1.0]))
    sdf_b = PlaneSDF(point=np.array([0.0, 0.0, 0.5 * overlap]), normal=np.array([0.0, 0.0, 1.0]))
    fields = sample_linear_pfc_balance_fields(
        grid,
        sdf_a,
        sdf_b,
        LinearPressureLaw(k_a),
        LinearPressureLaw(k_b),
        max_depth_a=1.0,
        max_depth_b=1.0,
    )
    return grid, fields


def _build_native_model(*, consistent: bool) -> NativeBandFlatContactModel:
    return NativeBandFlatContactModel(
        mass=1.0,
        grid=UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(4, 4, 20)),
        footprint=BoxFootprint(0.8, 0.8),
        law_a=LinearPressureLaw(200.0),
        law_b=LinearPressureLaw(200.0),
        config=NativeBandAccumulatorConfig(eta=8.0, band_half_width=12.0),
        max_depth_a=0.2,
        max_depth_b=0.2,
        consistent_traction_reconstruction=consistent,
    )


def test_projected_pressure_reconstruction_matches_local_normal_equilibrium_on_flat_contact() -> None:
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    _, fields = _build_flat_fields(overlap=overlap, k_a=k_a, k_b=k_b)
    traversal = build_higher_order_sparse_active_traversal(
        fields,
        NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4),
        local_normal_correction=True,
    )
    first_cell = traversal.active_cells[0]
    for sample in first_cell.cubature_samples:
        assert math.isclose(
            sample.pressure_projected_linearized,
            sample.pressure_local_normal,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        assert math.isclose(sample.projected_overlap, overlap, rel_tol=1e-12, abs_tol=1e-12)


def test_consistent_traction_reconstruction_improves_native_band_dynamic_impulse_without_hurting_force() -> None:
    baseline_model = _build_native_model(consistent=False)
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0 * baseline_model.effective_area,
        t_final=0.14,
    )

    hist_baseline = run_flat_impact_benchmark(
        setup,
        dt=0.02,
        scheme="event_aware_midpoint_work_consistent",
        model=_build_native_model(consistent=False),
    )
    err_baseline = benchmark_flat_impact_error(setup, hist_baseline)

    hist_consistent = run_flat_impact_benchmark(
        setup,
        dt=0.02,
        scheme="event_aware_midpoint_work_consistent",
        model=_build_native_model(consistent=True),
    )
    err_consistent = benchmark_flat_impact_error(setup, hist_consistent)

    assert err_consistent.force_error <= err_baseline.force_error + 5e-4
    assert err_consistent.impulse_error <= err_baseline.impulse_error + 5e-4
