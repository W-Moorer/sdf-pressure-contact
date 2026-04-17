from __future__ import annotations

import numpy as np

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import FlatImpactSetup, NativeBandFlatContactModel, run_flat_impact_benchmark
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def _build_native_model(*, use_continuity_warm_start: bool, boundary_only_update: bool) -> NativeBandFlatContactModel:
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
        use_continuity_warm_start=use_continuity_warm_start,
        boundary_only_update=boundary_only_update,
        continuity_dilation_radius=1,
    )


def test_continuity_driven_native_band_reduces_candidate_traversal_without_force_drift() -> None:
    dense_model = _build_native_model(use_continuity_warm_start=False, boundary_only_update=False)
    continuity_model = _build_native_model(use_continuity_warm_start=True, boundary_only_update=True)
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0 * continuity_model.effective_area,
        t_final=0.12,
    )

    hist_dense = run_flat_impact_benchmark(setup, dt=0.01, scheme="event_aware_midpoint", model=dense_model)
    hist_cont = run_flat_impact_benchmark(setup, dt=0.01, scheme="event_aware_midpoint", model=continuity_model)

    active_steps = hist_cont.active_measure > 0.0
    assert np.any(active_steps)

    dense_candidates = hist_dense.candidate_count[active_steps]
    cont_candidates = hist_cont.candidate_count[active_steps]
    assert np.nanmean(cont_candidates) < np.nanmean(dense_candidates)

    rel_force_diff = abs(hist_cont.forces[-1] - hist_dense.forces[-1]) / max(abs(hist_dense.forces[-1]), 1e-12)
    assert rel_force_diff < 0.05
