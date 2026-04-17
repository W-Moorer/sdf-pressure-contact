from __future__ import annotations

import numpy as np

from pfcsdf.contact.active_set import ActiveSetContinuityState, ActiveSetSnapshot, continuity_report
from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import FlatImpactSetup, NativeBandFlatContactModel, run_flat_impact_benchmark
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
    )


def test_active_set_continuity_report_tracks_retained_cells() -> None:
    prev_mask = np.zeros((2, 2, 2), dtype=bool)
    prev_mask[0, 0, 0] = True
    prev_mask[1, 0, 0] = True
    curr_mask = np.zeros((2, 2, 2), dtype=bool)
    curr_mask[1, 0, 0] = True
    curr_mask[1, 1, 0] = True

    state = ActiveSetContinuityState(previous=ActiveSetSnapshot(prev_mask, measure=2.0), step_index=1)
    next_state, report = continuity_report(state, ActiveSetSnapshot(curr_mask, measure=2.5))

    assert next_state.step_index == 2
    assert report.retained_count == 1
    assert report.entered_count == 1
    assert report.exited_count == 1
    assert 0.0 < report.jaccard_index < 1.0
    assert report.relative_measure_jump > 0.0
    assert np.array_equal(report.warm_start_mask, prev_mask)


def test_native_band_benchmark_records_continuity_history() -> None:
    model = _build_native_model()
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0 * model.effective_area,
        t_final=0.10,
    )
    hist = run_flat_impact_benchmark(setup, dt=0.01, scheme="event_aware_midpoint", model=model)

    finite = hist.continuity_jaccard[np.isfinite(hist.continuity_jaccard)]
    assert finite.size > 0
    assert np.nanmax(hist.active_measure) > 0.0
    assert np.all((finite >= 0.0) & (finite <= 1.0))
