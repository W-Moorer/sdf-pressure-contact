from __future__ import annotations

import numpy as np

from pfcsdf.contact.active_set import ActiveSetSnapshot
from pfcsdf.dynamics.benchmarks import FlatImpactSetup, NativeBandFlatContactModel, run_flat_impact_benchmark
from pfcsdf.dynamics.events import EventAwareControllerConfig
from pfcsdf.dynamics.integrators import event_aware_midpoint_step
from pfcsdf.dynamics.state import ModelEvaluation, NormalDynamicsState
from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


class PredictorCorrectorMockModel:
    def __init__(self) -> None:
        self.empty = ActiveSetSnapshot.empty((3, 3, 3))
        mask_predictor = np.zeros((3, 3, 3), dtype=bool)
        mask_predictor[1, 1, 1] = True
        self.predictor = ActiveSetSnapshot(mask_predictor, measure=1.0)
        mask_corrector = np.zeros((3, 3, 3), dtype=bool)
        mask_corrector[1, 2, 1] = True
        self.corrector = ActiveSetSnapshot(mask_corrector, measure=1.0)
        self.calls: list[dict] = []

    def evaluate_state(self, state: NormalDynamicsState, *, warm_start_snapshot=None, boundary_only_update=None, repair_mask=None):
        self.calls.append(
            {
                "time": state.time,
                "warm_start_snapshot": warm_start_snapshot,
                "boundary_only_update": boundary_only_update,
                "repair_mask": None if repair_mask is None else np.asarray(repair_mask, dtype=bool).copy(),
            }
        )
        if warm_start_snapshot is None:
            if state.time <= 1e-14:
                return ModelEvaluation(force=1.0, acceleration=0.0, active=True, active_measure=1.0, active_snapshot=self.predictor)
            return ModelEvaluation(force=1.0, acceleration=0.0, active=True, active_measure=1.0, active_snapshot=self.predictor)
        if repair_mask is None:
            return ModelEvaluation(force=1.0, acceleration=0.0, active=True, active_measure=1.0, active_snapshot=self.corrector)
        return ModelEvaluation(force=1.0, acceleration=0.0, active=True, active_measure=1.0, active_snapshot=self.corrector)



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



def test_predictor_snapshot_warms_corrector_and_triggers_local_repair() -> None:
    model = PredictorCorrectorMockModel()
    state = NormalDynamicsState(time=0.0, gap=1.0, velocity=0.0, mass=1.0)
    controller = EventAwareControllerConfig(
        max_depth=0,
        use_predictor_corrector_continuity=True,
        predictor_corrector_jaccard_tol=0.0 + 1e-12,
        predictor_corrector_mismatch_fraction_tol=1.0 - 1e-12,
    )

    result = event_aware_midpoint_step(state, 0.1, model, controller=controller)

    assert len(model.calls) == 4  # start, predictor, corrector, local repair
    assert model.calls[2]["warm_start_snapshot"] is model.predictor
    assert model.calls[2]["boundary_only_update"] is True
    assert model.calls[3]["repair_mask"] is not None
    assert result.diagnostics.predictor_corrector_jaccard == 0.0
    assert result.diagnostics.predictor_corrector_mismatch_fraction == 1.0



def test_predictor_corrector_mismatch_can_trigger_substep_when_enabled() -> None:
    model = PredictorCorrectorMockModel()
    state = NormalDynamicsState(time=0.0, gap=1.0, velocity=0.0, mass=1.0)

    disabled = event_aware_midpoint_step(
        state,
        0.1,
        model,
        controller=EventAwareControllerConfig(max_depth=1, use_predictor_corrector_continuity=False),
    )
    enabled = event_aware_midpoint_step(
        state,
        0.1,
        PredictorCorrectorMockModel(),
        controller=EventAwareControllerConfig(
            max_depth=1,
            use_predictor_corrector_continuity=True,
            predictor_corrector_jaccard_tol=0.5,
            predictor_corrector_mismatch_fraction_tol=0.5,
        ),
    )

    assert disabled.diagnostics.used_substeps == 1
    assert enabled.diagnostics.used_substeps > 1



def test_native_band_predictor_corrector_controller_exposes_finite_mismatch_metrics_without_force_drift() -> None:
    setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=64.0,
        t_final=0.08,
    )
    dense_controller = EventAwareControllerConfig(max_depth=6, use_predictor_corrector_continuity=False)
    pc_controller = EventAwareControllerConfig(
        max_depth=6,
        use_predictor_corrector_continuity=True,
        predictor_corrector_jaccard_tol=0.5,
        predictor_corrector_mismatch_fraction_tol=0.6,
        predictor_force_relative_jump_tol=0.8,
        predictor_active_measure_relative_jump_tol=0.8,
    )

    hist_off = run_flat_impact_benchmark(setup, dt=0.01, scheme="event_aware_midpoint", model=_build_native_model(), controller=dense_controller)
    hist_on = run_flat_impact_benchmark(setup, dt=0.01, scheme="event_aware_midpoint", model=_build_native_model(), controller=pc_controller)

    active_steps = hist_on.active_measure > 0.0
    assert np.any(active_steps)
    assert np.all(np.isfinite(hist_on.predictor_corrector_jaccard[active_steps]))
    assert np.all(np.isfinite(hist_on.predictor_corrector_mismatch_fraction[active_steps]))

    rel_force_diff = abs(hist_on.forces[-1] - hist_off.forces[-1]) / max(abs(hist_off.forces[-1]), 1e-12)
    assert rel_force_diff < 0.02
