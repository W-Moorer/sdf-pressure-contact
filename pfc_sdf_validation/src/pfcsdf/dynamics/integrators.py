from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from pfcsdf.contact.active_set import ActiveSetSnapshot, active_set_mismatch_report, dilate_mask

from .events import (
    EventAwareControllerConfig,
    EventAwareIndicators,
    event_aware_indicators,
    predict_contact_event,
    should_substep_event_aware,
)
from .state import IntegrationStepResult, ModelEvaluation, NormalDynamicsState, StepDiagnostics


class NormalAccelerationModel(Protocol):
    def __call__(self, state: NormalDynamicsState) -> tuple[float, float, bool] | ModelEvaluation:
        """Return either (force, acceleration, active) or a ModelEvaluation."""



def _normalize_evaluation(raw: tuple[float, float, bool] | ModelEvaluation) -> ModelEvaluation:
    if isinstance(raw, ModelEvaluation):
        return raw
    force, acceleration, active = raw
    return ModelEvaluation(force=float(force), acceleration=float(acceleration), active=bool(active))



def _evaluate_model(
    model: NormalAccelerationModel,
    state: NormalDynamicsState,
    *,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
    boundary_only_update: bool | None = None,
    repair_mask: np.ndarray | None = None,
) -> ModelEvaluation:
    if hasattr(model, "evaluate_state"):
        raw = model.evaluate_state(
            state,
            warm_start_snapshot=warm_start_snapshot,
            boundary_only_update=boundary_only_update,
            repair_mask=repair_mask,
        )
    else:
        raw = model(state)
    return _normalize_evaluation(raw)



def _merge_indicator_values(*values: float | None, mode: str = "max") -> float | None:
    finite = [float(v) for v in values if v is not None]
    if not finite:
        return None
    if mode == "min":
        return min(finite)
    return max(finite)



def semi_implicit_euler_step(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
) -> IntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    eval0 = _evaluate_model(model, state)
    prediction = predict_contact_event(state, dt, eval0.acceleration)
    v_next = state.velocity + dt * eval0.acceleration
    gap_next = state.gap + dt * v_next
    next_state = state.with_state(time=state.time + dt, gap=gap_next, velocity=v_next)
    return IntegrationStepResult(
        state=next_state,
        diagnostics=StepDiagnostics(
            force=eval0.force,
            acceleration=eval0.acceleration,
            active=eval0.active,
            onset_detected=prediction.onset,
            release_detected=prediction.release,
            onset_time_estimate=prediction.onset_time_estimate,
            release_time_estimate=prediction.release_time_estimate,
            active_measure=eval0.active_measure,
        ),
    )



def midpoint_contact_step(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
) -> IntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    eval0 = _evaluate_model(model, state)
    prediction = predict_contact_event(state, dt, eval0.acceleration)
    gap_mid = state.gap + 0.5 * dt * state.velocity
    vel_mid = state.velocity + 0.5 * dt * eval0.acceleration
    mid_state = state.with_state(time=state.time + 0.5 * dt, gap=gap_mid, velocity=vel_mid)
    eval_mid = _evaluate_model(model, mid_state)
    indicators = event_aware_indicators(eval0, eval_mid)
    gap_next = state.gap + dt * vel_mid
    vel_next = state.velocity + dt * eval_mid.acceleration
    next_state = state.with_state(time=state.time + dt, gap=gap_next, velocity=vel_next)
    return IntegrationStepResult(
        state=next_state,
        diagnostics=StepDiagnostics(
            force=eval_mid.force,
            acceleration=eval_mid.acceleration,
            active=eval_mid.active or eval0.active,
            onset_detected=prediction.onset,
            release_detected=prediction.release,
            onset_time_estimate=prediction.onset_time_estimate,
            release_time_estimate=prediction.release_time_estimate,
            active_measure=eval_mid.active_measure,
            relative_force_jump=indicators.relative_force_jump,
            relative_acceleration_jump=indicators.relative_acceleration_jump,
            active_measure_jump=indicators.active_measure_jump,
        ),
    )



def _predictor_corrector_event_aware_evaluations(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
    controller: EventAwareControllerConfig,
) -> tuple[ModelEvaluation, ModelEvaluation, ModelEvaluation, EventAwareIndicators, IntegrationStepResult | None]:
    eval0 = _evaluate_model(model, state)
    prediction = predict_contact_event(state, dt, eval0.acceleration)

    gap_mid = state.gap + 0.5 * dt * state.velocity
    vel_mid = state.velocity + 0.5 * dt * eval0.acceleration
    mid_state = state.with_state(time=state.time + 0.5 * dt, gap=gap_mid, velocity=vel_mid)

    if not controller.use_predictor_corrector_continuity:
        eval_predictor = eval0
        eval_mid = _evaluate_model(model, mid_state)
        indicators = event_aware_indicators(eval0, eval_mid)
        if controller.max_depth > 0 and should_substep_event_aware(state, dt, prediction, indicators, controller):
            child = EventAwareControllerConfig(
                max_depth=controller.max_depth - 1,
                min_dt=controller.min_dt,
                force_relative_jump_tol=controller.force_relative_jump_tol,
                acceleration_relative_jump_tol=controller.acceleration_relative_jump_tol,
                active_measure_relative_jump_tol=controller.active_measure_relative_jump_tol,
                predictor_force_relative_jump_tol=controller.predictor_force_relative_jump_tol,
                predictor_active_measure_relative_jump_tol=controller.predictor_active_measure_relative_jump_tol,
                predictor_corrector_jaccard_tol=controller.predictor_corrector_jaccard_tol,
                predictor_corrector_mismatch_fraction_tol=controller.predictor_corrector_mismatch_fraction_tol,
                work_mismatch_relative_tol=controller.work_mismatch_relative_tol,
                mismatch_repair_dilation_radius=controller.mismatch_repair_dilation_radius,
                use_predictor_corrector_continuity=controller.use_predictor_corrector_continuity,
            )
            left = event_aware_midpoint_step(state, 0.5 * dt, model, controller=child)
            right = event_aware_midpoint_step(left.state, 0.5 * dt, model, controller=child)
            onset_candidates = [t for t in (prediction.onset_time_estimate, left.diagnostics.onset_time_estimate, right.diagnostics.onset_time_estimate) if t is not None]
            release_candidates = [t for t in (prediction.release_time_estimate, left.diagnostics.release_time_estimate, right.diagnostics.release_time_estimate) if t is not None]
            return eval0, eval_predictor, eval_mid, indicators, IntegrationStepResult(
                state=right.state,
                diagnostics=StepDiagnostics(
                    force=right.diagnostics.force,
                    acceleration=right.diagnostics.acceleration,
                    active=right.diagnostics.active or left.diagnostics.active or eval0.active,
                    used_substeps=left.diagnostics.used_substeps + right.diagnostics.used_substeps,
                    onset_detected=prediction.onset or left.diagnostics.onset_detected or right.diagnostics.onset_detected,
                    release_detected=prediction.release or left.diagnostics.release_detected or right.diagnostics.release_detected,
                    onset_time_estimate=min(onset_candidates) if onset_candidates else None,
                    release_time_estimate=min(release_candidates) if release_candidates else None,
                    active_measure=right.diagnostics.active_measure,
                    relative_force_jump=_merge_indicator_values(indicators.relative_force_jump, left.diagnostics.relative_force_jump, right.diagnostics.relative_force_jump, mode="max"),
                    relative_acceleration_jump=_merge_indicator_values(indicators.relative_acceleration_jump, left.diagnostics.relative_acceleration_jump, right.diagnostics.relative_acceleration_jump, mode="max"),
                    active_measure_jump=_merge_indicator_values(indicators.active_measure_jump, left.diagnostics.active_measure_jump, right.diagnostics.active_measure_jump, mode="max"),
                    continuity_jaccard=right.diagnostics.continuity_jaccard,
                ),
            )
        return eval0, eval_predictor, eval_mid, indicators, None

    predictor_velocity = state.velocity + dt * eval0.acceleration
    predictor_state = state.with_state(
        time=state.time + dt,
        gap=prediction.gap_end,
        velocity=predictor_velocity,
    )
    eval_predictor = _evaluate_model(model, predictor_state)

    eval_mid = _evaluate_model(
        model,
        mid_state,
        warm_start_snapshot=eval_predictor.active_snapshot,
        boundary_only_update=True,
    )

    mismatch_report = None
    if eval_predictor.active_snapshot is not None and eval_mid.active_snapshot is not None:
        mismatch_report = active_set_mismatch_report(eval_predictor.active_snapshot, eval_mid.active_snapshot)
        if mismatch_report.mismatch_count > 0 and hasattr(model, "evaluate_state"):
            repair_mask = dilate_mask(
                mismatch_report.mismatch_mask,
                radius=controller.mismatch_repair_dilation_radius,
            )
            eval_repaired = _evaluate_model(
                model,
                mid_state,
                warm_start_snapshot=eval_mid.active_snapshot,
                boundary_only_update=True,
                repair_mask=repair_mask,
            )
            if eval_repaired.active_snapshot is not None:
                mismatch_report = active_set_mismatch_report(eval_predictor.active_snapshot, eval_repaired.active_snapshot)
            eval_mid = eval_repaired

    indicators = event_aware_indicators(eval0, eval_mid)
    indicators = EventAwareIndicators(
        relative_force_jump=indicators.relative_force_jump,
        relative_acceleration_jump=indicators.relative_acceleration_jump,
        active_measure_jump=indicators.active_measure_jump,
        predictor_force_jump=(
            abs(eval_mid.force - eval_predictor.force) / max(abs(eval_mid.force), abs(eval_predictor.force), 1e-14)
        ),
        predictor_active_measure_jump=(
            None
            if eval_mid.active_measure is None or eval_predictor.active_measure is None
            else abs(eval_mid.active_measure - eval_predictor.active_measure)
            / max(abs(eval_mid.active_measure), abs(eval_predictor.active_measure), 1e-14)
        ),
        predictor_corrector_jaccard=None if mismatch_report is None else mismatch_report.jaccard_index,
        predictor_corrector_mismatch_fraction=None if mismatch_report is None else mismatch_report.mismatch_fraction,
    )

    if controller.max_depth > 0 and should_substep_event_aware(state, dt, prediction, indicators, controller):
        child = EventAwareControllerConfig(
            max_depth=controller.max_depth - 1,
            min_dt=controller.min_dt,
            force_relative_jump_tol=controller.force_relative_jump_tol,
            acceleration_relative_jump_tol=controller.acceleration_relative_jump_tol,
            active_measure_relative_jump_tol=controller.active_measure_relative_jump_tol,
            predictor_force_relative_jump_tol=controller.predictor_force_relative_jump_tol,
            predictor_active_measure_relative_jump_tol=controller.predictor_active_measure_relative_jump_tol,
            predictor_corrector_jaccard_tol=controller.predictor_corrector_jaccard_tol,
            predictor_corrector_mismatch_fraction_tol=controller.predictor_corrector_mismatch_fraction_tol,
            mismatch_repair_dilation_radius=controller.mismatch_repair_dilation_radius,
            use_predictor_corrector_continuity=controller.use_predictor_corrector_continuity,
        )
        left = event_aware_midpoint_step(state, 0.5 * dt, model, controller=child)
        right = event_aware_midpoint_step(left.state, 0.5 * dt, model, controller=child)
        onset_candidates = [
            t
            for t in (
                prediction.onset_time_estimate,
                left.diagnostics.onset_time_estimate,
                right.diagnostics.onset_time_estimate,
            )
            if t is not None
        ]
        release_candidates = [
            t
            for t in (
                prediction.release_time_estimate,
                left.diagnostics.release_time_estimate,
                right.diagnostics.release_time_estimate,
            )
            if t is not None
        ]
        substep_result = IntegrationStepResult(
            state=right.state,
            diagnostics=StepDiagnostics(
                force=right.diagnostics.force,
                acceleration=right.diagnostics.acceleration,
                active=right.diagnostics.active or left.diagnostics.active or eval0.active,
                used_substeps=left.diagnostics.used_substeps + right.diagnostics.used_substeps,
                onset_detected=prediction.onset or left.diagnostics.onset_detected or right.diagnostics.onset_detected,
                release_detected=prediction.release or left.diagnostics.release_detected or right.diagnostics.release_detected,
                onset_time_estimate=min(onset_candidates) if onset_candidates else None,
                release_time_estimate=min(release_candidates) if release_candidates else None,
                active_measure=right.diagnostics.active_measure,
                relative_force_jump=_merge_indicator_values(
                    indicators.relative_force_jump,
                    left.diagnostics.relative_force_jump,
                    right.diagnostics.relative_force_jump,
                    mode="max",
                ),
                relative_acceleration_jump=_merge_indicator_values(
                    indicators.relative_acceleration_jump,
                    left.diagnostics.relative_acceleration_jump,
                    right.diagnostics.relative_acceleration_jump,
                    mode="max",
                ),
                active_measure_jump=_merge_indicator_values(
                    indicators.active_measure_jump,
                    left.diagnostics.active_measure_jump,
                    right.diagnostics.active_measure_jump,
                    mode="max",
                ),
                predictor_force_jump=_merge_indicator_values(
                    indicators.predictor_force_jump,
                    left.diagnostics.predictor_force_jump,
                    right.diagnostics.predictor_force_jump,
                    mode="max",
                ),
                predictor_active_measure_jump=_merge_indicator_values(
                    indicators.predictor_active_measure_jump,
                    left.diagnostics.predictor_active_measure_jump,
                    right.diagnostics.predictor_active_measure_jump,
                    mode="max",
                ),
                predictor_corrector_jaccard=_merge_indicator_values(
                    indicators.predictor_corrector_jaccard,
                    left.diagnostics.predictor_corrector_jaccard,
                    right.diagnostics.predictor_corrector_jaccard,
                    mode="min",
                ),
                predictor_corrector_mismatch_fraction=_merge_indicator_values(
                    indicators.predictor_corrector_mismatch_fraction,
                    left.diagnostics.predictor_corrector_mismatch_fraction,
                    right.diagnostics.predictor_corrector_mismatch_fraction,
                    mode="max",
                ),
                continuity_jaccard=right.diagnostics.continuity_jaccard,
            ),
        )
        return eval0, eval_predictor, eval_mid, indicators, substep_result

    return eval0, eval_predictor, eval_mid, indicators, None



def event_aware_midpoint_step(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
    *,
    controller: EventAwareControllerConfig | None = None,
) -> IntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if controller is None:
        controller = EventAwareControllerConfig()

    eval0, eval_predictor, eval_mid, indicators, substep_result = _predictor_corrector_event_aware_evaluations(
        state,
        dt,
        model,
        controller,
    )
    prediction = predict_contact_event(state, dt, eval0.acceleration)
    if substep_result is not None:
        return substep_result

    gap_mid = state.gap + 0.5 * dt * state.velocity
    vel_mid = state.velocity + 0.5 * dt * eval0.acceleration
    gap_next = state.gap + dt * vel_mid
    vel_next = state.velocity + dt * eval_mid.acceleration
    next_state = state.with_state(time=state.time + dt, gap=gap_next, velocity=vel_next)
    return IntegrationStepResult(
        state=next_state,
        diagnostics=StepDiagnostics(
            force=eval_mid.force,
            acceleration=eval_mid.acceleration,
            active=eval_mid.active or eval0.active,
            used_substeps=1,
            onset_detected=prediction.onset,
            release_detected=prediction.release,
            onset_time_estimate=prediction.onset_time_estimate,
            release_time_estimate=prediction.release_time_estimate,
            active_measure=eval_mid.active_measure,
            relative_force_jump=indicators.relative_force_jump,
            relative_acceleration_jump=indicators.relative_acceleration_jump,
            active_measure_jump=indicators.active_measure_jump,
            predictor_force_jump=indicators.predictor_force_jump,
            predictor_active_measure_jump=indicators.predictor_active_measure_jump,
            predictor_corrector_jaccard=indicators.predictor_corrector_jaccard,
            predictor_corrector_mismatch_fraction=indicators.predictor_corrector_mismatch_fraction,
            continuity_jaccard=indicators.predictor_corrector_jaccard,
        ),
    )



def midpoint_contact_substep(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
    *,
    max_depth: int = 8,
) -> IntegrationStepResult:
    controller = EventAwareControllerConfig(max_depth=max_depth)
    return event_aware_midpoint_step(state, dt, model, controller=controller)

def _force_triplet_evaluations(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
    eval0: ModelEvaluation,
    eval_mid: ModelEvaluation,
    *,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
) -> tuple[NormalDynamicsState, ModelEvaluation, float]:
    """Build a provisional end-state and evaluate the model there.

    This provides a generic three-point force sample (start/mid/end) for
    higher-order time integration without introducing geometry-specific rules.
    """
    gap_mid = state.gap + 0.5 * dt * state.velocity
    vel_mid = state.velocity + 0.5 * dt * eval_mid.acceleration
    gap_next_provisional = state.gap + dt * vel_mid
    vel_next_provisional = state.velocity + dt * eval_mid.acceleration
    end_state = state.with_state(
        time=state.time + dt,
        gap=gap_next_provisional,
        velocity=vel_next_provisional,
    )
    eval_end = _evaluate_model(
        model,
        end_state,
        warm_start_snapshot=warm_start_snapshot,
        boundary_only_update=True if warm_start_snapshot is not None else None,
    )
    return end_state, eval_end, vel_mid


def _simpson_average(a0: float, am: float, a1: float) -> float:
    return (a0 + 4.0 * am + a1) / 6.0


def _kinetic_energy(mass: float, velocity: float) -> float:
    return 0.5 * mass * velocity * velocity


def _relative_mismatch(a: float, b: float, *, atol: float = 1e-14) -> float:
    return abs(a - b) / max(abs(a), abs(b), atol)


def _triplet_work_estimate(
    state: NormalDynamicsState,
    dt: float,
    eval0: ModelEvaluation,
    eval_mid: ModelEvaluation,
    eval_end: ModelEvaluation,
    *,
    vel_mid: float,
    vel_end: float,
) -> float:
    power0 = eval0.force * state.velocity
    power_mid = eval_mid.force * vel_mid
    power_end = eval_end.force * vel_end
    return dt * _simpson_average(power0, power_mid, power_end)


def event_aware_midpoint_impulse_corrected_step(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
    *,
    controller: EventAwareControllerConfig | None = None,
) -> IntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if controller is None:
        controller = EventAwareControllerConfig()

    eval0, eval_predictor, eval_mid, indicators, substep_result = _predictor_corrector_event_aware_evaluations(
        state,
        dt,
        model,
        controller,
    )
    prediction = predict_contact_event(state, dt, eval0.acceleration)
    if substep_result is not None:
        return substep_result

    warm_start_snapshot = eval_mid.active_snapshot if eval_mid.active_snapshot is not None else eval_predictor.active_snapshot
    _, eval_end, vel_mid_consistent = _force_triplet_evaluations(
        state,
        dt,
        model,
        eval0,
        eval_mid,
        warm_start_snapshot=warm_start_snapshot,
    )

    # Simpson-style impulse correction for acceleration / force integration.
    acc_avg = _simpson_average(eval0.acceleration, eval_mid.acceleration, eval_end.acceleration)
    vel_next = state.velocity + dt * acc_avg
    gap_next = state.gap + dt * _simpson_average(state.velocity, vel_mid_consistent, vel_next)
    next_state = state.with_state(time=state.time + dt, gap=gap_next, velocity=vel_next)

    work_est = _triplet_work_estimate(
        state,
        dt,
        eval0,
        eval_mid,
        eval_end,
        vel_mid=vel_mid_consistent,
        vel_end=vel_next,
    )
    work_mismatch = _relative_mismatch(
        _kinetic_energy(state.mass, vel_next) - _kinetic_energy(state.mass, state.velocity),
        work_est,
    )

    force_out = eval_end.force
    active_measure = eval_end.active_measure

    return IntegrationStepResult(
        state=next_state,
        diagnostics=StepDiagnostics(
            force=force_out,
            acceleration=acc_avg,
            active=eval0.active or eval_mid.active or eval_end.active,
            used_substeps=1,
            onset_detected=prediction.onset,
            release_detected=prediction.release,
            onset_time_estimate=prediction.onset_time_estimate,
            release_time_estimate=prediction.release_time_estimate,
            active_measure=active_measure,
            relative_force_jump=indicators.relative_force_jump,
            relative_acceleration_jump=indicators.relative_acceleration_jump,
            active_measure_jump=indicators.active_measure_jump,
            predictor_force_jump=indicators.predictor_force_jump,
            predictor_active_measure_jump=indicators.predictor_active_measure_jump,
            predictor_corrector_jaccard=indicators.predictor_corrector_jaccard,
            predictor_corrector_mismatch_fraction=indicators.predictor_corrector_mismatch_fraction,
            continuity_jaccard=indicators.predictor_corrector_jaccard,
            work_mismatch=work_mismatch,
        ),
    )


def event_aware_midpoint_work_consistent_step(
    state: NormalDynamicsState,
    dt: float,
    model: NormalAccelerationModel,
    *,
    controller: EventAwareControllerConfig | None = None,
) -> IntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if controller is None:
        controller = EventAwareControllerConfig()

    eval0, eval_predictor, eval_mid, indicators, substep_result = _predictor_corrector_event_aware_evaluations(
        state,
        dt,
        model,
        controller,
    )
    prediction = predict_contact_event(state, dt, eval0.acceleration)
    if substep_result is not None:
        return substep_result

    warm_start_snapshot = eval_mid.active_snapshot if eval_mid.active_snapshot is not None else eval_predictor.active_snapshot
    _, eval_end, vel_mid_consistent = _force_triplet_evaluations(
        state,
        dt,
        model,
        eval0,
        eval_mid,
        warm_start_snapshot=warm_start_snapshot,
    )

    acc_avg = _simpson_average(eval0.acceleration, eval_mid.acceleration, eval_end.acceleration)
    vel_next_provisional = state.velocity + dt * acc_avg
    work_est = _triplet_work_estimate(
        state,
        dt,
        eval0,
        eval_mid,
        eval_end,
        vel_mid=vel_mid_consistent,
        vel_end=vel_next_provisional,
    )
    kinetic_start = _kinetic_energy(state.mass, state.velocity)
    kinetic_end_provisional = _kinetic_energy(state.mass, vel_next_provisional)
    work_mismatch = _relative_mismatch(kinetic_end_provisional - kinetic_start, work_est)

    work_indicators = EventAwareIndicators(
        relative_force_jump=indicators.relative_force_jump,
        relative_acceleration_jump=indicators.relative_acceleration_jump,
        active_measure_jump=indicators.active_measure_jump,
        predictor_force_jump=indicators.predictor_force_jump,
        predictor_active_measure_jump=indicators.predictor_active_measure_jump,
        predictor_corrector_jaccard=indicators.predictor_corrector_jaccard,
        predictor_corrector_mismatch_fraction=indicators.predictor_corrector_mismatch_fraction,
        work_mismatch=work_mismatch,
    )

    if controller.max_depth > 0 and should_substep_event_aware(state, dt, prediction, work_indicators, controller):
        child = EventAwareControllerConfig(
            max_depth=controller.max_depth - 1,
            min_dt=controller.min_dt,
            force_relative_jump_tol=controller.force_relative_jump_tol,
            acceleration_relative_jump_tol=controller.acceleration_relative_jump_tol,
            active_measure_relative_jump_tol=controller.active_measure_relative_jump_tol,
            predictor_force_relative_jump_tol=controller.predictor_force_relative_jump_tol,
            predictor_active_measure_relative_jump_tol=controller.predictor_active_measure_relative_jump_tol,
            predictor_corrector_jaccard_tol=controller.predictor_corrector_jaccard_tol,
            predictor_corrector_mismatch_fraction_tol=controller.predictor_corrector_mismatch_fraction_tol,
            work_mismatch_relative_tol=controller.work_mismatch_relative_tol,
            mismatch_repair_dilation_radius=controller.mismatch_repair_dilation_radius,
            use_predictor_corrector_continuity=controller.use_predictor_corrector_continuity,
        )
        left = event_aware_midpoint_work_consistent_step(state, 0.5 * dt, model, controller=child)
        right = event_aware_midpoint_work_consistent_step(left.state, 0.5 * dt, model, controller=child)
        onset_candidates = [t for t in (prediction.onset_time_estimate, left.diagnostics.onset_time_estimate, right.diagnostics.onset_time_estimate) if t is not None]
        release_candidates = [t for t in (prediction.release_time_estimate, left.diagnostics.release_time_estimate, right.diagnostics.release_time_estimate) if t is not None]
        return IntegrationStepResult(
            state=right.state,
            diagnostics=StepDiagnostics(
                force=right.diagnostics.force,
                acceleration=right.diagnostics.acceleration,
                active=right.diagnostics.active or left.diagnostics.active or eval0.active,
                used_substeps=left.diagnostics.used_substeps + right.diagnostics.used_substeps,
                onset_detected=prediction.onset or left.diagnostics.onset_detected or right.diagnostics.onset_detected,
                release_detected=prediction.release or left.diagnostics.release_detected or right.diagnostics.release_detected,
                onset_time_estimate=min(onset_candidates) if onset_candidates else None,
                release_time_estimate=min(release_candidates) if release_candidates else None,
                active_measure=right.diagnostics.active_measure,
                relative_force_jump=_merge_indicator_values(work_indicators.relative_force_jump, left.diagnostics.relative_force_jump, right.diagnostics.relative_force_jump, mode="max"),
                relative_acceleration_jump=_merge_indicator_values(work_indicators.relative_acceleration_jump, left.diagnostics.relative_acceleration_jump, right.diagnostics.relative_acceleration_jump, mode="max"),
                active_measure_jump=_merge_indicator_values(work_indicators.active_measure_jump, left.diagnostics.active_measure_jump, right.diagnostics.active_measure_jump, mode="max"),
                predictor_force_jump=_merge_indicator_values(work_indicators.predictor_force_jump, left.diagnostics.predictor_force_jump, right.diagnostics.predictor_force_jump, mode="max"),
                predictor_active_measure_jump=_merge_indicator_values(work_indicators.predictor_active_measure_jump, left.diagnostics.predictor_active_measure_jump, right.diagnostics.predictor_active_measure_jump, mode="max"),
                predictor_corrector_jaccard=_merge_indicator_values(work_indicators.predictor_corrector_jaccard, left.diagnostics.predictor_corrector_jaccard, right.diagnostics.predictor_corrector_jaccard, mode="min"),
                predictor_corrector_mismatch_fraction=_merge_indicator_values(work_indicators.predictor_corrector_mismatch_fraction, left.diagnostics.predictor_corrector_mismatch_fraction, right.diagnostics.predictor_corrector_mismatch_fraction, mode="max"),
                continuity_jaccard=right.diagnostics.continuity_jaccard,
                work_mismatch=_merge_indicator_values(work_indicators.work_mismatch, left.diagnostics.work_mismatch, right.diagnostics.work_mismatch, mode="max"),
            ),
        )

    kinetic_target = max(kinetic_start + work_est, 0.0)
    if kinetic_target <= 1e-16:
        vel_next = 0.0
    else:
        sign = np.sign(vel_next_provisional) if abs(vel_next_provisional) > 1e-14 else np.sign(state.velocity if abs(state.velocity) > 1e-14 else -1.0)
        vel_next = float(sign * np.sqrt(2.0 * kinetic_target / state.mass))
    gap_next = state.gap + dt * _simpson_average(state.velocity, vel_mid_consistent, vel_next)
    corrected_state = state.with_state(time=state.time + dt, gap=gap_next, velocity=vel_next)
    eval_corrected = _evaluate_model(
        model,
        corrected_state,
        warm_start_snapshot=warm_start_snapshot,
        boundary_only_update=True if warm_start_snapshot is not None else None,
    )

    return IntegrationStepResult(
        state=corrected_state,
        diagnostics=StepDiagnostics(
            force=eval_corrected.force,
            acceleration=(vel_next - state.velocity) / dt,
            active=eval0.active or eval_mid.active or eval_end.active or eval_corrected.active,
            used_substeps=1,
            onset_detected=prediction.onset,
            release_detected=prediction.release,
            onset_time_estimate=prediction.onset_time_estimate,
            release_time_estimate=prediction.release_time_estimate,
            active_measure=eval_corrected.active_measure,
            relative_force_jump=work_indicators.relative_force_jump,
            relative_acceleration_jump=work_indicators.relative_acceleration_jump,
            active_measure_jump=work_indicators.active_measure_jump,
            predictor_force_jump=work_indicators.predictor_force_jump,
            predictor_active_measure_jump=work_indicators.predictor_active_measure_jump,
            predictor_corrector_jaccard=work_indicators.predictor_corrector_jaccard,
            predictor_corrector_mismatch_fraction=work_indicators.predictor_corrector_mismatch_fraction,
            continuity_jaccard=work_indicators.predictor_corrector_jaccard,
            work_mismatch=work_mismatch,
        ),
    )


Integrator = Callable[[NormalDynamicsState, float, NormalAccelerationModel], IntegrationStepResult]
