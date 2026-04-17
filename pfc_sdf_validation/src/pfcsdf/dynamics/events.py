from __future__ import annotations

from dataclasses import dataclass
import math

from .state import ModelEvaluation, NormalDynamicsState


@dataclass(frozen=True)
class EventPrediction:
    gap_end: float
    active_start: bool
    active_end: bool
    onset: bool
    release: bool
    onset_time_estimate: float | None = None
    release_time_estimate: float | None = None


@dataclass(frozen=True)
class EventAwareControllerConfig:
    max_depth: int = 8
    min_dt: float = 1e-6
    force_relative_jump_tol: float = 0.20
    acceleration_relative_jump_tol: float = 0.20
    active_measure_relative_jump_tol: float = 0.15
    predictor_force_relative_jump_tol: float = 0.20
    predictor_active_measure_relative_jump_tol: float = 0.15
    predictor_corrector_jaccard_tol: float = 0.85
    predictor_corrector_mismatch_fraction_tol: float = 0.10
    work_mismatch_relative_tol: float = 0.01
    mismatch_repair_dilation_radius: int = 1
    use_predictor_corrector_continuity: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.min_dt <= 0.0:
            raise ValueError("min_dt must be positive")
        if self.force_relative_jump_tol <= 0.0:
            raise ValueError("force_relative_jump_tol must be positive")
        if self.acceleration_relative_jump_tol <= 0.0:
            raise ValueError("acceleration_relative_jump_tol must be positive")
        if self.active_measure_relative_jump_tol <= 0.0:
            raise ValueError("active_measure_relative_jump_tol must be positive")
        if self.predictor_force_relative_jump_tol <= 0.0:
            raise ValueError("predictor_force_relative_jump_tol must be positive")
        if self.predictor_active_measure_relative_jump_tol <= 0.0:
            raise ValueError("predictor_active_measure_relative_jump_tol must be positive")
        if not (0.0 < self.predictor_corrector_jaccard_tol <= 1.0):
            raise ValueError("predictor_corrector_jaccard_tol must lie in (0, 1]")
        if not (0.0 <= self.predictor_corrector_mismatch_fraction_tol < 1.0):
            raise ValueError("predictor_corrector_mismatch_fraction_tol must lie in [0, 1)")
        if self.work_mismatch_relative_tol <= 0.0:
            raise ValueError("work_mismatch_relative_tol must be positive")
        if self.mismatch_repair_dilation_radius < 0:
            raise ValueError("mismatch_repair_dilation_radius must be non-negative")


@dataclass(frozen=True)
class EventAwareIndicators:
    relative_force_jump: float
    relative_acceleration_jump: float
    active_measure_jump: float | None
    predictor_force_jump: float | None = None
    predictor_active_measure_jump: float | None = None
    predictor_corrector_jaccard: float | None = None
    predictor_corrector_mismatch_fraction: float | None = None
    work_mismatch: float | None = None



def ballistic_gap_prediction(state: NormalDynamicsState, dt: float, acceleration: float) -> float:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    return float(state.gap + dt * state.velocity + 0.5 * dt * dt * acceleration)



def _ballistic_zero_crossing_time(state: NormalDynamicsState, dt: float, acceleration: float) -> float | None:
    g = float(state.gap)
    v = float(state.velocity)
    a = float(acceleration)
    if abs(a) <= 1e-14:
        if abs(v) <= 1e-14:
            return None
        t = -g / v
        return t if 0.0 <= t <= dt else None

    A = 0.5 * a
    B = v
    C = g
    disc = B * B - 4.0 * A * C
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(max(disc, 0.0))
    roots = [(-B - sqrt_disc) / (2.0 * A), (-B + sqrt_disc) / (2.0 * A)]
    roots = [t for t in roots if 0.0 <= t <= dt]
    if not roots:
        return None
    return float(min(roots))



def predict_contact_event(state: NormalDynamicsState, dt: float, acceleration: float) -> EventPrediction:
    gap_end = ballistic_gap_prediction(state, dt, acceleration)
    active_start = bool(state.gap <= 0.0)
    active_end = bool(gap_end <= 0.0)
    onset = (not active_start) and active_end
    release = active_start and (not active_end)
    t_cross = _ballistic_zero_crossing_time(state, dt, acceleration)
    onset_time_estimate = state.time + t_cross if onset and t_cross is not None else None
    release_time_estimate = state.time + t_cross if release and t_cross is not None else None
    return EventPrediction(
        gap_end=gap_end,
        active_start=active_start,
        active_end=active_end,
        onset=onset,
        release=release,
        onset_time_estimate=onset_time_estimate,
        release_time_estimate=release_time_estimate,
    )



def need_substep_for_event(state: NormalDynamicsState, dt: float, acceleration: float) -> bool:
    pred = predict_contact_event(state, dt, acceleration)
    return bool(pred.onset or pred.release)



def _relative_jump(a: float, b: float, *, atol: float = 1e-14) -> float:
    denom = max(abs(a), abs(b), atol)
    return float(abs(b - a) / denom)



def event_aware_indicators(start_eval: ModelEvaluation, midpoint_eval: ModelEvaluation) -> EventAwareIndicators:
    force_jump = _relative_jump(start_eval.force, midpoint_eval.force)
    accel_jump = _relative_jump(start_eval.acceleration, midpoint_eval.acceleration)
    if start_eval.active_measure is None or midpoint_eval.active_measure is None:
        active_measure_jump = None
    else:
        active_measure_jump = _relative_jump(start_eval.active_measure, midpoint_eval.active_measure)
    return EventAwareIndicators(
        relative_force_jump=force_jump,
        relative_acceleration_jump=accel_jump,
        active_measure_jump=active_measure_jump,
    )



def should_substep_event_aware(
    state: NormalDynamicsState,
    dt: float,
    prediction: EventPrediction,
    indicators: EventAwareIndicators,
    config: EventAwareControllerConfig,
) -> bool:
    if dt <= config.min_dt:
        return False
    if prediction.onset or prediction.release:
        return True
    if indicators.relative_force_jump > config.force_relative_jump_tol:
        return True
    if indicators.relative_acceleration_jump > config.acceleration_relative_jump_tol:
        return True
    if indicators.active_measure_jump is not None and indicators.active_measure_jump > config.active_measure_relative_jump_tol:
        return True
    if indicators.predictor_force_jump is not None and indicators.predictor_force_jump > config.predictor_force_relative_jump_tol:
        return True
    if (
        indicators.predictor_active_measure_jump is not None
        and indicators.predictor_active_measure_jump > config.predictor_active_measure_relative_jump_tol
    ):
        return True
    if (
        indicators.predictor_corrector_jaccard is not None
        and indicators.predictor_corrector_jaccard < config.predictor_corrector_jaccard_tol
    ):
        return True
    if (
        indicators.predictor_corrector_mismatch_fraction is not None
        and indicators.predictor_corrector_mismatch_fraction > config.predictor_corrector_mismatch_fraction_tol
    ):
        return True
    if indicators.work_mismatch is not None and indicators.work_mismatch > config.work_mismatch_relative_tol:
        return True
    return False
