from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from pfcsdf.contact.active_set import ActiveSetMismatchReport
from pfcsdf.contact.wrench import PairWrench
from pfcsdf.dynamics.rigid_state import RigidBodyState
from pfcsdf.dynamics.rotation import rotation_angle_error


@dataclass(frozen=True)
class RigidEventAwareControllerConfig:
    max_depth: int = 7
    min_dt: float = 1e-5
    force_relative_jump_tol: float = 0.20
    torque_relative_jump_tol: float = 0.20
    active_measure_relative_jump_tol: float = 0.15
    predictor_corrector_mismatch_fraction_tol: float = 0.12
    predictor_corrector_jaccard_tol: float = 0.85
    orientation_mismatch_tol: float = 0.10
    work_mismatch_relative_tol: float = 0.02
    angular_work_mismatch_relative_tol: float = 0.02
    mismatch_repair_dilation_radius: int = 1


@dataclass(frozen=True)
class RigidControllerIndicators:
    force_jump: float
    torque_jump: float
    active_measure_jump: float
    predictor_corrector_jaccard: float
    predictor_corrector_mismatch_fraction: float
    orientation_mismatch: float
    work_mismatch: float
    angular_work_mismatch: float


def _relative_norm_jump(a: np.ndarray, b: np.ndarray, *, atol: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)), atol)
    return float(np.linalg.norm(b - a) / denom)


def rigid_controller_indicators(
    start_wrench: PairWrench,
    predictor_wrench: PairWrench,
    corrector_wrench: PairWrench,
    *,
    start_measure: float,
    predictor_measure: float,
    corrector_measure: float,
    mismatch: ActiveSetMismatchReport,
    predictor_state: RigidBodyState,
    corrector_state: RigidBodyState,
    linear_work_mismatch: float = 0.0,
    angular_work_mismatch: float = 0.0,
) -> RigidControllerIndicators:
    force_jump = max(
        _relative_norm_jump(start_wrench.force, corrector_wrench.force),
        _relative_norm_jump(predictor_wrench.force, corrector_wrench.force),
    )
    torque_jump = max(
        _relative_norm_jump(start_wrench.torque, corrector_wrench.torque),
        _relative_norm_jump(predictor_wrench.torque, corrector_wrench.torque),
    )
    def rel(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-12)
        return float(abs(b - a) / denom)
    active_measure_jump = max(rel(start_measure, corrector_measure), rel(predictor_measure, corrector_measure))
    orientation_mismatch = rotation_angle_error(predictor_state.rotation, corrector_state.rotation)
    return RigidControllerIndicators(
        force_jump=force_jump,
        torque_jump=torque_jump,
        active_measure_jump=active_measure_jump,
        predictor_corrector_jaccard=mismatch.jaccard_index,
        predictor_corrector_mismatch_fraction=mismatch.mismatch_fraction,
        orientation_mismatch=orientation_mismatch,
        work_mismatch=float(abs(linear_work_mismatch)),
        angular_work_mismatch=float(abs(angular_work_mismatch)),
    )


def should_substep_rigid(config: RigidEventAwareControllerConfig, dt: float, indicators: RigidControllerIndicators) -> bool:
    if dt <= config.min_dt:
        return False
    if indicators.force_jump > config.force_relative_jump_tol:
        return True
    if indicators.torque_jump > config.torque_relative_jump_tol:
        return True
    if indicators.active_measure_jump > config.active_measure_relative_jump_tol:
        return True
    if indicators.predictor_corrector_jaccard < config.predictor_corrector_jaccard_tol:
        return True
    if indicators.predictor_corrector_mismatch_fraction > config.predictor_corrector_mismatch_fraction_tol:
        return True
    if indicators.orientation_mismatch > config.orientation_mismatch_tol:
        return True
    if indicators.work_mismatch > config.work_mismatch_relative_tol:
        return True
    if indicators.angular_work_mismatch > config.angular_work_mismatch_relative_tol:
        return True
    return False
