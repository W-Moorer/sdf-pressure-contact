from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from pfcsdf.contact.wrench import PairWrench
from pfcsdf.dynamics.rotation import exp_so3, project_to_so3
from pfcsdf.dynamics.rigid_state import RigidBodyState

ArrayLike = np.ndarray


@runtime_checkable
class RigidWrenchModel(Protocol):
    def evaluate(self, state: RigidBodyState) -> PairWrench: ...


@dataclass(frozen=True)
class RigidModelEvaluation:
    wrench: PairWrench
    linear_acceleration: ArrayLike
    angular_acceleration: ArrayLike


@dataclass(frozen=True)
class RigidStepDiagnostics:
    force: ArrayLike
    torque: ArrayLike
    linear_acceleration: ArrayLike
    angular_acceleration: ArrayLike
    translational_kinetic_energy: float
    rotational_kinetic_energy: float
    total_kinetic_energy: float


@dataclass(frozen=True)
class RigidIntegrationStepResult:
    state: RigidBodyState
    diagnostics: RigidStepDiagnostics


@dataclass(frozen=True)
class ConstantWrenchModel:
    wrench: PairWrench

    def evaluate(self, state: RigidBodyState) -> PairWrench:
        return self.wrench


def _as_model(model_or_fn: RigidWrenchModel | Callable[[RigidBodyState], PairWrench]) -> Callable[[RigidBodyState], PairWrench]:
    if hasattr(model_or_fn, "evaluate"):
        return model_or_fn.evaluate  # type: ignore[return-value]
    return model_or_fn


def evaluate_rigid_dynamics(
    state: RigidBodyState,
    wrench: PairWrench,
    *,
    gravity: ArrayLike | None = None,
) -> RigidModelEvaluation:
    gravity_vec = np.zeros(3, dtype=float) if gravity is None else np.asarray(gravity, dtype=float).reshape(3)
    force = np.asarray(wrench.force, dtype=float).reshape(3)
    torque = np.asarray(wrench.torque, dtype=float).reshape(3)
    linear_acceleration = force / state.mass + gravity_vec
    Iw = state.inertia_world
    L_term = np.cross(state.angular_velocity, Iw @ state.angular_velocity)
    angular_acceleration = state.inverse_inertia_world @ (torque - L_term)
    return RigidModelEvaluation(wrench=wrench, linear_acceleration=linear_acceleration, angular_acceleration=angular_acceleration)


def _midpoint_rotation(rotation: ArrayLike, angular_velocity: ArrayLike, angular_acceleration: ArrayLike, dt: float) -> ArrayLike:
    omega_mid = np.asarray(angular_velocity, dtype=float) + 0.5 * dt * np.asarray(angular_acceleration, dtype=float)
    return project_to_so3(np.asarray(rotation, dtype=float) @ exp_so3(dt * omega_mid))


def step_rigid_midpoint(
    state: RigidBodyState,
    dt: float,
    model_or_fn: RigidWrenchModel | Callable[[RigidBodyState], PairWrench],
    *,
    gravity: ArrayLike | None = None,
) -> RigidIntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    evaluate = _as_model(model_or_fn)

    eval0 = evaluate_rigid_dynamics(state, evaluate(state), gravity=gravity)

    pos_mid = state.position + 0.5 * dt * state.linear_velocity
    vel_mid = state.linear_velocity + 0.5 * dt * eval0.linear_acceleration
    rot_mid = _midpoint_rotation(state.rotation, state.angular_velocity, eval0.angular_acceleration, 0.5 * dt)
    omega_mid = state.angular_velocity + 0.5 * dt * eval0.angular_acceleration
    mid_state = state.with_state(
        position=pos_mid,
        rotation=rot_mid,
        linear_velocity=vel_mid,
        angular_velocity=omega_mid,
        time=state.time + 0.5 * dt,
    )

    eval_mid = evaluate_rigid_dynamics(mid_state, evaluate(mid_state), gravity=gravity)

    pos_new = state.position + dt * (state.linear_velocity + 0.5 * dt * eval_mid.linear_acceleration)
    vel_new = state.linear_velocity + dt * eval_mid.linear_acceleration
    omega_new = state.angular_velocity + dt * eval_mid.angular_acceleration
    rot_new = _midpoint_rotation(state.rotation, state.angular_velocity, eval_mid.angular_acceleration, dt)

    new_state = state.with_state(
        position=pos_new,
        rotation=rot_new,
        linear_velocity=vel_new,
        angular_velocity=omega_new,
        time=state.time + dt,
    )

    Iw_new = new_state.inertia_world
    translational_ke = 0.5 * new_state.mass * float(np.dot(new_state.linear_velocity, new_state.linear_velocity))
    rotational_ke = 0.5 * float(new_state.angular_velocity @ (Iw_new @ new_state.angular_velocity))
    diagnostics = RigidStepDiagnostics(
        force=np.asarray(eval_mid.wrench.force, dtype=float),
        torque=np.asarray(eval_mid.wrench.torque, dtype=float),
        linear_acceleration=np.asarray(eval_mid.linear_acceleration, dtype=float),
        angular_acceleration=np.asarray(eval_mid.angular_acceleration, dtype=float),
        translational_kinetic_energy=translational_ke,
        rotational_kinetic_energy=rotational_ke,
        total_kinetic_energy=translational_ke + rotational_ke,
    )
    return RigidIntegrationStepResult(state=new_state, diagnostics=diagnostics)


def step_rigid_semi_implicit(
    state: RigidBodyState,
    dt: float,
    model_or_fn: RigidWrenchModel | Callable[[RigidBodyState], PairWrench],
    *,
    gravity: ArrayLike | None = None,
) -> RigidIntegrationStepResult:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    evaluate = _as_model(model_or_fn)
    eval0 = evaluate_rigid_dynamics(state, evaluate(state), gravity=gravity)
    vel_new = state.linear_velocity + dt * eval0.linear_acceleration
    omega_new = state.angular_velocity + dt * eval0.angular_acceleration
    pos_new = state.position + dt * vel_new
    rot_new = project_to_so3(state.rotation @ exp_so3(dt * omega_new))
    new_state = state.with_state(
        position=pos_new,
        rotation=rot_new,
        linear_velocity=vel_new,
        angular_velocity=omega_new,
        time=state.time + dt,
    )
    Iw_new = new_state.inertia_world
    translational_ke = 0.5 * new_state.mass * float(np.dot(new_state.linear_velocity, new_state.linear_velocity))
    rotational_ke = 0.5 * float(new_state.angular_velocity @ (Iw_new @ new_state.angular_velocity))
    diagnostics = RigidStepDiagnostics(
        force=np.asarray(eval0.wrench.force, dtype=float),
        torque=np.asarray(eval0.wrench.torque, dtype=float),
        linear_acceleration=np.asarray(eval0.linear_acceleration, dtype=float),
        angular_acceleration=np.asarray(eval0.angular_acceleration, dtype=float),
        translational_kinetic_energy=translational_ke,
        rotational_kinetic_energy=rotational_ke,
        total_kinetic_energy=translational_ke + rotational_ke,
    )
    return RigidIntegrationStepResult(state=new_state, diagnostics=diagnostics)
