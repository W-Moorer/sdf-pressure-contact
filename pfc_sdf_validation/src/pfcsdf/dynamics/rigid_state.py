from __future__ import annotations

from dataclasses import dataclass
import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class RigidBodyState:
    position: ArrayLike
    rotation: ArrayLike
    linear_velocity: ArrayLike
    angular_velocity: ArrayLike
    mass: float
    inertia_body: ArrayLike
    time: float = 0.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float).reshape(3)
        rotation = np.asarray(self.rotation, dtype=float).reshape(3, 3)
        linear_velocity = np.asarray(self.linear_velocity, dtype=float).reshape(3)
        angular_velocity = np.asarray(self.angular_velocity, dtype=float).reshape(3)
        inertia_body = np.asarray(self.inertia_body, dtype=float).reshape(3, 3)
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if np.linalg.det(rotation) <= 0.0:
            raise ValueError("rotation must have positive determinant")
        if not np.allclose(inertia_body, inertia_body.T, atol=1e-12):
            raise ValueError("inertia_body must be symmetric")
        eigvals = np.linalg.eigvalsh(inertia_body)
        if np.any(eigvals <= 0.0):
            raise ValueError("inertia_body must be positive definite")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "linear_velocity", linear_velocity)
        object.__setattr__(self, "angular_velocity", angular_velocity)
        object.__setattr__(self, "inertia_body", inertia_body)
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "time", float(self.time))

    def with_state(
        self,
        *,
        position: ArrayLike | None = None,
        rotation: ArrayLike | None = None,
        linear_velocity: ArrayLike | None = None,
        angular_velocity: ArrayLike | None = None,
        time: float | None = None,
    ) -> "RigidBodyState":
        return RigidBodyState(
            position=self.position if position is None else np.asarray(position, dtype=float),
            rotation=self.rotation if rotation is None else np.asarray(rotation, dtype=float),
            linear_velocity=self.linear_velocity if linear_velocity is None else np.asarray(linear_velocity, dtype=float),
            angular_velocity=self.angular_velocity if angular_velocity is None else np.asarray(angular_velocity, dtype=float),
            mass=self.mass,
            inertia_body=self.inertia_body,
            time=self.time if time is None else float(time),
        )

    @property
    def inertia_world(self) -> ArrayLike:
        return self.rotation @ self.inertia_body @ self.rotation.T

    @property
    def inverse_inertia_world(self) -> ArrayLike:
        return self.rotation @ np.linalg.inv(self.inertia_body) @ self.rotation.T

    @property
    def kinetic_energy(self) -> float:
        translational = 0.5 * self.mass * float(np.dot(self.linear_velocity, self.linear_velocity))
        rotational = 0.5 * float(self.angular_velocity @ (self.inertia_world @ self.angular_velocity))
        return translational + rotational

    @property
    def angular_momentum_world(self) -> ArrayLike:
        return self.inertia_world @ self.angular_velocity


def world_inertia(state: RigidBodyState) -> ArrayLike:
    return state.inertia_world


def world_inverse_inertia(state: RigidBodyState) -> ArrayLike:
    return state.inverse_inertia_world


def kinetic_energy(state: RigidBodyState) -> float:
    return state.kinetic_energy
