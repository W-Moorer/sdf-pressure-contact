from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned 3D bounding box for finite signed-distance geometry."""

    minimum: ArrayLike
    maximum: ArrayLike

    def __post_init__(self) -> None:
        minimum = np.asarray(self.minimum, dtype=float)
        maximum = np.asarray(self.maximum, dtype=float)
        if minimum.shape != (3,) or maximum.shape != (3,):
            raise ValueError("bounding box corners must have shape (3,)")
        if np.any(maximum < minimum):
            raise ValueError("bounding box maximum must be >= minimum component-wise")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)


@runtime_checkable
class SignedDistanceGeometry(Protocol):
    """Shared 3D signed-distance contract for engine-facing geometry objects."""

    def signed_distance(self, x: ArrayLike) -> float:
        """Return the signed distance at a 3D query point."""


@runtime_checkable
class DifferentiableSignedDistanceGeometry(SignedDistanceGeometry, Protocol):
    """Signed-distance geometry that can provide an analytic gradient."""

    def gradient(self, x: ArrayLike) -> ArrayLike:
        """Return the signed-distance gradient at a 3D query point."""


@runtime_checkable
class BoundedSignedDistanceGeometry(SignedDistanceGeometry, Protocol):
    """Finite signed-distance geometry with an axis-aligned bounding box."""

    def bounding_box(self) -> BoundingBox:
        """Return a finite axis-aligned 3D bounding box."""


def signed_distance_gradient(
    geometry: SignedDistanceGeometry,
    point: ArrayLike,
    *,
    eps: float = 1e-6,
) -> ArrayLike:
    """Query an SDF gradient, using finite differences if analytic gradients are unavailable.

    The shared geometry contract only requires ``signed_distance``. This helper keeps current
    native-band workflows working with analytic primitives today while also allowing future
    mesh-backed or adapter-based geometries to opt into analytic gradients later.
    """

    point = np.asarray(point, dtype=float)
    if point.shape != (3,):
        raise ValueError("point must have shape (3,)")

    if isinstance(geometry, DifferentiableSignedDistanceGeometry):
        return np.asarray(geometry.gradient(point), dtype=float)

    grad = np.zeros(3, dtype=float)
    for axis in range(3):
        step = np.zeros(3, dtype=float)
        step[axis] = eps
        plus = float(geometry.signed_distance(point + step))
        minus = float(geometry.signed_distance(point - step))
        grad[axis] = (plus - minus) / (2.0 * eps)
    return grad
