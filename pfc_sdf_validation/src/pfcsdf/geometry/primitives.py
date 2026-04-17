from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .base import BoundingBox

ArrayLike = np.ndarray


def _normalize(v: ArrayLike) -> ArrayLike:
    norm = np.linalg.norm(v)
    if norm <= 0.0:
        raise ValueError("Vector must be non-zero.")
    return v / norm


@dataclass(frozen=True)
class PlaneSDF:
    point: ArrayLike
    normal: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "point", np.asarray(self.point, dtype=float))
        object.__setattr__(self, "normal", _normalize(np.asarray(self.normal, dtype=float)))

    def signed_distance(self, x: ArrayLike) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.dot(x - self.point, self.normal))

    def gradient(self, x: ArrayLike | None = None) -> ArrayLike:
        return self.normal.copy()


@dataclass(frozen=True)
class SphereSDF:
    center: ArrayLike
    radius: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", np.asarray(self.center, dtype=float))
        if self.radius <= 0.0:
            raise ValueError("Radius must be positive.")

    def signed_distance(self, x: ArrayLike) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.linalg.norm(x - self.center) - self.radius)

    def gradient(self, x: ArrayLike) -> ArrayLike:
        x = np.asarray(x, dtype=float)
        r = x - self.center
        norm = np.linalg.norm(r)
        if norm <= 1e-14:
            raise ValueError("Gradient undefined at sphere center.")
        return r / norm

    def bounding_box(self) -> BoundingBox:
        extent = np.full(3, self.radius, dtype=float)
        return BoundingBox(minimum=self.center - extent, maximum=self.center + extent)


@dataclass(frozen=True)
class BoxFootprint:
    """Axis-aligned rectangular support region in the xy-plane."""

    lx: float
    ly: float

    @property
    def area(self) -> float:
        return self.lx * self.ly

    def contains_xy(self, x: ArrayLike) -> bool:
        x = np.asarray(x, dtype=float)
        return bool(abs(x[0]) <= 0.5 * self.lx and abs(x[1]) <= 0.5 * self.ly)
