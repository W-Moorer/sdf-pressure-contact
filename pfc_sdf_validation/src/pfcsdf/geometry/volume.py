from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pfcsdf.geometry.base import SignedDistanceGeometry
from pfcsdf.physics.depth import depth_from_phi

ArrayLike = np.ndarray


@dataclass(frozen=True)
class UniformGrid3D:
    """Uniform Cartesian grid for sampled 3D scalar fields."""

    origin: ArrayLike
    spacing: ArrayLike
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        spacing = np.asarray(self.spacing, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        if spacing.shape != (3,):
            raise ValueError("spacing must have shape (3,)")
        if np.any(spacing <= 0.0):
            raise ValueError("spacing must be strictly positive")
        if len(self.shape) != 3 or any(n < 2 for n in self.shape):
            raise ValueError("shape must be a length-3 tuple with each entry >= 2")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))

    @property
    def x_coords(self) -> ArrayLike:
        return self.origin[0] + self.spacing[0] * np.arange(self.shape[0], dtype=float)

    @property
    def y_coords(self) -> ArrayLike:
        return self.origin[1] + self.spacing[1] * np.arange(self.shape[1], dtype=float)

    @property
    def z_coords(self) -> ArrayLike:
        return self.origin[2] + self.spacing[2] * np.arange(self.shape[2], dtype=float)

    @property
    def cell_volume(self) -> float:
        return float(np.prod(self.spacing))

    def point(self, i: int, j: int, k: int) -> ArrayLike:
        return self.origin + self.spacing * np.array([i, j, k], dtype=float)

    def mesh_coordinates(self) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        return np.meshgrid(self.x_coords, self.y_coords, self.z_coords, indexing="ij")

    def stacked_points(self) -> ArrayLike:
        x, y, z = self.mesh_coordinates()
        return np.stack([x, y, z], axis=-1)


@dataclass(frozen=True)
class SampledScalarField3D:
    grid: UniformGrid3D
    values: ArrayLike

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.shape != self.grid.shape:
            raise ValueError("values shape must match grid shape")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class SampledBalanceField3D(SampledScalarField3D):
    pass


def sample_scalar_field(
    grid: UniformGrid3D,
    func: Callable[[ArrayLike], float],
) -> SampledScalarField3D:
    values = np.empty(grid.shape, dtype=float)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                values[i, j, k] = float(func(grid.point(i, j, k)))
    return SampledScalarField3D(grid=grid, values=values)


def narrow_band_mask(field: SampledScalarField3D, half_width: float) -> ArrayLike:
    if half_width <= 0.0:
        raise ValueError("half_width must be positive")
    return np.abs(field.values) <= half_width


def sample_linear_balance_field_from_sdfs(
    grid: UniformGrid3D,
    sdf_a: SignedDistanceGeometry,
    sdf_b: SignedDistanceGeometry,
    *,
    stiffness_a: float,
    stiffness_b: float,
    max_depth_a: float,
    max_depth_b: float,
) -> SampledBalanceField3D:
    """Sample a linear pressure-balance field from two 3D signed-distance geometries."""

    if stiffness_a <= 0.0 or stiffness_b <= 0.0:
        raise ValueError("stiffnesses must be positive")

    def balance(point: ArrayLike) -> float:
        phi_a = float(sdf_a.signed_distance(point))
        phi_b = float(sdf_b.signed_distance(point))
        depth_a = float(depth_from_phi(phi_a, max_depth_a))
        depth_b = float(depth_from_phi(phi_b, max_depth_b))
        return stiffness_a * depth_a - stiffness_b * depth_b

    sampled = sample_scalar_field(grid, balance)
    return SampledBalanceField3D(grid=sampled.grid, values=sampled.values)
