from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .base import BoundingBox
from .volume import UniformGrid3D

ArrayLike = np.ndarray


def _as_spacing_vector(spacing: float | ArrayLike) -> ArrayLike:
    spacing_arr = np.asarray(spacing, dtype=float)
    if spacing_arr.ndim == 0:
        return np.full(3, float(spacing_arr), dtype=float)
    if spacing_arr.shape != (3,):
        raise ValueError("spacing must be a scalar or have shape (3,)")
    return spacing_arr


@dataclass(frozen=True)
class GridSDFGeometry:
    """Signed-distance geometry backed by sampled values on a regular Cartesian grid.

    Distances are queried in the geometry's local frame using trilinear interpolation over the
    sampled nodal values. Out-of-bounds queries are handled deterministically by clamping the
    interpolation coordinates to the grid's local bounding box.
    """

    origin: ArrayLike
    spacing: float | ArrayLike
    values: ArrayLike
    grid: UniformGrid3D = field(init=False, repr=False)

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=float)
        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")

        spacing = _as_spacing_vector(self.spacing)
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 3:
            raise ValueError("values must be a 3D array")

        grid = UniformGrid3D(origin=origin, spacing=spacing, shape=tuple(int(v) for v in values.shape))
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "grid", grid)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    def bounding_box(self) -> BoundingBox:
        extent = self.grid.spacing * (np.array(self.grid.shape, dtype=float) - 1.0)
        return BoundingBox(minimum=self.grid.origin, maximum=self.grid.origin + extent)

    def _grid_coordinates(self, point: ArrayLike) -> ArrayLike:
        point = np.asarray(point, dtype=float)
        if point.shape != (3,):
            raise ValueError("point must have shape (3,)")
        return (point - self.grid.origin) / self.grid.spacing

    def _cell_coordinates(self, point: ArrayLike) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        coord_raw = self._grid_coordinates(point)
        coord_clamped = np.clip(coord_raw, 0.0, np.array(self.grid.shape, dtype=float) - 1.0)
        lower = np.floor(coord_clamped).astype(int)
        upper_bound = np.array(self.grid.shape, dtype=int) - 2
        lower = np.clip(lower, 0, upper_bound)
        frac = coord_clamped - lower.astype(float)
        outside = (coord_raw < 0.0) | (coord_raw > (np.array(self.grid.shape, dtype=float) - 1.0))
        return coord_raw, lower, frac, outside

    def _corner_values(self, lower: ArrayLike) -> tuple[float, float, float, float, float, float, float, float]:
        i, j, k = (int(lower[0]), int(lower[1]), int(lower[2]))
        return (
            float(self.values[i, j, k]),
            float(self.values[i + 1, j, k]),
            float(self.values[i, j + 1, k]),
            float(self.values[i + 1, j + 1, k]),
            float(self.values[i, j, k + 1]),
            float(self.values[i + 1, j, k + 1]),
            float(self.values[i, j + 1, k + 1]),
            float(self.values[i + 1, j + 1, k + 1]),
        )

    def signed_distance(self, x: ArrayLike) -> float:
        _, lower, frac, _ = self._cell_coordinates(x)
        tx, ty, tz = (float(frac[0]), float(frac[1]), float(frac[2]))
        v000, v100, v010, v110, v001, v101, v011, v111 = self._corner_values(lower)

        c00 = (1.0 - tx) * v000 + tx * v100
        c10 = (1.0 - tx) * v010 + tx * v110
        c01 = (1.0 - tx) * v001 + tx * v101
        c11 = (1.0 - tx) * v011 + tx * v111
        c0 = (1.0 - ty) * c00 + ty * c10
        c1 = (1.0 - ty) * c01 + ty * c11
        return float((1.0 - tz) * c0 + tz * c1)

    def gradient(self, x: ArrayLike) -> ArrayLike:
        _, lower, frac, outside = self._cell_coordinates(x)
        tx, ty, tz = (float(frac[0]), float(frac[1]), float(frac[2]))
        v000, v100, v010, v110, v001, v101, v011, v111 = self._corner_values(lower)

        dc00_dtx = v100 - v000
        dc10_dtx = v110 - v010
        dc01_dtx = v101 - v001
        dc11_dtx = v111 - v011
        dc0_dtx = (1.0 - ty) * dc00_dtx + ty * dc10_dtx
        dc1_dtx = (1.0 - ty) * dc01_dtx + ty * dc11_dtx
        dvalue_dtx = (1.0 - tz) * dc0_dtx + tz * dc1_dtx

        dc0_dty = ((1.0 - tx) * v010 + tx * v110) - ((1.0 - tx) * v000 + tx * v100)
        dc1_dty = ((1.0 - tx) * v011 + tx * v111) - ((1.0 - tx) * v001 + tx * v101)
        dvalue_dty = (1.0 - tz) * dc0_dty + tz * dc1_dty

        dvalue_dtz = (
            (1.0 - ty) * ((1.0 - tx) * v001 + tx * v101)
            + ty * ((1.0 - tx) * v011 + tx * v111)
            - ((1.0 - ty) * ((1.0 - tx) * v000 + tx * v100) + ty * ((1.0 - tx) * v010 + tx * v110))
        )

        grad = np.array(
            [
                dvalue_dtx / self.grid.spacing[0],
                dvalue_dty / self.grid.spacing[1],
                dvalue_dtz / self.grid.spacing[2],
            ],
            dtype=float,
        )
        grad[outside] = 0.0
        return grad
