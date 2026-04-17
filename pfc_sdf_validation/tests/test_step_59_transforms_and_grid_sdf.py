from __future__ import annotations

import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry import GridSDFGeometry, TransformedGeometry
from pfcsdf.geometry.primitives import PlaneSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def _rotation_y(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=float,
    )


def _rotation_z(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def _sample_grid_sdf(
    grid: UniformGrid3D,
    func,
) -> GridSDFGeometry:
    values = np.empty(grid.shape, dtype=float)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(grid.shape[2]):
                values[i, j, k] = float(func(grid.point(i, j, k)))
    return GridSDFGeometry(origin=grid.origin, spacing=grid.spacing, values=values)


def test_transformed_geometry_distance_and_gradient_follow_rigid_motion() -> None:
    local_plane = PlaneSDF(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    transform = TransformedGeometry(
        geometry=local_plane,
        rotation=_rotation_y(0.5 * math.pi),
        translation=np.array([1.0, -2.0, 0.5]),
    )

    assert math.isclose(transform.signed_distance(np.array([3.0, -2.0, 0.5])), 2.0, abs_tol=1e-12)
    assert math.isclose(transform.signed_distance(np.array([-2.0, -2.0, 0.5])), -3.0, abs_tol=1e-12)
    np.testing.assert_allclose(transform.gradient(np.array([4.0, 1.5, -6.0])), np.array([1.0, 0.0, 0.0]), atol=1e-12)


def test_grid_sdf_geometry_reproduces_linear_field_and_clamps_outside() -> None:
    grid = UniformGrid3D(
        origin=np.array([-1.0, 0.5, -0.25]),
        spacing=np.array([0.4, 0.5, 0.25]),
        shape=(6, 5, 7),
    )
    geometry = _sample_grid_sdf(grid, lambda p: 1.5 * p[0] - 0.25 * p[1] + 2.0 * p[2] - 0.3)

    query = np.array([-0.34, 1.15, 0.28])
    expected = 1.5 * query[0] - 0.25 * query[1] + 2.0 * query[2] - 0.3
    assert math.isclose(geometry.signed_distance(query), expected, rel_tol=1e-12, abs_tol=1e-12)
    np.testing.assert_allclose(geometry.gradient(query), np.array([1.5, -0.25, 2.0]), atol=1e-12)

    bbox = geometry.bounding_box()
    np.testing.assert_allclose(bbox.minimum, np.array([-1.0, 0.5, -0.25]))
    np.testing.assert_allclose(bbox.maximum, np.array([1.0, 2.5, 1.25]))

    outside_query = np.array([-0.34, 1.15, 4.0])
    clamped_query = np.array([-0.34, 1.15, 1.25])
    clamped_expected = 1.5 * clamped_query[0] - 0.25 * clamped_query[1] + 2.0 * clamped_query[2] - 0.3
    assert math.isclose(geometry.signed_distance(outside_query), clamped_expected, rel_tol=1e-12, abs_tol=1e-12)
    np.testing.assert_allclose(geometry.gradient(outside_query), np.array([1.5, -0.25, 0.0]), atol=1e-12)


def test_transformed_geometry_bounding_box_is_world_space_aabb() -> None:
    geometry = GridSDFGeometry(
        origin=np.zeros(3),
        spacing=np.array([1.0, 1.0, 1.0]),
        values=np.zeros((3, 5, 7), dtype=float),
    )
    transform = TransformedGeometry(
        geometry=geometry,
        rotation=_rotation_z(0.5 * math.pi),
        translation=np.array([10.0, 20.0, 30.0]),
    )

    bbox = transform.bounding_box()
    np.testing.assert_allclose(bbox.minimum, np.array([6.0, 20.0, 30.0]), atol=1e-12)
    np.testing.assert_allclose(bbox.maximum, np.array([10.0, 22.0, 36.0]), atol=1e-12)


def test_native_band_accepts_grid_backed_geometry() -> None:
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    dx = dy = 0.05
    dz = 0.01
    nx, ny, nz = 40, 24, 81
    grid = UniformGrid3D(
        origin=np.array([-0.5 * nx * dx, -0.5 * ny * dy, -0.5 * nz * dz]),
        spacing=np.array([dx, dy, dz]),
        shape=(nx, ny, nz),
    )
    plane_a = PlaneSDF(point=np.array([0.0, 0.0, -0.5 * overlap]), normal=np.array([0.0, 0.0, -1.0]))
    plane_b = PlaneSDF(point=np.array([0.0, 0.0, 0.5 * overlap]), normal=np.array([0.0, 0.0, 1.0]))
    sdf_a = _sample_grid_sdf(grid, plane_a.signed_distance)
    sdf_b = _sample_grid_sdf(grid, plane_b.signed_distance)
    law_a = LinearPressureLaw(k_a)
    law_b = LinearPressureLaw(k_b)

    fields = sample_linear_pfc_balance_fields(
        grid,
        sdf_a,
        sdf_b,
        law_a,
        law_b,
        max_depth_a=1.0,
        max_depth_b=1.0,
    )
    result = accumulate_sdf_native_band_wrench(fields, NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4))

    area = grid.shape[0] * grid.spacing[0] * grid.shape[1] * grid.spacing[1]
    k_eq = k_a * k_b / (k_a + k_b)
    expected_force = area * k_eq * overlap

    assert math.isclose(result.wrench.force[2], expected_force, rel_tol=2e-2, abs_tol=1e-8)
    assert result.active_count < 0.2 * result.total_count
