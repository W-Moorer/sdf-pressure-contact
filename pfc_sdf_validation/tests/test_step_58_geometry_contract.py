from __future__ import annotations

import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry import (
    BoundedSignedDistanceGeometry,
    DifferentiableSignedDistanceGeometry,
    SignedDistanceGeometry,
    signed_distance_gradient,
)
from pfcsdf.geometry.primitives import PlaneSDF, SphereSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


class SignedDistanceOnlyPlane:
    def __init__(self, *, point_z: float, normal_z: float) -> None:
        self.point_z = float(point_z)
        self.normal_z = float(normal_z)

    def signed_distance(self, x: np.ndarray) -> float:
        return float((np.asarray(x, dtype=float)[2] - self.point_z) * self.normal_z)


def test_primitives_satisfy_shared_geometry_contract() -> None:
    plane = PlaneSDF(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    sphere = SphereSDF(center=[1.0, -2.0, 3.0], radius=2.5)

    assert isinstance(plane, SignedDistanceGeometry)
    assert isinstance(plane, DifferentiableSignedDistanceGeometry)
    assert isinstance(sphere, SignedDistanceGeometry)
    assert isinstance(sphere, DifferentiableSignedDistanceGeometry)
    assert isinstance(sphere, BoundedSignedDistanceGeometry)

    bbox = sphere.bounding_box()
    np.testing.assert_allclose(bbox.minimum, np.array([-1.5, -4.5, 0.5]))
    np.testing.assert_allclose(bbox.maximum, np.array([3.5, 0.5, 5.5]))


def test_signed_distance_gradient_falls_back_for_distance_only_geometry() -> None:
    geometry = SignedDistanceOnlyPlane(point_z=0.25, normal_z=1.0)
    grad = signed_distance_gradient(geometry, np.array([0.4, -0.2, 1.3]))
    np.testing.assert_allclose(grad, np.array([0.0, 0.0, 1.0]), atol=1e-6)


def test_native_band_accepts_shared_geometry_contract_objects() -> None:
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    dx = dy = 0.05
    dz = 0.01
    nx, ny, nz = 40, 24, 81
    grid = UniformGrid3D(
        origin=np.array([-0.5 * nx * dx + 0.5 * dx, -0.5 * ny * dy + 0.5 * dy, -0.5 * nz * dz + 0.5 * dz]),
        spacing=np.array([dx, dy, dz]),
        shape=(nx, ny, nz),
    )
    sdf_a = SignedDistanceOnlyPlane(point_z=-0.5 * overlap, normal_z=-1.0)
    sdf_b = SignedDistanceOnlyPlane(point_z=0.5 * overlap, normal_z=1.0)
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
    result = accumulate_sdf_native_band_wrench(
        fields,
        NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4),
    )

    area = grid.shape[0] * grid.spacing[0] * grid.shape[1] * grid.spacing[1]
    k_eq = k_a * k_b / (k_a + k_b)
    expected_force = area * k_eq * overlap

    assert math.isclose(result.wrench.force[2], expected_force, rel_tol=2e-2, abs_tol=1e-8)
    assert result.active_count < 0.2 * result.total_count
