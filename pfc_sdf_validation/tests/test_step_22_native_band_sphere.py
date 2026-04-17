import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry.primitives import PlaneSDF, SphereSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw
from pfcsdf.solvers.static import compute_sphere_plane_contact_linear_exact


def test_sdf_native_band_approximates_sphere_plane_linear_force():
    radius = 1.2
    overlap = 0.18
    k_sphere = 12.0
    k_plane = 18.0

    dx = dy = 0.03
    dz = 0.01
    nx = ny = 121
    nz = 80
    grid = UniformGrid3D(
        origin=np.array([-0.5 * nx * dx + 0.5 * dx, -0.5 * ny * dy + 0.5 * dy, -0.25 + 0.5 * dz]),
        spacing=np.array([dx, dy, dz]),
        shape=(nx, ny, nz),
    )

    sphere = SphereSDF(center=np.array([0.0, 0.0, radius - overlap]), radius=radius)
    plane = PlaneSDF(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    law_sphere = LinearPressureLaw(k_sphere)
    law_plane = LinearPressureLaw(k_plane)

    fields = sample_linear_pfc_balance_fields(
        grid, sphere, plane, law_sphere, law_plane, max_depth_a=2.0, max_depth_b=2.0
    )
    config = NativeBandAccumulatorConfig(eta=0.4, band_half_width=0.6)
    result = accumulate_sdf_native_band_wrench(fields, config)

    exact = compute_sphere_plane_contact_linear_exact(
        overlap, radius, law_sphere, law_plane, normal=np.array([0.0, 0.0, 1.0])
    )

    assert math.isclose(result.wrench.force[0], 0.0, abs_tol=1e-10)
    assert math.isclose(result.wrench.force[1], 0.0, abs_tol=1e-10)
    assert math.isclose(result.wrench.force[2], exact.wrench.force[2], rel_tol=6e-2, abs_tol=5e-3)
    assert np.allclose(result.wrench.torque, np.zeros(3), atol=5e-3)
    assert result.active_count < 0.15 * result.total_count
