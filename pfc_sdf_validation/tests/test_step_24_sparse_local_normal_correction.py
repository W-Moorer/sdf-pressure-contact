import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_sparse_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry.primitives import PlaneSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def build_flat_fields(*, overlap: float, k_a: float, k_b: float):
    dx = dy = 0.05
    dz = 0.01
    nx, ny, nz = 40, 24, 81
    grid = UniformGrid3D(
        origin=np.array([-0.5 * nx * dx + 0.5 * dx, -0.5 * ny * dy + 0.5 * dy, -0.5 * nz * dz + 0.5 * dz]),
        spacing=np.array([dx, dy, dz]),
        shape=(nx, ny, nz),
    )
    sdf_a = PlaneSDF(point=np.array([0.0, 0.0, -0.5 * overlap]), normal=np.array([0.0, 0.0, -1.0]))
    sdf_b = PlaneSDF(point=np.array([0.0, 0.0, 0.5 * overlap]), normal=np.array([0.0, 0.0, 1.0]))
    law_a = LinearPressureLaw(k_a)
    law_b = LinearPressureLaw(k_b)
    fields = sample_linear_pfc_balance_fields(
        grid, sdf_a, sdf_b, law_a, law_b, max_depth_a=1.0, max_depth_b=1.0
    )
    return grid, fields


def test_sparse_local_normal_accumulator_recovers_flat_force_measure():
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    grid, fields = build_flat_fields(overlap=overlap, k_a=k_a, k_b=k_b)
    config = NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4)

    result = accumulate_sparse_sdf_native_band_wrench(fields, config, local_normal_correction=True)

    area = grid.shape[0] * grid.spacing[0] * grid.shape[1] * grid.spacing[1]
    k_eq = k_a * k_b / (k_a + k_b)
    expected_force = area * k_eq * overlap

    assert math.isclose(result.wrench.force[0], 0.0, abs_tol=1e-12)
    assert math.isclose(result.wrench.force[1], 0.0, abs_tol=1e-12)
    assert math.isclose(result.wrench.force[2], expected_force, rel_tol=2e-2, abs_tol=1e-8)
    assert np.allclose(result.wrench.torque, np.zeros(3), atol=1e-12)
    assert math.isclose(result.weighted_measure, area, rel_tol=2e-2, abs_tol=1e-8)


def test_sparse_local_normal_accumulator_recovers_offset_torque_for_flat_contact():
    overlap = 0.08
    k_a = 10.0
    k_b = 15.0
    _, fields = build_flat_fields(overlap=overlap, k_a=k_a, k_b=k_b)
    config = NativeBandAccumulatorConfig(eta=0.2, band_half_width=0.3)
    reference = np.array([0.13, -0.17, 0.0])

    result = accumulate_sparse_sdf_native_band_wrench(
        fields,
        config,
        reference=reference,
        local_normal_correction=True,
        use_projected_points=True,
    )

    expected_torque = np.cross(-reference, result.wrench.force)
    assert np.allclose(result.wrench.torque, expected_torque, atol=1e-10)
