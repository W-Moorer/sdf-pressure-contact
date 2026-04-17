import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_higher_order_sparse_sdf_native_band_wrench,
    accumulate_sparse_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry.primitives import PlaneSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def build_underresolved_flat_fields(*, overlap: float, k_a: float, k_b: float):
    dx = dy = 0.08
    dz = 0.02
    nx, ny, nz = 20, 12, 21
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


def test_higher_order_sparse_band_recovers_underresolved_flat_contact_measure_and_force():
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    grid, fields = build_underresolved_flat_fields(overlap=overlap, k_a=k_a, k_b=k_b)
    config = NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4)

    first_order = accumulate_sparse_sdf_native_band_wrench(
        fields,
        config,
        local_normal_correction=True,
        use_projected_points=True,
    )
    higher_order = accumulate_higher_order_sparse_sdf_native_band_wrench(
        fields,
        config,
        local_normal_correction=True,
        use_projected_points=True,
    )

    area = grid.shape[0] * grid.spacing[0] * grid.shape[1] * grid.spacing[1]
    k_eq = k_a * k_b / (k_a + k_b)
    expected_force = area * k_eq * overlap

    assert first_order.weighted_measure < 0.25 * area
    assert math.isclose(higher_order.weighted_measure, area, rel_tol=4e-2, abs_tol=1e-8)
    assert math.isclose(higher_order.wrench.force[2], expected_force, rel_tol=4e-2, abs_tol=1e-8)


def test_higher_order_sparse_band_preserves_offset_torque_relation_for_flat_contact():
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    _, fields = build_underresolved_flat_fields(overlap=overlap, k_a=k_a, k_b=k_b)
    config = NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4)
    reference = np.array([0.11, -0.09, 0.03])

    result = accumulate_higher_order_sparse_sdf_native_band_wrench(
        fields,
        config,
        reference=reference,
        local_normal_correction=True,
        use_projected_points=True,
    )

    expected_torque = np.cross(-reference, result.wrench.force)
    assert np.allclose(result.wrench.torque, expected_torque, atol=1e-10)
