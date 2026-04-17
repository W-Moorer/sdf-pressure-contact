import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    PerCellCubatureRule,
    build_higher_order_sparse_active_traversal,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry.primitives import PlaneSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def build_flat_fields(*, overlap: float, k_a: float, k_b: float):
    dx = dy = 0.05
    dz = 0.01
    nx, ny, nz = 16, 12, 41
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


def test_gauss_legendre_tensor_rule_is_symmetric_and_normalized():
    rule = PerCellCubatureRule.gauss_legendre_tensor_2()
    assert rule.offsets.shape == (8, 3)
    assert rule.weights.shape == (8,)
    assert math.isclose(float(np.sum(rule.weights)), 1.0, abs_tol=1e-12)
    assert np.allclose(np.sum(rule.weights[:, None] * rule.offsets, axis=0), np.zeros(3), atol=1e-12)
    second_moment = np.sum(rule.weights * rule.offsets[:, 0] ** 2)
    assert math.isclose(float(second_moment), 1.0 / 12.0, rel_tol=1e-12, abs_tol=1e-12)


def test_higher_order_sparse_traversal_builds_cubature_samples_per_active_cell():
    overlap = 0.08
    grid, fields = build_flat_fields(overlap=overlap, k_a=10.0, k_b=15.0)
    config = NativeBandAccumulatorConfig(eta=0.2, band_half_width=0.3)
    rule = PerCellCubatureRule.gauss_legendre_tensor_2()

    traversal = build_higher_order_sparse_active_traversal(
        fields,
        config,
        cubature_rule=rule,
        local_normal_correction=True,
    )

    assert traversal.active_count > 0
    first_cell = traversal.active_cells[0]
    assert len(first_cell.cubature_samples) == 8

    k_a = 10.0
    k_b = 15.0
    k_eq = k_a * k_b / (k_a + k_b)
    expected_pressure = k_eq * overlap
    expected_balance_plane_z = 0.5 * overlap * (k_b - k_a) / (k_a + k_b)
    for sample in first_cell.cubature_samples:
        assert math.isclose(sample.projected_point[2], expected_balance_plane_z, abs_tol=1e-12)
        assert math.isclose(sample.pressure_local_normal, expected_pressure, rel_tol=1e-12, abs_tol=1e-12)
        assert sample.weight >= 0.0

    area = grid.shape[0] * grid.spacing[0] * grid.shape[1] * grid.spacing[1]
    total_weight = sum(cell.integrated_weight for cell in traversal.active_cells)
    assert math.isclose(total_weight, area, rel_tol=3e-2, abs_tol=1e-8)
