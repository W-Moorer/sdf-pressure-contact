import math

import numpy as np

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    active_contact_mask,
    build_sparse_active_traversal,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry.primitives import PlaneSDF
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def build_flat_fields(*, overlap: float, k_a: float, k_b: float):
    dx = dy = 0.08
    dz = 0.02
    nx, ny, nz = 18, 14, 41
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


def test_sparse_active_traversal_matches_active_mask_count():
    _, fields = build_flat_fields(overlap=0.12, k_a=14.0, k_b=21.0)
    config = NativeBandAccumulatorConfig(eta=0.2, band_half_width=0.35)

    traversal = build_sparse_active_traversal(fields, config)
    dense_mask = active_contact_mask(fields, config)

    assert traversal.active_count == int(np.count_nonzero(dense_mask))
    assert traversal.active_count < 0.12 * traversal.total_count


def test_sparse_cells_project_to_flat_equal_pressure_plane_and_correct_column_pressure():
    overlap = 0.12
    k_a = 14.0
    k_b = 21.0
    _, fields = build_flat_fields(overlap=overlap, k_a=k_a, k_b=k_b)
    config = NativeBandAccumulatorConfig(eta=0.2, band_half_width=0.35)
    traversal = build_sparse_active_traversal(fields, config, local_normal_correction=True)

    expected_z = ((k_b - k_a) / (k_a + k_b)) * (0.5 * overlap)
    k_eq = k_a * k_b / (k_a + k_b)
    expected_pressure = k_eq * overlap

    sample = traversal.active_cells[len(traversal.active_cells) // 2]
    assert math.isclose(sample.projected_point[2], expected_z, rel_tol=0.0, abs_tol=1e-12)
    assert np.allclose(sample.normal, np.array([0.0, 0.0, 1.0]), atol=1e-12)
    assert math.isclose(sample.pressure_local_normal, expected_pressure, rel_tol=0.0, abs_tol=1e-12)
