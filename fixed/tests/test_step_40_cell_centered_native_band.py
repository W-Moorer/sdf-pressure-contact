from __future__ import annotations

import math

import numpy as np

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig, accumulate_sparse_sdf_native_band_wrench, sample_linear_pfc_balance_fields
from pfcsdf.dynamics.benchmarks import _LowerOverlapPlane, _UpperOverlapPlane
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def test_uniform_grid_cell_centers_are_offset_by_half_spacing() -> None:
    grid = UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(4, 4, 20))
    assert np.allclose(grid.cell_center_origin, np.array([-0.3, -0.3, -0.095]))
    assert np.allclose(grid.cell_center_point(0, 0, 0), np.array([-0.3, -0.3, -0.095]))
    assert np.allclose(grid.cell_center_point(3, 3, 19), np.array([0.3, 0.3, 0.095]))


def test_native_band_flat_force_uses_cell_centered_support_area() -> None:
    grid = UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(4, 4, 20))
    law = LinearPressureLaw(200.0)
    overlap = 0.05
    fields = sample_linear_pfc_balance_fields(
        grid,
        _UpperOverlapPlane(overlap),
        _LowerOverlapPlane(overlap),
        law,
        law,
        max_depth_a=0.2,
        max_depth_b=0.2,
    )
    footprint = BoxFootprint(0.8, 0.8)
    extra_mask = np.zeros(grid.shape, dtype=bool)
    centers = grid.stacked_cell_centers()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            extra_mask[i, j, :] = footprint.contains_xy(centers[i, j, 0])

    result = accumulate_sparse_sdf_native_band_wrench(
        fields,
        NativeBandAccumulatorConfig(eta=8.0, band_half_width=12.0),
        extra_mask=extra_mask,
        local_normal_correction=True,
        use_projected_points=True,
    )

    expected_area = footprint.area
    expected_force = expected_area * 100.0 * overlap
    active_area = float(np.count_nonzero(extra_mask[:, :, 0]) * grid.spacing[0] * grid.spacing[1])

    assert math.isclose(active_area, expected_area, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.wrench.force[2], expected_force, rel_tol=2e-3, abs_tol=1e-10)
