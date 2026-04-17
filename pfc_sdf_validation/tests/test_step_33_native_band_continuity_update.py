from __future__ import annotations

import numpy as np

from pfcsdf.contact.active_set import ActiveSetSnapshot
from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    build_higher_order_sparse_active_traversal,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.dynamics.benchmarks import _LowerOverlapPlane, _UpperOverlapPlane
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def _build_flat_fields(overlap: float):
    grid = UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(5, 5, 21))
    law = LinearPressureLaw(200.0)
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
    points = grid.stacked_points()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            extra_mask[i, j, :] = footprint.contains_xy(points[i, j, 0])
    return fields, extra_mask


def test_warm_start_boundary_only_update_recovers_dense_active_mask() -> None:
    config = NativeBandAccumulatorConfig(eta=8.0, band_half_width=12.0)
    prev_fields, extra_mask = _build_flat_fields(overlap=0.06)
    prev_traversal = build_higher_order_sparse_active_traversal(
        prev_fields,
        config,
        extra_mask=extra_mask,
        local_normal_correction=True,
    )
    snapshot = ActiveSetSnapshot(prev_traversal.active_mask, measure=1.0)

    curr_fields, _ = _build_flat_fields(overlap=0.07)
    dense = build_higher_order_sparse_active_traversal(
        curr_fields,
        config,
        extra_mask=extra_mask,
        local_normal_correction=True,
    )
    warm = build_higher_order_sparse_active_traversal(
        curr_fields,
        config,
        extra_mask=extra_mask,
        local_normal_correction=True,
        warm_start_snapshot=snapshot,
        continuity_dilation_radius=1,
        boundary_only_update=True,
    )

    assert np.array_equal(warm.active_mask, dense.active_mask)
    assert warm.candidate_count < dense.candidate_count
    assert warm.recompute_count < warm.candidate_count
    assert warm.retained_count > 0
