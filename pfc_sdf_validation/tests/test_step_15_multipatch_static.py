from __future__ import annotations

import numpy as np

from pfcsdf.contact.patch import (
    StaticPatchTask,
    build_rectangular_planar_patch,
    clip_planar_patch_with_halfspace,
    evaluate_static_patch,
    evaluate_static_patch_collection,
)
from pfcsdf.geometry.polygon import HalfSpace2D
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.physics.pressure import LinearPressureLaw


def test_multipatch_collection_matches_single_patch_for_exact_partition() -> None:
    patch = build_rectangular_planar_patch(BoxFootprint(lx=2.0, ly=1.0))
    left = clip_planar_patch_with_halfspace(patch, HalfSpace2D(normal=np.array([-1.0, 0.0]), offset=0.0))
    right = clip_planar_patch_with_halfspace(patch, HalfSpace2D(normal=np.array([1.0, 0.0]), offset=0.0))
    assert left is not None and right is not None

    law_a = LinearPressureLaw(stiffness=8.0)
    law_b = LinearPressureLaw(stiffness=12.0)
    overlap = 0.2

    single = evaluate_static_patch(patch, lambda local_xy, world_xyz: overlap, law_a, law_b)
    multi = evaluate_static_patch_collection(
        [
            StaticPatchTask(left, lambda local_xy, world_xyz: overlap),
            StaticPatchTask(right, lambda local_xy, world_xyz: overlap),
        ],
        law_a,
        law_b,
    )

    assert np.isclose(multi.total_area, patch.area)
    assert np.allclose(multi.wrench.force, single.wrench.force, atol=1e-12)
    assert np.allclose(multi.wrench.torque, single.wrench.torque, atol=1e-12)



def test_multipatch_collection_accumulates_piecewise_constant_wrench_correctly() -> None:
    patch = build_rectangular_planar_patch(BoxFootprint(lx=2.0, ly=1.0))
    left = clip_planar_patch_with_halfspace(patch, HalfSpace2D(normal=np.array([-1.0, 0.0]), offset=0.0))
    right = clip_planar_patch_with_halfspace(patch, HalfSpace2D(normal=np.array([1.0, 0.0]), offset=0.0))
    assert left is not None and right is not None

    law_a = LinearPressureLaw(stiffness=10.0)
    law_b = LinearPressureLaw(stiffness=15.0)
    k_eq = law_a.stiffness * law_b.stiffness / (law_a.stiffness + law_b.stiffness)

    overlap_left = 0.2
    overlap_right = 0.4
    multi = evaluate_static_patch_collection(
        [
            StaticPatchTask(left, lambda local_xy, world_xyz: overlap_left),
            StaticPatchTask(right, lambda local_xy, world_xyz: overlap_right),
        ],
        law_a,
        law_b,
        reference=np.zeros(3),
    )

    f_left = k_eq * left.area * overlap_left
    f_right = k_eq * right.area * overlap_right
    expected_force = np.array([0.0, 0.0, f_left + f_right])
    expected_torque = np.array([0.0, 0.5 * (f_left - f_right), 0.0])

    assert np.allclose(multi.wrench.force, expected_force, atol=1e-12)
    assert np.allclose(multi.wrench.torque, expected_torque, atol=1e-12)
