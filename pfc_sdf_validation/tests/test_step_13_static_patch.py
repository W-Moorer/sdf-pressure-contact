from __future__ import annotations

import numpy as np

from pfcsdf.contact.patch import build_rectangular_planar_patch, evaluate_static_patch
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.physics.pressure import LinearPressureLaw
from pfcsdf.solvers.static import compute_uniform_flat_contact


def test_patch_evaluator_matches_uniform_flat_contact_result() -> None:
    footprint = BoxFootprint(lx=2.0, ly=1.5)
    patch = build_rectangular_planar_patch(footprint)
    law_a = LinearPressureLaw(stiffness=8.0)
    law_b = LinearPressureLaw(stiffness=12.0)
    overlap = 0.2

    result_patch = evaluate_static_patch(
        patch,
        lambda local_xy, world_xyz: overlap,
        law_a,
        law_b,
    )
    result_flat = compute_uniform_flat_contact(overlap, footprint, law_a, law_b)

    assert np.allclose(result_patch.wrench.force, result_flat.wrench.force, atol=1e-12)
    assert np.allclose(result_patch.wrench.torque, result_flat.wrench.torque, atol=1e-12)
    assert np.isclose(result_patch.mean_pressure, result_flat.column.pressure, atol=1e-12)


def test_patch_evaluator_recovers_affine_pressure_resultants_and_moment() -> None:
    footprint = BoxFootprint(lx=2.0, ly=1.0)
    patch = build_rectangular_planar_patch(footprint)
    law_a = LinearPressureLaw(stiffness=10.0)
    law_b = LinearPressureLaw(stiffness=15.0)
    k_eq = law_a.stiffness * law_b.stiffness / (law_a.stiffness + law_b.stiffness)

    g0 = 0.3
    ax = 0.1
    by = -0.2

    result = evaluate_static_patch(
        patch,
        lambda local_xy, world_xyz: g0 + ax * local_xy[0] + by * local_xy[1],
        law_a,
        law_b,
        reference=patch.world_centroid,
    )

    area = footprint.area
    expected_force = np.array([0.0, 0.0, k_eq * g0 * area])
    expected_torque = np.array(
        [k_eq * by * area * footprint.ly**2 / 12.0, -k_eq * ax * area * footprint.lx**2 / 12.0, 0.0]
    )

    assert np.allclose(result.wrench.force, expected_force, atol=1e-12)
    assert np.allclose(result.wrench.torque, expected_torque, atol=1e-12)
