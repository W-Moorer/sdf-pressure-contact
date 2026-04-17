import numpy as np

from pfcsdf.physics import LinearPressureLaw
from pfcsdf.solvers import (
    compute_sphere_plane_contact_linear_exact,
    compute_sphere_plane_contact_linear_quadrature,
)


def test_sphere_plane_linear_exact_recovers_closed_form_force():
    overlap = 0.05
    radius = 1.5
    law_a = LinearPressureLaw(stiffness=1200.0)
    law_b = LinearPressureLaw(stiffness=1800.0)

    result = compute_sphere_plane_contact_linear_exact(overlap, radius, law_a, law_b)

    k_eq = law_a.stiffness * law_b.stiffness / (law_a.stiffness + law_b.stiffness)
    expected_force = np.pi * k_eq * overlap**2 * (radius - overlap / 3.0)
    expected_peak = k_eq * overlap

    assert np.isclose(result.peak_pressure, expected_peak)
    np.testing.assert_allclose(result.wrench.force, np.array([0.0, 0.0, expected_force]))
    np.testing.assert_allclose(result.wrench.torque, np.zeros(3))


def test_sphere_plane_quadrature_matches_closed_form_force():
    overlap = 0.04
    radius = 2.0
    law_a = LinearPressureLaw(stiffness=800.0)
    law_b = LinearPressureLaw(stiffness=2000.0)

    exact = compute_sphere_plane_contact_linear_exact(overlap, radius, law_a, law_b)
    numeric = compute_sphere_plane_contact_linear_quadrature(
        overlap,
        radius,
        law_a,
        law_b,
        n_rings=6000,
    )

    assert np.isclose(numeric.contact_radius, exact.contact_radius)
    assert np.isclose(numeric.peak_pressure, exact.peak_pressure)
    np.testing.assert_allclose(numeric.wrench.force, exact.wrench.force, rtol=5e-4, atol=1e-8)
    np.testing.assert_allclose(numeric.wrench.torque, np.zeros(3), atol=1e-12)
