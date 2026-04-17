import numpy as np

from pfcsdf.geometry import BoxFootprint
from pfcsdf.physics import LinearPressureLaw
from pfcsdf.solvers import compute_uniform_flat_contact


def test_uniform_flat_contact_recovers_force_area_keq_delta():
    overlap = 0.02
    law_a = LinearPressureLaw(stiffness=1000.0)
    law_b = LinearPressureLaw(stiffness=3000.0)
    footprint = BoxFootprint(lx=0.4, ly=0.5)

    result = compute_uniform_flat_contact(overlap, footprint, law_a, law_b)

    k_eq = law_a.stiffness * law_b.stiffness / (law_a.stiffness + law_b.stiffness)
    expected_force = footprint.area * k_eq * overlap

    assert np.isclose(result.column.pressure, k_eq * overlap)
    np.testing.assert_allclose(result.wrench.force, np.array([0.0, 0.0, expected_force]))
    np.testing.assert_allclose(result.wrench.torque, np.zeros(3))
