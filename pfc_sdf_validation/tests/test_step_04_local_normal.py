import numpy as np

from pfcsdf.physics import LinearPressureLaw
from pfcsdf.contact import solve_column_equilibrium


def test_local_normal_equilibrium_matches_linear_closed_form():
    overlap = 0.3
    law_a = LinearPressureLaw(stiffness=100.0)
    law_b = LinearPressureLaw(stiffness=300.0)

    result = solve_column_equilibrium(overlap, law_a, law_b)

    expected_da = law_b.stiffness / (law_a.stiffness + law_b.stiffness) * overlap
    expected_db = law_a.stiffness / (law_a.stiffness + law_b.stiffness) * overlap
    expected_p = law_a.stiffness * expected_da

    assert np.isclose(result.depth_a, expected_da)
    assert np.isclose(result.depth_b, expected_db)
    assert np.isclose(result.pressure, expected_p)
    assert np.isclose(result.depth_a + result.depth_b, overlap)
