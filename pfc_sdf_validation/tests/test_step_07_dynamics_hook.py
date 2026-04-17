import numpy as np

from pfcsdf.solvers import midpoint_step_1d


def test_midpoint_step_matches_constant_acceleration_solution():
    a = -9.81
    result = midpoint_step_1d(
        z=1.0,
        v=2.0,
        dt=0.1,
        acceleration=lambda z, v: a,
    )
    expected_z = 1.0 + 2.0 * 0.1 + 0.5 * a * 0.1**2
    expected_v = 2.0 + a * 0.1
    assert np.isclose(result.z_next, expected_z)
    assert np.isclose(result.v_next, expected_v)
