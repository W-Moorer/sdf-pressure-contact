import numpy as np

from pfcsdf.contact import accumulate_uniform_pressure_wrench


def test_uniform_pressure_wrench_has_expected_force_and_torque():
    result = accumulate_uniform_pressure_wrench(
        pressure=5.0,
        area=2.0,
        normal=np.array([0.0, 0.0, 1.0]),
        center=np.array([1.0, 0.0, 0.0]),
        reference=np.array([0.0, 0.0, 0.0]),
    )
    np.testing.assert_allclose(result.force, np.array([0.0, 0.0, 10.0]))
    np.testing.assert_allclose(result.torque, np.array([0.0, -10.0, 0.0]))
