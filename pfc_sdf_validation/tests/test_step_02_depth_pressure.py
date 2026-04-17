import numpy as np

from pfcsdf.physics import depth_from_phi, LinearPressureLaw


def test_depth_from_phi_clamps_correctly():
    phi = np.array([1.0, -0.2, -5.0])
    depth = depth_from_phi(phi, max_depth=1.0)
    np.testing.assert_allclose(depth, np.array([0.0, 0.2, 1.0]))


def test_linear_pressure_law():
    law = LinearPressureLaw(stiffness=10.0)
    depth = np.array([0.0, 0.1, 0.2])
    np.testing.assert_allclose(law.pressure(depth), np.array([0.0, 1.0, 2.0]))
