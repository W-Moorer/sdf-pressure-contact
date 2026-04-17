import numpy as np

from pfcsdf.contact import balance_value, balance_gradient


def test_balance_value_zero_at_equal_pressure():
    assert balance_value(3.0, 3.0) == 0.0
    assert balance_value(4.0, 1.0) == 3.0


def test_balance_gradient_returns_unit_normal():
    grad = balance_gradient(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, -1.0]))
    np.testing.assert_allclose(grad, np.array([0.0, 0.0, 1.0]))
