import numpy as np

from pfcsdf.geometry.primitives import PlaneSDF, SphereSDF
from pfcsdf.geometry.volume import (
    UniformGrid3D,
    narrow_band_mask,
    sample_linear_balance_field_from_sdfs,
    sample_scalar_field,
)


def test_uniform_grid_sampling_and_narrow_band_mask() -> None:
    grid = UniformGrid3D(origin=np.array([-1.0, -1.0, -1.0]), spacing=np.array([0.5, 0.5, 0.5]), shape=(5, 5, 5))
    field = sample_scalar_field(grid, lambda p: p[0] + 2.0 * p[1] - p[2])

    assert field.values.shape == (5, 5, 5)
    np.testing.assert_allclose(field.values[2, 1, 4], 0.0 + 2.0 * (-0.5) - 1.0)

    mask = narrow_band_mask(field, half_width=0.26)
    assert mask.shape == field.values.shape
    assert mask[2, 2, 2]
    assert not mask[4, 4, 0]


def test_linear_balance_field_from_sdfs_has_central_sign_change() -> None:
    grid = UniformGrid3D(origin=np.array([-0.5, -0.5, -0.5]), spacing=np.array([0.25, 0.25, 0.125]), shape=(5, 5, 9))
    plane = PlaneSDF(point=np.zeros(3), normal=np.array([0.0, 0.0, 1.0]))
    sphere = SphereSDF(center=np.array([0.0, 0.0, 0.75]), radius=1.0)

    balance = sample_linear_balance_field_from_sdfs(
        grid,
        plane,
        sphere,
        stiffness_a=1.0,
        stiffness_b=1.0,
        max_depth_a=2.0,
        max_depth_b=2.0,
    )

    central_column = balance.values[2, 2, :]
    has_sign_change = np.any(central_column[:-1] * central_column[1:] < 0.0)
    hits_zero_level = np.any(np.isclose(central_column, 0.0, atol=1e-12))
    assert has_sign_change or hits_zero_level
    assert np.min(np.abs(central_column)) < 0.2
