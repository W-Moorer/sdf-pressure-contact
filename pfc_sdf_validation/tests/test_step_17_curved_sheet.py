import numpy as np

from pfcsdf.contact.reconstruct import integrate_uniform_pressure_on_curved_sheet, reconstruct_curved_sheets_from_sampled_field
from pfcsdf.geometry.volume import UniformGrid3D, sample_scalar_field


def test_single_curved_sheet_reconstruction_and_uniform_pressure_wrench() -> None:
    grid = UniformGrid3D(origin=np.array([-1.0, -1.0, -0.2]), spacing=np.array([0.1, 0.1, 0.05]), shape=(21, 21, 13))

    def paraboloid_field(point: np.ndarray) -> float:
        x, y, z = point
        surface_z = 0.2 * (x**2 + y**2)
        return z - surface_z

    sampled = sample_scalar_field(grid, paraboloid_field)
    reconstruction = reconstruct_curved_sheets_from_sampled_field(sampled, band_half_width=0.06)

    assert reconstruction.num_components == 1
    assert reconstruction.root_mask.all()

    component = reconstruction.components[0]
    assert component.num_faces > 0
    assert component.area > 4.0
    np.testing.assert_allclose(component.centroid[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(component.centroid[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(component.centroid[2], 2.0 / 15.0, atol=2.5e-2)
    np.testing.assert_allclose(component.mean_normal, np.array([0.0, 0.0, 1.0]), atol=2.0e-2)

    wrench = integrate_uniform_pressure_on_curved_sheet(component, pressure=2.0, reference=np.zeros(3))
    np.testing.assert_allclose(wrench.force[:2], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(wrench.force[2], 8.0, atol=1e-12)
    np.testing.assert_allclose(wrench.torque, np.zeros(3), atol=1e-12)
