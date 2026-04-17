import numpy as np

from pfcsdf.contact.marching import (
    integrate_uniform_pressure_on_triangle_mesh,
    reconstruct_sheet_mesh_marching_cubes,
)
from pfcsdf.geometry.volume import UniformGrid3D, sample_scalar_field


def test_marching_cubes_extracts_single_sphere_component() -> None:
    radius = 0.75
    grid = UniformGrid3D(origin=np.array([-1.2, -1.2, -1.2]), spacing=np.array([0.08, 0.08, 0.08]), shape=(31, 31, 31))

    field = sample_scalar_field(grid, lambda p: np.linalg.norm(p) - radius)
    reconstruction = reconstruct_sheet_mesh_marching_cubes(field)

    assert reconstruction.num_components == 1
    component = reconstruction.components[0]
    assert component.num_faces > 0
    assert component.num_vertices > 0

    exact_area = 4.0 * np.pi * radius**2
    assert np.isclose(component.area, exact_area, rtol=0.08)
    assert np.allclose(component.centroid, np.zeros(3), atol=2.5e-2)

    wrench = integrate_uniform_pressure_on_triangle_mesh(component, pressure=1.7)
    assert np.allclose(wrench.force, np.zeros(3), atol=7.5e-2)
    assert np.allclose(wrench.torque, np.zeros(3), atol=7.5e-2)
