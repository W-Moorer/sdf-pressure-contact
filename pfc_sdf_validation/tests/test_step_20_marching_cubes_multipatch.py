import numpy as np

from pfcsdf.contact.marching import (
    integrate_uniform_pressure_on_mesh_collection,
    reconstruct_sheet_mesh_marching_cubes,
)
from pfcsdf.geometry.volume import UniformGrid3D, sample_scalar_field


def test_marching_cubes_splits_two_disconnected_spheres() -> None:
    radius = 0.55
    center_a = np.array([-0.8, 0.0, 0.0])
    center_b = np.array([0.8, 0.0, 0.0])

    grid = UniformGrid3D(origin=np.array([-1.8, -1.1, -1.1]), spacing=np.array([0.08, 0.08, 0.08]), shape=(46, 28, 28))

    def union_sdf(point: np.ndarray) -> float:
        phi_a = np.linalg.norm(point - center_a) - radius
        phi_b = np.linalg.norm(point - center_b) - radius
        return min(phi_a, phi_b)

    field = sample_scalar_field(grid, union_sdf)
    reconstruction = reconstruct_sheet_mesh_marching_cubes(field)

    assert reconstruction.num_components == 2

    centroids = sorted((component.centroid for component in reconstruction.components), key=lambda c: c[0])
    assert np.allclose(centroids[0], center_a, atol=4.5e-2)
    assert np.allclose(centroids[1], center_b, atol=4.5e-2)

    wrench = integrate_uniform_pressure_on_mesh_collection(reconstruction.components, pressure=2.0)
    assert np.allclose(wrench.force, np.zeros(3), atol=1.0e-1)
    assert np.allclose(wrench.torque, np.zeros(3), atol=1.0e-1)
