import numpy as np

from pfcsdf.contact.reconstruct import (
    integrate_uniform_pressure_on_component_collection,
    integrate_uniform_pressure_on_curved_sheet,
    reconstruct_curved_sheets_from_sampled_field,
)
from pfcsdf.geometry.volume import UniformGrid3D, sample_scalar_field


def test_disconnected_curved_sheets_are_split_into_two_components_and_sum_cleanly() -> None:
    grid = UniformGrid3D(origin=np.array([-3.0, -1.5, 0.0]), spacing=np.array([0.1, 0.1, 0.05]), shape=(61, 31, 9))

    centers = (-1.2, 1.2)
    radius = 0.7
    base = 0.2
    curvature = 0.2

    def twin_caps_field(point: np.ndarray) -> float:
        x, y, z = point
        surfaces: list[float] = []
        for cx in centers:
            rx = x - cx
            r2 = rx * rx + y * y
            if r2 <= radius * radius:
                surfaces.append(base + curvature * r2)
        if not surfaces:
            return 1.0
        return z - min(surfaces)

    sampled = sample_scalar_field(grid, twin_caps_field)
    reconstruction = reconstruct_curved_sheets_from_sampled_field(sampled, band_half_width=0.06)

    assert reconstruction.num_components == 2
    x_centroids = sorted(component.centroid[0] for component in reconstruction.components)
    assert x_centroids[0] < -0.5
    assert x_centroids[1] > 0.5

    collection_wrench = integrate_uniform_pressure_on_component_collection(
        reconstruction.components,
        pressure=3.0,
        reference=np.zeros(3),
    )
    summed_wrench = reconstruction.components[0]
    total_force = np.zeros(3)
    total_torque = np.zeros(3)
    for component in reconstruction.components:
        wrench = integrate_uniform_pressure_on_curved_sheet(component, pressure=3.0, reference=np.zeros(3))
        total_force += wrench.force
        total_torque += wrench.torque

    np.testing.assert_allclose(collection_wrench.force, total_force, atol=1e-12)
    np.testing.assert_allclose(collection_wrench.torque, total_torque, atol=1e-12)
    np.testing.assert_allclose(collection_wrench.force[:2], np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(collection_wrench.torque, np.zeros(3), atol=1e-12)
    assert collection_wrench.force[2] > 0.0
