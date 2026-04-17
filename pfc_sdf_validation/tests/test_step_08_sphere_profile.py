import numpy as np

from pfcsdf.solvers import sphere_plane_contact_radius_exact, sphere_plane_local_overlap


def test_sphere_plane_contact_radius_matches_geometry_identity():
    R = 2.0
    overlap = 0.3
    a = sphere_plane_contact_radius_exact(R, overlap)
    assert np.isclose(a * a, 2.0 * R * overlap - overlap * overlap)


def test_sphere_local_overlap_is_peak_at_center_and_zero_at_edge():
    R = 3.0
    overlap = 0.2
    a = sphere_plane_contact_radius_exact(R, overlap)

    assert np.isclose(sphere_plane_local_overlap(0.0, R, overlap), overlap)
    assert np.isclose(sphere_plane_local_overlap(a, R, overlap), 0.0, atol=1e-12)
    assert np.isclose(sphere_plane_local_overlap(1.1 * a, R, overlap), 0.0, atol=1e-12)
