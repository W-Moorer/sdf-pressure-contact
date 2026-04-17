import numpy as np

from pfcsdf.geometry import PlaneSDF, SphereSDF, BoxFootprint


def test_plane_sdf_distance_and_gradient():
    plane = PlaneSDF(point=[0.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0])
    assert np.isclose(plane.signed_distance([0.0, 0.0, 2.0]), 2.0)
    assert np.isclose(plane.signed_distance([0.0, 0.0, -3.0]), -3.0)
    np.testing.assert_allclose(plane.gradient(), np.array([0.0, 0.0, 1.0]))


def test_sphere_sdf_distance_and_gradient():
    sphere = SphereSDF(center=[0.0, 0.0, 0.0], radius=2.0)
    assert np.isclose(sphere.signed_distance([0.0, 0.0, 2.0]), 0.0)
    assert np.isclose(sphere.signed_distance([0.0, 0.0, 1.0]), -1.0)
    np.testing.assert_allclose(sphere.gradient([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 1.0]))


def test_box_footprint_area_and_membership():
    box = BoxFootprint(lx=2.0, ly=4.0)
    assert np.isclose(box.area, 8.0)
    assert box.contains_xy([0.5, 1.5, 99.0])
    assert not box.contains_xy([2.0, 0.0, 0.0])
