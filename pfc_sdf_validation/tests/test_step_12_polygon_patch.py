from __future__ import annotations

import numpy as np

from pfcsdf.contact.patch import (
    build_rectangular_planar_patch,
    integrate_over_planar_patch,
    polygon_quadrature_degree2,
)
from pfcsdf.geometry.polygon import ConvexPolygon2D, triangle_area
from pfcsdf.geometry.primitives import BoxFootprint


def test_convex_polygon_area_centroid_and_orientation_normalization() -> None:
    vertices_cw = np.array(
        [
            [-1.0, 0.0],
            [0.0, 2.0],
            [2.0, 1.0],
            [1.0, -1.0],
        ],
        dtype=float,
    )[::-1]
    polygon = ConvexPolygon2D(vertices_cw)
    assert polygon.area > 0.0
    assert np.allclose(polygon.centroid, np.array([0.5, 0.5]))


def test_polygon_triangulation_area_and_degree2_quadrature() -> None:
    polygon = ConvexPolygon2D(
        np.array(
            [
                [-1.0, -0.5],
                [1.5, -0.75],
                [2.0, 0.5],
                [0.5, 1.75],
                [-1.5, 1.0],
            ],
            dtype=float,
        )
    )
    triangles = polygon.triangles_from_centroid()
    area_sum = sum(triangle_area(*tri) for tri in triangles)
    assert np.isclose(area_sum, polygon.area)

    patch = build_rectangular_planar_patch(BoxFootprint(lx=2.0, ly=1.0))
    qps = polygon_quadrature_degree2(patch.polygon)
    assert np.isclose(sum(qp.weight for qp in qps), patch.area)

    integral_affine = integrate_over_planar_patch(
        patch,
        lambda local_xy, world_xyz: 2.0 + 3.0 * local_xy[0] - 4.0 * local_xy[1],
    )
    assert np.isclose(integral_affine, patch.area * 2.0)
