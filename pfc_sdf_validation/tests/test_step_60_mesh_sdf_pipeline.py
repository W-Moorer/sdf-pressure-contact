from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry import (
    GridSDFGeometry,
    SignedDistanceGeometry,
    TransformedGeometry,
    TriangleMesh,
    build_mesh_grid_sdf,
    inspect_triangle_mesh,
    load_obj_triangle_mesh,
    mesh_to_grid_sdf,
    triangle_mesh_aabb,
)
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


def _cube_obj_text() -> str:
    return "\n".join(
        [
            "# unit cube centered at the origin",
            "v -0.5 -0.5 -0.5",
            "v 0.5 -0.5 -0.5",
            "v 0.5 0.5 -0.5",
            "v -0.5 0.5 -0.5",
            "v -0.5 -0.5 0.5",
            "v 0.5 -0.5 0.5",
            "v 0.5 0.5 0.5",
            "v -0.5 0.5 0.5",
            "f 1 3 2",
            "f 1 4 3",
            "f 5 6 7",
            "f 5 7 8",
            "f 1 2 6",
            "f 1 6 5",
            "f 4 8 7",
            "f 4 7 3",
            "f 1 5 8",
            "f 1 8 4",
            "f 2 3 7",
            "f 2 7 6",
            "",
        ]
    )


def _cube_mesh() -> TriangleMesh:
    return TriangleMesh(
        vertices=np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ],
            dtype=float,
        ),
        faces=np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [3, 7, 6],
                [3, 6, 2],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=int,
        ),
    )


def _box_signed_distance(point: np.ndarray, half_extent: np.ndarray) -> float:
    q = np.abs(np.asarray(point, dtype=float)) - np.asarray(half_extent, dtype=float)
    outside = np.maximum(q, 0.0)
    inside = min(max(q[0], max(q[1], q[2])), 0.0)
    return float(np.linalg.norm(outside) + inside)


def test_load_obj_triangle_mesh_parses_minimal_cube(tmp_path: Path) -> None:
    path = tmp_path / "cube.obj"
    path.write_text(_cube_obj_text(), encoding="utf-8")

    mesh = load_obj_triangle_mesh(path)

    assert mesh.num_vertices == 8
    assert mesh.num_faces == 12
    np.testing.assert_allclose(mesh.vertices[0], np.array([-0.5, -0.5, -0.5]))
    np.testing.assert_array_equal(mesh.faces[0], np.array([0, 2, 1]))


def test_mesh_preprocess_reports_closed_oriented_cube_aabb() -> None:
    mesh = _cube_mesh()

    bbox = triangle_mesh_aabb(mesh)
    report = inspect_triangle_mesh(mesh)

    np.testing.assert_allclose(bbox.minimum, np.array([-0.5, -0.5, -0.5]))
    np.testing.assert_allclose(bbox.maximum, np.array([0.5, 0.5, 0.5]))
    assert report.is_closed_edge_manifold
    assert report.has_consistent_orientation
    assert report.has_no_degenerate_faces
    assert report.is_signed_distance_ready


def test_mesh_to_grid_sdf_matches_closed_cube_field_approximately() -> None:
    mesh = _cube_mesh()
    result = build_mesh_grid_sdf(mesh, spacing=0.1, padding=0.2)
    geometry = result.geometry

    assert isinstance(geometry, GridSDFGeometry)
    assert isinstance(geometry, SignedDistanceGeometry)
    np.testing.assert_allclose(geometry.bounding_box().minimum, np.array([-0.7, -0.7, -0.7]))
    np.testing.assert_allclose(geometry.bounding_box().maximum, np.array([0.7, 0.7, 0.7]))

    queries = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.7, 0.0, 0.0]),
        np.array([0.58, 0.58, 0.0]),
        np.array([0.2, -0.35, 0.49]),
    ]
    for query in queries:
        expected = _box_signed_distance(query, np.array([0.5, 0.5, 0.5]))
        actual = geometry.signed_distance(query)
        assert math.isclose(actual, expected, abs_tol=6e-2)


def test_mesh_to_grid_sdf_rejects_open_mesh_by_default() -> None:
    open_mesh = TriangleMesh(
        vertices=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        faces=np.array([[0, 1, 2]], dtype=int),
    )

    with pytest.raises(ValueError, match="closed edge-manifold triangle mesh"):
        mesh_to_grid_sdf(open_mesh, spacing=0.1, padding=0.1)


def test_native_band_smoke_with_mesh_backed_grid_sdf() -> None:
    cube_sdf = mesh_to_grid_sdf(_cube_mesh(), spacing=0.1, padding=0.2)
    lower = TransformedGeometry.identity(cube_sdf)
    upper = TransformedGeometry.from_translation(cube_sdf, np.array([0.0, 0.0, 0.9]))

    grid = UniformGrid3D(
        origin=np.array([-0.5, -0.5, 0.3]),
        spacing=np.array([0.1, 0.1, 0.05]),
        shape=(10, 10, 6),
    )
    fields = sample_linear_pfc_balance_fields(
        grid,
        lower,
        upper,
        LinearPressureLaw(12.0),
        LinearPressureLaw(18.0),
        max_depth_a=1.0,
        max_depth_b=1.0,
    )
    result = accumulate_sdf_native_band_wrench(
        fields,
        NativeBandAccumulatorConfig(eta=0.12, band_half_width=0.25),
    )

    assert result.active_count > 0
    assert np.isfinite(result.wrench.force).all()
    assert np.isfinite(result.wrench.torque).all()
    assert abs(result.wrench.force[2]) > 1e-6
    assert abs(result.wrench.force[0]) < 1e-6
    assert abs(result.wrench.force[1]) < 1e-6
