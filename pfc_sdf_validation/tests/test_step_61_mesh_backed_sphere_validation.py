from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import (
    MeshNativeBandSphereContactModel,
    NativeBandSphereContactModel,
    SphereImpactSetup,
)
from pfcsdf.geometry import (
    GridSDFGeometry,
    TransformedGeometry,
    build_mesh_asset_sdf_geometry,
    recommend_mesh_sdf_spacing_from_native_band,
)
from pfcsdf.geometry.primitives import SphereSDF
from pfcsdf.geometry.volume import UniformGrid3D


def _write_uv_sphere_obj(path: Path, *, radius: float, segments: int = 12, stacks: int = 6) -> None:
    vertices: list[list[float]] = [[0.0, 0.0, radius]]
    for i in range(1, stacks):
        theta = math.pi * i / stacks
        z = radius * math.cos(theta)
        rxy = radius * math.sin(theta)
        for j in range(segments):
            phi = 2.0 * math.pi * j / segments
            vertices.append([rxy * math.cos(phi), rxy * math.sin(phi), z])
    vertices.append([0.0, 0.0, -radius])

    top = 0
    bottom = len(vertices) - 1

    def ring_start(i: int) -> int:
        return 1 + (i - 1) * segments

    faces: list[list[int]] = []
    for j in range(segments):
        a = ring_start(1) + j
        b = ring_start(1) + (j + 1) % segments
        faces.append([top, a, b])
    for i in range(1, stacks - 1):
        rs = ring_start(i)
        ns = ring_start(i + 1)
        for j in range(segments):
            a = rs + j
            b = rs + (j + 1) % segments
            c = ns + j
            d = ns + (j + 1) % segments
            faces.append([a, c, d])
            faces.append([a, d, b])
    last = ring_start(stacks - 1)
    for j in range(segments):
        a = last + j
        b = last + (j + 1) % segments
        faces.append([a, b, bottom])

    vertices_arr = np.asarray(vertices, dtype=float)
    faces_arr = np.asarray(faces, dtype=int)
    oriented_faces: list[list[int]] = []
    for face in faces_arr:
        p0, p1, p2 = vertices_arr[face]
        normal = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if float(np.dot(normal, centroid)) < 0.0:
            oriented_faces.append([int(face[0]), int(face[2]), int(face[1])])
        else:
            oriented_faces.append([int(face[0]), int(face[1]), int(face[2])])

    lines: list[str] = ["# uv sphere approximating a sphere"]
    faces_arr = np.asarray(oriented_faces, dtype=int)
    lines.extend(f"v {v[0]} {v[1]} {v[2]}" for v in vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces_arr)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_grid(radius: float) -> UniformGrid3D:
    return UniformGrid3D(
        origin=np.array([-1.35 * radius, -1.35 * radius, -0.2]),
        spacing=np.array([0.07, 0.07, 0.035]),
        shape=(39, 39, 38),
    )


@pytest.fixture(scope="module")
def sphere_asset_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    asset_path = tmp_path_factory.mktemp("mesh_sphere") / "uv_sphere.obj"
    _write_uv_sphere_obj(asset_path, radius=1.0)
    return asset_path


@pytest.fixture(scope="module")
def sphere_mesh_result(sphere_asset_path: Path):
    return build_mesh_asset_sdf_geometry(
        sphere_asset_path,
        spacing=0.12,
        padding=0.12,
        translation=np.array([0.0, 0.0, 1.2]),
    )


def test_mesh_asset_factory_recommends_mesh_sdf_spacing_from_native_band() -> None:
    recommended = recommend_mesh_sdf_spacing_from_native_band(np.array([0.1, 0.1, 0.05]))
    assert math.isclose(recommended, 0.09, abs_tol=1e-12)


def test_mesh_asset_factory_builds_world_wrapped_geometry_from_obj(sphere_mesh_result) -> None:
    result = sphere_mesh_result

    assert isinstance(result.local_geometry, GridSDFGeometry)
    assert isinstance(result.geometry, TransformedGeometry)
    assert result.validation.is_signed_distance_ready
    assert result.mesh.num_faces == 120
    assert np.allclose(result.sdf_spacing, np.array([0.12, 0.12, 0.12]))
    assert result.recommended_sdf_spacing is None
    assert not result.used_recommended_spacing_policy
    assert math.isclose(result.geometry.signed_distance(np.array([0.0, 0.0, 1.2])), -0.85, abs_tol=0.12)


def test_mesh_asset_factory_can_use_recommended_spacing_policy(sphere_asset_path: Path) -> None:
    result = build_mesh_asset_sdf_geometry(
        sphere_asset_path,
        native_band_spacing=np.array([0.1, 0.1, 0.05]),
        padding=0.12,
    )

    assert result.used_recommended_spacing_policy
    assert math.isclose(result.recommended_sdf_spacing, 0.09, abs_tol=1e-12)
    assert np.allclose(result.sdf_spacing, np.array([0.09, 0.09, 0.09]))


def test_mesh_backed_sphere_static_force_is_close_to_analytic_native_band(sphere_mesh_result) -> None:
    radius = 1.0
    overlap = 0.14
    mesh_result = sphere_mesh_result

    setup = SphereImpactSetup(
        initial_gap=0.04,
        initial_velocity=-0.8,
        mass=1.0,
        sphere_radius=radius,
        sphere_stiffness=12.0,
        plane_stiffness=18.0,
        t_final=0.24,
    )
    grid = _build_grid(radius)
    config = NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4)

    analytic_model = NativeBandSphereContactModel(setup, grid, config, max_depth_a=2.0, max_depth_b=2.0)
    mesh_model = MeshNativeBandSphereContactModel(
        setup,
        grid,
        config,
        sphere_geometry_local=mesh_result.local_geometry,
        max_depth_a=2.0,
        max_depth_b=2.0,
    )

    gap = -overlap
    force_analytic = analytic_model.evaluate(gap).force
    force_mesh = mesh_model.evaluate(gap).force

    rel = abs(force_mesh - force_analytic) / force_analytic
    assert force_mesh > 0.0
    assert rel < 0.5


def test_mesh_backed_sphere_sdf_tracks_analytic_sphere_samples(sphere_mesh_result) -> None:
    radius = 1.0
    mesh_result = sphere_mesh_result
    analytic = SphereSDF(center=np.zeros(3), radius=radius)

    queries = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.4, -0.3, 0.2]),
        np.array([1.15, 0.0, 0.0]),
    ]
    for query in queries:
        analytic_sd = analytic.signed_distance(query)
        mesh_sd = mesh_result.local_geometry.signed_distance(query)
        assert math.isclose(mesh_sd, analytic_sd, abs_tol=0.18)
