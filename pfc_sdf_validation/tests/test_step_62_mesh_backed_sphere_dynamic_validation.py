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
    validate_mesh_native_band_sphere_dynamic_against_analytic,
)
from pfcsdf.geometry import build_mesh_asset_sdf_geometry
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
    lines.extend(f"v {v[0]} {v[1]} {v[2]}" for v in vertices_arr)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in oriented_faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def dynamic_mesh_sphere_asset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    asset_path = tmp_path_factory.mktemp("mesh_sphere_dynamic") / "uv_sphere.obj"
    _write_uv_sphere_obj(asset_path, radius=1.0)
    return asset_path


@pytest.fixture(scope="module")
def dynamic_mesh_sphere_comparison(dynamic_mesh_sphere_asset: Path):
    mesh_result = build_mesh_asset_sdf_geometry(
        dynamic_mesh_sphere_asset,
        spacing=0.12,
        padding=0.12,
    )
    setup = SphereImpactSetup(
        initial_gap=0.0,
        initial_velocity=-1.0,
        mass=1.0,
        sphere_radius=1.0,
        sphere_stiffness=12.0,
        plane_stiffness=18.0,
        t_final=0.06,
    )
    grid = UniformGrid3D(
        origin=np.array([-1.1, -1.1, -0.14]),
        spacing=np.array([0.1, 0.1, 0.05]),
        shape=(23, 23, 24),
    )
    config = NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4)
    analytic_model = NativeBandSphereContactModel(
        setup,
        grid,
        config,
        max_depth_a=2.0,
        max_depth_b=2.0,
    )
    mesh_model = MeshNativeBandSphereContactModel(
        setup,
        grid,
        config,
        sphere_geometry_local=mesh_result.local_geometry,
        max_depth_a=2.0,
        max_depth_b=2.0,
    )
    return validate_mesh_native_band_sphere_dynamic_against_analytic(
        setup,
        dt=0.02,
        scheme="midpoint",
        analytic_model=analytic_model,
        mesh_model=mesh_model,
        dt_ref=2e-3,
    )


def test_mesh_backed_sphere_dynamic_validation_runs(dynamic_mesh_sphere_comparison) -> None:
    comparison = dynamic_mesh_sphere_comparison

    assert len(comparison.analytic_history.times) == len(comparison.mesh_history.times)
    assert len(comparison.mesh_history.times) == 4
    assert np.max(comparison.mesh_history.forces) > 0.0
    assert comparison.mesh_history.onset_time is not None
    assert comparison.analytic_history.onset_time is not None


def test_mesh_backed_sphere_dynamic_validation_stays_in_family(dynamic_mesh_sphere_comparison) -> None:
    comparison = dynamic_mesh_sphere_comparison

    assert comparison.peak_force_relative_difference < 1.0
    assert comparison.impulse_relative_difference < 1.0
    assert comparison.max_penetration_relative_difference < 1e-3
    assert comparison.onset_time_difference <= 0.02 + 1e-12

    assert comparison.mesh_error.peak_force_error <= comparison.analytic_error.peak_force_error + 0.1
    assert comparison.mesh_error.max_penetration_error <= comparison.analytic_error.max_penetration_error + 1e-3

