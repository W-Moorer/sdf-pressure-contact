from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import (
    SphereImpactSetup,
    validate_mesh_native_band_sphere_dynamic_from_asset,
)
from pfcsdf.geometry.mesh_io import build_uv_sphere_triangle_mesh
from pfcsdf.geometry.volume import UniformGrid3D


def _write_triangle_mesh_obj(path: Path) -> None:
    mesh = build_uv_sphere_triangle_mesh(radius=1.0, segments=12, stacks=6)
    lines = ["# uv sphere approximating a sphere"]
    lines.extend(f"v {v[0]} {v[1]} {v[2]}" for v in mesh.vertices)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def policy_recheck_asset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    asset_path = tmp_path_factory.mktemp("mesh_sphere_policy_recheck") / "uv_sphere.obj"
    _write_triangle_mesh_obj(asset_path)
    return asset_path


@pytest.fixture(scope="module")
def policy_recheck_results(policy_recheck_asset: Path):
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
        origin=np.array([-0.6, -0.6, -0.14]),
        spacing=np.array([0.1, 0.1, 0.05]),
        shape=(12, 12, 24),
    )
    config = NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4)

    coarse = validate_mesh_native_band_sphere_dynamic_from_asset(
        policy_recheck_asset,
        setup,
        grid=grid,
        config=config,
        dt=0.02,
        scheme="midpoint",
        mesh_sdf_spacing=0.12,
        padding=0.12,
        dt_ref=2e-3,
    )
    refined = validate_mesh_native_band_sphere_dynamic_from_asset(
        policy_recheck_asset,
        setup,
        grid=grid,
        config=config,
        dt=0.015,
        scheme="midpoint",
        padding=0.12,
        dt_ref=2e-3,
    )
    return coarse, refined


def test_mesh_backed_sphere_dynamic_policy_recheck_runs(policy_recheck_results) -> None:
    coarse, refined = policy_recheck_results

    assert coarse.comparison.mesh_history.onset_time is not None
    assert refined.comparison.mesh_history.onset_time is not None
    assert coarse.mesh_geometry.used_recommended_spacing_policy is False
    assert refined.mesh_geometry.used_recommended_spacing_policy is True
    assert coarse.dt == 0.02
    assert refined.dt == 0.015
    assert refined.mesh_geometry.recommended_sdf_spacing == pytest.approx(0.09, abs=1e-12)
    assert refined.comparison.peak_force_relative_difference < 1.0
    assert refined.comparison.impulse_relative_difference < 1.0


def test_mesh_backed_sphere_dynamic_policy_recheck_is_more_informative(policy_recheck_results) -> None:
    coarse, refined = policy_recheck_results

    assert refined.comparison.peak_force_relative_difference < coarse.comparison.peak_force_relative_difference
    assert refined.comparison.impulse_relative_difference < coarse.comparison.impulse_relative_difference
    assert refined.comparison.max_penetration_relative_difference <= coarse.comparison.max_penetration_relative_difference + 1e-4
