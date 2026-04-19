from __future__ import annotations

import math

import numpy as np

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import SphereImpactSetup, run_mesh_backed_sphere_sensitivity_study


def _study_setup() -> SphereImpactSetup:
    return SphereImpactSetup(
        initial_gap=0.0,
        initial_velocity=-1.0,
        mass=1.0,
        sphere_radius=1.0,
        sphere_stiffness=12.0,
        plane_stiffness=18.0,
        t_final=0.06,
    )


def _study_kwargs() -> dict:
    return {
        "dt": 0.02,
        "scheme": "midpoint",
        "config": NativeBandAccumulatorConfig(eta=0.25, band_half_width=0.4),
        "baseline_native_band_origin": np.array([-0.6, -0.6, -0.14]),
        "baseline_native_band_spacing": np.array([0.1, 0.1, 0.05]),
        "baseline_native_band_shape": (12, 12, 24),
    }


def test_mesh_backed_sphere_sensitivity_study_returns_expected_structure() -> None:
    study = run_mesh_backed_sphere_sensitivity_study(
        _study_setup(),
        mesh_sdf_spacing_levels=(0.12,),
        mesh_resolution_levels=((12, 6),),
        native_band_spacing_levels=((0.1, 0.1, 0.05),),
        dynamic_axes=(),
        **_study_kwargs(),
    )

    assert study.axes_scanned == ("mesh_sdf_spacing", "mesh_resolution", "native_band_spacing")
    assert len(study.cases) == 3

    mesh_sdf_case = study.axis_cases("mesh_sdf_spacing")[0]
    mesh_resolution_case = study.axis_cases("mesh_resolution")[0]
    native_band_case = study.axis_cases("native_band_spacing")[0]

    assert mesh_sdf_case.native_band_shape == (12, 12, 24)
    assert mesh_resolution_case.mesh_segments == 12
    assert mesh_resolution_case.mesh_stacks == 6
    assert native_band_case.native_band_spacing == (0.1, 0.1, 0.05)
    assert mesh_resolution_case.peak_force_relative_difference is None
    assert mesh_resolution_case.impulse_relative_difference is None
    assert mesh_resolution_case.static_force_relative_difference >= 0.0


def test_mesh_backed_sphere_sensitivity_study_grid_sdf_spacing_has_dynamic_metrics() -> None:
    study = run_mesh_backed_sphere_sensitivity_study(
        _study_setup(),
        mesh_sdf_spacing_levels=(0.16, 0.12),
        mesh_resolution_levels=((12, 6),),
        native_band_spacing_levels=((0.1, 0.1, 0.05),),
        dynamic_axes=("mesh_sdf_spacing",),
        **_study_kwargs(),
    )

    coarse_case, fine_case = study.axis_cases("mesh_sdf_spacing")

    assert coarse_case.peak_force_relative_difference is not None
    assert fine_case.peak_force_relative_difference is not None
    assert coarse_case.impulse_relative_difference is not None
    assert fine_case.impulse_relative_difference is not None
    assert math.isclose(coarse_case.onset_time_difference, 0.02, abs_tol=1e-12)
    assert math.isclose(fine_case.onset_time_difference, 0.02, abs_tol=1e-12)
    assert coarse_case.peak_force_relative_difference < 1.0
    assert fine_case.peak_force_relative_difference < 1.0
    assert fine_case.static_force_relative_difference < coarse_case.static_force_relative_difference


def test_mesh_backed_sphere_sensitivity_study_native_band_refinement_exposes_force_gap() -> None:
    study = run_mesh_backed_sphere_sensitivity_study(
        _study_setup(),
        mesh_sdf_spacing_levels=(0.12,),
        mesh_resolution_levels=((12, 6),),
        native_band_spacing_levels=((0.1, 0.1, 0.05), (0.08, 0.08, 0.04)),
        dynamic_axes=(),
        **_study_kwargs(),
    )

    coarse_case, fine_case = study.axis_cases("native_band_spacing")

    assert fine_case.native_band_shape[0] > coarse_case.native_band_shape[0]
    assert fine_case.mesh_static_force > coarse_case.mesh_static_force
    assert fine_case.static_force_relative_difference > coarse_case.static_force_relative_difference
