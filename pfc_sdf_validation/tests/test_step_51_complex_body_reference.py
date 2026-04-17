
from __future__ import annotations

import numpy as np

from pfcsdf.geometry.complex_bodies import build_capsule_flat_edge_body_profile
from pfcsdf.dynamics.benchmarks_complex import ComplexBodyDropSetup, benchmark_complex_body_error, run_complex_body_drop_benchmark, run_high_resolution_complex_reference
from pfcsdf.dynamics.events import EventAwareControllerConfig


def test_complex_body_reference_is_more_accurate_than_baseline() -> None:
    profile = build_capsule_flat_edge_body_profile(nx=51, width=0.5)
    setup = ComplexBodyDropSetup(
        initial_height=0.17,
        initial_velocity=-0.65,
        initial_angle=np.deg2rad(8.0),
        initial_omega=0.0,
        mass=2.0,
        inertia=0.045,
        contact_stiffness=2800.0,
        gravity=9.81,
        t_final=0.30,
    )
    ref = run_high_resolution_complex_reference(setup, profile)
    baseline = run_complex_body_drop_benchmark(setup, profile, dt=0.01, controller=EventAwareControllerConfig(max_depth=0, force_relative_jump_tol=10.0, active_measure_relative_jump_tol=10.0, predictor_corrector_mismatch_fraction_tol=0.99), continuity_enabled=False)
    full = run_complex_body_drop_benchmark(setup, profile, dt=0.01, controller=EventAwareControllerConfig(max_depth=6), continuity_enabled=True)
    err_b = benchmark_complex_body_error(baseline, ref)
    err_f = benchmark_complex_body_error(full, ref)
    assert err_f.z_rms <= err_b.z_rms * 1.05
    assert err_f.force_rms <= err_b.force_rms * 1.05
