from __future__ import annotations

import numpy as np

from pfcsdf.dynamics.benchmarks_complex_6dof import (
    benchmark_complex_rigid_6dof_error,
    default_complex_6dof_setup,
    run_complex_rigid_6dof_benchmark,
    run_high_resolution_complex_6dof_reference,
)
from pfcsdf.dynamics.rigid_controller import RigidEventAwareControllerConfig


def test_complex_6dof_benchmark_runs_and_has_finite_outputs():
    setup, support_cloud = default_complex_6dof_setup()
    controller = RigidEventAwareControllerConfig(max_depth=5)
    hist = run_complex_rigid_6dof_benchmark(setup, support_cloud, dt=0.01, controller=controller, continuity_enabled=True)
    assert hist.times.size > 5
    assert np.all(np.isfinite(hist.position))
    assert np.all(np.isfinite(hist.force))
    assert np.max(np.linalg.norm(hist.torque, axis=1)) > 1e-6
    assert np.any(hist.active_measure > 0.0)


def test_complex_6dof_matches_high_res_reference_reasonably():
    setup, support_cloud = default_complex_6dof_setup()
    controller = RigidEventAwareControllerConfig(max_depth=5)
    hist = run_complex_rigid_6dof_benchmark(setup, support_cloud, dt=0.01, controller=controller, continuity_enabled=True)
    ref = run_high_resolution_complex_6dof_reference(setup, support_cloud)
    err = benchmark_complex_rigid_6dof_error(hist, ref)
    assert err.position_rms < 0.1
    assert err.orientation_rms < 0.3
    assert err.force_rms < 50.0
    assert err.torque_rms < 20.0
