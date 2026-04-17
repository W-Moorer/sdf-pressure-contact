
from __future__ import annotations

import numpy as np

from pfcsdf.geometry.complex_bodies import build_capsule_flat_edge_body_profile
from pfcsdf.dynamics.benchmarks_complex import ComplexBodyDropSetup, run_complex_body_drop_benchmark
from pfcsdf.dynamics.events import EventAwareControllerConfig


def test_complex_body_benchmark_runs_and_generates_torque() -> None:
    profile = build_capsule_flat_edge_body_profile(nx=41, width=0.5)
    setup = ComplexBodyDropSetup(
        initial_height=0.16,
        initial_velocity=-0.6,
        initial_angle=np.deg2rad(7.0),
        initial_omega=0.0,
        mass=2.0,
        inertia=0.045,
        contact_stiffness=2600.0,
        gravity=9.81,
        t_final=0.35,
    )
    hist = run_complex_body_drop_benchmark(setup, profile, dt=0.01, controller=EventAwareControllerConfig(max_depth=5), continuity_enabled=True)
    assert hist.times.size > 10
    assert np.max(np.abs(hist.torque_y)) > 1e-6
    assert np.max(hist.active_components) >= 1
