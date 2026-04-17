from __future__ import annotations

import numpy as np

from pfcsdf.dynamics.rigid_state import RigidBodyState, kinetic_energy, world_inertia
from pfcsdf.dynamics.rotation import exp_so3, project_to_so3, rotation_angle_error


def _make_state() -> RigidBodyState:
    return RigidBodyState(
        position=np.array([0.0, 0.0, 1.0]),
        rotation=np.eye(3),
        linear_velocity=np.array([1.0, 0.0, -2.0]),
        angular_velocity=np.array([0.0, 0.5, 0.0]),
        mass=2.0,
        inertia_body=np.diag([0.2, 0.3, 0.4]),
        time=0.0,
    )


def test_rigid_body_state_world_inertia_and_energy_are_finite() -> None:
    state = _make_state()
    Iw = world_inertia(state)
    assert Iw.shape == (3, 3)
    assert np.allclose(Iw, np.diag([0.2, 0.3, 0.4]))
    ke = kinetic_energy(state)
    assert np.isfinite(ke)
    assert ke > 0.0


def test_project_to_so3_recovers_rotation_matrix() -> None:
    R = exp_so3(np.array([0.0, 0.3, 0.1]))
    R_bad = R.copy()
    R_bad[0, 0] += 1e-4
    R_proj = project_to_so3(R_bad)
    assert np.allclose(R_proj.T @ R_proj, np.eye(3), atol=1e-10)
    assert np.linalg.det(R_proj) > 0.0
    assert rotation_angle_error(R, R_proj) < 1e-3
