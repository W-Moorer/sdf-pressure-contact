from __future__ import annotations

import numpy as np

from pfcsdf.contact.wrench import PairWrench, make_point_force_wrench, shift_wrench_reference
from pfcsdf.dynamics.rigid_integrators import ConstantWrenchModel, step_rigid_midpoint, step_rigid_semi_implicit
from pfcsdf.dynamics.rigid_state import RigidBodyState
from pfcsdf.dynamics.rotation import rotation_angle_error


def _make_state() -> RigidBodyState:
    return RigidBodyState(
        position=np.array([0.0, 0.0, 0.0]),
        rotation=np.eye(3),
        linear_velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        mass=2.0,
        inertia_body=np.diag([0.5, 0.6, 0.7]),
        time=0.0,
    )


def test_pure_force_drives_translation_without_rotation() -> None:
    state = _make_state()
    wrench = PairWrench(force=np.array([0.0, 0.0, 4.0]), torque=np.zeros(3))
    result = step_rigid_midpoint(state, 0.1, ConstantWrenchModel(wrench), gravity=np.zeros(3))
    expected_v = np.array([0.0, 0.0, 0.2])
    expected_x = np.array([0.0, 0.0, 0.01])
    assert np.allclose(result.state.linear_velocity, expected_v, atol=1e-12)
    assert np.allclose(result.state.position, expected_x, atol=1e-12)
    assert np.allclose(result.state.angular_velocity, np.zeros(3), atol=1e-12)
    assert rotation_angle_error(np.eye(3), result.state.rotation) < 1e-12


def test_pure_torque_drives_rotation_without_translation() -> None:
    state = _make_state()
    wrench = PairWrench(force=np.zeros(3), torque=np.array([0.0, 1.2, 0.0]))
    result = step_rigid_midpoint(state, 0.1, ConstantWrenchModel(wrench), gravity=np.zeros(3))
    expected_omega_y = 0.1 * (1.2 / 0.6)
    assert np.allclose(result.state.linear_velocity, np.zeros(3), atol=1e-12)
    assert abs(result.state.angular_velocity[1] - expected_omega_y) < 1e-12
    assert rotation_angle_error(np.eye(3), result.state.rotation) > 0.0


def test_wrench_reference_shift_is_consistent() -> None:
    force = np.array([0.0, 0.0, 5.0])
    application = np.array([0.3, 0.0, 0.0])
    about_a = np.zeros(3)
    about_b = np.array([0.1, 0.0, 0.0])
    wrench_a = make_point_force_wrench(force, application, about_a)
    wrench_b = shift_wrench_reference(wrench_a, about_a, about_b)
    direct_b = make_point_force_wrench(force, application, about_b)
    assert np.allclose(wrench_b.force, direct_b.force)
    assert np.allclose(wrench_b.torque, direct_b.torque)


def test_midpoint_is_more_accurate_than_semi_implicit_for_constant_force() -> None:
    state = _make_state()
    force = np.array([0.0, 0.0, 4.0])
    model = ConstantWrenchModel(PairWrench(force=force, torque=np.zeros(3)))
    dt = 0.2
    semi = step_rigid_semi_implicit(state, dt, model, gravity=np.zeros(3))
    mid = step_rigid_midpoint(state, dt, model, gravity=np.zeros(3))
    a = force / state.mass
    x_exact = 0.5 * a * dt * dt
    semi_err = np.linalg.norm(semi.state.position - x_exact)
    mid_err = np.linalg.norm(mid.state.position - x_exact)
    assert mid_err <= semi_err + 1e-14
