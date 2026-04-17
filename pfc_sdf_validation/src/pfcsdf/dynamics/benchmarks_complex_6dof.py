from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from pfcsdf.contact.active_set import (
    ActiveSetContinuityState,
    ActiveSetSnapshot,
    active_set_mismatch_report,
    continuity_report,
    continuity_update_plan,
    repair_update_plan,
    transport_snapshot_by_rigid_motion,
)
from pfcsdf.contact.wrench import PairWrench
from pfcsdf.dynamics.rigid_controller import (
    RigidControllerIndicators,
    RigidEventAwareControllerConfig,
    rigid_controller_indicators,
    should_substep_rigid,
)
from pfcsdf.dynamics.rigid_integrators import evaluate_rigid_dynamics
from pfcsdf.dynamics.rigid_state import RigidBodyState
from pfcsdf.dynamics.rotation import exp_so3, project_to_so3, rotation_angle_error
from pfcsdf.geometry.complex_bodies import SupportCloud3D, build_capsule_flat_edge_body_cloud


@dataclass(frozen=True)
class RigidContactEvaluation:
    wrench: PairWrench
    linear_acceleration: np.ndarray
    angular_acceleration: np.ndarray
    active_measure: float
    active_snapshot: ActiveSetSnapshot
    active_components: int
    candidate_count: int
    recompute_count: int


@dataclass(frozen=True)
class ComplexRigid6DoFSetup:
    initial_position: np.ndarray
    initial_rotation: np.ndarray
    initial_linear_velocity: np.ndarray
    initial_angular_velocity: np.ndarray
    mass: float
    inertia_body: np.ndarray
    contact_stiffness: float
    gravity: np.ndarray
    t_final: float

    def initial_state(self) -> RigidBodyState:
        return RigidBodyState(
            position=np.asarray(self.initial_position, dtype=float),
            rotation=np.asarray(self.initial_rotation, dtype=float),
            linear_velocity=np.asarray(self.initial_linear_velocity, dtype=float),
            angular_velocity=np.asarray(self.initial_angular_velocity, dtype=float),
            mass=float(self.mass),
            inertia_body=np.asarray(self.inertia_body, dtype=float),
            time=0.0,
        )


@dataclass(frozen=True)
class ComplexRigid6DoFHistory:
    times: np.ndarray
    position: np.ndarray
    linear_velocity: np.ndarray
    orientation_error: np.ndarray
    angular_velocity: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    active_measure: np.ndarray
    active_components: np.ndarray
    candidate_count: np.ndarray
    recompute_count: np.ndarray
    mismatch_fraction: np.ndarray
    continuity_jaccard: np.ndarray
    used_substeps: np.ndarray
    controller_triggered: np.ndarray
    total_energy: np.ndarray


@dataclass(frozen=True)
class ComplexRigid6DoFErrorSummary:
    position_rms: float
    orientation_rms: float
    force_rms: float
    torque_rms: float
    peak_force_error: float
    peak_torque_error: float
    release_timing_error: float
    mean_component_count_error: float


class ComplexRigid6DoFContactModel:
    def __init__(
        self,
        support_cloud: SupportCloud3D,
        *,
        mass: float,
        inertia_body: np.ndarray,
        contact_stiffness: float,
        gravity: np.ndarray | None = None,
        continuity_enabled: bool = True,
        boundary_only_update: bool = True,
        continuity_dilation_radius: int = 1,
    ) -> None:
        self.support_cloud = support_cloud
        self.mass = float(mass)
        self.inertia_body = np.asarray(inertia_body, dtype=float)
        self.contact_stiffness = float(contact_stiffness)
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=float) if gravity is None else np.asarray(gravity, dtype=float)
        self.continuity_enabled = bool(continuity_enabled)
        self.boundary_only_update = bool(boundary_only_update)
        self.continuity_dilation_radius = int(continuity_dilation_radius)
        self._continuity_state = ActiveSetContinuityState()
        self.last_continuity_report = None

    def reset_continuity(self) -> None:
        self._continuity_state = ActiveSetContinuityState()
        self.last_continuity_report = None

    @property
    def warm_start_snapshot(self) -> ActiveSetSnapshot | None:
        if not self.continuity_enabled:
            return None
        return self._continuity_state.previous

    def commit_active_snapshot(self, snapshot: ActiveSetSnapshot | None) -> None:
        if snapshot is None:
            return
        self._continuity_state, self.last_continuity_report = continuity_report(self._continuity_state, snapshot)

    @staticmethod
    def _component_count(mask2d: np.ndarray) -> int:
        mask2d = np.asarray(mask2d, dtype=bool)
        visited = np.zeros_like(mask2d, dtype=bool)
        count = 0
        nx, ny = mask2d.shape
        for i in range(nx):
            for j in range(ny):
                if not mask2d[i, j] or visited[i, j]:
                    continue
                count += 1
                stack = [(i, j)]
                visited[i, j] = True
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        xi, yj = x + dx, y + dy
                        if 0 <= xi < nx and 0 <= yj < ny and mask2d[xi, yj] and not visited[xi, yj]:
                            visited[xi, yj] = True
                            stack.append((xi, yj))
        return count

    def _world_points(self, state: RigidBodyState, indices: np.ndarray) -> np.ndarray:
        pts_b = self.support_cloud.body_points[indices]
        return state.position[None, :] + (state.rotation @ pts_b.T).T

    def evaluate(
        self,
        state: RigidBodyState,
        *,
        warm_start_snapshot: ActiveSetSnapshot | None = None,
        repair_mask: np.ndarray | None = None,
    ) -> RigidContactEvaluation:
        nx, ny = self.support_cloud.grid_shape
        support_mask = np.ones((nx, ny, 1), dtype=bool)
        update_plan = None
        if repair_mask is not None and warm_start_snapshot is not None:
            update_plan = repair_update_plan(warm_start_snapshot.mask, repair_mask, extra_mask=support_mask)
            candidate_mask = update_plan.candidate_mask
            retained_mask = update_plan.retained_interior_mask
        elif warm_start_snapshot is not None and warm_start_snapshot.active_count > 0 and self.boundary_only_update:
            update_plan = continuity_update_plan(warm_start_snapshot.mask, extra_mask=support_mask, dilation_radius=self.continuity_dilation_radius)
            candidate_mask = update_plan.candidate_mask
            retained_mask = update_plan.retained_interior_mask
        else:
            candidate_mask = support_mask.copy()
            retained_mask = np.zeros_like(support_mask, dtype=bool)
        candidate_indices = np.flatnonzero(candidate_mask[:, :, 0].reshape(-1))
        world_pts = self._world_points(state, candidate_indices)
        overlap = np.maximum(-world_pts[:, 2], 0.0)
        weights = self.support_cloud.weights[candidate_indices]
        force_mag = self.contact_stiffness * weights * overlap
        forces = np.zeros((candidate_indices.size, 3), dtype=float)
        forces[:, 2] = force_mag
        rel = world_pts - state.position[None, :]
        torques = np.cross(rel, forces)
        active_mask = np.zeros((nx, ny, 1), dtype=bool)
        active_flat = overlap > 0.0
        active_mask[:, :, 0].reshape(-1)[candidate_indices] = active_flat
        force = np.sum(forces, axis=0)
        torque = np.sum(torques, axis=0)
        if np.any(retained_mask):
            retained_indices = np.flatnonzero(retained_mask[:, :, 0].reshape(-1))
            retained_pts = self._world_points(state, retained_indices)
            overlap_r = np.maximum(-retained_pts[:, 2], 0.0)
            weights_r = self.support_cloud.weights[retained_indices]
            fmag_r = self.contact_stiffness * weights_r * overlap_r
            forces_r = np.zeros((retained_indices.size, 3), dtype=float)
            forces_r[:, 2] = fmag_r
            rel_r = retained_pts - state.position[None, :]
            torques_r = np.cross(rel_r, forces_r)
            active_mask[:, :, 0].reshape(-1)[retained_indices] = overlap_r > 0.0
            force += np.sum(forces_r, axis=0)
            torque += np.sum(torques_r, axis=0)
        active_measure = float(np.sum(self.support_cloud.weights[active_mask[:, :, 0].reshape(-1)]))
        snapshot = ActiveSetSnapshot(mask=active_mask, measure=active_measure)
        dyn = evaluate_rigid_dynamics(state, PairWrench(force=force, torque=torque), gravity=self.gravity)
        recompute_count = int(np.count_nonzero(candidate_mask)) if update_plan is None else update_plan.recompute_count
        return RigidContactEvaluation(
            wrench=PairWrench(force=force, torque=torque),
            linear_acceleration=dyn.linear_acceleration,
            angular_acceleration=dyn.angular_acceleration,
            active_measure=active_measure,
            active_snapshot=snapshot,
            active_components=self._component_count(active_mask[:, :, 0]),
            candidate_count=int(np.count_nonzero(candidate_mask)),
            recompute_count=recompute_count,
        )


def _kinetic_energy(state: RigidBodyState, gravity: np.ndarray) -> float:
    return state.kinetic_energy - state.mass * float(np.dot(gravity, state.position))


def _predict_state(state: RigidBodyState, lin_acc: np.ndarray, ang_acc: np.ndarray, dt: float) -> RigidBodyState:
    vel1 = state.linear_velocity + dt * lin_acc
    pos1 = state.position + dt * state.linear_velocity + 0.5 * dt * dt * lin_acc
    omega1 = state.angular_velocity + dt * ang_acc
    omega_mid = state.angular_velocity + 0.5 * dt * ang_acc
    rot1 = project_to_so3(state.rotation @ exp_so3(dt * omega_mid))
    return state.with_state(position=pos1, rotation=rot1, linear_velocity=vel1, angular_velocity=omega1, time=state.time + dt)


def _mid_state(state: RigidBodyState, pred_state: RigidBodyState) -> RigidBodyState:
    omega_mid = 0.5 * (state.angular_velocity + pred_state.angular_velocity)
    rot_mid = project_to_so3(state.rotation @ exp_so3(0.5 * rotation_angle_error(np.eye(3), state.rotation.T @ pred_state.rotation) * _safe_dir(omega_mid)))
    return state.with_state(
        position=0.5 * (state.position + pred_state.position),
        rotation=rot_mid,
        linear_velocity=0.5 * (state.linear_velocity + pred_state.linear_velocity),
        angular_velocity=omega_mid,
        time=state.time + 0.5 * (pred_state.time - state.time),
    )


def _safe_dir(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.asarray(v, dtype=float) / n


def _accepted_state(state: RigidBodyState, dt: float, start_eval: RigidContactEvaluation, corr_eval: RigidContactEvaluation, pred_eval: RigidContactEvaluation) -> RigidBodyState:
    a0, am, a1 = start_eval.linear_acceleration, corr_eval.linear_acceleration, pred_eval.linear_acceleration
    al0, alm, al1 = start_eval.angular_acceleration, corr_eval.angular_acceleration, pred_eval.angular_acceleration
    vel1 = state.linear_velocity + dt * (a0 + 4.0 * am + a1) / 6.0
    omega1 = state.angular_velocity + dt * (al0 + 4.0 * alm + al1) / 6.0
    pos1 = state.position + 0.5 * dt * (state.linear_velocity + vel1)
    omega_mid = state.angular_velocity + 0.5 * dt * alm
    rot1 = project_to_so3(state.rotation @ exp_so3(dt * 0.5 * (omega_mid + omega1)))
    return state.with_state(position=pos1, rotation=rot1, linear_velocity=vel1, angular_velocity=omega1, time=state.time + dt)


def integrate_complex_rigid_step(
    model: ComplexRigid6DoFContactModel,
    state: RigidBodyState,
    dt: float,
    controller: RigidEventAwareControllerConfig,
    *,
    depth: int = 0,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
) -> tuple[RigidBodyState, dict[str, Any]]:
    start_eval = model.evaluate(state, warm_start_snapshot=warm_start_snapshot)
    pred_state = _predict_state(state, start_eval.linear_acceleration, start_eval.angular_acceleration, dt)
    pred_warm = transport_snapshot_by_rigid_motion(start_eval.active_snapshot, previous_rotation=state.rotation, previous_position=state.position, new_rotation=pred_state.rotation, new_position=pred_state.position)
    pred_eval = model.evaluate(pred_state, warm_start_snapshot=pred_warm)
    mid_state = state.with_state(
        position=0.5 * (state.position + pred_state.position),
        rotation=project_to_so3(state.rotation @ exp_so3(0.5 * dt * (state.angular_velocity + 0.5 * dt * start_eval.angular_acceleration))),
        linear_velocity=0.5 * (state.linear_velocity + pred_state.linear_velocity),
        angular_velocity=0.5 * (state.angular_velocity + pred_state.angular_velocity),
        time=state.time + 0.5 * dt,
    )
    corr_warm = transport_snapshot_by_rigid_motion(pred_eval.active_snapshot, previous_rotation=pred_state.rotation, previous_position=pred_state.position, new_rotation=mid_state.rotation, new_position=mid_state.position)
    corr_eval = model.evaluate(mid_state, warm_start_snapshot=corr_warm)
    mismatch = active_set_mismatch_report(pred_eval.active_snapshot, corr_eval.active_snapshot)
    linear_work_mismatch = float(np.dot(corr_eval.wrench.force, pred_state.position - state.position) - (state.kinetic_energy - pred_state.kinetic_energy))
    angular_work_mismatch = float(np.dot(corr_eval.wrench.torque, pred_state.angular_velocity - state.angular_velocity))
    indicators = rigid_controller_indicators(
        start_eval.wrench,
        pred_eval.wrench,
        corr_eval.wrench,
        start_measure=start_eval.active_measure,
        predictor_measure=pred_eval.active_measure,
        corrector_measure=corr_eval.active_measure,
        mismatch=mismatch,
        predictor_state=pred_state,
        corrector_state=mid_state,
        linear_work_mismatch=linear_work_mismatch,
        angular_work_mismatch=angular_work_mismatch,
    )
    controller_trigger = False
    if depth < controller.max_depth and should_substep_rigid(controller, dt, indicators):
        controller_trigger = True
    if controller_trigger:
        half = 0.5 * dt
        s_mid, d_a = integrate_complex_rigid_step(model, state, half, controller, depth=depth + 1, warm_start_snapshot=warm_start_snapshot)
        s_end, d_b = integrate_complex_rigid_step(model, s_mid, half, controller, depth=depth + 1, warm_start_snapshot=d_a['accepted_snapshot'])
        merged = {
            'force': d_b['force'],
            'torque': d_b['torque'],
            'active_measure': d_b['active_measure'],
            'active_components': d_b['active_components'],
            'candidate_count': d_a['candidate_count'] + d_b['candidate_count'],
            'recompute_count': d_a['recompute_count'] + d_b['recompute_count'],
            'mismatch_fraction': max(d_a['mismatch_fraction'], d_b['mismatch_fraction']),
            'continuity_jaccard': d_b['continuity_jaccard'],
            'used_substeps': d_a['used_substeps'] + d_b['used_substeps'],
            'controller_triggered': True,
            'accepted_snapshot': d_b['accepted_snapshot'],
            'orientation_error': d_b['orientation_error'],
        }
        return s_end, merged
    # local repair if mismatch exists
    repair_mask = mismatch.mismatch_mask if mismatch.mismatch_count > 0 else None
    accepted_state = _accepted_state(state, dt, start_eval, corr_eval, pred_eval)
    end_warm = transport_snapshot_by_rigid_motion(corr_eval.active_snapshot, previous_rotation=mid_state.rotation, previous_position=mid_state.position, new_rotation=accepted_state.rotation, new_position=accepted_state.position)
    final_eval = model.evaluate(accepted_state, warm_start_snapshot=end_warm, repair_mask=repair_mask)
    orientation_err = rotation_angle_error(np.eye(3), state.rotation.T @ accepted_state.rotation)
    return accepted_state, {
        'force': final_eval.wrench.force,
        'torque': final_eval.wrench.torque,
        'active_measure': final_eval.active_measure,
        'active_components': final_eval.active_components,
        'candidate_count': start_eval.candidate_count + pred_eval.candidate_count + corr_eval.candidate_count + final_eval.candidate_count,
        'recompute_count': start_eval.recompute_count + pred_eval.recompute_count + corr_eval.recompute_count + final_eval.recompute_count,
        'mismatch_fraction': mismatch.mismatch_fraction,
        'continuity_jaccard': mismatch.jaccard_index,
        'used_substeps': 1,
        'controller_triggered': False,
        'accepted_snapshot': final_eval.active_snapshot,
        'orientation_error': orientation_err,
    }


def run_complex_rigid_6dof_benchmark(
    setup: ComplexRigid6DoFSetup,
    support_cloud: SupportCloud3D,
    *,
    dt: float,
    controller: RigidEventAwareControllerConfig,
    continuity_enabled: bool = True,
) -> ComplexRigid6DoFHistory:
    model = ComplexRigid6DoFContactModel(
        support_cloud,
        mass=setup.mass,
        inertia_body=setup.inertia_body,
        contact_stiffness=setup.contact_stiffness,
        gravity=setup.gravity,
        continuity_enabled=continuity_enabled,
    )
    state = setup.initial_state()
    init_eval = model.evaluate(state)
    model.commit_active_snapshot(init_eval.active_snapshot)
    t = [state.time]
    pos = [state.position.copy()]
    vel = [state.linear_velocity.copy()]
    ori = [0.0]
    omg = [state.angular_velocity.copy()]
    force = [init_eval.wrench.force.copy()]
    torque = [init_eval.wrench.torque.copy()]
    ameas = [init_eval.active_measure]
    acomp = [init_eval.active_components]
    cand = [init_eval.candidate_count]
    rec = [init_eval.recompute_count]
    mis = [0.0]
    jac = [1.0]
    subs = [1]
    trig = [False]
    energy = [_kinetic_energy(state, setup.gravity)]
    steps = int(np.ceil(setup.t_final / dt))
    for _ in range(steps):
        step_dt = min(dt, setup.t_final - state.time)
        if step_dt <= 0.0:
            break
        state, diag = integrate_complex_rigid_step(model, state, step_dt, controller, warm_start_snapshot=model.warm_start_snapshot)
        model.commit_active_snapshot(diag['accepted_snapshot'])
        t.append(state.time)
        pos.append(state.position.copy())
        vel.append(state.linear_velocity.copy())
        ori.append(diag['orientation_error'])
        omg.append(state.angular_velocity.copy())
        force.append(np.asarray(diag['force'], dtype=float))
        torque.append(np.asarray(diag['torque'], dtype=float))
        ameas.append(diag['active_measure'])
        acomp.append(diag['active_components'])
        cand.append(diag['candidate_count'])
        rec.append(diag['recompute_count'])
        mis.append(diag['mismatch_fraction'])
        jac.append(diag['continuity_jaccard'])
        subs.append(diag['used_substeps'])
        trig.append(diag['controller_triggered'])
        energy.append(_kinetic_energy(state, setup.gravity))
    return ComplexRigid6DoFHistory(
        times=np.asarray(t),
        position=np.asarray(pos),
        linear_velocity=np.asarray(vel),
        orientation_error=np.asarray(ori),
        angular_velocity=np.asarray(omg),
        force=np.asarray(force),
        torque=np.asarray(torque),
        active_measure=np.asarray(ameas),
        active_components=np.asarray(acomp),
        candidate_count=np.asarray(cand),
        recompute_count=np.asarray(rec),
        mismatch_fraction=np.asarray(mis),
        continuity_jaccard=np.asarray(jac),
        used_substeps=np.asarray(subs),
        controller_triggered=np.asarray(trig),
        total_energy=np.asarray(energy),
    )


def _interp(ref_t: np.ndarray, ref_y: np.ndarray, t: np.ndarray) -> np.ndarray:
    if ref_y.ndim == 1:
        return np.interp(t, ref_t, ref_y)
    out = np.vstack([np.interp(t, ref_t, ref_y[:, i]) for i in range(ref_y.shape[1])]).T
    return out


def benchmark_complex_rigid_6dof_error(history: ComplexRigid6DoFHistory, reference: ComplexRigid6DoFHistory) -> ComplexRigid6DoFErrorSummary:
    pos_ref = _interp(reference.times, reference.position, history.times)
    force_ref = _interp(reference.times, reference.force, history.times)
    torque_ref = _interp(reference.times, reference.torque, history.times)
    comp_ref = _interp(reference.times, reference.active_components.astype(float), history.times)
    ori_ref = _interp(reference.times, reference.orientation_error, history.times)
    position_rms = float(np.sqrt(np.mean(np.sum((history.position - pos_ref) ** 2, axis=1))))
    orientation_rms = float(np.sqrt(np.mean((history.orientation_error - ori_ref) ** 2)))
    force_rms = float(np.sqrt(np.mean(np.sum((history.force - force_ref) ** 2, axis=1))))
    torque_rms = float(np.sqrt(np.mean(np.sum((history.torque - torque_ref) ** 2, axis=1))))
    peak_force_error = float(abs(np.max(np.linalg.norm(history.force, axis=1)) - np.max(np.linalg.norm(force_ref, axis=1))) / max(np.max(np.linalg.norm(force_ref, axis=1)), 1e-12))
    peak_torque_error = float(abs(np.max(np.linalg.norm(history.torque, axis=1)) - np.max(np.linalg.norm(torque_ref, axis=1))) / max(np.max(np.linalg.norm(torque_ref, axis=1)), 1e-12))
    release_h = history.times[np.where(np.linalg.norm(history.force, axis=1) <= 1e-8)[0]]
    release_r = reference.times[np.where(np.linalg.norm(reference.force, axis=1) <= 1e-8)[0]]
    release_h_t = float(release_h[0]) if release_h.size else float(history.times[-1])
    release_r_t = float(release_r[0]) if release_r.size else float(reference.times[-1])
    mean_comp_err = float(np.mean(np.abs(history.active_components.astype(float) - comp_ref)))
    return ComplexRigid6DoFErrorSummary(position_rms, orientation_rms, force_rms, torque_rms, peak_force_error, peak_torque_error, abs(release_h_t - release_r_t), mean_comp_err)


def run_high_resolution_complex_6dof_reference(setup: ComplexRigid6DoFSetup, support_cloud: SupportCloud3D) -> ComplexRigid6DoFHistory:
    controller = RigidEventAwareControllerConfig(
        max_depth=7,
        min_dt=2e-4,
        force_relative_jump_tol=0.10,
        torque_relative_jump_tol=0.10,
        active_measure_relative_jump_tol=0.10,
        predictor_corrector_mismatch_fraction_tol=0.07,
        orientation_mismatch_tol=0.08,
    )
    dt = min(0.004, setup.t_final / 220.0)
    return run_complex_rigid_6dof_benchmark(setup, support_cloud, dt=dt, controller=controller, continuity_enabled=True)


def export_complex_rigid_6dof_outputs(history: ComplexRigid6DoFHistory, reference: ComplexRigid6DoFHistory, outdir: str | Path) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8,5))
    plt.plot(reference.times, np.linalg.norm(reference.force, axis=1), label='reference force')
    plt.plot(history.times, np.linalg.norm(history.force, axis=1), label='method force')
    plt.plot(reference.times, np.linalg.norm(reference.torque, axis=1), label='reference torque')
    plt.plot(history.times, np.linalg.norm(history.torque, axis=1), label='method torque')
    plt.xlabel('time'); plt.legend(); plt.tight_layout(); plt.savefig(out/'complex6dof_force_torque_time.pdf'); plt.close()
    plt.figure(figsize=(8,5))
    plt.plot(history.times, history.position[:,2], label='z')
    plt.plot(history.times, history.orientation_error, label='orientation err')
    plt.xlabel('time'); plt.legend(); plt.tight_layout(); plt.savefig(out/'complex6dof_pose_time.pdf'); plt.close()
    plt.figure(figsize=(8,5))
    plt.plot(history.times, history.active_measure, label='active measure')
    plt.plot(history.times, history.mismatch_fraction, label='mismatch fraction')
    plt.plot(history.times, history.continuity_jaccard, label='continuity jaccard')
    plt.xlabel('time'); plt.legend(); plt.tight_layout(); plt.savefig(out/'complex6dof_controller_stats.pdf'); plt.close()


def default_complex_6dof_setup() -> tuple[ComplexRigid6DoFSetup, SupportCloud3D]:
    support_cloud = build_capsule_flat_edge_body_cloud(nx=17, ny=11)
    # inertia for asymmetric medium-sized rigid body
    inertia_body = np.diag([0.018, 0.024, 0.012])
    setup = ComplexRigid6DoFSetup(
        initial_position=np.array([0.0, 0.0, 0.20]),
        initial_rotation=project_to_so3(exp_so3(np.array([0.10, 0.16, 0.02]))),
        initial_linear_velocity=np.array([0.10, 0.02, -1.10]),
        initial_angular_velocity=np.array([0.0, 0.0, 0.0]),
        mass=1.2,
        inertia_body=inertia_body,
        contact_stiffness=1600.0,
        gravity=np.array([0.0, 0.0, -9.81]),
        t_final=0.22,
    )
    return setup, support_cloud
