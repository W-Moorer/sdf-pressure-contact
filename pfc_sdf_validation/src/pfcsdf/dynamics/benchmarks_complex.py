
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pfcsdf.contact.active_set import ActiveSetContinuityState, ActiveSetSnapshot, active_set_mismatch_report, continuity_report, continuity_update_plan
from pfcsdf.dynamics.events import EventAwareControllerConfig
from pfcsdf.geometry.complex_bodies import SupportProfile2D


@dataclass(frozen=True)
class PlanarRigidBodyState:
    time: float
    z: float
    theta: float
    vz: float
    omega: float
    mass: float
    inertia: float

    def with_state(self, *, time: float | None = None, z: float | None = None, theta: float | None = None, vz: float | None = None, omega: float | None = None) -> 'PlanarRigidBodyState':
        return PlanarRigidBodyState(
            time=self.time if time is None else float(time),
            z=self.z if z is None else float(z),
            theta=self.theta if theta is None else float(theta),
            vz=self.vz if vz is None else float(vz),
            omega=self.omega if omega is None else float(omega),
            mass=self.mass,
            inertia=self.inertia,
        )


@dataclass(frozen=True)
class PlanarContactEvaluation:
    force_z: float
    torque_y: float
    az: float
    alpha: float
    active: bool
    active_measure: float
    active_snapshot: ActiveSetSnapshot
    active_components: int
    candidate_count: int
    recompute_count: int


@dataclass(frozen=True)
class ComplexBodyDropSetup:
    initial_height: float
    initial_velocity: float
    initial_angle: float
    initial_omega: float
    mass: float
    inertia: float
    contact_stiffness: float
    gravity: float
    t_final: float

    def initial_state(self) -> PlanarRigidBodyState:
        return PlanarRigidBodyState(
            time=0.0,
            z=float(self.initial_height),
            theta=float(self.initial_angle),
            vz=float(self.initial_velocity),
            omega=float(self.initial_omega),
            mass=float(self.mass),
            inertia=float(self.inertia),
        )


@dataclass(frozen=True)
class ComplexDynamicHistory:
    times: np.ndarray
    z: np.ndarray
    theta: np.ndarray
    vz: np.ndarray
    omega: np.ndarray
    force_z: np.ndarray
    torque_y: np.ndarray
    active_measure: np.ndarray
    active_components: np.ndarray
    candidate_count: np.ndarray
    recompute_count: np.ndarray
    continuity_jaccard: np.ndarray
    mismatch_fraction: np.ndarray
    used_substeps: np.ndarray
    controller_triggered: np.ndarray
    total_energy: np.ndarray


@dataclass(frozen=True)
class ComplexErrorSummary:
    z_rms: float
    theta_rms: float
    force_rms: float
    torque_rms: float
    peak_force_error: float
    peak_torque_error: float
    release_timing_error: float
    mean_component_count_error: float


class ComplexBodyContactModel:
    def __init__(
        self,
        profile: SupportProfile2D,
        *,
        mass: float,
        inertia: float,
        contact_stiffness: float,
        gravity: float = 9.81,
        continuity_enabled: bool = True,
        boundary_only_update: bool = True,
        continuity_dilation_radius: int = 1,
    ) -> None:
        self.profile = profile
        self.mass = float(mass)
        self.inertia = float(inertia)
        self.contact_stiffness = float(contact_stiffness)
        self.gravity = float(gravity)
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

    def _kinematics(self, state: PlanarRigidBodyState, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.profile.xs[indices]
        z_local = self.profile.zs[indices]
        c = np.cos(state.theta)
        s = np.sin(state.theta)
        x_rel = x * c - z_local * s
        z_world = state.z + x * s + z_local * c
        return x_rel, z_world, self.profile.weights[indices]

    @staticmethod
    def _component_count(mask_1d: np.ndarray) -> int:
        count = 0
        prev = False
        for value in mask_1d.astype(bool):
            if value and not prev:
                count += 1
            prev = bool(value)
        return count

    def evaluate(self, state: PlanarRigidBodyState, *, warm_start_snapshot: ActiveSetSnapshot | None = None) -> PlanarContactEvaluation:
        n = self.profile.n
        support_mask = np.ones((n, 1, 1), dtype=bool)
        if warm_start_snapshot is not None and warm_start_snapshot.active_count > 0 and self.boundary_only_update:
            plan = continuity_update_plan(warm_start_snapshot.mask, extra_mask=support_mask, dilation_radius=self.continuity_dilation_radius)
            candidate_mask = plan.candidate_mask
            retained_mask = plan.retained_interior_mask
            recompute_mask = plan.recompute_mask
        else:
            candidate_mask = support_mask.copy()
            retained_mask = np.zeros_like(support_mask, dtype=bool)
            recompute_mask = candidate_mask.copy()
        candidate_indices = np.flatnonzero(candidate_mask[:, 0, 0])
        x_rel, z_world, weights = self._kinematics(state, candidate_indices)
        overlap = np.maximum(-z_world, 0.0)
        forces = self.contact_stiffness * weights * overlap
        torques = x_rel * forces
        active_mask = np.zeros((n, 1, 1), dtype=bool)
        active_mask[candidate_indices, 0, 0] = overlap > 0.0
        # retained interior is assumed to stay active only if current state also overlaps
        if np.any(retained_mask):
            retained_indices = np.flatnonzero(retained_mask[:, 0, 0])
            x_rel_r, z_world_r, weights_r = self._kinematics(state, retained_indices)
            overlap_r = np.maximum(-z_world_r, 0.0)
            forces_r = self.contact_stiffness * weights_r * overlap_r
            torques_r = x_rel_r * forces_r
            active_mask[retained_indices, 0, 0] = overlap_r > 0.0
            force_z = float(np.sum(forces) + np.sum(forces_r))
            torque_y = float(np.sum(torques) + np.sum(torques_r))
        else:
            force_z = float(np.sum(forces))
            torque_y = float(np.sum(torques))
        active_measure = float(np.sum(self.profile.weights[active_mask[:, 0, 0]]))
        snapshot = ActiveSetSnapshot(mask=active_mask, measure=active_measure)
        return PlanarContactEvaluation(
            force_z=force_z,
            torque_y=torque_y,
            az=force_z / self.mass - self.gravity,
            alpha=torque_y / self.inertia,
            active=force_z > 0.0,
            active_measure=active_measure,
            active_snapshot=snapshot,
            active_components=self._component_count(active_mask[:, 0, 0]),
            candidate_count=int(np.count_nonzero(candidate_mask)),
            recompute_count=int(np.count_nonzero(recompute_mask)),
        )


def _relative_jump(a: float, b: float, *, atol: float = 1e-14) -> float:
    denom = max(abs(a), abs(b), atol)
    return float(abs(b - a) / denom)


def _simpson_step(state: PlanarRigidBodyState, dt: float, a0: tuple[float, float], am: tuple[float, float], a1: tuple[float, float]) -> PlanarRigidBodyState:
    az0, al0 = a0
    azm, alm = am
    az1, al1 = a1
    vz1 = state.vz + dt * (az0 + 4.0 * azm + az1) / 6.0
    omega1 = state.omega + dt * (al0 + 4.0 * alm + al1) / 6.0
    z1 = state.z + 0.5 * dt * (state.vz + vz1)
    th1 = state.theta + 0.5 * dt * (state.omega + omega1)
    return state.with_state(time=state.time + dt, z=z1, theta=th1, vz=vz1, omega=omega1)


def integrate_complex_body_step(
    model: ComplexBodyContactModel,
    state: PlanarRigidBodyState,
    dt: float,
    controller: EventAwareControllerConfig,
    *,
    depth: int = 0,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
) -> tuple[PlanarRigidBodyState, dict[str, Any]]:
    start_eval = model.evaluate(state, warm_start_snapshot=warm_start_snapshot)
    pred_state = state.with_state(
        time=state.time + dt,
        z=state.z + dt * state.vz + 0.5 * dt * dt * start_eval.az,
        theta=state.theta + dt * state.omega + 0.5 * dt * dt * start_eval.alpha,
        vz=state.vz + dt * start_eval.az,
        omega=state.omega + dt * start_eval.alpha,
    )
    pred_eval = model.evaluate(pred_state, warm_start_snapshot=start_eval.active_snapshot)
    mid_state = state.with_state(
        time=state.time + 0.5 * dt,
        z=0.5 * (state.z + pred_state.z),
        theta=0.5 * (state.theta + pred_state.theta),
        vz=0.5 * (state.vz + pred_state.vz),
        omega=0.5 * (state.omega + pred_state.omega),
    )
    corr_eval = model.evaluate(mid_state, warm_start_snapshot=pred_eval.active_snapshot)
    mismatch = active_set_mismatch_report(pred_eval.active_snapshot, corr_eval.active_snapshot)
    force_jump = max(_relative_jump(start_eval.force_z, corr_eval.force_z), _relative_jump(pred_eval.force_z, corr_eval.force_z))
    active_jump = max(_relative_jump(start_eval.active_measure, corr_eval.active_measure), _relative_jump(pred_eval.active_measure, corr_eval.active_measure))
    controller_trigger = False
    if dt > controller.min_dt and depth < controller.max_depth:
        if force_jump > controller.force_relative_jump_tol or active_jump > controller.active_measure_relative_jump_tol or mismatch.mismatch_fraction > controller.predictor_corrector_mismatch_fraction_tol:
            controller_trigger = True
    if controller_trigger:
        half = 0.5 * dt
        mid_result, diag_a = integrate_complex_body_step(model, state, half, controller, depth=depth + 1, warm_start_snapshot=warm_start_snapshot)
        end_result, diag_b = integrate_complex_body_step(model, mid_result, half, controller, depth=depth + 1, warm_start_snapshot=diag_a['accepted_snapshot'])
        merged = {
            'force_z': diag_b['force_z'],
            'torque_y': diag_b['torque_y'],
            'active_measure': diag_b['active_measure'],
            'active_components': diag_b['active_components'],
            'candidate_count': diag_a['candidate_count'] + diag_b['candidate_count'],
            'recompute_count': diag_a['recompute_count'] + diag_b['recompute_count'],
            'used_substeps': diag_a['used_substeps'] + diag_b['used_substeps'],
            'continuity_jaccard': diag_b['continuity_jaccard'],
            'mismatch_fraction': max(diag_a['mismatch_fraction'], diag_b['mismatch_fraction']),
            'controller_triggered': True,
            'accepted_snapshot': diag_b['accepted_snapshot'],
        }
        return end_result, merged
    end_state = _simpson_step(state, dt, (start_eval.az, start_eval.alpha), (corr_eval.az, corr_eval.alpha), (pred_eval.az, pred_eval.alpha))
    final_eval = model.evaluate(end_state, warm_start_snapshot=corr_eval.active_snapshot)
    return end_state, {
        'force_z': final_eval.force_z,
        'torque_y': final_eval.torque_y,
        'active_measure': final_eval.active_measure,
        'active_components': final_eval.active_components,
        'candidate_count': start_eval.candidate_count + pred_eval.candidate_count + corr_eval.candidate_count + final_eval.candidate_count,
        'recompute_count': start_eval.recompute_count + pred_eval.recompute_count + corr_eval.recompute_count + final_eval.recompute_count,
        'used_substeps': 1,
        'continuity_jaccard': mismatch.jaccard_index,
        'mismatch_fraction': mismatch.mismatch_fraction,
        'controller_triggered': False,
        'accepted_snapshot': final_eval.active_snapshot,
    }


def run_complex_body_drop_benchmark(
    setup: ComplexBodyDropSetup,
    profile: SupportProfile2D,
    *,
    dt: float,
    controller: EventAwareControllerConfig,
    continuity_enabled: bool = True,
) -> ComplexDynamicHistory:
    model = ComplexBodyContactModel(
        profile,
        mass=setup.mass,
        inertia=setup.inertia,
        contact_stiffness=setup.contact_stiffness,
        gravity=setup.gravity,
        continuity_enabled=continuity_enabled,
    )
    state = setup.initial_state()
    times = [state.time]
    z_hist = [state.z]
    th_hist = [state.theta]
    vz_hist = [state.vz]
    om_hist = [state.omega]
    init_eval = model.evaluate(state)
    force_hist = [init_eval.force_z]
    torque_hist = [init_eval.torque_y]
    active_measure_hist = [init_eval.active_measure]
    active_components_hist = [init_eval.active_components]
    cand_hist = [init_eval.candidate_count]
    rec_hist = [init_eval.recompute_count]
    jacc_hist = [1.0]
    mis_hist = [0.0]
    sub_hist = [1]
    trig_hist = [False]
    energy_hist = [0.5 * setup.mass * state.vz**2 + 0.5 * setup.inertia * state.omega**2 + setup.mass * setup.gravity * state.z]
    model.commit_active_snapshot(init_eval.active_snapshot)
    n_steps = int(np.ceil(setup.t_final / dt))
    for _ in range(n_steps):
        step_dt = min(dt, setup.t_final - state.time)
        if step_dt <= 0.0:
            break
        next_state, diag = integrate_complex_body_step(model, state, step_dt, controller, warm_start_snapshot=model.warm_start_snapshot)
        model.commit_active_snapshot(diag['accepted_snapshot'])
        state = next_state
        times.append(state.time)
        z_hist.append(state.z)
        th_hist.append(state.theta)
        vz_hist.append(state.vz)
        om_hist.append(state.omega)
        force_hist.append(diag['force_z'])
        torque_hist.append(diag['torque_y'])
        active_measure_hist.append(diag['active_measure'])
        active_components_hist.append(diag['active_components'])
        cand_hist.append(diag['candidate_count'])
        rec_hist.append(diag['recompute_count'])
        jacc_hist.append(diag['continuity_jaccard'])
        mis_hist.append(diag['mismatch_fraction'])
        sub_hist.append(diag['used_substeps'])
        trig_hist.append(diag['controller_triggered'])
        energy_hist.append(0.5 * setup.mass * state.vz**2 + 0.5 * setup.inertia * state.omega**2 + setup.mass * setup.gravity * state.z)
    return ComplexDynamicHistory(
        times=np.asarray(times),
        z=np.asarray(z_hist),
        theta=np.asarray(th_hist),
        vz=np.asarray(vz_hist),
        omega=np.asarray(om_hist),
        force_z=np.asarray(force_hist),
        torque_y=np.asarray(torque_hist),
        active_measure=np.asarray(active_measure_hist),
        active_components=np.asarray(active_components_hist),
        candidate_count=np.asarray(cand_hist),
        recompute_count=np.asarray(rec_hist),
        continuity_jaccard=np.asarray(jacc_hist),
        mismatch_fraction=np.asarray(mis_hist),
        used_substeps=np.asarray(sub_hist),
        controller_triggered=np.asarray(trig_hist),
        total_energy=np.asarray(energy_hist),
    )


def _interp(reference_t: np.ndarray, reference_y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.interp(t, reference_t, reference_y)


def benchmark_complex_body_error(history: ComplexDynamicHistory, reference: ComplexDynamicHistory) -> ComplexErrorSummary:
    z_ref = _interp(reference.times, reference.z, history.times)
    th_ref = _interp(reference.times, reference.theta, history.times)
    f_ref = _interp(reference.times, reference.force_z, history.times)
    tau_ref = _interp(reference.times, reference.torque_y, history.times)
    comp_ref = _interp(reference.times, reference.active_components.astype(float), history.times)
    z_rms = float(np.sqrt(np.mean((history.z - z_ref) ** 2)))
    theta_rms = float(np.sqrt(np.mean((history.theta - th_ref) ** 2)))
    force_rms = float(np.sqrt(np.mean((history.force_z - f_ref) ** 2)))
    torque_rms = float(np.sqrt(np.mean((history.torque_y - tau_ref) ** 2)))
    peak_force_error = float(abs(history.force_z.max() - f_ref.max()) / max(abs(f_ref.max()), 1e-12))
    peak_torque_error = float(abs(np.max(np.abs(history.torque_y)) - np.max(np.abs(tau_ref))) / max(np.max(np.abs(tau_ref)), 1e-12))
    release_hist = history.times[np.where(history.force_z <= 1e-8)[0]]
    release_ref = reference.times[np.where(reference.force_z <= 1e-8)[0]]
    release_h = float(release_hist[0]) if release_hist.size > 0 else float(history.times[-1])
    release_r = float(release_ref[0]) if release_ref.size > 0 else float(reference.times[-1])
    mean_comp_err = float(np.mean(np.abs(history.active_components.astype(float) - comp_ref)))
    return ComplexErrorSummary(
        z_rms=z_rms,
        theta_rms=theta_rms,
        force_rms=force_rms,
        torque_rms=torque_rms,
        peak_force_error=peak_force_error,
        peak_torque_error=peak_torque_error,
        release_timing_error=abs(release_h - release_r),
        mean_component_count_error=mean_comp_err,
    )


def run_high_resolution_complex_reference(setup: ComplexBodyDropSetup, profile: SupportProfile2D) -> ComplexDynamicHistory:
    controller = EventAwareControllerConfig(
        max_depth=10,
        min_dt=1e-5,
        force_relative_jump_tol=0.08,
        active_measure_relative_jump_tol=0.08,
        predictor_corrector_mismatch_fraction_tol=0.05,
    )
    dt = min(0.002, setup.t_final / 400.0)
    return run_complex_body_drop_benchmark(setup, profile, dt=dt, controller=controller, continuity_enabled=True)
