from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import (
    BodySource,
    AggregatedContact,
    World,
    ForceAssemblyResult,
    BodyState6D,
    Pose6D,
    integrate_quaternion,
    SimulationLogEntry,
)


class ContactManager:
    def __init__(self, evaluator):
        self.evaluator = evaluator

    def compute_all_contacts(self, world: World):
        body_sources = [BodySource(i, b) for i, b in enumerate(world.bodies)]
        domain_sources = list(world.domain_sources)
        names = [s.name for s in body_sources] + [s.name for s in domain_sources]
        out = {name: AggregatedContact(np.zeros(3), np.zeros(3), []) for name in names}

        for ds in domain_sources:
            for bs in body_sources:
                if not bs.is_dynamic:
                    continue
                rec = self.evaluator.compute_source_pair(ds, bs, 'source_source')
                if rec is None:
                    continue
                out[ds.name].pair_records.append(rec)
                out[bs.name].pair_records.append(rec)
                out[ds.name].total_force += rec.contribution_a.force
                out[ds.name].total_moment += rec.contribution_a.moment
                out[bs.name].total_force += rec.contribution_b.force
                out[bs.name].total_moment += rec.contribution_b.moment
                for agg, contrib in ((out[ds.name], rec.contribution_a), (out[bs.name], rec.contribution_b)):
                    agg.num_pair_patch_points += contrib.meta['num_pair_patch_points']
                    agg.num_pair_sheet_points += contrib.meta['num_pair_sheet_points']
                    agg.num_pair_tractions += contrib.meta['num_pair_tractions']

        for i in range(len(body_sources)):
            for j in range(i + 1, len(body_sources)):
                rec = self.evaluator.compute_source_pair(body_sources[i], body_sources[j], 'source_source')
                if rec is None:
                    continue
                for name, contrib in ((body_sources[i].name, rec.contribution_a), (body_sources[j].name, rec.contribution_b)):
                    out[name].pair_records.append(rec)
                    out[name].total_force += contrib.force
                    out[name].total_moment += contrib.moment
                    out[name].num_pair_patch_points += contrib.meta['num_pair_patch_points']
                    out[name].num_pair_sheet_points += contrib.meta['num_pair_sheet_points']
                    out[name].num_pair_tractions += contrib.meta['num_pair_tractions']
        return out


@dataclass
class IntegratorConfig:
    dt: float = 0.02
    newton_max_iter: int = 6
    newton_tol: float = 1.0e-8
    fd_eps: float = 1.0e-5
    line_search_factors: tuple[float, ...] = (1.0, 0.8, 0.5, 0.25, 0.1, 0.05)
    scheme: str = 'implicit_midpoint'            # default tuned scheme
    jacobian_mode: str = 'central'               # default tuned Jacobian
    predictor_mode: str = 'explicit_force'       # default tuned predictor
    linear_regularization: float = 1.0e-10
    collect_diagnostics: bool = False
    adaptive_substepping: bool = True
    max_substep_levels: int = 4
    acceptance_abs_residual: float = 5.0e-7
    acceptance_rel_factor: float = 0.8
    min_dt_fraction: float = 1.0 / 16.0
    max_line_search_trials: int = 8
    onset_focus_enabled: bool = True
    onset_force_threshold: float = 1.0e-8
    onset_refine_substeps: int = 2
    onset_local_refine_steps: int = 2
    onset_state_tol: float = 2.5e-4
    onset_velocity_tol: float = 2.5e-3
    onset_force_rel_tol: float = 8.0e-2
    onset_force_abs_tol: float = 1.0e-3
    onset_bisect_iters: int = 6
    onset_aligned_post_fraction: float = 0.5
    onset_microstep_enabled: bool = True
    onset_microstep_fraction: float = 0.8
    onset_microstep_min_fraction: float = 0.10
    early_active_interval_enabled: bool = True
    early_active_interval_samples: int = 4
    early_active_interval_report_force: bool = True
    equivalent_contact_state_enabled: bool = True
    equivalent_contact_state_fraction: float = 0.5
    equivalent_contact_state_zero_velocity: bool = True
    equivalent_contact_state_report_force: bool = False
    impulse_matched_contact_state_enabled: bool = True
    impulse_matched_contact_state_report_force: bool = False
    impulse_match_bisect_iters: int = 10


def accumulate_all_body_forces(world: World, cm: ContactManager):
    contacts = cm.compute_all_contacts(world)
    out = []
    for body in world.bodies:
        agg = contacts[body.name]
        F_ext = body.inertia.mass * world.gravity - body.linear_damping * body.state.linear_velocity
        M_ext = -body.angular_damping * body.state.angular_velocity
        out.append(
            ForceAssemblyResult(
                total_force=F_ext + agg.total_force,
                total_moment=M_ext + agg.total_moment,
                contact_force=agg.total_force.copy(),
                contact_moment=agg.total_moment.copy(),
                contact_meta={
                    'num_pairs': len(agg.pair_records),
                    'num_pair_patch_points': agg.num_pair_patch_points,
                    'num_pair_sheet_points': agg.num_pair_sheet_points,
                    'num_pair_tractions': agg.num_pair_tractions,
                },
            )
        )
    return out


class GlobalImplicitSystemSolver6D:
    def __init__(self, contact_manager: ContactManager, integ_cfg: IntegratorConfig | None = None):
        self.contact_manager = contact_manager
        self.cfg = integ_cfg or IntegratorConfig()
        self.last_step_diagnostics: dict | None = None
        self._onset_focus_steps_remaining: int = 0
        self._last_contact_summary: dict | None = None

    def _dynamic_indices(self, world: World):
        return [i for i, b in enumerate(world.bodies) if not b.is_static]

    def _pack_unknowns(self, world: World, dyn):
        if not dyn:
            return np.zeros(0)
        return np.concatenate([
            np.concatenate([world.bodies[i].state.linear_velocity, world.bodies[i].state.angular_velocity])
            for i in dyn
        ])

    def _predict_initial_unknowns(self, world: World, dyn, dt: float):
        U0 = self._pack_unknowns(world, dyn)
        if self.cfg.predictor_mode != 'explicit_force' or len(U0) == 0:
            return U0
        forces = accumulate_all_body_forces(world, self.contact_manager)
        offs = {bi: 6 * k for k, bi in enumerate(dyn)}
        Up = U0.copy()
        for bi in dyn:
            off = offs[bi]
            b = world.bodies[bi]
            f = forces[bi]
            v0 = b.state.linear_velocity
            w0 = b.state.angular_velocity
            dv = dt * f.total_force / max(b.inertia.mass, 1.0e-15)
            I0 = b.inertia.inertia_world(b.state.pose.orientation)
            try:
                dw = dt * np.linalg.solve(I0, f.total_moment)
            except np.linalg.LinAlgError:
                dw = dt * (np.linalg.pinv(I0) @ f.total_moment)
            Up[off:off + 3] = v0 + dv
            Up[off + 3:off + 6] = w0 + dw
        return Up

    def _trial_state_components(self, b, v: np.ndarray, w: np.ndarray, dt: float):
        x0 = b.state.pose.position
        q0 = b.state.pose.orientation
        v0 = b.state.linear_velocity
        w0 = b.state.angular_velocity
        scheme = self.cfg.scheme
        if scheme == 'implicit_midpoint':
            v_mid = 0.5 * (v0 + v)
            w_mid = 0.5 * (w0 + w)
            x_eval = x0 + 0.5 * dt * v_mid
            x_final = x0 + dt * v_mid
            q_eval = integrate_quaternion(q0, w_mid, 0.5 * dt)
            q_final = integrate_quaternion(q0, w_mid, dt)
            v_eval = v_mid
            w_eval = w_mid
        elif scheme == 'backward_euler':
            x_eval = x0 + dt * v
            x_final = x_eval
            q_eval = integrate_quaternion(q0, w, dt)
            q_final = q_eval
            v_eval = v
            w_eval = w
        else:
            raise ValueError(f'Unknown integrator scheme: {scheme}')
        return x_eval, q_eval, v_eval, w_eval, x_final, q_final

    def _make_world_from_unknowns(self, world: World, dyn, U, dt: float, *, evaluation: bool):
        offs = {bi: 6 * k for k, bi in enumerate(dyn)}
        new_bodies = []
        for i, b in enumerate(world.bodies):
            if i not in offs:
                new_bodies.append(b)
                continue
            off = offs[i]
            v = U[off:off + 3]
            w = U[off + 3:off + 6]
            x_eval, q_eval, v_eval, w_eval, x_final, q_final = self._trial_state_components(b, v, w, dt)
            if evaluation:
                pose = Pose6D(x_eval.copy(), q_eval.copy())
                lin = v_eval.copy()
                ang = w_eval.copy()
            else:
                pose = Pose6D(x_final.copy(), q_final.copy())
                lin = v.copy()
                ang = w.copy()
            ns = BodyState6D(pose, lin, ang)
            new_bodies.append(b.clone_with_state(ns))
        return World(domain=world.domain, gravity=world.gravity.copy(), bodies=new_bodies, domain_sources=world.domain_sources)

    def _make_evaluation_world(self, world: World, dyn, U, dt: float):
        return self._make_world_from_unknowns(world, dyn, U, dt, evaluation=True)

    def _make_final_world(self, world: World, dyn, U, dt: float):
        return self._make_world_from_unknowns(world, dyn, U, dt, evaluation=False)

    def _eval_global_residual(self, world: World, dyn, U, dt: float):
        ew = self._make_evaluation_world(world, dyn, U, dt)
        forces = accumulate_all_body_forces(ew, self.contact_manager)
        offs = {bi: 6 * k for k, bi in enumerate(dyn)}
        blocks = []
        for bi in dyn:
            off = offs[bi]
            b0 = world.bodies[bi]
            ev = ew.bodies[bi]
            f = forces[bi]
            v0 = b0.state.linear_velocity
            w0 = b0.state.angular_velocity
            v = U[off:off + 3]
            w = U[off + 3:off + 6]
            I_eval = ev.inertia.inertia_world(ev.state.pose.orientation)
            Rv = b0.inertia.mass * (v - v0) - dt * f.total_force
            Rw = I_eval @ (w - w0) + dt * np.cross(ev.state.angular_velocity, I_eval @ ev.state.angular_velocity) - dt * f.total_moment
            blocks.append(np.concatenate([Rv, Rw]))
        return (np.concatenate(blocks) if blocks else np.zeros(0)), ew, forces

    def _fd_step_for(self, value: float) -> float:
        return self.cfg.fd_eps * max(1.0, abs(float(value)))

    def _build_fd_jacobian(self, world: World, dyn, U, dt: float, R_base=None):
        ndof = len(U)
        if ndof == 0:
            return np.zeros((0, 0))
        J = np.zeros((ndof, ndof))
        mode = self.cfg.jacobian_mode
        if R_base is None:
            R_base, _, _ = self._eval_global_residual(world, dyn, U, dt)
        for k in range(ndof):
            h = self._fd_step_for(U[k])
            if mode == 'central':
                Up = U.copy(); Up[k] += h
                Um = U.copy(); Um[k] -= h
                Rp, _, _ = self._eval_global_residual(world, dyn, Up, dt)
                Rm, _, _ = self._eval_global_residual(world, dyn, Um, dt)
                J[:, k] = (Rp - Rm) / (2.0 * h)
            elif mode == 'forward':
                Up = U.copy(); Up[k] += h
                Rp, _, _ = self._eval_global_residual(world, dyn, Up, dt)
                J[:, k] = (Rp - R_base) / h
            else:
                raise ValueError(f'Unknown jacobian_mode: {mode}')
        return J

    def _solve_linear_step(self, J: np.ndarray, R: np.ndarray):
        if J.size == 0:
            return np.zeros(0), {'solve_kind': 'empty'}
        reg = float(self.cfg.linear_regularization)
        try:
            cond = float(np.linalg.cond(J))
        except np.linalg.LinAlgError:
            cond = float('inf')
        meta = {'cond': cond, 'reg': reg, 'solve_kind': 'direct'}
        try:
            if reg > 0.0:
                dU = np.linalg.solve(J + reg * np.eye(J.shape[0]), -R)
                meta['solve_kind'] = 'regularized_direct'
            else:
                dU = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError:
            dU, *_ = np.linalg.lstsq(J, -R, rcond=None)
            meta['solve_kind'] = 'lstsq'
        return dU, meta

    def linearize_current_step(self, world: World):
        dyn = self._dynamic_indices(world)
        U = self._predict_initial_unknowns(world, dyn, self.cfg.dt)
        R, ew, forces = self._eval_global_residual(world, dyn, U, self.cfg.dt)
        J = self._build_fd_jacobian(world, dyn, U, self.cfg.dt, R)
        try:
            sing = np.linalg.svd(J, compute_uv=False)
            sigma_min = float(sing[-1]) if len(sing) else float('nan')
            sigma_max = float(sing[0]) if len(sing) else float('nan')
        except np.linalg.LinAlgError:
            sigma_min = float('nan')
            sigma_max = float('nan')
        return {
            'dyn': dyn,
            'U': U,
            'residual_norm': float(np.linalg.norm(R)),
            'residual': R,
            'jacobian': J,
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'cond': float(np.linalg.cond(J)) if J.size else float('nan'),
            'evaluation_world': ew,
            'forces': forces,
        }

    def _snapshot_world(self, world: World):
        return [b.state.copy() for b in world.bodies]

    def _restore_world(self, world: World, snapshot) -> None:
        for b, s in zip(world.bodies, snapshot):
            b.state = s.copy()

    def _summary_from_infos(self, world: World, infos) -> dict:
        info_map = {int(x['body_index']): x for x in infos}
        bodies = {}
        any_active = False
        primary_index = None
        for i, b in enumerate(world.bodies):
            if b.is_static:
                continue
            if primary_index is None:
                primary_index = i
            force = np.asarray(info_map.get(i, {}).get('contact_force', np.zeros(3)), dtype=float)
            active = bool(np.linalg.norm(force) > self.cfg.onset_force_threshold)
            bodies[i] = {
                'active': active,
                'force': force,
                'moment': np.asarray(info_map.get(i, {}).get('contact_moment', np.zeros(3)), dtype=float),
                'y': float(b.state.pose.position[1]),
                'vy': float(b.state.linear_velocity[1]),
            }
            any_active = any_active or active
        return {'any_active': any_active, 'bodies': bodies, 'primary_index': primary_index}

    def _world_contact_summary(self, world: World) -> dict:
        contacts = self.contact_manager.compute_all_contacts(world)
        bodies = {}
        any_active = False
        primary_index = None
        for i, b in enumerate(world.bodies):
            if b.is_static:
                continue
            if primary_index is None:
                primary_index = i
            agg = contacts[b.name]
            force = agg.total_force.copy()
            active = bool(agg.num_pair_tractions > 0 or np.linalg.norm(force) > self.cfg.onset_force_threshold)
            bodies[i] = {
                'active': active,
                'force': force,
                'moment': agg.total_moment.copy(),
                'y': float(b.state.pose.position[1]),
                'vy': float(b.state.linear_velocity[1]),
            }
            any_active = any_active or active
        return {'any_active': any_active, 'bodies': bodies, 'primary_index': primary_index}

    def _state_snapshot_error(self, world: World, snap_a, snap_b, infos_a, infos_b) -> dict:
        info_a = {int(x['body_index']): x for x in infos_a}
        info_b = {int(x['body_index']): x for x in infos_b}
        pos_err = 0.0
        vel_err = 0.0
        force_abs = 0.0
        force_rel = 0.0
        for i, b in enumerate(world.bodies):
            if b.is_static:
                continue
            sa = snap_a[i]
            sb = snap_b[i]
            pos_err = max(pos_err, float(np.linalg.norm(sa.pose.position - sb.pose.position)))
            vel_err = max(vel_err, float(np.linalg.norm(sa.linear_velocity - sb.linear_velocity)))
            fa = np.asarray(info_a.get(i, {}).get('contact_force', np.zeros(3)), dtype=float)
            fb = np.asarray(info_b.get(i, {}).get('contact_force', np.zeros(3)), dtype=float)
            diff = float(np.linalg.norm(fa - fb))
            force_abs = max(force_abs, diff)
            scale = max(float(np.linalg.norm(fa)), float(np.linalg.norm(fb)), 1.0e-12)
            force_rel = max(force_rel, diff / scale)
        return {
            'pos_err': pos_err,
            'vel_err': vel_err,
            'force_abs_err': force_abs,
            'force_rel_err': force_rel,
        }

    def _should_accept_local_refine(self, metrics: dict, coarse_summary: dict, fine_summary: dict) -> bool:
        if coarse_summary.get('any_active') != fine_summary.get('any_active'):
            return True
        return (
            metrics['pos_err'] > self.cfg.onset_state_tol
            or metrics['vel_err'] > self.cfg.onset_velocity_tol
            or metrics['force_rel_err'] > self.cfg.onset_force_rel_tol
            or metrics['force_abs_err'] > self.cfg.onset_force_abs_tol
        )

    def _snapshot_payload(self, snapshot, infos) -> dict:
        info_map = {int(x['body_index']): x for x in infos}
        bodies = {}
        for i, state in enumerate(snapshot):
            bodies[i] = {
                'position': state.pose.position.copy(),
                'orientation': state.pose.orientation.copy(),
                'linear_velocity': state.linear_velocity.copy(),
                'angular_velocity': state.angular_velocity.copy(),
                'contact_force': np.asarray(info_map.get(i, {}).get('contact_force', np.zeros(3)), dtype=float),
                'contact_moment': np.asarray(info_map.get(i, {}).get('contact_moment', np.zeros(3)), dtype=float),
            }
        return {'bodies': bodies}

    def _force_maps_from_infos(self, infos):
        force_map = {}
        moment_map = {}
        for x in infos:
            bi = int(x['body_index'])
            force_map[bi] = np.asarray(x.get('contact_force', np.zeros(3)), dtype=float).copy()
            moment_map[bi] = np.asarray(x.get('contact_moment', np.zeros(3)), dtype=float).copy()
        return force_map, moment_map

    def _payload_from_snapshot_and_maps(self, snapshot, force_map, moment_map, *, time_offset: float, dt: float | None = None):
        bodies = {}
        for i, state in enumerate(snapshot):
            bodies[i] = {
                'position': state.pose.position.copy(),
                'orientation': state.pose.orientation.copy(),
                'linear_velocity': state.linear_velocity.copy(),
                'angular_velocity': state.angular_velocity.copy(),
                'contact_force': np.asarray(force_map.get(i, np.zeros(3)), dtype=float).copy(),
                'contact_moment': np.asarray(moment_map.get(i, np.zeros(3)), dtype=float).copy(),
            }
        payload = {'bodies': bodies, 'time_offset': float(time_offset)}
        if dt is not None:
            payload['dt'] = float(dt)
        return payload

    def _replace_infos_with_payload(self, infos, payload):
        bodies = (payload or {}).get('bodies', {}) if isinstance(payload, dict) else {}
        out = []
        for x in infos:
            bi = int(x['body_index'])
            y = dict(x)
            body = bodies.get(bi, {})
            if 'contact_force' in body:
                y['contact_force'] = np.asarray(body['contact_force'], dtype=float).copy()
            if 'contact_moment' in body:
                y['contact_moment'] = np.asarray(body['contact_moment'], dtype=float).copy()
            out.append(y)
        return out


    def _primary_dynamic_body_index(self, world: World) -> int | None:
        for i, b in enumerate(world.bodies):
            if not b.is_static:
                return i
        return None

    def _interpolate_snapshots(self, snap_a, snap_b, alpha: float):
        a = float(np.clip(alpha, 0.0, 1.0))
        out = []
        for sa, sb in zip(snap_a, snap_b):
            pos = (1.0 - a) * sa.pose.position + a * sb.pose.position
            q = (1.0 - a) * sa.pose.orientation + a * sb.pose.orientation
            qn = np.linalg.norm(q)
            if qn <= 1.0e-15:
                q = sa.pose.orientation.copy()
            else:
                q = q / qn
            lin = (1.0 - a) * sa.linear_velocity + a * sb.linear_velocity
            ang = (1.0 - a) * sa.angular_velocity + a * sb.angular_velocity
            out.append(BodyState6D(Pose6D(pos.astype(float).copy(), q.astype(float).copy()), lin.astype(float).copy(), ang.astype(float).copy()))
        return out

    def _scalar_force_from_payload(self, payload, body_index: int, direction: np.ndarray) -> float:
        body = ((payload or {}).get('bodies') or {}).get(body_index, {}) if isinstance(payload, dict) else {}
        force = np.asarray(body.get('contact_force', np.zeros(3)), dtype=float)
        return float(np.dot(force, direction))

    def _impulse_matched_equivalent_payload(self, world: World, onset_snapshot, end_snapshot, avg_force_map, *, time_offset: float, duration: float):
        if (not self.cfg.impulse_matched_contact_state_enabled) or duration <= 1.0e-12:
            return None
        body_index = self._primary_dynamic_body_index(world)
        if body_index is None:
            return None
        target_force = np.asarray(avg_force_map.get(body_index, np.zeros(3)), dtype=float)
        target_norm = float(np.linalg.norm(target_force))
        if target_norm <= self.cfg.onset_force_threshold:
            return None
        direction = target_force / target_norm
        snapshot_keep = self._snapshot_world(world)
        try:
            payload0 = self._evaluate_static_contact_payload(world, onset_snapshot, time_offset=time_offset, dt=0.0)
            payload1 = self._evaluate_static_contact_payload(world, end_snapshot, time_offset=time_offset + duration, dt=duration)
            f0 = self._scalar_force_from_payload(payload0, body_index, direction)
            f1 = self._scalar_force_from_payload(payload1, body_index, direction)
            target_scalar = float(np.dot(target_force, direction))
            lo, hi = 0.0, 1.0
            pay_lo, pay_hi = payload0, payload1
            flo, fhi = f0, f1
            if min(flo, fhi) <= target_scalar <= max(flo, fhi) and abs(fhi - flo) > 1.0e-12:
                best_alpha = 0.5
                best_payload = None
                best_err = float('inf')
                for _ in range(max(int(self.cfg.impulse_match_bisect_iters), 1)):
                    mid = 0.5 * (lo + hi)
                    snap_mid = self._interpolate_snapshots(onset_snapshot, end_snapshot, mid)
                    pay_mid = self._evaluate_static_contact_payload(world, snap_mid, time_offset=time_offset + mid * duration, dt=mid * duration)
                    fmid = self._scalar_force_from_payload(pay_mid, body_index, direction)
                    err = abs(fmid - target_scalar)
                    if err < best_err:
                        best_err = err
                        best_alpha = mid
                        best_payload = pay_mid
                    if (flo <= target_scalar <= fmid) or (fmid <= target_scalar <= flo):
                        hi = mid
                        pay_hi = pay_mid
                        fhi = fmid
                    else:
                        lo = mid
                        pay_lo = pay_mid
                        flo = fmid
                if best_payload is not None:
                    best_payload['impulse_match_alpha'] = float(best_alpha)
                    best_payload['impulse_match_target_force'] = target_force.copy()
                    return best_payload
            # fallback: choose closer endpoint static state
            if abs(f1 - target_scalar) <= abs(f0 - target_scalar):
                payload1['impulse_match_alpha'] = 1.0
                payload1['impulse_match_target_force'] = target_force.copy()
                return payload1
            payload0['impulse_match_alpha'] = 0.0
            payload0['impulse_match_target_force'] = target_force.copy()
            return payload0
        finally:
            self._restore_world(world, snapshot_keep)

    def _evaluate_static_contact_payload(self, world: World, snapshot, *, time_offset: float, dt: float | None = None):
        snapshot_keep = self._snapshot_world(world)
        self._restore_world(world, snapshot)
        if self.cfg.equivalent_contact_state_zero_velocity:
            for b in world.bodies:
                if not b.is_static:
                    b.state.linear_velocity[:] = 0.0
                    b.state.angular_velocity[:] = 0.0
        contacts = self.contact_manager.compute_all_contacts(world)
        force_map = {}
        moment_map = {}
        for i, b in enumerate(world.bodies):
            if b.is_static:
                continue
            agg = contacts[b.name]
            force_map[i] = agg.total_force.copy()
            moment_map[i] = agg.total_moment.copy()
        snap = self._snapshot_world(world)
        payload = self._payload_from_snapshot_and_maps(snap, force_map, moment_map, time_offset=time_offset, dt=dt)
        self._restore_world(world, snapshot_keep)
        return payload

    def _reconstruct_active_interval_payload(self, world: World, onset_snapshot, onset_infos, remaining_dt: float, base_time_offset: float):
        if (not self.cfg.early_active_interval_enabled) or remaining_dt <= 1.0e-12:
            return None
        snapshot_keep = self._snapshot_world(world)
        n = max(int(self.cfg.early_active_interval_samples), 1)
        dt_seg = remaining_dt / n
        start_force_map, start_moment_map = self._force_maps_from_infos(onset_infos)
        self._restore_world(world, onset_snapshot)
        records = []
        cur_time = float(base_time_offset)
        for k in range(n):
            ok, infos_seg, _ = self._single_step_attempt(world, dt_seg, force_accept=True)
            if not ok:
                break
            snap_seg = self._snapshot_world(world)
            cur_time += dt_seg
            force_map, moment_map = self._force_maps_from_infos(infos_seg)
            records.append({
                'time': cur_time,
                'snapshot': snap_seg,
                'infos': infos_seg,
                'force_map': force_map,
                'moment_map': moment_map,
            })
        if not records:
            self._restore_world(world, snapshot_keep)
            return None

        # midpoint state/force
        target_time = float(base_time_offset + 0.5 * remaining_dt)
        self._restore_world(world, onset_snapshot)
        ok_mid, infos_mid, _ = self._single_step_attempt(world, 0.5 * remaining_dt, force_accept=True)
        midpoint_snapshot = None
        if ok_mid:
            snap_mid = self._snapshot_world(world)
            midpoint_snapshot = snap_mid
            mid_force_map, mid_moment_map = self._force_maps_from_infos(infos_mid)
            midpoint_payload = self._payload_from_snapshot_and_maps(snap_mid, mid_force_map, mid_moment_map, time_offset=target_time, dt=0.5 * remaining_dt)
        else:
            rec_mid = min(records, key=lambda r: abs(r['time'] - target_time))
            midpoint_snapshot = rec_mid['snapshot']
            midpoint_payload = self._payload_from_snapshot_and_maps(rec_mid['snapshot'], rec_mid['force_map'], rec_mid['moment_map'], time_offset=rec_mid['time'])

        equivalent_payload = None
        if self.cfg.equivalent_contact_state_enabled and midpoint_snapshot is not None:
            equivalent_payload = self._evaluate_static_contact_payload(
                world,
                midpoint_snapshot,
                time_offset=float(base_time_offset + float(self.cfg.equivalent_contact_state_fraction) * remaining_dt),
                dt=float(self.cfg.equivalent_contact_state_fraction) * remaining_dt,
            )

        # trapezoidal average of force/moment over active remainder using onset state as left endpoint
        avg_force_map = {i: np.zeros(3, dtype=float) for i in range(len(snapshot_keep))}
        avg_moment_map = {i: np.zeros(3, dtype=float) for i in range(len(snapshot_keep))}
        prev_time = float(base_time_offset)
        prev_force_map = start_force_map
        prev_moment_map = start_moment_map
        for rec in records:
            dt_loc = max(float(rec['time'] - prev_time), 0.0)
            for i in range(len(snapshot_keep)):
                f0 = np.asarray(prev_force_map.get(i, np.zeros(3)), dtype=float)
                f1 = np.asarray(rec['force_map'].get(i, np.zeros(3)), dtype=float)
                m0 = np.asarray(prev_moment_map.get(i, np.zeros(3)), dtype=float)
                m1 = np.asarray(rec['moment_map'].get(i, np.zeros(3)), dtype=float)
                avg_force_map[i] += 0.5 * (f0 + f1) * dt_loc
                avg_moment_map[i] += 0.5 * (m0 + m1) * dt_loc
            prev_time = float(rec['time'])
            prev_force_map = rec['force_map']
            prev_moment_map = rec['moment_map']
        duration = max(float(records[-1]['time'] - base_time_offset), 1.0e-15)
        for i in avg_force_map:
            avg_force_map[i] = avg_force_map[i] / duration
            avg_moment_map[i] = avg_moment_map[i] / duration
        avg_snapshot = snap_mid if ok_mid else rec_mid['snapshot']
        average_payload = self._payload_from_snapshot_and_maps(avg_snapshot, avg_force_map, avg_moment_map, time_offset=float(base_time_offset + 0.5 * duration), dt=duration)
        impulse_payload = self._impulse_matched_equivalent_payload(
            world,
            onset_snapshot,
            records[-1]['snapshot'],
            avg_force_map,
            time_offset=float(base_time_offset),
            duration=float(duration),
        )
        self._restore_world(world, snapshot_keep)
        out = {
            'midpoint': midpoint_payload,
            'average': average_payload,
            'duration': float(duration),
            'samples': len(records),
        }
        if equivalent_payload is not None:
            out['equivalent_state'] = equivalent_payload
        if impulse_payload is not None:
            out['impulse_matched_state'] = impulse_payload
        return out

    def _execute_uniform_substeps(self, world: World, dt: float, n_parts: int, *, reason: str, force_accept: bool = False):
        sub_dt = dt / max(int(n_parts), 1)
        part_diags = []
        part_summaries = []
        part_records = []
        infos = []
        for part in range(max(int(n_parts), 1)):
            part_start_snapshot = self._snapshot_world(world)
            ok, infos, diag = self._single_step_attempt(world, sub_dt, force_accept=force_accept)
            if not ok:
                return False, [], {
                    'dt': dt,
                    'substeps': part + 1,
                    'accepted': False,
                    'refine_reason': reason,
                    'part_diags': part_diags,
                }, part_summaries, part_records
            part_end_snapshot = self._snapshot_world(world)
            summary = self._summary_from_infos(world, infos)
            part_diags.append(diag)
            part_summaries.append(summary)
            part_records.append({
                'part_index': part,
                'dt': sub_dt,
                'start_snapshot': part_start_snapshot,
                'end_snapshot': part_end_snapshot,
                'summary': summary,
                'infos': infos,
            })
        merged = {
            'dt': dt,
            'substeps': int(sum(int(d.get('substeps', 1)) for d in part_diags)),
            'accepted': True,
            'refine_reason': reason,
            'part_diags': part_diags,
            'final_residual_norm': float(part_diags[-1].get('final_residual_norm', float('nan'))) if part_diags else float('nan'),
        }
        return True, infos, merged, part_summaries, part_records

    def _localize_onset_sample(self, world: World, start_snapshot, part_dt: float, prefix_dt: float) -> dict | None:
        snapshot_keep = self._snapshot_world(world)
        lo = 0.0
        hi = part_dt
        best = None
        best_infos = []
        onset_time = None
        onset_snapshot = None
        onset_infos = []
        for _ in range(max(int(self.cfg.onset_bisect_iters), 1)):
            mid = 0.5 * (lo + hi)
            self._restore_world(world, start_snapshot)
            ok, infos_mid, _ = self._single_step_attempt(world, mid, force_accept=True)
            if not ok:
                lo = mid
                continue
            summary_mid = self._summary_from_infos(world, infos_mid)
            if summary_mid.get('any_active', False):
                hi = mid
                best = self._snapshot_world(world)
                best_infos = infos_mid
                onset_time = mid
                onset_snapshot = self._snapshot_world(world)
                onset_infos = infos_mid
            else:
                lo = mid
        if best is None or onset_snapshot is None or onset_time is None:
            self._restore_world(world, snapshot_keep)
            return None

        payload = self._snapshot_payload(onset_snapshot, onset_infos)
        payload['time_offset'] = float(prefix_dt + onset_time)

        # Optional aligned sample later within the same localized part, still reconstructed from the localized onset state.
        if self.cfg.onset_aligned_post_fraction > 0.0:
            remaining = max(part_dt - onset_time, 0.0)
            post_dt = remaining * float(self.cfg.onset_aligned_post_fraction)
            if post_dt > 1.0e-12:
                self._restore_world(world, onset_snapshot)
                ok2, infos2, _ = self._single_step_attempt(world, post_dt, force_accept=True)
                if ok2:
                    aligned_snapshot = self._snapshot_world(world)
                    payload['onset_aligned'] = self._snapshot_payload(aligned_snapshot, infos2)
                    payload['onset_aligned']['time_offset'] = float(prefix_dt + onset_time + post_dt)

        if self.cfg.onset_microstep_enabled:
            remaining = max(part_dt - onset_time, 0.0)
            micro_dt = min(remaining, max(part_dt * float(self.cfg.onset_microstep_min_fraction), remaining * float(self.cfg.onset_microstep_fraction)))
            if micro_dt > 1.0e-12:
                self._restore_world(world, onset_snapshot)
                ok3, infos3, _ = self._single_step_attempt(world, micro_dt, force_accept=True)
                if ok3:
                    micro_snapshot = self._snapshot_world(world)
                    payload['onset_microstep'] = self._snapshot_payload(micro_snapshot, infos3)
                    payload['onset_microstep']['time_offset'] = float(prefix_dt + onset_time + micro_dt)
                    payload['onset_microstep']['dt'] = float(micro_dt)

        active_interval = self._reconstruct_active_interval_payload(world, onset_snapshot, onset_infos, max(part_dt - onset_time, 0.0), float(prefix_dt + onset_time))
        if active_interval is not None:
            payload['onset_active_interval_mid'] = active_interval['midpoint']
            payload['onset_active_interval_avg'] = active_interval['average']
            if 'equivalent_state' in active_interval:
                payload['onset_equivalent_state'] = active_interval['equivalent_state']
            if 'impulse_matched_state' in active_interval:
                payload['onset_impulse_matched_state'] = active_interval['impulse_matched_state']
            payload['onset_active_interval_duration'] = float(active_interval['duration'])
            payload['onset_active_interval_samples'] = int(active_interval['samples'])

        self._restore_world(world, snapshot_keep)
        return payload

    def _onset_offset_from_part_summaries(self, dt: float, part_summaries: list[dict]) -> float:
        if not part_summaries:
            return 0.5 * dt
        sub_dt = dt / len(part_summaries)
        for idx, summ in enumerate(part_summaries):
            if summ.get('any_active'):
                return max(0.0, min(dt, (idx + 0.5) * sub_dt))
        return dt

    def _accept_step(self, initial_norm: float, final_norm: float) -> bool:
        if not np.isfinite(final_norm):
            return False
        thresh = max(self.cfg.acceptance_abs_residual, self.cfg.acceptance_rel_factor * max(initial_norm, self.cfg.newton_tol))
        return final_norm <= thresh

    def _generate_line_search_factors(self):
        base = list(self.cfg.line_search_factors)
        if not base:
            base = [1.0]
        out = []
        seen = set()
        for a in base:
            aa = float(a)
            if aa not in seen:
                out.append(aa)
                seen.add(aa)
        while len(out) < self.cfg.max_line_search_trials:
            out.append(max(out[-1] * 0.5, 1.0e-4))
            if out[-1] <= 1.0e-4:
                break
        return tuple(out)

    def _single_step_attempt(self, world: World, dt: float, *, force_accept: bool = False):
        dyn = self._dynamic_indices(world)
        if not dyn:
            diag = {
                'dt': dt,
                'iterations': [],
                'accepted': True,
                'forced_accept': False,
                'final_residual_norm': 0.0,
                'scheme': self.cfg.scheme,
                'jacobian_mode': self.cfg.jacobian_mode,
                'predictor_mode': self.cfg.predictor_mode,
                'substeps': 1,
                'adaptive_depth': 0,
            }
            return True, [], diag
        U = self._predict_initial_unknowns(world, dyn, dt)
        iter_diags = []
        accepted_any = False
        R0, _, _ = self._eval_global_residual(world, dyn, U, dt)
        initial_norm = float(np.linalg.norm(R0))
        best_U = U.copy()
        best_norm = initial_norm
        line_factors = self._generate_line_search_factors()

        for it in range(self.cfg.newton_max_iter):
            R, _, _ = self._eval_global_residual(world, dyn, U, dt)
            Rn = float(np.linalg.norm(R))
            if Rn < best_norm:
                best_norm = Rn
                best_U = U.copy()
            if self.cfg.collect_diagnostics:
                iter_diags.append({'iter': it, 'residual_norm': Rn})
            if Rn < self.cfg.newton_tol:
                accepted_any = True
                break
            J = self._build_fd_jacobian(world, dyn, U, dt, R)
            dU, lin_meta = self._solve_linear_step(J, R)
            curr = Rn
            accepted = False
            best_trial = None
            for alpha in line_factors:
                Un = U + alpha * dU
                Rnext, _, _ = self._eval_global_residual(world, dyn, Un, dt)
                norm_next = float(np.linalg.norm(Rnext))
                if best_trial is None or norm_next < best_trial[0]:
                    best_trial = (norm_next, alpha, Un)
                armijo_target = curr * max(0.05, 1.0 - 0.25 * alpha)
                if norm_next < min(curr, armijo_target):
                    U = Un
                    accepted = True
                    accepted_any = True
                    break
            if self.cfg.collect_diagnostics:
                iter_diags[-1].update({
                    'jacobian_cond': float(lin_meta.get('cond', float('nan'))),
                    'linear_solve_kind': lin_meta.get('solve_kind', 'unknown'),
                    'accepted': accepted,
                    'step_norm': float(np.linalg.norm(dU)),
                    'best_trial_norm': float(best_trial[0]) if best_trial is not None else float('nan'),
                })
            if not accepted:
                if best_trial is not None and best_trial[0] < curr:
                    U = best_trial[2]
                    accepted_any = True
                else:
                    break

        Rf, _, _ = self._eval_global_residual(world, dyn, best_U, dt)
        final_norm = float(np.linalg.norm(Rf))
        success = self._accept_step(initial_norm, final_norm)
        forced = False
        if (not success) and force_accept and np.isfinite(final_norm):
            success = True
            forced = True

        diag = {
            'dt': dt,
            'iterations': iter_diags,
            'accepted': success and (accepted_any or initial_norm < self.cfg.newton_tol),
            'forced_accept': forced,
            'initial_residual_norm': initial_norm,
            'final_residual_norm': final_norm,
            'best_residual_norm': best_norm,
            'scheme': self.cfg.scheme,
            'jacobian_mode': self.cfg.jacobian_mode,
            'predictor_mode': self.cfg.predictor_mode,
            'substeps': 1,
            'adaptive_depth': 0,
        }

        if not success:
            return False, [], diag

        fw = self._make_final_world(world, dyn, best_U, dt)
        world.bodies = fw.bodies
        final_forces = accumulate_all_body_forces(world, self.contact_manager)
        infos = []
        for i, b in enumerate(world.bodies):
            if b.is_static:
                continue
            f = final_forces[i]
            infos.append({'body_index': i, 'contact_force': f.contact_force.copy(), 'contact_moment': f.contact_moment.copy(), **f.contact_meta})
        return True, infos, diag

    def _step_recursive(self, world: World, dt: float, depth: int = 0):
        if dt < self.cfg.dt * self.cfg.min_dt_fraction:
            success, infos, diag = self._single_step_attempt(world, dt, force_accept=True)
            diag['adaptive_depth'] = depth
            return success, infos, diag

        snapshot = self._snapshot_world(world)
        success, infos, diag = self._single_step_attempt(world, dt)
        diag['adaptive_depth'] = depth
        if success or (not self.cfg.adaptive_substepping) or depth >= self.cfg.max_substep_levels:
            if (not success) and depth >= self.cfg.max_substep_levels:
                self._restore_world(world, snapshot)
                success, infos, diag = self._single_step_attempt(world, dt, force_accept=True)
                diag['adaptive_depth'] = depth
            return success, infos, diag

        self._restore_world(world, snapshot)
        success_a, _, diag_a = self._step_recursive(world, 0.5 * dt, depth + 1)
        if not success_a:
            self._restore_world(world, snapshot)
            return False, [], {'dt': dt, 'substeps': 2, 'adaptive_depth': depth, 'first_half': diag_a, 'accepted': False}
        success_b, infos_b, diag_b = self._step_recursive(world, 0.5 * dt, depth + 1)
        if not success_b:
            self._restore_world(world, snapshot)
            return False, [], {'dt': dt, 'substeps': 2, 'adaptive_depth': depth, 'first_half': diag_a, 'second_half': diag_b, 'accepted': False}
        merged = {
            'dt': dt,
            'substeps': int(diag_a.get('substeps', 1)) + int(diag_b.get('substeps', 1)),
            'adaptive_depth': depth,
            'accepted': True,
            'scheme': self.cfg.scheme,
            'jacobian_mode': self.cfg.jacobian_mode,
            'predictor_mode': self.cfg.predictor_mode,
            'first_half': diag_a,
            'second_half': diag_b,
            'final_residual_norm': float(diag_b.get('final_residual_norm', float('nan'))),
        }
        return True, infos_b, merged

    def step_world(self, world: World):
        snapshot0 = self._snapshot_world(world)
        if self.cfg.onset_focus_enabled:
            start_summary = self._last_contact_summary if self._last_contact_summary is not None else self._world_contact_summary(world)
        else:
            start_summary = {'any_active': False}
        success, infos, diag = self._step_recursive(world, self.cfg.dt, 0)
        if not success:
            self.last_step_diagnostics = diag
            return []
        end_summary = self._summary_from_infos(world, infos) if self.cfg.onset_focus_enabled else {'any_active': False}
        final_snapshot = self._snapshot_world(world)
        final_infos = infos
        final_diag = dict(diag)

        onset_event = self.cfg.onset_focus_enabled and (not start_summary.get('any_active', False)) and end_summary.get('any_active', False)
        release_event = self.cfg.onset_focus_enabled and start_summary.get('any_active', False) and (not end_summary.get('any_active', False))

        if onset_event:
            self._restore_world(world, snapshot0)
            ok_ref, infos_ref, diag_ref, part_summaries, part_records = self._execute_uniform_substeps(
                world,
                self.cfg.dt,
                max(2, int(self.cfg.onset_refine_substeps)),
                reason='onset_uniform_refine',
                force_accept=True,
            )
            if ok_ref:
                final_snapshot = self._snapshot_world(world)
                final_infos = infos_ref
                final_diag = diag_ref
                final_diag['used_local_refinement'] = True
                final_diag['onset_time_offset'] = self._onset_offset_from_part_summaries(self.cfg.dt, part_summaries)
                onset_payload = None
                for rec in part_records:
                    if rec['summary'].get('any_active', False):
                        onset_payload = self._localize_onset_sample(
                            world,
                            rec['start_snapshot'],
                            float(rec['dt']),
                            float(rec['part_index']) * float(rec['dt']),
                        )
                        break
                if onset_payload is not None:
                    final_diag['onset_state'] = onset_payload
                    if 'onset_aligned' in onset_payload:
                        final_diag['onset_aligned'] = onset_payload['onset_aligned']
                    else:
                        final_diag['onset_aligned'] = onset_payload
                    if 'onset_microstep' in onset_payload:
                        final_diag['onset_microstep'] = onset_payload['onset_microstep']
                    if 'onset_active_interval_mid' in onset_payload:
                        final_diag['onset_active_interval_mid'] = onset_payload['onset_active_interval_mid']
                    if 'onset_active_interval_avg' in onset_payload:
                        final_diag['onset_active_interval_avg'] = onset_payload['onset_active_interval_avg']
                    if 'onset_equivalent_state' in onset_payload:
                        final_diag['onset_equivalent_state'] = onset_payload['onset_equivalent_state']
                    if 'onset_impulse_matched_state' in onset_payload:
                        final_diag['onset_impulse_matched_state'] = onset_payload['onset_impulse_matched_state']
                    if self.cfg.impulse_matched_contact_state_report_force and 'onset_impulse_matched_state' in onset_payload:
                        final_infos = self._replace_infos_with_payload(final_infos, onset_payload['onset_impulse_matched_state'])
                    elif self.cfg.equivalent_contact_state_report_force and 'onset_equivalent_state' in onset_payload:
                        final_infos = self._replace_infos_with_payload(final_infos, onset_payload['onset_equivalent_state'])
                    elif self.cfg.early_active_interval_report_force and 'onset_active_interval_avg' in onset_payload:
                        final_infos = self._replace_infos_with_payload(final_infos, onset_payload['onset_active_interval_avg'])
                    final_diag['onset_time_offset'] = float(onset_payload.get('time_offset', final_diag['onset_time_offset']))
                end_summary = self._summary_from_infos(world, final_infos)
            else:
                self._restore_world(world, final_snapshot)
                final_infos = infos
                final_diag = dict(diag)
                final_diag['used_local_refinement'] = False
                final_diag['onset_time_offset'] = self.cfg.dt
            self._onset_focus_steps_remaining = max(self._onset_focus_steps_remaining, int(self.cfg.onset_local_refine_steps))
        elif self.cfg.onset_focus_enabled and self._onset_focus_steps_remaining > 0 and (start_summary.get('any_active', False) or end_summary.get('any_active', False)):
            coarse_snapshot = final_snapshot
            coarse_infos = final_infos
            coarse_diag = dict(final_diag)
            coarse_end_summary = end_summary
            self._restore_world(world, snapshot0)
            ok_ref, infos_ref, diag_ref, _, _ = self._execute_uniform_substeps(
                world,
                self.cfg.dt,
                2,
                reason='first_contact_cycle_halfstep_consistency',
                force_accept=True,
            )
            if ok_ref:
                fine_snapshot = self._snapshot_world(world)
                fine_end_summary = self._summary_from_infos(world, infos_ref)
                metrics = self._state_snapshot_error(world, coarse_snapshot, fine_snapshot, coarse_infos, infos_ref)
                if self._should_accept_local_refine(metrics, coarse_end_summary, fine_end_summary):
                    final_snapshot = fine_snapshot
                    final_infos = infos_ref
                    final_diag = diag_ref
                    final_diag['used_local_refinement'] = True
                    final_diag['consistency_metrics'] = metrics
                    end_summary = fine_end_summary
                else:
                    self._restore_world(world, coarse_snapshot)
                    final_snapshot = coarse_snapshot
                    final_infos = coarse_infos
                    final_diag = coarse_diag
                    final_diag['used_local_refinement'] = False
                    final_diag['consistency_metrics'] = metrics
                    end_summary = coarse_end_summary
            else:
                self._restore_world(world, coarse_snapshot)
                final_snapshot = coarse_snapshot
                final_infos = coarse_infos
                final_diag = coarse_diag
                final_diag['used_local_refinement'] = False
                end_summary = coarse_end_summary
            self._onset_focus_steps_remaining = max(self._onset_focus_steps_remaining - 1, 0)
        elif not end_summary.get('any_active', False):
            self._onset_focus_steps_remaining = 0

        if release_event and not end_summary.get('any_active', False):
            self._onset_focus_steps_remaining = 0

        self._last_contact_summary = end_summary if self.cfg.onset_focus_enabled else None
        self.last_step_diagnostics = final_diag
        return final_infos


class Simulator:
    def __init__(self, world: World, solver: GlobalImplicitSystemSolver6D):
        self.world = world
        self.solver = solver
        self.time = 0.0
        self.log = []

    def step(self):
        dt = self.solver.cfg.dt
        infos = self.solver.step_world(self.world)
        info_map = {x['body_index']: x for x in infos}
        for i, b in enumerate(self.world.bodies):
            if b.is_static:
                continue
            info = info_map.get(i, {
                'contact_force': np.zeros(3),
                'num_pairs': 0,
                'num_pair_patch_points': 0,
                'num_pair_sheet_points': 0,
                'num_pair_tractions': 0,
            })
            self.log.append(
                SimulationLogEntry(
                    time=self.time + dt,
                    body_name=b.name,
                    position=b.state.pose.position.copy(),
                    linear_velocity=b.state.linear_velocity.copy(),
                    angular_velocity=b.state.angular_velocity.copy(),
                    contact_force=info['contact_force'].copy(),
                    num_pairs=int(info['num_pairs']),
                    num_pair_patch_points=int(info['num_pair_patch_points']),
                    num_pair_sheet_points=int(info['num_pair_sheet_points']),
                    num_pair_tractions=int(info['num_pair_tractions']),
                    marker_positions=b.world_markers(),
                )
            )
        self.time += dt

    def run(self, total_time: float):
        nsteps = int(round(total_time / self.solver.cfg.dt))
        for _ in range(nsteps):
            self.step()
        return self.log
