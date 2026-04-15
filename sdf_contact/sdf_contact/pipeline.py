from __future__ import annotations

from dataclasses import dataclass
import math
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
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25)
    scheme: str = 'backward_euler'          # {'backward_euler', 'implicit_midpoint'}
    jacobian_mode: str = 'forward'          # {'forward', 'central'}
    predictor_mode: str = 'current_velocity' # {'current_velocity', 'explicit_force'}
    linear_regularization: float = 1.0e-10
    collect_diagnostics: bool = False


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

    def _dynamic_indices(self, world: World):
        return [i for i, b in enumerate(world.bodies) if not b.is_static]

    def _pack_unknowns(self, world: World, dyn):
        if not dyn:
            return np.zeros(0)
        return np.concatenate([
            np.concatenate([world.bodies[i].state.linear_velocity, world.bodies[i].state.angular_velocity])
            for i in dyn
        ])

    def _predict_initial_unknowns(self, world: World, dyn):
        U0 = self._pack_unknowns(world, dyn)
        if self.cfg.predictor_mode != 'explicit_force' or len(U0) == 0:
            return U0
        forces = accumulate_all_body_forces(world, self.contact_manager)
        dt = self.cfg.dt
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
                dw = dt * np.linalg.pinv(I0) @ f.total_moment
            Up[off:off + 3] = v0 + dv
            Up[off + 3:off + 6] = w0 + dw
        return Up

    def _trial_state_components(self, b, v: np.ndarray, w: np.ndarray):
        dt = self.cfg.dt
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

    def _make_world_from_unknowns(self, world: World, dyn, U, *, evaluation: bool):
        offs = {bi: 6 * k for k, bi in enumerate(dyn)}
        new_bodies = []
        for i, b in enumerate(world.bodies):
            if i not in offs:
                new_bodies.append(b)
                continue
            off = offs[i]
            v = U[off:off + 3]
            w = U[off + 3:off + 6]
            x_eval, q_eval, v_eval, w_eval, x_final, q_final = self._trial_state_components(b, v, w)
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

    def _make_evaluation_world(self, world: World, dyn, U):
        return self._make_world_from_unknowns(world, dyn, U, evaluation=True)

    def _make_final_world(self, world: World, dyn, U):
        return self._make_world_from_unknowns(world, dyn, U, evaluation=False)

    def _eval_global_residual(self, world: World, dyn, U):
        ew = self._make_evaluation_world(world, dyn, U)
        forces = accumulate_all_body_forces(ew, self.contact_manager)
        dt = self.cfg.dt
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

    def _build_fd_jacobian(self, world: World, dyn, U, R_base=None):
        ndof = len(U)
        if ndof == 0:
            return np.zeros((0, 0))
        J = np.zeros((ndof, ndof))
        mode = self.cfg.jacobian_mode
        if R_base is None:
            R_base, _, _ = self._eval_global_residual(world, dyn, U)
        for k in range(ndof):
            h = self._fd_step_for(U[k])
            if mode == 'central':
                Up = U.copy(); Up[k] += h
                Um = U.copy(); Um[k] -= h
                Rp, _, _ = self._eval_global_residual(world, dyn, Up)
                Rm, _, _ = self._eval_global_residual(world, dyn, Um)
                J[:, k] = (Rp - Rm) / (2.0 * h)
            elif mode == 'forward':
                Up = U.copy(); Up[k] += h
                Rp, _, _ = self._eval_global_residual(world, dyn, Up)
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
        U = self._predict_initial_unknowns(world, dyn)
        R, ew, forces = self._eval_global_residual(world, dyn, U)
        J = self._build_fd_jacobian(world, dyn, U, R)
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

    def step_world(self, world: World):
        dyn = self._dynamic_indices(world)
        if not dyn:
            self.last_step_diagnostics = {'iterations': [], 'accepted': True, 'final_residual_norm': 0.0}
            return []
        U = self._predict_initial_unknowns(world, dyn)
        iter_diags = []
        accepted_any = False
        for it in range(self.cfg.newton_max_iter):
            R, _, _ = self._eval_global_residual(world, dyn, U)
            Rn = float(np.linalg.norm(R))
            if self.cfg.collect_diagnostics:
                iter_diags.append({'iter': it, 'residual_norm': Rn})
            if Rn < self.cfg.newton_tol:
                accepted_any = True
                break
            J = self._build_fd_jacobian(world, dyn, U, R)
            dU, lin_meta = self._solve_linear_step(J, R)
            curr = Rn
            accepted = False
            best_trial = None
            for alpha in self.cfg.line_search_factors:
                Un = U + alpha * dU
                Rnext, _, _ = self._eval_global_residual(world, dyn, Un)
                norm_next = float(np.linalg.norm(Rnext))
                if best_trial is None or norm_next < best_trial[0]:
                    best_trial = (norm_next, alpha, Un)
                if norm_next < curr:
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
                    U = U - 0.1 * R
        Rf, _, _ = self._eval_global_residual(world, dyn, U)
        self.last_step_diagnostics = {
            'iterations': iter_diags,
            'accepted': accepted_any,
            'final_residual_norm': float(np.linalg.norm(Rf)),
            'scheme': self.cfg.scheme,
            'jacobian_mode': self.cfg.jacobian_mode,
            'predictor_mode': self.cfg.predictor_mode,
        }
        fw = self._make_final_world(world, dyn, U)
        world.bodies = fw.bodies
        final_forces = accumulate_all_body_forces(world, self.contact_manager)
        infos = []
        for i, b in enumerate(world.bodies):
            if b.is_static:
                continue
            f = final_forces[i]
            infos.append({'body_index': i, 'contact_force': f.contact_force.copy(), 'contact_moment': f.contact_moment.copy(), **f.contact_meta})
        return infos


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
            info = info_map[i]
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
