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
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25)


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

    def _dynamic_indices(self, world: World):
        return [i for i, b in enumerate(world.bodies) if not b.is_static]

    def _pack_unknowns(self, world: World, dyn):
        if not dyn:
            return np.zeros(0)
        return np.concatenate([
            np.concatenate([world.bodies[i].state.linear_velocity, world.bodies[i].state.angular_velocity])
            for i in dyn
        ])

    def _make_trial_world(self, world: World, dyn, U):
        dt = self.cfg.dt
        offs = {bi: 6 * k for k, bi in enumerate(dyn)}
        new_bodies = []
        for i, b in enumerate(world.bodies):
            if i not in offs:
                new_bodies.append(b)
                continue
            off = offs[i]
            v = U[off:off + 3]
            w = U[off + 3:off + 6]
            x = b.state.pose.position + dt * v
            q = integrate_quaternion(b.state.pose.orientation, w, dt)
            ns = BodyState6D(Pose6D(x.copy(), q.copy()), v.copy(), w.copy())
            new_bodies.append(b.clone_with_state(ns))
        return World(domain=world.domain, gravity=world.gravity.copy(), bodies=new_bodies, domain_sources=world.domain_sources)

    def _eval_global_residual(self, world: World, dyn, U):
        tw = self._make_trial_world(world, dyn, U)
        forces = accumulate_all_body_forces(tw, self.contact_manager)
        dt = self.cfg.dt
        blocks = []
        for bi in dyn:
            b0 = world.bodies[bi]
            bt = tw.bodies[bi]
            f = forces[bi]
            v0 = b0.state.linear_velocity
            w0 = b0.state.angular_velocity
            v = bt.state.linear_velocity
            w = bt.state.angular_velocity
            Iw = bt.inertia.inertia_world(bt.state.pose.orientation)
            Rv = bt.inertia.mass * (v - v0) - dt * f.total_force
            Rw = Iw @ (w - w0) + dt * np.cross(w, Iw @ w) - dt * f.total_moment
            blocks.append(np.concatenate([Rv, Rw]))
        return (np.concatenate(blocks) if blocks else np.zeros(0)), tw, forces

    def step_world(self, world: World):
        dyn = self._dynamic_indices(world)
        if not dyn:
            return []
        U = self._pack_unknowns(world, dyn)
        for _ in range(self.cfg.newton_max_iter):
            R, _, _ = self._eval_global_residual(world, dyn, U)
            if np.linalg.norm(R) < self.cfg.newton_tol:
                break
            ndof = len(U)
            J = np.zeros((ndof, ndof))
            eps = self.cfg.fd_eps
            for k in range(ndof):
                Up = U.copy()
                Up[k] += eps
                Rp, _, _ = self._eval_global_residual(world, dyn, Up)
                J[:, k] = (Rp - R) / eps
            try:
                dU = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dU = -R
            curr = np.linalg.norm(R)
            accepted = False
            for alpha in self.cfg.line_search_factors:
                Un = U + alpha * dU
                Rn, _, _ = self._eval_global_residual(world, dyn, Un)
                if np.linalg.norm(Rn) < curr:
                    U = Un
                    accepted = True
                    break
            if not accepted:
                U = U - 0.2 * R
        fw = self._make_trial_world(world, dyn, U)
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
