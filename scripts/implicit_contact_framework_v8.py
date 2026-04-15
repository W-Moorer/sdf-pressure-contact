from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from implicit_contact_framework_v7 import (
    Marker,
    Pose6D,
    BodyState6D,
    SpatialInertia,
    RigidBody6D,
    DomainSpec,
    World,
    SphereGeometry,
    UnifiedPairPatchConfig,
    SheetExtractConfig,
    ContactModelConfig,
    UnifiedContactManager,
)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n <= 1e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rv)
    if theta <= 1.0e-15:
        return quat_normalize(np.array([1.0, 0.5*rv[0], 0.5*rv[1], 0.5*rv[2]], dtype=float))
    axis = rv / theta
    h = 0.5 * theta
    return np.array([math.cos(h), *(math.sin(h) * axis)], dtype=float)


def integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    dq = quat_from_rotvec(dt * omega)
    return quat_normalize(quat_mul(dq, q))


def inertia_world(inertia: SpatialInertia, q: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(q)
    return R @ inertia.inertia_body @ R.T


def copy_state(state: BodyState6D) -> BodyState6D:
    return BodyState6D(
        pose=Pose6D(state.pose.position.copy(), state.pose.orientation.copy()),
        linear_velocity=state.linear_velocity.copy(),
        angular_velocity=state.angular_velocity.copy(),
    )


def clone_body_with_state(body: RigidBody6D, state: BodyState6D) -> RigidBody6D:
    return RigidBody6D(
        name=body.name,
        inertia=body.inertia,
        geometry=body.geometry,
        state=copy_state(state),
        markers=body.markers,
        is_static=body.is_static,
        linear_damping=body.linear_damping,
        angular_damping=body.angular_damping,
    )


@dataclass
class IntegratorConfig:
    dt: float = 0.01
    newton_max_iter: int = 8
    newton_tol: float = 1.0e-8
    fd_eps: float = 1.0e-5
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


@dataclass
class ForceAssemblyResult:
    total_force: np.ndarray
    total_moment: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    contact_meta: dict


def accumulate_all_body_forces(world: World, contact_manager: UnifiedContactManager) -> list[ForceAssemblyResult]:
    contacts = contact_manager.compute_all_contacts(world)
    out: list[ForceAssemblyResult] = []
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
    """
    Global multi-body Newton solver that directly calls the unified source-source
    UnifiedContactManager from v7.
    """
    def __init__(self, contact_manager: UnifiedContactManager, cfg: IntegratorConfig):
        self.contact_manager = contact_manager
        self.cfg = cfg

    def _dynamic_indices(self, world: World) -> list[int]:
        return [i for i, b in enumerate(world.bodies) if not b.is_static]

    def _pack_unknowns(self, world: World, dyn: list[int]) -> np.ndarray:
        blocks = []
        for i in dyn:
            b = world.bodies[i]
            blocks.append(np.concatenate([b.state.linear_velocity, b.state.angular_velocity]))
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)

    def _make_trial_world(self, world: World, dyn: list[int], U: np.ndarray) -> World:
        dt = self.cfg.dt
        offset_map = {bi: 6*k for k, bi in enumerate(dyn)}
        new_bodies: list[RigidBody6D] = []
        for i, body in enumerate(world.bodies):
            if i not in offset_map:
                new_bodies.append(body)
                continue
            off = offset_map[i]
            v = U[off:off+3]
            w = U[off+3:off+6]
            x0 = body.state.pose.position
            q0 = body.state.pose.orientation
            x = x0 + dt * v
            q = integrate_quaternion(q0, w, dt)
            new_state = BodyState6D(Pose6D(x.copy(), q.copy()), v.copy(), w.copy())
            new_bodies.append(clone_body_with_state(body, new_state))
        return World(domain=world.domain, gravity=world.gravity.copy(), bodies=new_bodies)

    def _eval_global_residual(self, world: World, dyn: list[int], U: np.ndarray):
        trial_world = self._make_trial_world(world, dyn, U)
        forces = accumulate_all_body_forces(trial_world, self.contact_manager)
        dt = self.cfg.dt
        blocks = []
        for bi in dyn:
            b0 = world.bodies[bi]
            bt = trial_world.bodies[bi]
            f = forces[bi]
            v0 = b0.state.linear_velocity
            w0 = b0.state.angular_velocity
            v = bt.state.linear_velocity
            w = bt.state.angular_velocity
            Iw = inertia_world(bt.inertia, bt.state.pose.orientation)
            Rv = bt.inertia.mass * (v - v0) - dt * f.total_force
            Rw = Iw @ (w - w0) + dt * np.cross(w, Iw @ w) - dt * f.total_moment
            blocks.append(np.concatenate([Rv, Rw]))
        R = np.concatenate(blocks) if blocks else np.zeros(0, dtype=float)
        return R, trial_world, forces

    def step_world(self, world: World) -> list[dict]:
        dyn = self._dynamic_indices(world)
        if not dyn:
            return []
        U = self._pack_unknowns(world, dyn)
        for _ in range(self.cfg.newton_max_iter):
            R, _, _ = self._eval_global_residual(world, dyn, U)
            if np.linalg.norm(R) < self.cfg.newton_tol:
                break
            ndof = len(U)
            J = np.zeros((ndof, ndof), dtype=float)
            eps = self.cfg.fd_eps
            for k in range(ndof):
                Up = U.copy(); Up[k] += eps
                Rp, _, _ = self._eval_global_residual(world, dyn, Up)
                J[:, k] = (Rp - R) / eps
            try:
                dU = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dU = -R
            current = np.linalg.norm(R)
            accepted = False
            for alpha in self.cfg.line_search_factors:
                Un = U + alpha * dU
                Rn, _, _ = self._eval_global_residual(world, dyn, Un)
                if np.linalg.norm(Rn) < current:
                    U = Un
                    accepted = True
                    break
            if not accepted:
                U = U - 0.2 * R
        final_world = self._make_trial_world(world, dyn, U)
        for i in range(len(world.bodies)):
            world.bodies[i] = final_world.bodies[i]
        final_forces = accumulate_all_body_forces(world, self.contact_manager)
        infos = []
        for i, body in enumerate(world.bodies):
            if body.is_static:
                continue
            f = final_forces[i]
            infos.append({
                'body_index': i,
                'contact_force': f.contact_force.copy(),
                'contact_moment': f.contact_moment.copy(),
                **f.contact_meta,
            })
        return infos


@dataclass
class SimulationLogEntry:
    time: float
    body_name: str
    position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    num_pairs: int
    num_pair_patch_points: int
    num_pair_sheet_points: int
    num_pair_tractions: int
    marker_positions: dict[str, np.ndarray]


class Simulator:
    def __init__(self, world: World, solver: GlobalImplicitSystemSolver6D):
        self.world = world
        self.solver = solver
        self.time = 0.0
        self.log: list[SimulationLogEntry] = []

    def step(self) -> None:
        dt = self.solver.cfg.dt
        info_list = self.solver.step_world(self.world)
        info_map = {info['body_index']: info for info in info_list}
        for i, body in enumerate(self.world.bodies):
            if body.is_static:
                continue
            info = info_map.get(i)
            if info is None:
                continue
            self.log.append(SimulationLogEntry(
                time=self.time + dt,
                body_name=body.name,
                position=body.state.pose.position.copy(),
                linear_velocity=body.state.linear_velocity.copy(),
                angular_velocity=body.state.angular_velocity.copy(),
                contact_force=info['contact_force'].copy(),
                contact_moment=info['contact_moment'].copy(),
                num_pairs=int(info['num_pairs']),
                num_pair_patch_points=int(info['num_pair_patch_points']),
                num_pair_sheet_points=int(info['num_pair_sheet_points']),
                num_pair_tractions=int(info['num_pair_tractions']),
                marker_positions={k: v.copy() for k, v in body.world_markers().items()},
            ))
        self.time += dt

    def run(self, total_time: float) -> list[SimulationLogEntry]:
        nsteps = int(round(total_time / self.solver.cfg.dt))
        for _ in range(nsteps):
            self.step()
        return self.log
