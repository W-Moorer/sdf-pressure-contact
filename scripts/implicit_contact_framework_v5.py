#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implicit_contact_framework_v5.py

Global multi-body implicit Newton solver.

Upgrade over v4:
    ImplicitSystemSolver6D is no longer "per-body block implicit".
    It now solves one monolithic nonlinear system per time step:

        U = [v_1, w_1, v_2, w_2, ..., v_N, w_N]

    for all dynamic bodies at once.

Architecture:
    - multi-body pair ContactManager from v4 is preserved
    - residual assembly is now GLOBAL
    - world update is committed GLOBALLY after Newton convergence

Supported contact families:
    1) body-domain
    2) sphere-sphere body-body

Important honesty note:
    - This is a true global Newton solve over all dynamic bodies in the current world.
    - Jacobian is assembled by finite differences, so it is reference-quality, not optimized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Sequence
import math
import numpy as np


# ============================================================
# Math helpers
# ============================================================

def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n <= 1.0e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q) @ v


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


# ============================================================
# Core data
# ============================================================

@dataclass
class Marker:
    name: str
    local_position: np.ndarray


@dataclass
class Pose6D:
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class BodyState6D:
    pose: Pose6D
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def copy(self) -> "BodyState6D":
        return BodyState6D(
            pose=Pose6D(self.pose.position.copy(), self.pose.orientation.copy()),
            linear_velocity=self.linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
        )


@dataclass
class SpatialInertia:
    mass: float
    inertia_body: np.ndarray

    def inertia_world(self, q: np.ndarray) -> np.ndarray:
        R = quat_to_rotmat(q)
        return R @ self.inertia_body @ R.T


class Geometry(Protocol):
    def phi_world(self, x, y, z, pose: Pose6D):
        ...

    def footprint_bbox_world(self, pose: Pose6D, domain: "DomainSpec") -> tuple[float, float, float, float]:
        ...


@dataclass
class RigidBody6D:
    name: str
    inertia: SpatialInertia
    geometry: Geometry
    state: BodyState6D
    markers: list[Marker] = field(default_factory=list)
    is_static: bool = False
    linear_damping: float = 0.0
    angular_damping: float = 0.0

    def world_markers(self) -> dict[str, np.ndarray]:
        return {
            m.name: self.state.pose.position + rotate_vec(self.state.pose.orientation, m.local_position)
            for m in self.markers
        }

    def clone_with_state(self, state: BodyState6D) -> "RigidBody6D":
        return RigidBody6D(
            name=self.name,
            inertia=self.inertia,
            geometry=self.geometry,
            state=state.copy(),
            markers=self.markers,
            is_static=self.is_static,
            linear_damping=self.linear_damping,
            angular_damping=self.angular_damping,
        )


@dataclass
class DomainSpec:
    cube_size: float
    cube_height: float
    top_y: float = 0.0

    @property
    def x_bounds(self) -> tuple[float, float]:
        h = self.cube_size / 2.0
        return -h, h

    @property
    def z_bounds(self) -> tuple[float, float]:
        h = self.cube_size / 2.0
        return -h, h

    @property
    def y_bounds(self) -> tuple[float, float]:
        return self.top_y - self.cube_height, self.top_y


@dataclass
class World:
    domain: DomainSpec
    gravity: np.ndarray
    bodies: list[RigidBody6D]


# ============================================================
# Configs and contact data
# ============================================================

@dataclass
class PatchBuildConfig:
    Nxz: int = 20
    quad_order: int = 3
    bbox_padding_cells: int = 1


@dataclass
class SheetExtractConfig:
    bisection_steps: int = 22
    normal_step: float = 1.0e-6


@dataclass
class ContactModelConfig:
    stiffness_k: float = 10000.0
    damping_c: float = 120.0
    top_y: float = 0.0
    sphere_sphere_stiffness: float = 22000.0
    sphere_sphere_damping: float = 180.0
    sphere_pair_margin: float = 0.03


@dataclass
class IntegratorConfig:
    dt: float = 0.01
    newton_max_iter: int = 8
    newton_tol: float = 1.0e-8
    fd_eps: float = 1.0e-5
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


@dataclass
class ProjectedPatchPoint:
    x: float
    z: float
    projected_weight: float
    cell_i: int = -1
    cell_j: int = -1


@dataclass
class ContactPatches:
    samples: list[ProjectedPatchPoint]
    metadata: dict = field(default_factory=dict)


@dataclass
class SheetPoint:
    position: np.ndarray
    normal: np.ndarray
    projected_weight: float
    surface_weight: float
    source_index: int = -1


@dataclass
class Sheet:
    samples: list[SheetPoint]
    metadata: dict = field(default_factory=dict)


@dataclass
class TractionSample:
    position: np.ndarray
    normal: np.ndarray
    pressure: float
    traction: np.ndarray
    area_weight: float
    force: np.ndarray
    moment_about_com: np.ndarray


@dataclass
class TractionField:
    samples: list[TractionSample]
    metadata: dict = field(default_factory=dict)


@dataclass
class Wrench:
    force: np.ndarray
    moment: np.ndarray


@dataclass
class PairWrenchContribution:
    pair_kind: str
    force: np.ndarray
    moment: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class PairRecord:
    pair_kind: str
    body_i: int
    body_j: Optional[int]
    contribution_i: PairWrenchContribution
    contribution_j: Optional[PairWrenchContribution]
    meta: dict = field(default_factory=dict)


@dataclass
class AggregatedBodyContact:
    total_force: np.ndarray
    total_moment: np.ndarray
    pair_records: list[PairRecord]
    num_projected_points: int = 0
    num_sheet_points: int = 0
    num_tractions: int = 0


# ============================================================
# Example geometry
# ============================================================

class SphereGeometry:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def phi_world(self, x, y, z, pose: Pose6D):
        cx, cy, cz = pose.position
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - self.radius

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        c = pose.position
        d = np.array([x, y, z], dtype=float) - c
        n = np.linalg.norm(d)
        if n <= 1.0e-15:
            return np.array([0.0, -1.0, 0.0], dtype=float)
        return d / n

    def footprint_bbox_world(self, pose: Pose6D, domain: DomainSpec) -> tuple[float, float, float, float]:
        cx, cy, cz = pose.position
        delta = max(0.0, domain.top_y - (cy - self.radius))
        a = math.sqrt(max(0.0, 2.0 * self.radius * delta - delta * delta))
        return (cx - a, cx + a, cz - a, cz + a)


# ============================================================
# Domain contact pipeline
# ============================================================

def numerical_normal(geometry: Geometry, x: float, y: float, z: float, pose: Pose6D, h: float) -> np.ndarray:
    px = (geometry.phi_world(x + h, y, z, pose) - geometry.phi_world(x - h, y, z, pose)) / (2.0 * h)
    py = (geometry.phi_world(x, y + h, z, pose) - geometry.phi_world(x, y - h, z, pose)) / (2.0 * h)
    pz = (geometry.phi_world(x, y, z + h, pose) - geometry.phi_world(x, y, z - h, pose)) / (2.0 * h)
    g = np.array([px, py, pz], dtype=float)
    ng = np.linalg.norm(g)
    if ng <= 1.0e-15:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return g / ng


def build_contact_patches(body: RigidBody6D, domain: DomainSpec, cfg: PatchBuildConfig) -> ContactPatches:
    x_min, x_max = domain.x_bounds
    z_min, z_max = domain.z_bounds
    dx = (x_max - x_min) / cfg.Nxz
    dz = (z_max - z_min) / cfg.Nxz

    bx0, bx1, bz0, bz1 = body.geometry.footprint_bbox_world(body.state.pose, domain)
    i_min = max(0, int(math.floor((bx0 - x_min) / dx)) - cfg.bbox_padding_cells)
    i_max = min(cfg.Nxz - 1, int(math.floor((bx1 - x_min) / dx)) + cfg.bbox_padding_cells)
    j_min = max(0, int(math.floor((bz0 - z_min) / dz)) - cfg.bbox_padding_cells)
    j_max = min(cfg.Nxz - 1, int(math.floor((bz1 - z_min) / dz)) + cfg.bbox_padding_cells)

    xi, wi = np.polynomial.legendre.leggauss(cfg.quad_order)
    top_y = domain.top_y
    samples: list[ProjectedPatchPoint] = []
    for i in range(i_min, i_max + 1):
        xl = x_min + i * dx
        xr = xl + dx
        for j in range(j_min, j_max + 1):
            zl = z_min + j * dz
            zr = zl + dz
            for a, wa in zip(xi, wi):
                xq = 0.5 * (xr - xl) * a + 0.5 * (xr + xl)
                for b, wb in zip(xi, wi):
                    zq = 0.5 * (zr - zl) * b + 0.5 * (zr + zl)
                    w_proj = 0.25 * (xr - xl) * (zr - zl) * wa * wb
                    if body.geometry.phi_world(xq, top_y, zq, body.state.pose) <= 0.0:
                        samples.append(ProjectedPatchPoint(float(xq), float(zq), float(w_proj), i, j))
    return ContactPatches(samples=samples, metadata={"num_projected_points": len(samples)})


def extract_sheet(body: RigidBody6D, domain: DomainSpec, patches: ContactPatches, cfg: SheetExtractConfig) -> Sheet:
    y_min, y_max = domain.y_bounds
    samples: list[SheetPoint] = []
    for idx, pp in enumerate(patches.samples):
        phi_top = body.geometry.phi_world(pp.x, y_max, pp.z, body.state.pose)
        phi_bot = body.geometry.phi_world(pp.x, y_min, pp.z, body.state.pose)
        if not (phi_top <= 0.0 and phi_bot >= 0.0):
            continue

        yl, yr = y_min, y_max
        for _ in range(cfg.bisection_steps):
            ym = 0.5 * (yl + yr)
            fm = body.geometry.phi_world(pp.x, ym, pp.z, body.state.pose)
            if fm > 0.0:
                yl = ym
            else:
                yr = ym
        y_sigma = 0.5 * (yl + yr)

        if hasattr(body.geometry, "normal_world"):
            n = np.asarray(body.geometry.normal_world(pp.x, y_sigma, pp.z, body.state.pose), dtype=float)
            nn = np.linalg.norm(n)
            if nn > 1.0e-15:
                n = n / nn
            else:
                n = numerical_normal(body.geometry, pp.x, y_sigma, pp.z, body.state.pose, cfg.normal_step)
        else:
            n = numerical_normal(body.geometry, pp.x, y_sigma, pp.z, body.state.pose, cfg.normal_step)

        if abs(n[1]) < 1.0e-12:
            continue

        dA = pp.projected_weight / abs(n[1])
        samples.append(SheetPoint(np.array([pp.x, y_sigma, pp.z], dtype=float), n, pp.projected_weight, float(dA), idx))
    return Sheet(samples=samples, metadata={"num_sheet_points": len(samples)})


def integrate_tractions(body: RigidBody6D, sheet: Sheet, contact_cfg: ContactModelConfig) -> TractionField:
    samples: list[TractionSample] = []
    for sp in sheet.samples:
        y = float(sp.position[1])
        depth = max(0.0, contact_cfg.top_y - y)
        r = sp.position - body.state.pose.position
        v_point = body.state.linear_velocity + np.cross(body.state.angular_velocity, r)
        vn = float(np.dot(v_point, sp.normal))
        q = contact_cfg.stiffness_k * depth + contact_cfg.damping_c * max(0.0, -vn)
        traction = -q * sp.normal
        force = traction * sp.surface_weight
        moment = np.cross(r, force)
        samples.append(TractionSample(sp.position.copy(), sp.normal.copy(), float(q), traction, float(sp.surface_weight), force, moment))
    return TractionField(samples=samples, metadata={"num_traction_samples": len(samples)})


def assemble_wrench(body: RigidBody6D, tractions: TractionField) -> Wrench:
    F = np.zeros(3, dtype=float)
    M = np.zeros(3, dtype=float)
    for ts in tractions.samples:
        F += ts.force
        M += ts.moment_about_com
    return Wrench(F, M)


# ============================================================
# Multi-body pair manager
# ============================================================

class ContactManager:
    def __init__(self, patch_cfg: PatchBuildConfig, sheet_cfg: SheetExtractConfig, contact_cfg: ContactModelConfig):
        self.patch_cfg = patch_cfg
        self.sheet_cfg = sheet_cfg
        self.contact_cfg = contact_cfg

    def _compute_body_domain_pair(self, body_i: int, body: RigidBody6D, domain: DomainSpec) -> Optional[PairRecord]:
        patches = build_contact_patches(body, domain, self.patch_cfg)
        sheet = extract_sheet(body, domain, patches, self.sheet_cfg)
        tractions = integrate_tractions(body, sheet, self.contact_cfg)
        wrench = assemble_wrench(body, tractions)
        if np.linalg.norm(wrench.force) <= 1.0e-14 and len(sheet.samples) == 0:
            return None

        contrib_i = PairWrenchContribution(
            "body_domain",
            wrench.force.copy(),
            wrench.moment.copy(),
            {
                "num_projected_points": patches.metadata["num_projected_points"],
                "num_sheet_points": sheet.metadata["num_sheet_points"],
                "num_tractions": tractions.metadata["num_traction_samples"],
            },
        )
        return PairRecord("body_domain", body_i, None, contrib_i, None, {"domain_top_y": domain.top_y})

    def _compute_sphere_sphere_pair(self, i: int, bi: RigidBody6D, j: int, bj: RigidBody6D) -> Optional[PairRecord]:
        if not isinstance(bi.geometry, SphereGeometry) or not isinstance(bj.geometry, SphereGeometry):
            return None

        ci = bi.state.pose.position
        cj = bj.state.pose.position
        d = cj - ci
        dist = np.linalg.norm(d)
        ri = bi.geometry.radius
        rj = bj.geometry.radius

        if dist > ri + rj + self.contact_cfg.sphere_pair_margin:
            return None

        n = np.array([0.0, 1.0, 0.0], dtype=float) if dist <= 1.0e-15 else d / dist
        depth = max(0.0, ri + rj - dist)

        rel_v = bj.state.linear_velocity - bi.state.linear_velocity
        rel_w_term = np.cross(bj.state.angular_velocity, -rj * n) - np.cross(bi.state.angular_velocity, ri * n)
        vn = float(np.dot(rel_v + rel_w_term, n))

        q = self.contact_cfg.sphere_sphere_stiffness * depth + self.contact_cfg.sphere_sphere_damping * max(0.0, -vn)
        if q <= 1.0e-14:
            return None

        Fi = -q * n
        Fj = +q * n
        Mi = np.zeros(3, dtype=float)
        Mj = np.zeros(3, dtype=float)

        return PairRecord(
            "sphere_sphere",
            i,
            j,
            PairWrenchContribution("sphere_sphere", Fi, Mi, {"depth": depth, "vn": vn}),
            PairWrenchContribution("sphere_sphere", Fj, Mj, {"depth": depth, "vn": vn}),
            {"depth": depth, "vn": vn},
        )

    def compute_all_contacts(self, world: World) -> list[AggregatedBodyContact]:
        out = [
            AggregatedBodyContact(
                total_force=np.zeros(3, dtype=float),
                total_moment=np.zeros(3, dtype=float),
                pair_records=[],
                num_projected_points=0,
                num_sheet_points=0,
                num_tractions=0,
            )
            for _ in world.bodies
        ]

        # body-domain
        for i, body in enumerate(world.bodies):
            if body.is_static:
                continue
            rec = self._compute_body_domain_pair(i, body, world.domain)
            if rec is None:
                continue
            out[i].total_force += rec.contribution_i.force
            out[i].total_moment += rec.contribution_i.moment
            out[i].pair_records.append(rec)
            out[i].num_projected_points += rec.contribution_i.meta["num_projected_points"]
            out[i].num_sheet_points += rec.contribution_i.meta["num_sheet_points"]
            out[i].num_tractions += rec.contribution_i.meta["num_tractions"]

        # body-body
        n = len(world.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                if world.bodies[i].is_static and world.bodies[j].is_static:
                    continue
                rec = self._compute_sphere_sphere_pair(i, world.bodies[i], j, world.bodies[j])
                if rec is None:
                    continue

                out[i].total_force += rec.contribution_i.force
                out[i].total_moment += rec.contribution_i.moment
                out[i].pair_records.append(rec)

                out[j].total_force += rec.contribution_j.force
                out[j].total_moment += rec.contribution_j.moment
                out[j].pair_records.append(rec)

        return out


# ============================================================
# Global force assembly
# ============================================================

@dataclass
class ForceAssemblyResult:
    total_force: np.ndarray
    total_moment: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    contact_meta: dict


def accumulate_all_body_forces(world: World, contact_manager: ContactManager) -> list[ForceAssemblyResult]:
    contacts = contact_manager.compute_all_contacts(world)
    out: list[ForceAssemblyResult] = []

    for body, c in zip(world.bodies, contacts):
        F_ext = body.inertia.mass * world.gravity - body.linear_damping * body.state.linear_velocity
        M_ext = -body.angular_damping * body.state.angular_velocity
        out.append(
            ForceAssemblyResult(
                total_force=F_ext + c.total_force,
                total_moment=M_ext + c.total_moment,
                contact_force=c.total_force.copy(),
                contact_moment=c.total_moment.copy(),
                contact_meta={
                    "num_projected_points": c.num_projected_points,
                    "num_sheet_points": c.num_sheet_points,
                    "num_tractions": c.num_tractions,
                    "num_pairs": len(c.pair_records),
                },
            )
        )

    return out


# ============================================================
# Global implicit Newton solver
# ============================================================

class GlobalImplicitSystemSolver6D:
    """
    Monolithic multi-body implicit Newton solver.

    Unknown:
        U = [v_1, w_1, v_2, w_2, ..., v_nd, w_nd]
    for all dynamic bodies.
    """
    def __init__(self, contact_manager: ContactManager, cfg: IntegratorConfig):
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
        new_bodies: list[RigidBody6D] = []
        offset_map = {bi: 6 * k for k, bi in enumerate(dyn)}

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

            new_state = BodyState6D(
                pose=Pose6D(position=x.copy(), orientation=q.copy()),
                linear_velocity=v.copy(),
                angular_velocity=w.copy(),
            )
            new_bodies.append(body.clone_with_state(new_state))

        return World(domain=world.domain, gravity=world.gravity.copy(), bodies=new_bodies)

    def _eval_global_residual(self, world: World, dyn: list[int], U: np.ndarray):
        trial_world = self._make_trial_world(world, dyn, U)
        forces = accumulate_all_body_forces(trial_world, self.contact_manager)

        dt = self.cfg.dt
        blocks = []
        for k, bi in enumerate(dyn):
            b0 = world.bodies[bi]
            bt = trial_world.bodies[bi]
            f = forces[bi]

            v0 = b0.state.linear_velocity
            w0 = b0.state.angular_velocity
            v = bt.state.linear_velocity
            w = bt.state.angular_velocity
            Iw = bt.inertia.inertia_world(bt.state.pose.orientation)

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
                Up = U.copy()
                Up[k] += eps
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

        # Commit the converged global state
        final_world = self._make_trial_world(world, dyn, U)
        for i in range(len(world.bodies)):
            world.bodies[i] = final_world.bodies[i]

        # Compute final forces/meta on committed world
        final_forces = accumulate_all_body_forces(world, self.contact_manager)

        per_body_info = []
        for i, body in enumerate(world.bodies):
            if body.is_static:
                continue
            f = final_forces[i]
            per_body_info.append({
                "body_index": i,
                "contact_force": f.contact_force.copy(),
                "contact_moment": f.contact_moment.copy(),
                "num_projected_points": f.contact_meta["num_projected_points"],
                "num_sheet_points": f.contact_meta["num_sheet_points"],
                "num_tractions": f.contact_meta["num_tractions"],
                "num_pairs": f.contact_meta["num_pairs"],
            })
        return per_body_info


# ============================================================
# Simulator
# ============================================================

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
        info_map = {info["body_index"]: info for info in info_list}

        for i, body in enumerate(self.world.bodies):
            if body.is_static:
                continue
            info = info_map.get(i, None)
            if info is None:
                continue
            self.log.append(
                SimulationLogEntry(
                    time=self.time + dt,
                    body_name=body.name,
                    position=body.state.pose.position.copy(),
                    linear_velocity=body.state.linear_velocity.copy(),
                    angular_velocity=body.state.angular_velocity.copy(),
                    contact_force=info["contact_force"].copy(),
                    contact_moment=info["contact_moment"].copy(),
                    num_pairs=int(info["num_pairs"]),
                    marker_positions={k: v.copy() for k, v in body.world_markers().items()},
                )
            )
        self.time += dt

    def run(self, total_time: float) -> list[SimulationLogEntry]:
        nsteps = int(round(total_time / self.solver.cfg.dt))
        for _ in range(nsteps):
            self.step()
        return self.log
