#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implicit_contact_framework_v3.py

Multi-body + full Newton 6-DOF implicit contact framework.

This version upgrades the previous skeleton by replacing the rotational placeholder
with a genuine coupled implicit Newton solve for

    unknowns: u = [v_{n+1}, omega_{n+1}] in R^6

using the residuals

    R_v = m (v_{n+1} - v_n) - dt * F(x_{n+1}, q_{n+1}, v_{n+1}, omega_{n+1})

    R_w = I(q_{n+1}) (omega_{n+1} - omega_n)
          + dt * (omega_{n+1} x (I(q_{n+1}) omega_{n+1}))
          - dt * M(x_{n+1}, q_{n+1}, v_{n+1}, omega_{n+1})

with kinematics

    x_{n+1} = x_n + dt * v_{n+1}
    q_{n+1} = exp(dt * omega_{n+1}) q_n

The current contact manager still handles
    dynamic body vs fixed compliant cube domain
but the solver itself is fully 6-DOF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional, Sequence
import math
import numpy as np


# ============================================================
# Math helpers
# ============================================================

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


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


def quat_conj(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=float)


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


def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


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
    orientation: np.ndarray  # quaternion [w,x,y,z]

    def normalized(self) -> "Pose6D":
        return Pose6D(self.position.copy(), quat_normalize(self.orientation.copy()))


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
    inertia_body: np.ndarray  # 3x3 inertia in body frame

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
        out = {}
        for m in self.markers:
            out[m.name] = self.state.pose.position + rotate_vec(self.state.pose.orientation, m.local_position)
        return out

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
# Contact pipeline data
# ============================================================

@dataclass
class PatchBuildConfig:
    Nxz: int = 28
    quad_order: int = 3
    bbox_padding_cells: int = 1


@dataclass
class SheetExtractConfig:
    bisection_steps: int = 24
    normal_step: float = 1.0e-6


@dataclass
class ContactModelConfig:
    stiffness_k: float = 10000.0
    damping_c: float = 120.0
    top_y: float = 0.0


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


# ============================================================
# Example geometries
# ============================================================

class SphereGeometry:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def phi_world(self, x, y, z, pose: Pose6D):
        cx, cy, cz = pose.position
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - self.radius

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        cx, cy, cz = pose.position
        d = np.array([x - cx, y - cy, z - cz], dtype=float)
        n = np.linalg.norm(d)
        if n <= 1.0e-15:
            return np.array([0.0, -1.0, 0.0], dtype=float)
        return d / n

    def footprint_bbox_world(self, pose: Pose6D, domain: DomainSpec) -> tuple[float, float, float, float]:
        cx, cy, cz = pose.position
        delta = max(0.0, domain.top_y - (cy - self.radius))
        a = math.sqrt(max(0.0, 2.0 * self.radius * delta - delta * delta))
        return (cx - a, cx + a, cz - a, cz + a)


class BoxGeometry:
    """Oriented box geometry with exact SDF."""
    def __init__(self, half_extents: Sequence[float]):
        self.half_extents = np.asarray(half_extents, dtype=float)

    def _to_local(self, x, y, z, pose: Pose6D):
        p = np.stack([np.asarray(x), np.asarray(y), np.asarray(z)], axis=0)
        c = pose.position.reshape(3, *([1] * (p.ndim - 1))) if np.ndim(x) > 0 else pose.position
        R = quat_to_rotmat(pose.orientation)
        if np.ndim(x) == 0:
            return R.T @ (np.array([x, y, z], dtype=float) - pose.position)
        return R.T @ (p - c)

    def phi_world(self, x, y, z, pose: Pose6D):
        local = self._to_local(x, y, z, pose)
        q = np.abs(local) - self.half_extents.reshape(3, *([1] * (local.ndim - 1))) if local.ndim > 1 else np.abs(local) - self.half_extents
        ox = np.maximum(q[0], 0.0)
        oy = np.maximum(q[1], 0.0)
        oz = np.maximum(q[2], 0.0)
        outside = np.sqrt(ox*ox + oy*oy + oz*oz)
        inside = np.minimum(np.maximum.reduce([q[0], q[1], q[2]]), 0.0)
        return outside + inside

    def footprint_bbox_world(self, pose: Pose6D, domain: DomainSpec) -> tuple[float, float, float, float]:
        hx, hy, hz = self.half_extents
        corners_local = np.array([
            [-hx, -hy, -hz], [-hx, -hy, hz], [-hx, hy, -hz], [-hx, hy, hz],
            [hx, -hy, -hz], [hx, -hy, hz], [hx, hy, -hz], [hx, hy, hz],
        ], dtype=float)
        R = quat_to_rotmat(pose.orientation)
        corners_world = (R @ corners_local.T).T + pose.position
        return (
            float(np.min(corners_world[:, 0])),
            float(np.max(corners_world[:, 0])),
            float(np.min(corners_world[:, 2])),
            float(np.max(corners_world[:, 2])),
        )


# ============================================================
# Contact pipeline
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
            n = n / nn if nn > 1.0e-15 else numerical_normal(body.geometry, pp.x, y_sigma, pp.z, body.state.pose, cfg.normal_step)
        else:
            n = numerical_normal(body.geometry, pp.x, y_sigma, pp.z, body.state.pose, cfg.normal_step)

        if abs(n[1]) < 1.0e-12:
            continue

        dA = pp.projected_weight / abs(n[1])
        samples.append(
            SheetPoint(
                position=np.array([pp.x, y_sigma, pp.z], dtype=float),
                normal=n,
                projected_weight=pp.projected_weight,
                surface_weight=float(dA),
                source_index=idx,
            )
        )

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
        moment_about_com = np.cross(r, force)
        samples.append(
            TractionSample(
                position=sp.position.copy(),
                normal=sp.normal.copy(),
                pressure=float(q),
                traction=np.asarray(traction, dtype=float),
                area_weight=float(sp.surface_weight),
                force=np.asarray(force, dtype=float),
                moment_about_com=np.asarray(moment_about_com, dtype=float),
            )
        )

    return TractionField(samples=samples, metadata={"num_traction_samples": len(samples)})


def assemble_wrench(body: RigidBody6D, tractions: TractionField) -> Wrench:
    F = np.zeros(3, dtype=float)
    M = np.zeros(3, dtype=float)
    for ts in tractions.samples:
        F += ts.force
        M += ts.moment_about_com
    return Wrench(force=F, moment=M)


class ContactManager:
    def __init__(self, patch_cfg: PatchBuildConfig, sheet_cfg: SheetExtractConfig, contact_cfg: ContactModelConfig):
        self.patch_cfg = patch_cfg
        self.sheet_cfg = sheet_cfg
        self.contact_cfg = contact_cfg

    def compute_body_contact(self, body: RigidBody6D, domain: DomainSpec) -> dict:
        patches = build_contact_patches(body, domain, self.patch_cfg)
        sheet = extract_sheet(body, domain, patches, self.sheet_cfg)
        tractions = integrate_tractions(body, sheet, self.contact_cfg)
        wrench = assemble_wrench(body, tractions)
        return {"patches": patches, "sheet": sheet, "tractions": tractions, "wrench": wrench}


# ============================================================
# Force accumulation
# ============================================================

@dataclass
class ForceAssemblyResult:
    total_force: np.ndarray
    total_moment: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    contact_meta: dict


def accumulate_body_forces(body: RigidBody6D, world: World, contact_manager: ContactManager) -> ForceAssemblyResult:
    contact = contact_manager.compute_body_contact(body, world.domain)
    F_contact = contact["wrench"].force
    M_contact = contact["wrench"].moment
    F_ext = body.inertia.mass * world.gravity - body.linear_damping * body.state.linear_velocity
    M_ext = -body.angular_damping * body.state.angular_velocity
    return ForceAssemblyResult(
        total_force=F_ext + F_contact,
        total_moment=M_ext + M_contact,
        contact_force=F_contact,
        contact_moment=M_contact,
        contact_meta={
            "num_projected_points": contact["patches"].metadata["num_projected_points"],
            "num_sheet_points": contact["sheet"].metadata["num_sheet_points"],
            "num_tractions": contact["tractions"].metadata["num_traction_samples"],
        },
    )


# ============================================================
# Full Newton 6-DOF implicit solver
# ============================================================

@dataclass
class IntegratorConfig:
    dt: float = 0.01
    newton_max_iter: int = 10
    newton_tol: float = 1.0e-8
    fd_eps: float = 1.0e-5
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


class ImplicitSystemSolver6D:
    """
    Full Newton implicit 6-DOF stepper.

    Unknown per body:
        u = [v_{n+1}, omega_{n+1}] in R^6

    Kinematics:
        x_{n+1} = x_n + dt * v_{n+1}
        q_{n+1} = exp(dt * omega_{n+1}) q_n

    Residual:
        R_v = m (v - v_n) - dt F(x(v), q(w), v, w)
        R_w = I(q)(w - w_n) + dt * w x (I(q)w) - dt M(x(v), q(w), v, w)
    """
    def __init__(self, contact_manager: ContactManager, cfg: IntegratorConfig):
        self.contact_manager = contact_manager
        self.cfg = cfg

    def _make_trial_state(self, body: RigidBody6D, x0: np.ndarray, q0: np.ndarray, u: np.ndarray) -> BodyState6D:
        dt = self.cfg.dt
        v = u[:3]
        w = u[3:]
        x = x0 + dt * v
        q = integrate_quaternion(q0, w, dt)
        return BodyState6D(pose=Pose6D(position=x, orientation=q), linear_velocity=v.copy(), angular_velocity=w.copy())

    def _eval_residual(self, body: RigidBody6D, world: World, u: np.ndarray, x0: np.ndarray, q0: np.ndarray, v0: np.ndarray, w0: np.ndarray) -> tuple[np.ndarray, ForceAssemblyResult, BodyState6D]:
        state = self._make_trial_state(body, x0, q0, u)
        trial_body = body.clone_with_state(state)
        forces = accumulate_body_forces(trial_body, world, self.contact_manager)

        v = state.linear_velocity
        w = state.angular_velocity
        Iw = body.inertia.inertia_world(state.pose.orientation)

        Rv = body.inertia.mass * (v - v0) - self.cfg.dt * forces.total_force
        Rw = Iw @ (w - w0) + self.cfg.dt * np.cross(w, Iw @ w) - self.cfg.dt * forces.total_moment
        R = np.concatenate([Rv, Rw])
        return R, forces, state

    def step_body(self, body: RigidBody6D, world: World) -> tuple[RigidBody6D, dict]:
        if body.is_static:
            return body, {"contact_force": np.zeros(3), "contact_moment": np.zeros(3), "num_projected_points": 0, "num_sheet_points": 0, "num_tractions": 0}

        x0 = body.state.pose.position.copy()
        q0 = body.state.pose.orientation.copy()
        v0 = body.state.linear_velocity.copy()
        w0 = body.state.angular_velocity.copy()

        u = np.concatenate([v0, w0]).astype(float)

        for _ in range(self.cfg.newton_max_iter):
            R, _, _ = self._eval_residual(body, world, u, x0, q0, v0, w0)
            if np.linalg.norm(R) < self.cfg.newton_tol:
                break

            J = np.zeros((6, 6), dtype=float)
            eps = self.cfg.fd_eps
            for k in range(6):
                up = u.copy()
                up[k] += eps
                Rp, _, _ = self._eval_residual(body, world, up, x0, q0, v0, w0)
                J[:, k] = (Rp - R) / eps

            try:
                du = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                du = -R

            current = np.linalg.norm(R)
            accepted = False
            for alpha in self.cfg.line_search_factors:
                un = u + alpha * du
                Rn, _, _ = self._eval_residual(body, world, un, x0, q0, v0, w0)
                if np.linalg.norm(Rn) < current:
                    u = un
                    accepted = True
                    break
            if not accepted:
                u = u - 0.2 * R

        _, final_forces, final_state = self._eval_residual(body, world, u, x0, q0, v0, w0)
        body.state = final_state

        return body, {
            "contact_force": final_forces.contact_force.copy(),
            "contact_moment": final_forces.contact_moment.copy(),
            "num_projected_points": final_forces.contact_meta["num_projected_points"],
            "num_sheet_points": final_forces.contact_meta["num_sheet_points"],
            "num_tractions": final_forces.contact_meta["num_tractions"],
        }


# ============================================================
# Simulator and logging
# ============================================================

@dataclass
class SimulationLogEntry:
    time: float
    body_name: str
    position: np.ndarray
    orientation: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    marker_positions: dict[str, np.ndarray]


class Simulator:
    def __init__(self, world: World, solver: ImplicitSystemSolver6D):
        self.world = world
        self.solver = solver
        self.time = 0.0
        self.log: list[SimulationLogEntry] = []

    def step(self) -> None:
        dt = self.solver.cfg.dt
        for i, body in enumerate(self.world.bodies):
            if body.is_static:
                continue
            updated, info = self.solver.step_body(body, self.world)
            self.world.bodies[i] = updated
            self.log.append(
                SimulationLogEntry(
                    time=self.time + dt,
                    body_name=updated.name,
                    position=updated.state.pose.position.copy(),
                    orientation=updated.state.pose.orientation.copy(),
                    linear_velocity=updated.state.linear_velocity.copy(),
                    angular_velocity=updated.state.angular_velocity.copy(),
                    contact_force=info["contact_force"].copy(),
                    contact_moment=info["contact_moment"].copy(),
                    marker_positions={k: v.copy() for k, v in updated.world_markers().items()},
                )
            )
        self.time += dt

    def run(self, total_time: float) -> list[SimulationLogEntry]:
        nsteps = int(round(total_time / self.solver.cfg.dt))
        for _ in range(nsteps):
            self.step()
        return self.log
