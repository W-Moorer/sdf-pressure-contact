#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal but structured Python framework for implicit rigid-body contact simulation
based on the pipeline

    build_contact_patches()
    -> extract_sheet()
    -> integrate_tractions()
    -> assemble_wrench()

Current reference scope:
    - one or more dynamic rigid bodies
    - fixed compliant cube domain
    - local-normal contact traction model
    - implicit translational update (rotation placeholders kept in data model)

This is intentionally a clean reference implementation, not a production engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, Optional
import math
import numpy as np


# ============================================================
# Core data types
# ============================================================

@dataclass
class Marker:
    name: str
    local_position: np.ndarray


@dataclass
class Pose:
    position: np.ndarray
    # quaternion kept for future extension; current demo uses translation only
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=float))


@dataclass
class RigidBody:
    name: str
    mass: float
    geometry: "Geometry"
    pose: Pose
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    markers: list[Marker] = field(default_factory=list)
    is_static: bool = False
    linear_damping: float = 0.0

    def world_markers(self) -> dict[str, np.ndarray]:
        # Rotation is intentionally omitted in this minimal demo framework.
        return {m.name: self.pose.position + m.local_position for m in self.markers}

    def clone_with_state(self, position: np.ndarray, linear_velocity: np.ndarray) -> "RigidBody":
        return RigidBody(
            name=self.name,
            mass=self.mass,
            geometry=self.geometry,
            pose=Pose(position=position.copy(), orientation=self.pose.orientation.copy()),
            linear_velocity=linear_velocity.copy(),
            angular_velocity=self.angular_velocity.copy(),
            markers=self.markers,
            is_static=self.is_static,
            linear_damping=self.linear_damping,
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
    bodies: list[RigidBody]


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


@dataclass
class TractionField:
    samples: list[TractionSample]
    metadata: dict = field(default_factory=dict)


@dataclass
class Wrench:
    force: np.ndarray
    moment: np.ndarray


@dataclass
class SimulationLogEntry:
    time: float
    position: np.ndarray
    linear_velocity: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    gap: float
    num_patches: int
    num_sheet_points: int
    num_tractions: int
    marker_positions: dict[str, np.ndarray]


# ============================================================
# Geometry protocol and concrete geometry
# ============================================================

class Geometry(Protocol):
    def phi_world(self, x, y, z, pose: Pose):
        ...

    def footprint_bbox_world(self, pose: Pose, domain: DomainSpec) -> tuple[float, float, float, float]:
        ...

    # optional:
    # def normal_world(self, x: float, y: float, z: float, pose: Pose) -> np.ndarray:
    #     ...


class SphereGeometry:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def phi_world(self, x, y, z, pose: Pose):
        cx, cy, cz = pose.position
        return np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) - self.radius

    def normal_world(self, x: float, y: float, z: float, pose: Pose) -> np.ndarray:
        cx, cy, cz = pose.position
        d = np.array([x - cx, y - cy, z - cz], dtype=float)
        n = np.linalg.norm(d)
        if n <= 1.0e-15:
            return np.array([0.0, -1.0, 0.0], dtype=float)
        return d / n

    def footprint_bbox_world(self, pose: Pose, domain: DomainSpec) -> tuple[float, float, float, float]:
        cx, cy, cz = pose.position
        delta = max(0.0, domain.top_y - (cy - self.radius))
        a = math.sqrt(max(0.0, 2.0 * self.radius * delta - delta * delta))
        return (cx - a, cx + a, cz - a, cz + a)


# ============================================================
# Contact pipeline helpers
# ============================================================

def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


def numerical_normal(geometry: Geometry, x: float, y: float, z: float, pose: Pose, h: float) -> np.ndarray:
    px = (geometry.phi_world(x + h, y, z, pose) - geometry.phi_world(x - h, y, z, pose)) / (2.0 * h)
    py = (geometry.phi_world(x, y + h, z, pose) - geometry.phi_world(x, y - h, z, pose)) / (2.0 * h)
    pz = (geometry.phi_world(x, y, z + h, pose) - geometry.phi_world(x, y, z - h, pose)) / (2.0 * h)
    g = np.array([px, py, pz], dtype=float)
    ng = np.linalg.norm(g)
    if ng <= 1.0e-15:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return g / ng


def build_contact_patches(
    body: RigidBody,
    domain: DomainSpec,
    cfg: PatchBuildConfig,
) -> ContactPatches:
    """
    Build projected contact patches on the x-z top slice of the compliant cube.
    """
    x_min, x_max = domain.x_bounds
    z_min, z_max = domain.z_bounds
    dx = (x_max - x_min) / cfg.Nxz
    dz = (z_max - z_min) / cfg.Nxz

    bx0, bx1, bz0, bz1 = body.geometry.footprint_bbox_world(body.pose, domain)
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
                    if body.geometry.phi_world(xq, top_y, zq, body.pose) <= 0.0:
                        samples.append(ProjectedPatchPoint(float(xq), float(zq), float(w_proj), i, j))

    return ContactPatches(
        samples=samples,
        metadata={
            "num_projected_points": len(samples),
            "Nxz": cfg.Nxz,
            "quad_order": cfg.quad_order,
        },
    )


def extract_sheet(
    body: RigidBody,
    domain: DomainSpec,
    patches: ContactPatches,
    cfg: SheetExtractConfig,
) -> Sheet:
    """
    Extract the actual sheet points by root solving along y.
    """
    y_min, y_max = domain.y_bounds
    samples: list[SheetPoint] = []

    rejected_no_bracket = 0
    rejected_small_ny = 0

    for idx, pp in enumerate(patches.samples):
        phi_top = body.geometry.phi_world(pp.x, y_max, pp.z, body.pose)
        phi_bot = body.geometry.phi_world(pp.x, y_min, pp.z, body.pose)
        if not (phi_top <= 0.0 and phi_bot >= 0.0):
            rejected_no_bracket += 1
            continue

        yl, yr = y_min, y_max
        for _ in range(cfg.bisection_steps):
            ym = 0.5 * (yl + yr)
            fm = body.geometry.phi_world(pp.x, ym, pp.z, body.pose)
            if fm > 0.0:
                yl = ym
            else:
                yr = ym
        y_sigma = 0.5 * (yl + yr)

        if hasattr(body.geometry, "normal_world"):
            n = np.asarray(body.geometry.normal_world(pp.x, y_sigma, pp.z, body.pose), dtype=float)
            nn = np.linalg.norm(n)
            if nn <= 1.0e-15:
                n = numerical_normal(body.geometry, pp.x, y_sigma, pp.z, body.pose, cfg.normal_step)
            else:
                n = n / nn
        else:
            n = numerical_normal(body.geometry, pp.x, y_sigma, pp.z, body.pose, cfg.normal_step)

        if abs(n[1]) < 1.0e-12:
            rejected_small_ny += 1
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

    return Sheet(
        samples=samples,
        metadata={
            "num_sheet_points": len(samples),
            "rejected_no_bracket": rejected_no_bracket,
            "rejected_small_ny": rejected_small_ny,
        },
    )


def integrate_tractions(
    body: RigidBody,
    sheet: Sheet,
    contact_cfg: ContactModelConfig,
) -> TractionField:
    """
    Integrate local tractions on the extracted sheet.

    Current normal law:
        q = k * depth + c * max(0, -v_n)
        t = -q n
    """
    samples: list[TractionSample] = []

    for sp in sheet.samples:
        y = float(sp.position[1])
        depth = max(0.0, contact_cfg.top_y - y)
        vn = float(np.dot(body.linear_velocity, sp.normal))

        q = contact_cfg.stiffness_k * depth + contact_cfg.damping_c * max(0.0, -vn)
        traction = -q * sp.normal
        force = traction * sp.surface_weight

        samples.append(
            TractionSample(
                position=sp.position.copy(),
                normal=sp.normal.copy(),
                pressure=float(q),
                traction=np.asarray(traction, dtype=float),
                area_weight=float(sp.surface_weight),
                force=np.asarray(force, dtype=float),
            )
        )

    return TractionField(
        samples=samples,
        metadata={
            "num_traction_samples": len(samples),
            "stiffness_k": contact_cfg.stiffness_k,
            "damping_c": contact_cfg.damping_c,
        },
    )


def assemble_wrench(
    body: RigidBody,
    tractions: TractionField,
    ref_point: Optional[Sequence[float]] = None,
) -> Wrench:
    """
    Assemble the final wrench around a reference point.
    """
    ref = body.pose.position if ref_point is None else np.asarray(ref_point, dtype=float)

    F = np.zeros(3, dtype=float)
    M = np.zeros(3, dtype=float)

    for ts in tractions.samples:
        F += ts.force
        arm = ts.position - ref
        M += np.cross(arm, ts.force)

    return Wrench(force=F, moment=M)


def compute_contact_response(
    body: RigidBody,
    domain: DomainSpec,
    patch_cfg: PatchBuildConfig,
    sheet_cfg: SheetExtractConfig,
    contact_cfg: ContactModelConfig,
) -> dict:
    """
    Convenience wrapper for the four-stage contact pipeline.
    """
    patches = build_contact_patches(body, domain, patch_cfg)
    sheet = extract_sheet(body, domain, patches, sheet_cfg)
    tractions = integrate_tractions(body, sheet, contact_cfg)
    wrench = assemble_wrench(body, tractions, ref_point=body.pose.position)

    return {
        "patches": patches,
        "sheet": sheet,
        "tractions": tractions,
        "wrench": wrench,
    }


# ============================================================
# Optional baseline
# ============================================================

def direct_band_baseline(
    body: RigidBody,
    domain: DomainSpec,
    contact_cfg: ContactModelConfig,
    N: int = 64,
    eta_factor: float = 1.5,
    ref_point: Optional[Sequence[float]] = None,
) -> Wrench:
    """
    Direct voxel narrow-band baseline, useful for comparisons.
    """
    x_min, x_max = domain.x_bounds
    z_min, z_max = domain.z_bounds
    y_min, y_max = domain.y_bounds

    xs = np.linspace(x_min, x_max, N, endpoint=False) + (x_max - x_min) / N / 2.0
    ys = np.linspace(y_min, y_max, N, endpoint=False) + (y_max - y_min) / N / 2.0
    zs = np.linspace(z_min, z_max, N, endpoint=False) + (z_max - z_min) / N / 2.0

    dx = (x_max - x_min) / N
    dy = (y_max - y_min) / N
    dz = (z_max - z_min) / N
    dV = dx * dy * dz

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    phi = body.geometry.phi_world(X, Y, Z, body.pose)

    h = max(dx, dy, dz)
    eta = eta_factor * h
    band = delta_cosine(phi, eta)

    px = (body.geometry.phi_world(X + h, Y, Z, body.pose) - body.geometry.phi_world(X - h, Y, Z, body.pose)) / (2.0 * h)
    py = (body.geometry.phi_world(X, Y + h, Z, body.pose) - body.geometry.phi_world(X, Y - h, Z, body.pose)) / (2.0 * h)
    pz = (body.geometry.phi_world(X, Y, Z + h, body.pose) - body.geometry.phi_world(X, Y, Z - h, body.pose)) / (2.0 * h)
    gnorm = np.sqrt(px * px + py * py + pz * pz) + 1.0e-15
    nx = px / gnorm
    ny = py / gnorm
    nz = pz / gnorm

    depth = np.maximum(0.0, contact_cfg.top_y - Y)
    p_static = contact_cfg.stiffness_k * depth
    vn = body.linear_velocity[0] * nx + body.linear_velocity[1] * ny + body.linear_velocity[2] * nz
    p_damp = contact_cfg.damping_c * np.maximum(0.0, -vn)
    q = p_static + p_damp

    tx = -q * nx
    ty = -q * ny
    tz = -q * nz

    Fx = float(np.sum(tx * band) * dV)
    Fy = float(np.sum(ty * band) * dV)
    Fz = float(np.sum(tz * band) * dV)

    ref = body.pose.position if ref_point is None else np.asarray(ref_point, dtype=float)
    Mx = float(np.sum(((Y - ref[1]) * tz - (Z - ref[2]) * ty) * band) * dV)
    My = float(np.sum(((Z - ref[2]) * tx - (X - ref[0]) * tz) * band) * dV)
    Mz = float(np.sum(((X - ref[0]) * ty - (Y - ref[1]) * tx) * band) * dV)

    return Wrench(force=np.array([Fx, Fy, Fz], dtype=float), moment=np.array([Mx, My, Mz], dtype=float))


# ============================================================
# Implicit integrator
# ============================================================

class ImplicitEulerIntegrator:
    """
    Minimal implicit translational integrator.

    Unknown:
        x_{n+1}

    With:
        v_{n+1} = (x_{n+1} - x_n) / dt

    Residual:
        R(x) = x - x_n - dt v_n - dt^2 / m * F_total(x, v_{n+1})

    This is enough for the centered falling-sphere demo.
    """
    def __init__(
        self,
        patch_cfg: PatchBuildConfig,
        sheet_cfg: SheetExtractConfig,
        contact_cfg: ContactModelConfig,
        cfg: IntegratorConfig,
    ):
        self.patch_cfg = patch_cfg
        self.sheet_cfg = sheet_cfg
        self.contact_cfg = contact_cfg
        self.cfg = cfg

    def _evaluate_total_force(
        self,
        body: RigidBody,
        world: World,
        trial_position: np.ndarray,
        trial_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        trial_body = body.clone_with_state(trial_position, trial_velocity)
        contact = compute_contact_response(
            trial_body,
            world.domain,
            self.patch_cfg,
            self.sheet_cfg,
            self.contact_cfg,
        )
        F_contact = contact["wrench"].force
        F_total = body.mass * world.gravity - body.linear_damping * trial_velocity + F_contact
        return F_total, F_contact, contact

    def step_body(self, body: RigidBody, world: World) -> tuple[RigidBody, dict]:
        if body.is_static:
            return body, {"contact_force": np.zeros(3), "contact_moment": np.zeros(3), "patches": 0, "sheet_points": 0, "tractions": 0}

        dt = self.cfg.dt
        x0 = body.pose.position.copy()
        v0 = body.linear_velocity.copy()
        m = body.mass

        x = x0 + dt * v0  # predictor

        for _ in range(self.cfg.newton_max_iter):
            v = (x - x0) / dt
            F_total, _, _ = self._evaluate_total_force(body, world, x, v)
            R = x - x0 - dt * v0 - (dt * dt / m) * F_total

            if np.linalg.norm(R) < self.cfg.newton_tol:
                break

            J = np.eye(3)
            eps = self.cfg.fd_eps
            for k in range(3):
                xp = x.copy()
                xp[k] += eps
                vp = (xp - x0) / dt
                Fp, _, _ = self._evaluate_total_force(body, world, xp, vp)
                Rp = xp - x0 - dt * v0 - (dt * dt / m) * Fp
                J[:, k] = (Rp - R) / eps

            try:
                dx = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                dx = -R

            current_norm = np.linalg.norm(R)
            accepted = False
            for alpha in self.cfg.line_search_factors:
                xn = x + alpha * dx
                vn = (xn - x0) / dt
                Fn, _, _ = self._evaluate_total_force(body, world, xn, vn)
                Rn = xn - x0 - dt * v0 - (dt * dt / m) * Fn
                if np.linalg.norm(Rn) < current_norm:
                    x = xn
                    accepted = True
                    break

            if not accepted:
                x = x - 0.2 * R

        v = (x - x0) / dt

        # Commit final state
        body.pose.position = x
        body.linear_velocity = v

        final_contact = compute_contact_response(
            body,
            world.domain,
            self.patch_cfg,
            self.sheet_cfg,
            self.contact_cfg,
        )

        return body, {
            "contact_force": final_contact["wrench"].force.copy(),
            "contact_moment": final_contact["wrench"].moment.copy(),
            "patches": final_contact["patches"].metadata["num_projected_points"],
            "sheet_points": final_contact["sheet"].metadata["num_sheet_points"],
            "tractions": final_contact["tractions"].metadata["num_traction_samples"],
        }


# ============================================================
# Simulator
# ============================================================

class Simulator:
    def __init__(self, world: World, integrator: ImplicitEulerIntegrator):
        self.world = world
        self.integrator = integrator
        self.time = 0.0
        self.log: list[SimulationLogEntry] = []

    def step(self) -> None:
        dt = self.integrator.cfg.dt
        for i, body in enumerate(self.world.bodies):
            if body.is_static:
                continue

            updated, info = self.integrator.step_body(body, self.world)
            self.world.bodies[i] = updated

            # gap to top face for sphere-like interpretation: lowest point - top_y
            gap = 0.0
            if hasattr(updated.geometry, "radius"):
                gap = (updated.pose.position[1] - updated.geometry.radius) - self.world.domain.top_y

            self.log.append(
                SimulationLogEntry(
                    time=self.time + dt,
                    position=updated.pose.position.copy(),
                    linear_velocity=updated.linear_velocity.copy(),
                    contact_force=info["contact_force"].copy(),
                    contact_moment=info["contact_moment"].copy(),
                    gap=float(gap),
                    num_patches=int(info["patches"]),
                    num_sheet_points=int(info["sheet_points"]),
                    num_tractions=int(info["tractions"]),
                    marker_positions={k: v.copy() for k, v in updated.world_markers().items()},
                )
            )

        self.time += dt

    def run(self, total_time: float) -> list[SimulationLogEntry]:
        num_steps = int(round(total_time / self.integrator.cfg.dt))
        for _ in range(num_steps):
            self.step()
        return self.log
