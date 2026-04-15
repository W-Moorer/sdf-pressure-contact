#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implicit_contact_framework_v6.py

Upgrade over v5:
    body-body contact is no longer hard-coded as sphere-sphere point contact.
    It is extended to a more general body-body local patch / sheet extraction pipeline.

Pair-local pipeline for body-body:
    build_pair_contact_patches()
    -> extract_pair_sheet()
    -> integrate_pair_tractions()
    -> assemble_pair_wrench()

Current reference implementation assumptions:
    - best suited for smooth SDF bodies
    - pair local normal initialized from COM-to-COM direction
    - pair patch built on a local tangent plane
    - pair sheet extracted by root solves along the pair normal on both bodies

This gives a real architectural extension from point pair contact to distributed pair contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Optional
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


def orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / max(np.linalg.norm(n), 1.0e-15)
    if abs(n[2]) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=float)
    t1 = np.cross(n, a)
    t1 /= max(np.linalg.norm(t1), 1.0e-15)
    t2 = np.cross(n, t1)
    t2 /= max(np.linalg.norm(t2), 1.0e-15)
    return t1, t2


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

    # optional but strongly recommended
    # def normal_world(self, x, y, z, pose: Pose6D) -> np.ndarray:
    #     ...
    # def bounding_radius(self) -> float:
    #     ...


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
        return {m.name: self.state.pose.position + rotate_vec(self.state.pose.orientation, m.local_position) for m in self.markers}

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
# Configs
# ============================================================

@dataclass
class PatchBuildConfig:
    Nxz: int = 20
    quad_order: int = 3
    bbox_padding_cells: int = 1


@dataclass
class BodyBodyPatchBuildConfig:
    Nuv: int = 14
    quad_order: int = 3
    radius_scale: float = 1.5
    min_patch_radius: float = 0.01
    max_patch_radius: float = 0.35
    ray_span_scale: float = 1.2


@dataclass
class SheetExtractConfig:
    bisection_steps: int = 22
    normal_step: float = 1.0e-6


@dataclass
class ContactModelConfig:
    stiffness_k: float = 10000.0
    damping_c: float = 120.0
    top_y: float = 0.0
    pair_stiffness_k: float = 18000.0
    pair_damping_c: float = 140.0


@dataclass
class IntegratorConfig:
    dt: float = 0.01
    newton_max_iter: int = 8
    newton_tol: float = 1.0e-8
    fd_eps: float = 1.0e-5
    line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.1)


# ============================================================
# Contact data
# ============================================================

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
class PairPatchSample:
    uv: np.ndarray
    world_seed: np.ndarray
    projected_weight: float


@dataclass
class PairContactPatches:
    samples: list[PairPatchSample]
    center: np.ndarray
    normal0: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    radius: float
    metadata: dict = field(default_factory=dict)


@dataclass
class PairSheetPoint:
    midpoint: np.ndarray
    point_i: np.ndarray
    point_j: np.ndarray
    normal_i: np.ndarray
    normal_j: np.ndarray
    sheet_normal: np.ndarray
    projected_weight: float
    surface_weight: float
    source_index: int = -1


@dataclass
class PairSheet:
    samples: list[PairSheetPoint]
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
class PairTractionSample:
    midpoint: np.ndarray
    sheet_normal: np.ndarray
    pressure: float
    area_weight: float
    force_on_i: np.ndarray
    force_on_j: np.ndarray
    moment_i: np.ndarray
    moment_j: np.ndarray


@dataclass
class PairTractionField:
    samples: list[PairTractionSample]
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
    num_pair_patch_points: int = 0
    num_pair_sheet_points: int = 0
    num_pair_tractions: int = 0


# ============================================================
# Example geometry
# ============================================================

class SphereGeometry:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def bounding_radius(self) -> float:
        return self.radius

    def phi_world(self, x, y, z, pose: Pose6D):
        cx, cy, cz = pose.position
        return np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2) - self.radius

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        c = pose.position
        d = np.array([x, y, z], dtype=float) - c
        n = np.linalg.norm(d)
        if n <= 1.0e-15:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return d / n

    def footprint_bbox_world(self, pose: Pose6D, domain: DomainSpec) -> tuple[float, float, float, float]:
        cx, cy, cz = pose.position
        delta = max(0.0, domain.top_y - (cy - self.radius))
        a = math.sqrt(max(0.0, 2.0 * self.radius * delta - delta * delta))
        return (cx - a, cx + a, cz - a, cz + a)


# ============================================================
# Helpers
# ============================================================

def numerical_normal(geometry: Geometry, x: float, y: float, z: float, pose: Pose6D, h: float) -> np.ndarray:
    px = (geometry.phi_world(x + h, y, z, pose) - geometry.phi_world(x - h, y, z, pose)) / (2.0 * h)
    py = (geometry.phi_world(x, y + h, z, pose) - geometry.phi_world(x, y - h, z, pose)) / (2.0 * h)
    pz = (geometry.phi_world(x, y, z + h, pose) - geometry.phi_world(x, y, z - h, pose)) / (2.0 * h)
    g = np.array([px, py, pz], dtype=float)
    ng = np.linalg.norm(g)
    if ng <= 1.0e-15:
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return g / ng


def get_geometry_normal(body: RigidBody6D, x: float, y: float, z: float, h: float) -> np.ndarray:
    if hasattr(body.geometry, "normal_world"):
        n = np.asarray(body.geometry.normal_world(x, y, z, body.state.pose), dtype=float)
        nn = np.linalg.norm(n)
        if nn > 1.0e-15:
            return n / nn
    return numerical_normal(body.geometry, x, y, z, body.state.pose, h)


# ============================================================
# Body-domain pipeline
# ============================================================

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

        n = get_geometry_normal(body, pp.x, y_sigma, pp.z, cfg.normal_step)
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
# Body-body local patch / sheet pipeline
# ============================================================

def estimate_pair_patch_geometry(
    body_i: RigidBody6D,
    body_j: RigidBody6D,
    cfg: BodyBodyPatchBuildConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    """
    Estimate a local pair frame and patch radius.

    Returns:
        center0, n0, t1, t2, patch_radius, ray_span
    """
    ci = body_i.state.pose.position
    cj = body_j.state.pose.position
    d = cj - ci
    dist = np.linalg.norm(d)
    if dist <= 1.0e-15:
        n0 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        n0 = d / dist

    if not hasattr(body_i.geometry, "bounding_radius") or not hasattr(body_j.geometry, "bounding_radius"):
        return None

    ri = float(body_i.geometry.bounding_radius())
    rj = float(body_j.geometry.bounding_radius())
    overlap = max(0.0, ri + rj - dist)
    if overlap <= 0.0:
        return None

    # support points along the initial pair normal
    xi0 = ci + ri * n0
    xj0 = cj - rj * n0
    center0 = 0.5 * (xi0 + xj0)

    reff = (ri * rj) / max(ri + rj, 1.0e-15)
    a = math.sqrt(max(0.0, 2.0 * reff * overlap))
    patch_radius = min(cfg.max_patch_radius, max(cfg.min_patch_radius, cfg.radius_scale * a))
    ray_span = cfg.ray_span_scale * (ri + rj + patch_radius)

    t1, t2 = orthonormal_basis_from_normal(n0)
    return center0, n0, t1, t2, patch_radius, ray_span


def build_pair_contact_patches(
    body_i: RigidBody6D,
    body_j: RigidBody6D,
    cfg: BodyBodyPatchBuildConfig,
) -> PairContactPatches | None:
    """
    Build pair-local projected contact patches on a tangent plane.

    Acceptance criterion:
        the seed point lies inside both bodies, i.e.
            phi_i(seed) < 0 and phi_j(seed) < 0
    """
    geom = estimate_pair_patch_geometry(body_i, body_j, cfg)
    if geom is None:
        return None
    center0, n0, t1, t2, radius, ray_span = geom

    xi, wi = np.polynomial.legendre.leggauss(cfg.quad_order)
    du = 2.0 * radius / cfg.Nuv
    dv = du

    samples: list[PairPatchSample] = []

    for iu in range(cfg.Nuv):
        ul = -radius + iu * du
        ur = ul + du
        for iv in range(cfg.Nuv):
            vl = -radius + iv * dv
            vr = vl + dv
            for a, wa in zip(xi, wi):
                u = 0.5 * (ur - ul) * a + 0.5 * (ur + ul)
                for b, wb in zip(xi, wi):
                    v = 0.5 * (vr - vl) * b + 0.5 * (vr + vl)
                    seed = center0 + u * t1 + v * t2
                    w_proj = 0.25 * (ur - ul) * (vr - vl) * wa * wb
                    phi_i = body_i.geometry.phi_world(seed[0], seed[1], seed[2], body_i.state.pose)
                    phi_j = body_j.geometry.phi_world(seed[0], seed[1], seed[2], body_j.state.pose)
                    if phi_i < 0.0 and phi_j < 0.0:
                        samples.append(PairPatchSample(np.array([u, v], dtype=float), seed.astype(float), float(w_proj)))

    if len(samples) == 0:
        return None

    return PairContactPatches(
        samples=samples,
        center=center0.copy(),
        normal0=n0.copy(),
        tangent1=t1.copy(),
        tangent2=t2.copy(),
        radius=float(radius),
        metadata={"num_pair_patch_points": len(samples), "ray_span": ray_span},
    )


def _ray_root_positive(body: RigidBody6D, seed: np.ndarray, n0: np.ndarray, smax: float, bisection_steps: int) -> np.ndarray | None:
    f0 = body.geometry.phi_world(seed[0], seed[1], seed[2], body.state.pose)
    f1p = body.geometry.phi_world(*(seed + smax * n0), body.state.pose)
    if not (f0 <= 0.0 and f1p >= 0.0):
        return None

    sl, sr = 0.0, smax
    for _ in range(bisection_steps):
        sm = 0.5 * (sl + sr)
        pm = seed + sm * n0
        fm = body.geometry.phi_world(pm[0], pm[1], pm[2], body.state.pose)
        if fm > 0.0:
            sr = sm
        else:
            sl = sm
    return seed + 0.5 * (sl + sr) * n0


def _ray_root_negative(body: RigidBody6D, seed: np.ndarray, n0: np.ndarray, smax: float, bisection_steps: int) -> np.ndarray | None:
    f0 = body.geometry.phi_world(seed[0], seed[1], seed[2], body.state.pose)
    f1n = body.geometry.phi_world(*(seed - smax * n0), body.state.pose)
    if not (f0 <= 0.0 and f1n >= 0.0):
        return None

    sl, sr = 0.0, smax
    for _ in range(bisection_steps):
        sm = 0.5 * (sl + sr)
        pm = seed - sm * n0
        fm = body.geometry.phi_world(pm[0], pm[1], pm[2], body.state.pose)
        if fm > 0.0:
            sr = sm
        else:
            sl = sm
    return seed - 0.5 * (sl + sr) * n0


def extract_pair_sheet(
    body_i: RigidBody6D,
    body_j: RigidBody6D,
    patches: PairContactPatches,
    cfg_patch: BodyBodyPatchBuildConfig,
    cfg_sheet: SheetExtractConfig,
) -> PairSheet:
    """
    Extract opposing surface points on the two bodies by root solving along the pair normal.

    Reference orientation:
        n0 points from body_i toward body_j

    We therefore solve:
        body_i root on +n0 side
        body_j root on -n0 side
    """
    samples: list[PairSheetPoint] = []
    n0 = patches.normal0
    smax = float(patches.metadata["ray_span"])

    for idx, pp in enumerate(patches.samples):
        xi = _ray_root_positive(body_i, pp.world_seed, n0, smax, cfg_sheet.bisection_steps)
        xj = _ray_root_negative(body_j, pp.world_seed, n0, smax, cfg_sheet.bisection_steps)
        if xi is None or xj is None:
            continue

        ni = get_geometry_normal(body_i, xi[0], xi[1], xi[2], cfg_sheet.normal_step)
        nj = get_geometry_normal(body_j, xj[0], xj[1], xj[2], cfg_sheet.normal_step)

        ns = ni - nj
        nsn = np.linalg.norm(ns)
        if nsn <= 1.0e-12:
            ns = n0.copy()
        else:
            ns = ns / nsn

        cos_i = abs(float(np.dot(ni, n0)))
        cos_j = abs(float(np.dot(nj, n0)))
        if cos_i < 1.0e-12 or cos_j < 1.0e-12:
            continue

        dA = 0.5 * (pp.projected_weight / cos_i + pp.projected_weight / cos_j)
        mid = 0.5 * (xi + xj)

        samples.append(
            PairSheetPoint(
                midpoint=mid.astype(float),
                point_i=xi.astype(float),
                point_j=xj.astype(float),
                normal_i=ni.astype(float),
                normal_j=nj.astype(float),
                sheet_normal=ns.astype(float),
                projected_weight=pp.projected_weight,
                surface_weight=float(dA),
                source_index=idx,
            )
        )

    return PairSheet(samples=samples, metadata={"num_pair_sheet_points": len(samples)})


def integrate_pair_tractions(
    body_i: RigidBody6D,
    body_j: RigidBody6D,
    pair_sheet: PairSheet,
    cfg_contact: ContactModelConfig,
) -> PairTractionField:
    """
    Integrate symmetric pair tractions on the extracted pair sheet.

    Pressure law:
        q = k_pair * 0.5 * (d_i + d_j) + c_pair * max(0, -v_rel_n)

    with
        d_i = -phi_i(midpoint), d_j = -phi_j(midpoint)
        n = sheet_normal pointing roughly from i to j
    """
    samples: list[PairTractionSample] = []

    for sp in pair_sheet.samples:
        m = sp.midpoint
        n = sp.sheet_normal

        di = max(0.0, -float(body_i.geometry.phi_world(m[0], m[1], m[2], body_i.state.pose)))
        dj = max(0.0, -float(body_j.geometry.phi_world(m[0], m[1], m[2], body_j.state.pose)))
        depth = 0.5 * (di + dj)

        ri = m - body_i.state.pose.position
        rj = m - body_j.state.pose.position
        vi = body_i.state.linear_velocity + np.cross(body_i.state.angular_velocity, ri)
        vj = body_j.state.linear_velocity + np.cross(body_j.state.angular_velocity, rj)
        vrel_n = float(np.dot(vj - vi, n))

        q = cfg_contact.pair_stiffness_k * depth + cfg_contact.pair_damping_c * max(0.0, -vrel_n)
        fi = -q * n * sp.surface_weight
        fj = +q * n * sp.surface_weight
        mi = np.cross(ri, fi)
        mj = np.cross(rj, fj)

        samples.append(
            PairTractionSample(
                midpoint=m.copy(),
                sheet_normal=n.copy(),
                pressure=float(q),
                area_weight=float(sp.surface_weight),
                force_on_i=fi.astype(float),
                force_on_j=fj.astype(float),
                moment_i=mi.astype(float),
                moment_j=mj.astype(float),
            )
        )

    return PairTractionField(samples=samples, metadata={"num_pair_tractions": len(samples)})


def assemble_pair_wrench(pair_tr: PairTractionField) -> tuple[Wrench, Wrench]:
    Fi = np.zeros(3, dtype=float)
    Fj = np.zeros(3, dtype=float)
    Mi = np.zeros(3, dtype=float)
    Mj = np.zeros(3, dtype=float)

    for ts in pair_tr.samples:
        Fi += ts.force_on_i
        Fj += ts.force_on_j
        Mi += ts.moment_i
        Mj += ts.moment_j

    return Wrench(Fi, Mi), Wrench(Fj, Mj)


# ============================================================
# Multi-body contact manager
# ============================================================

class ContactManager:
    """
    Multi-body contact manager with:
        - body-domain pipeline
        - body-body local patch/sheet pipeline
    """
    def __init__(
        self,
        patch_cfg: PatchBuildConfig,
        pair_patch_cfg: BodyBodyPatchBuildConfig,
        sheet_cfg: SheetExtractConfig,
        contact_cfg: ContactModelConfig,
    ):
        self.patch_cfg = patch_cfg
        self.pair_patch_cfg = pair_patch_cfg
        self.sheet_cfg = sheet_cfg
        self.contact_cfg = contact_cfg

    def _compute_body_domain_pair(self, body_i: int, body: RigidBody6D, domain: DomainSpec) -> PairRecord | None:
        patches = build_contact_patches(body, domain, self.patch_cfg)
        sheet = extract_sheet(body, domain, patches, self.sheet_cfg)
        tr = integrate_tractions(body, sheet, self.contact_cfg)
        wrench = assemble_wrench(body, tr)
        if np.linalg.norm(wrench.force) <= 1.0e-14 and len(sheet.samples) == 0:
            return None

        return PairRecord(
            pair_kind="body_domain",
            body_i=body_i,
            body_j=None,
            contribution_i=PairWrenchContribution(
                "body_domain",
                wrench.force.copy(),
                wrench.moment.copy(),
                {
                    "num_projected_points": patches.metadata["num_projected_points"],
                    "num_sheet_points": sheet.metadata["num_sheet_points"],
                    "num_tractions": tr.metadata["num_traction_samples"],
                },
            ),
            contribution_j=None,
            meta={},
        )

    def _compute_body_body_pair(self, i: int, bi: RigidBody6D, j: int, bj: RigidBody6D) -> PairRecord | None:
        pair_patches = build_pair_contact_patches(bi, bj, self.pair_patch_cfg)
        if pair_patches is None:
            return None

        pair_sheet = extract_pair_sheet(bi, bj, pair_patches, self.pair_patch_cfg, self.sheet_cfg)
        if len(pair_sheet.samples) == 0:
            return None

        pair_tr = integrate_pair_tractions(bi, bj, pair_sheet, self.contact_cfg)
        if len(pair_tr.samples) == 0:
            return None

        wi, wj = assemble_pair_wrench(pair_tr)

        return PairRecord(
            pair_kind="body_body_local_patch",
            body_i=i,
            body_j=j,
            contribution_i=PairWrenchContribution(
                "body_body_local_patch",
                wi.force.copy(),
                wi.moment.copy(),
                {
                    "num_pair_patch_points": pair_patches.metadata["num_pair_patch_points"],
                    "num_pair_sheet_points": pair_sheet.metadata["num_pair_sheet_points"],
                    "num_pair_tractions": pair_tr.metadata["num_pair_tractions"],
                },
            ),
            contribution_j=PairWrenchContribution(
                "body_body_local_patch",
                wj.force.copy(),
                wj.moment.copy(),
                {
                    "num_pair_patch_points": pair_patches.metadata["num_pair_patch_points"],
                    "num_pair_sheet_points": pair_sheet.metadata["num_pair_sheet_points"],
                    "num_pair_tractions": pair_tr.metadata["num_pair_tractions"],
                },
            ),
            meta={"patch_radius": pair_patches.radius},
        )

    def compute_all_contacts(self, world: World) -> list[AggregatedBodyContact]:
        out = [
            AggregatedBodyContact(
                total_force=np.zeros(3, dtype=float),
                total_moment=np.zeros(3, dtype=float),
                pair_records=[],
            )
            for _ in world.bodies
        ]

        # body-domain pairs
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

        # body-body local patch pairs
        n = len(world.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                if world.bodies[i].is_static and world.bodies[j].is_static:
                    continue
                rec = self._compute_body_body_pair(i, world.bodies[i], j, world.bodies[j])
                if rec is None:
                    continue

                out[i].total_force += rec.contribution_i.force
                out[i].total_moment += rec.contribution_i.moment
                out[i].pair_records.append(rec)
                out[i].num_pair_patch_points += rec.contribution_i.meta["num_pair_patch_points"]
                out[i].num_pair_sheet_points += rec.contribution_i.meta["num_pair_sheet_points"]
                out[i].num_pair_tractions += rec.contribution_i.meta["num_pair_tractions"]

                out[j].total_force += rec.contribution_j.force
                out[j].total_moment += rec.contribution_j.moment
                out[j].pair_records.append(rec)
                out[j].num_pair_patch_points += rec.contribution_j.meta["num_pair_patch_points"]
                out[j].num_pair_sheet_points += rec.contribution_j.meta["num_pair_sheet_points"]
                out[j].num_pair_tractions += rec.contribution_j.meta["num_pair_tractions"]

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
                    "num_pair_patch_points": c.num_pair_patch_points,
                    "num_pair_sheet_points": c.num_pair_sheet_points,
                    "num_pair_tractions": c.num_pair_tractions,
                    "num_pairs": len(c.pair_records),
                },
            )
        )
    return out


# ============================================================
# Global implicit Newton solver
# ============================================================

class GlobalImplicitSystemSolver6D:
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
            new_state = BodyState6D(Pose6D(x.copy(), q.copy()), v.copy(), w.copy())
            new_bodies.append(body.clone_with_state(new_state))

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

        # commit global state
        final_world = self._make_trial_world(world, dyn, U)
        for i in range(len(world.bodies)):
            world.bodies[i] = final_world.bodies[i]

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
                "num_pair_patch_points": f.contact_meta["num_pair_patch_points"],
                "num_pair_sheet_points": f.contact_meta["num_pair_sheet_points"],
                "num_pair_tractions": f.contact_meta["num_pair_tractions"],
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
                    num_pair_patch_points=int(info["num_pair_patch_points"]),
                    num_pair_sheet_points=int(info["num_pair_sheet_points"]),
                    num_pair_tractions=int(info["num_pair_tractions"]),
                    marker_positions={k: v.copy() for k, v in body.world_markers().items()},
                )
            )
        self.time += dt

    def run(self, total_time: float) -> list[SimulationLogEntry]:
        nsteps = int(round(total_time / self.solver.cfg.dt))
        for _ in range(nsteps):
            self.step()
        return self.log
