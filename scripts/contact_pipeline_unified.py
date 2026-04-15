#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified reference implementation for 3D contact based on the pipeline

    build_contact_patches()
    -> extract_sheet()
    -> integrate_tractions()
    -> assemble_wrench()

The current reference builder is:
    top-slice x-z quadrature + y-direction root solve + local-normal traction accumulation

This is the generic, reusable algorithmic skeleton distilled from the previous validation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Protocol, Sequence
import math
import numpy as np


def delta_cosine(s: np.ndarray, eta: float) -> np.ndarray:
    a = np.abs(s)
    out = np.zeros_like(s, dtype=float)
    mask = a <= eta
    out[mask] = 0.5 / eta * (1.0 + np.cos(np.pi * s[mask] / eta))
    return out


# ============================================================
# Data containers
# ============================================================

@dataclass
class DomainSpec:
    """Computational domain for the compliant cube."""
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
class PatchBuildConfig:
    """
    Configuration for build_contact_patches().

    patch_builder:
        - "top_slice_quadrature": evaluate the top-slice footprint using x-z Gauss quadrature
    """
    Nxz: int = 48
    quad_order: int = 3
    bbox_padding_cells: int = 1
    patch_builder: str = "top_slice_quadrature"


@dataclass
class SheetExtractConfig:
    """Configuration for extract_sheet()."""
    bisection_steps: int = 30
    normal_step: float = 1.0e-6
    top_y: float = 0.0


@dataclass
class HydrostaticPressureModel:
    """
    Simple reference pressure law used throughout the validation campaign.

    p(y) = k * max(0, top_y - y)
    """
    k: float
    top_y: float = 0.0

    def pressure(self, y: float) -> float:
        return self.k * max(0.0, self.top_y - y)


@dataclass
class ProjectedPatchPoint:
    """
    A projected sample in the x-z footprint.

    projected_weight:
        projected area weight in x-z, i.e. dA_proj
    """
    x: float
    z: float
    projected_weight: float
    cell_i: int = -1
    cell_j: int = -1


@dataclass
class ContactPatches:
    """Output of build_contact_patches()."""
    samples: list[ProjectedPatchPoint]
    metadata: dict = field(default_factory=dict)


@dataclass
class SheetPoint:
    """
    A point on the extracted contact sheet.

    projected_weight:
        projected area weight dA_proj in x-z

    surface_weight:
        actual sheet area weight dA = dA_proj / |n_y|
    """
    position: np.ndarray
    normal: np.ndarray
    projected_weight: float
    surface_weight: float
    source_index: int = -1


@dataclass
class Sheet:
    """Output of extract_sheet()."""
    samples: list[SheetPoint]
    metadata: dict = field(default_factory=dict)


@dataclass
class TractionSample:
    """A local traction contribution on the contact sheet."""
    position: np.ndarray
    normal: np.ndarray
    pressure: float
    traction: np.ndarray
    area_weight: float
    force: np.ndarray


@dataclass
class TractionField:
    """Output of integrate_tractions()."""
    samples: list[TractionSample]
    metadata: dict = field(default_factory=dict)


@dataclass
class Wrench:
    """Final assembled wrench."""
    force: np.ndarray
    moment: np.ndarray


# ============================================================
# Shape protocol
# ============================================================

class ContactShape(Protocol):
    """
    Required geometry interface.

    The shape is assumed rigid. The large cube is the compliant body.
    """
    def phi(self, x, y, z, delta: float):
        ...

    def footprint_bbox(self, delta: float) -> tuple[float, float, float, float]:
        ...

    # optional:
    # def normal(self, x: float, y: float, z: float, delta: float) -> np.ndarray: ...


# ============================================================
# Helper functions
# ============================================================

def numerical_normal(shape: ContactShape, x: float, y: float, z: float, delta: float, h: float) -> np.ndarray:
    """Central-difference fallback for shape normals."""
    px = (shape.phi(x + h, y, z, delta) - shape.phi(x - h, y, z, delta)) / (2.0 * h)
    py = (shape.phi(x, y + h, z, delta) - shape.phi(x, y - h, z, delta)) / (2.0 * h)
    pz = (shape.phi(x, y, z + h, delta) - shape.phi(x, y, z - h, delta)) / (2.0 * h)
    g = np.array([px, py, pz], dtype=float)
    ng = np.linalg.norm(g)
    if ng <= 1.0e-15:
        return np.array([0.0, -1.0, 0.0], dtype=float)
    return g / ng


def exact_force_from_submerged_volume(volume: float, k: float) -> float:
    return k * volume


def segment_area_circle(radius: float, depth: float) -> float:
    if depth <= 0.0:
        return 0.0
    if depth >= 2.0 * radius:
        return math.pi * radius * radius
    return radius * radius * math.acos((radius - depth) / radius) - (radius - depth) * math.sqrt(max(0.0, 2.0 * radius * depth - depth * depth))


# ============================================================
# Core unified interface
# ============================================================

def build_contact_patches(
    shape: ContactShape,
    delta: float,
    domain: DomainSpec,
    cfg: PatchBuildConfig,
) -> ContactPatches:
    """
    Build projected contact patches in x-z.

    Current reference implementation:
        top-slice quadrature on x-z cells
        accept a quadrature point if phi(x, top_y, z) <= 0

    This function deliberately returns *projected* samples only.
    It does not yet know where the true sheet point lies in y.
    """
    if cfg.patch_builder != "top_slice_quadrature":
        raise ValueError(f"Unsupported patch_builder: {cfg.patch_builder}")

    x_min, x_max = domain.x_bounds
    z_min, z_max = domain.z_bounds
    dx = (x_max - x_min) / cfg.Nxz
    dz = (z_max - z_min) / cfg.Nxz

    bx0, bx1, bz0, bz1 = shape.footprint_bbox(delta)
    i_min = max(0, int(math.floor((bx0 - x_min) / dx)) - cfg.bbox_padding_cells)
    i_max = min(cfg.Nxz - 1, int(math.floor((bx1 - x_min) / dx)) + cfg.bbox_padding_cells)
    j_min = max(0, int(math.floor((bz0 - z_min) / dz)) - cfg.bbox_padding_cells)
    j_max = min(cfg.Nxz - 1, int(math.floor((bz1 - z_min) / dz)) + cfg.bbox_padding_cells)

    xi, wi = np.polynomial.legendre.leggauss(cfg.quad_order)

    samples: list[ProjectedPatchPoint] = []
    top_y = domain.top_y

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

                    if shape.phi(xq, top_y, zq, delta) <= 0.0:
                        samples.append(
                            ProjectedPatchPoint(
                                x=float(xq),
                                z=float(zq),
                                projected_weight=float(w_proj),
                                cell_i=i,
                                cell_j=j,
                            )
                        )

    return ContactPatches(
        samples=samples,
        metadata={
            "patch_builder": cfg.patch_builder,
            "Nxz": cfg.Nxz,
            "quad_order": cfg.quad_order,
            "num_projected_points": len(samples),
            "x_bounds": domain.x_bounds,
            "z_bounds": domain.z_bounds,
        },
    )


def extract_sheet(
    shape: ContactShape,
    delta: float,
    domain: DomainSpec,
    patches: ContactPatches,
    cfg: SheetExtractConfig,
) -> Sheet:
    """
    Extract the actual sheet points from projected patch samples.

    For each projected x-z point:
        solve phi(x, y, z) = 0 by bisection on y in [top_y - cube_height, top_y].

    Then compute:
        normal
        surface area weight dA = dA_proj / |n_y|
    """
    y_min, y_max = domain.y_bounds
    top_y = cfg.top_y

    samples: list[SheetPoint] = []
    ny_too_small = 0
    rejected_no_bracket = 0

    for idx, pp in enumerate(patches.samples):
        phi_top = shape.phi(pp.x, top_y, pp.z, delta)
        phi_bot = shape.phi(pp.x, y_min, pp.z, delta)

        if not (phi_top <= 0.0 and phi_bot >= 0.0):
            rejected_no_bracket += 1
            continue

        yl = y_min
        yr = top_y
        for _ in range(cfg.bisection_steps):
            ym = 0.5 * (yl + yr)
            fm = shape.phi(pp.x, ym, pp.z, delta)
            if fm > 0.0:
                yl = ym
            else:
                yr = ym

        y_sigma = 0.5 * (yl + yr)

        if hasattr(shape, "normal"):
            n = np.asarray(shape.normal(pp.x, y_sigma, pp.z, delta), dtype=float)
            nn = np.linalg.norm(n)
            if nn <= 1.0e-15:
                n = numerical_normal(shape, pp.x, y_sigma, pp.z, delta, cfg.normal_step)
            else:
                n = n / nn
        else:
            n = numerical_normal(shape, pp.x, y_sigma, pp.z, delta, cfg.normal_step)

        if abs(n[1]) < 1.0e-12:
            ny_too_small += 1
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
            "rejected_small_ny": ny_too_small,
            "bisection_steps": cfg.bisection_steps,
        },
    )


def integrate_tractions(
    sheet: Sheet,
    pressure_model: HydrostaticPressureModel,
) -> TractionField:
    """
    Integrate local tractions on the extracted sheet.

    Reference traction law:
        t = -p(y) n
    """
    samples: list[TractionSample] = []

    for sp in sheet.samples:
        y = float(sp.position[1])
        p = pressure_model.pressure(y)
        traction = -p * sp.normal
        force = traction * sp.surface_weight

        samples.append(
            TractionSample(
                position=sp.position.copy(),
                normal=sp.normal.copy(),
                pressure=float(p),
                traction=np.asarray(traction, dtype=float),
                area_weight=float(sp.surface_weight),
                force=np.asarray(force, dtype=float),
            )
        )

    return TractionField(
        samples=samples,
        metadata={
            "num_traction_samples": len(samples),
            "pressure_model": "hydrostatic_linear",
            "k": pressure_model.k,
        },
    )


def assemble_wrench(
    tractions: TractionField,
    ref_point: Sequence[float] = (0.0, 0.0, 0.0),
) -> Wrench:
    """
    Assemble the final force and moment.
    """
    ref = np.asarray(ref_point, dtype=float)
    F = np.zeros(3, dtype=float)
    M = np.zeros(3, dtype=float)

    for ts in tractions.samples:
        F += ts.force
        arm = ts.position - ref
        M += np.cross(arm, ts.force)

    return Wrench(force=F, moment=M)


def run_contact_pipeline(
    shape: ContactShape,
    delta: float,
    domain: DomainSpec,
    pressure_model: HydrostaticPressureModel,
    patch_cfg: PatchBuildConfig | None = None,
    sheet_cfg: SheetExtractConfig | None = None,
    ref_point: Sequence[float] = (0.0, 0.0, 0.0),
) -> dict:
    """
    Convenience wrapper for the whole pipeline.

    Returns:
        {
            "patches": ContactPatches,
            "sheet": Sheet,
            "tractions": TractionField,
            "wrench": Wrench,
        }
    """
    patch_cfg = patch_cfg or PatchBuildConfig()
    sheet_cfg = sheet_cfg or SheetExtractConfig(top_y=domain.top_y)

    patches = build_contact_patches(shape, delta, domain, patch_cfg)
    sheet = extract_sheet(shape, delta, domain, patches, sheet_cfg)
    tractions = integrate_tractions(sheet, pressure_model)
    wrench = assemble_wrench(tractions, ref_point=ref_point)

    return {
        "patches": patches,
        "sheet": sheet,
        "tractions": tractions,
        "wrench": wrench,
    }


# ============================================================
# Optional baseline for comparisons
# ============================================================

def direct_band_baseline(
    shape: ContactShape,
    delta: float,
    domain: DomainSpec,
    pressure_model: HydrostaticPressureModel,
    N: int = 64,
    eta_factor: float = 1.5,
    ref_point: Sequence[float] = (0.0, 0.0, 0.0),
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
    phi = shape.phi(X, Y, Z, delta)

    h = max(dx, dy, dz)
    eta = eta_factor * h
    band = delta_cosine(phi, eta)

    px = (shape.phi(X + h, Y, Z, delta) - shape.phi(X - h, Y, Z, delta)) / (2.0 * h)
    py = (shape.phi(X, Y + h, Z, delta) - shape.phi(X, Y - h, Z, delta)) / (2.0 * h)
    pz = (shape.phi(X, Y, Z + h, delta) - shape.phi(X, Y, Z - h, delta)) / (2.0 * h)

    gnorm = np.sqrt(px * px + py * py + pz * pz) + 1.0e-15
    nx = px / gnorm
    ny = py / gnorm
    nz = pz / gnorm

    pressure = np.vectorize(pressure_model.pressure)(Y)
    tx = -pressure * nx
    ty = -pressure * ny
    tz = -pressure * nz

    Fx = float(np.sum(tx * band) * dV)
    Fy = float(np.sum(ty * band) * dV)
    Fz = float(np.sum(tz * band) * dV)

    ref = np.asarray(ref_point, dtype=float)
    Mx = float(np.sum(((Y - ref[1]) * tz - (Z - ref[2]) * ty) * band) * dV)
    My = float(np.sum(((Z - ref[2]) * tx - (X - ref[0]) * tz) * band) * dV)
    Mz = float(np.sum(((X - ref[0]) * ty - (Y - ref[1]) * tx) * band) * dV)

    return Wrench(force=np.array([Fx, Fy, Fz], dtype=float), moment=np.array([Mx, My, Mz], dtype=float))


# ============================================================
# Example shapes
# ============================================================

@dataclass
class SmallBoxPunch:
    lx: float
    lz: float
    rigid_height: float

    def __post_init__(self):
        self.hx = self.lx / 2.0
        self.hz = self.lz / 2.0
        self.hy = self.rigid_height / 2.0

    def footprint_bbox(self, delta: float) -> tuple[float, float, float, float]:
        return (-self.hx, self.hx, -self.hz, self.hz)

    def phi(self, x, y, z, delta: float):
        cx, cy, cz = 0.0, self.hy - delta, 0.0
        qx = np.abs(x - cx) - self.hx
        qy = np.abs(y - cy) - self.hy
        qz = np.abs(z - cz) - self.hz
        ox = np.maximum(qx, 0.0)
        oy = np.maximum(qy, 0.0)
        oz = np.maximum(qz, 0.0)
        outside = np.sqrt(ox * ox + oy * oy + oz * oz)
        inside = np.minimum(np.maximum.reduce([qx, qy, qz]), 0.0)
        return outside + inside

    def normal(self, x: float, y: float, z: float, delta: float) -> np.ndarray:
        # In the shallow contact region, the active sheet is the bottom face.
        return np.array([0.0, -1.0, 0.0], dtype=float)

    def exact_force(self, k: float, delta: float) -> float:
        return k * self.lx * self.lz * delta


@dataclass
class AnnularFlatPunch:
    ri: float
    ro: float
    rigid_height: float

    def __post_init__(self):
        self.hy = self.rigid_height / 2.0

    def footprint_bbox(self, delta: float) -> tuple[float, float, float, float]:
        return (-self.ro, self.ro, -self.ro, self.ro)

    def _sd_annulus_2d(self, x, z):
        r = np.sqrt(np.asarray(x) * np.asarray(x) + np.asarray(z) * np.asarray(z))
        out = np.empty_like(r, dtype=float)
        m1 = r < self.ri
        m2 = (r >= self.ri) & (r <= self.ro)
        m3 = r > self.ro
        out[m1] = self.ri - r[m1]
        out[m2] = -np.minimum(r[m2] - self.ri, self.ro - r[m2])
        out[m3] = r[m3] - self.ro
        return out

    def phi(self, x, y, z, delta: float):
        sd2 = self._sd_annulus_2d(x, z)
        yc = self.hy - delta
        qy = np.abs(np.asarray(y) - yc) - self.hy
        ox = np.maximum(sd2, 0.0)
        oy = np.maximum(qy, 0.0)
        outside = np.sqrt(ox * ox + oy * oy)
        inside = np.minimum(np.maximum(sd2, qy), 0.0)
        return outside + inside

    def normal(self, x: float, y: float, z: float, delta: float) -> np.ndarray:
        # In the shallow contact region, the active sheet is the bottom annular face.
        return np.array([0.0, -1.0, 0.0], dtype=float)

    def exact_force(self, k: float, delta: float) -> float:
        return k * math.pi * (self.ro * self.ro - self.ri * self.ri) * delta


@dataclass
class HorizontalCylinder:
    radius: float
    length: float

    def footprint_bbox(self, delta: float) -> tuple[float, float, float, float]:
        a = math.sqrt(max(0.0, 2.0 * self.radius * delta - delta * delta))
        return (-a, a, -self.length / 2.0, self.length / 2.0)

    def phi(self, x, y, z, delta: float):
        cx, cy, cz = 0.0, self.radius - delta, 0.0
        d_rad = np.sqrt((np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2) - self.radius
        d_ax = np.abs(np.asarray(z) - cz) - self.length / 2.0
        ox = np.maximum(d_rad, 0.0)
        oy = np.maximum(d_ax, 0.0)
        outside = np.sqrt(ox * ox + oy * oy)
        inside = np.minimum(np.maximum(d_rad, d_ax), 0.0)
        return outside + inside

    def normal(self, x: float, y: float, z: float, delta: float) -> np.ndarray:
        cx, cy = 0.0, self.radius - delta
        dx = x - cx
        dy = y - cy
        rr = math.sqrt(dx * dx + dy * dy)
        if rr <= 1.0e-15:
            return np.array([0.0, -1.0, 0.0], dtype=float)
        return np.array([dx / rr, dy / rr, 0.0], dtype=float)

    def exact_force(self, k: float, delta: float) -> float:
        return k * self.length * segment_area_circle(self.radius, delta)
