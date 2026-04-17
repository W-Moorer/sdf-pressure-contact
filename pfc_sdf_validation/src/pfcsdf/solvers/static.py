from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from pfcsdf.contact.local_normal import ColumnEquilibrium, solve_column_equilibrium
from pfcsdf.contact.wrench import (
    PairWrench,
    accumulate_axisymmetric_pressure_wrench,
    accumulate_uniform_pressure_wrench,
)
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.physics.pressure import LinearPressureLaw


@dataclass(frozen=True)
class FlatContactResult:
    column: ColumnEquilibrium
    wrench: PairWrench
    area: float


@dataclass(frozen=True)
class SpherePlaneContactResult:
    overlap: float
    contact_radius: float
    peak_pressure: float
    wrench: PairWrench


def compute_uniform_flat_contact(
    overlap: float,
    footprint: BoxFootprint,
    law_a,
    law_b,
    *,
    normal: np.ndarray | None = None,
    center: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> FlatContactResult:
    """Minimal flat-punch validator: uniform overlap over a rectangular footprint."""
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    if center is None:
        center = np.zeros(3)
    if reference is None:
        reference = np.zeros(3)

    column = solve_column_equilibrium(overlap, law_a, law_b)
    wrench = accumulate_uniform_pressure_wrench(
        pressure=column.pressure,
        area=footprint.area,
        normal=normal,
        center=center,
        reference=reference,
    )
    return FlatContactResult(column=column, wrench=wrench, area=footprint.area)


def sphere_plane_contact_radius_exact(sphere_radius: float, overlap: float) -> float:
    if sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be positive")
    if overlap < 0.0:
        raise ValueError("overlap must be non-negative")
    if overlap == 0.0:
        return 0.0
    if overlap >= 2.0 * sphere_radius:
        raise ValueError("overlap must be smaller than the sphere diameter")
    return math.sqrt(max(0.0, 2.0 * sphere_radius * overlap - overlap * overlap))



def sphere_plane_local_overlap(r: float | np.ndarray, sphere_radius: float, overlap: float) -> float | np.ndarray:
    r = np.asarray(r, dtype=float)
    if sphere_radius <= 0.0:
        raise ValueError("sphere_radius must be positive")
    if overlap < 0.0:
        raise ValueError("overlap must be non-negative")
    if overlap == 0.0:
        return np.zeros_like(r)

    inside = np.maximum(0.0, sphere_radius * sphere_radius - r * r)
    sag = sphere_radius - np.sqrt(inside)
    local_overlap = overlap - sag
    local_overlap = np.where(r <= sphere_plane_contact_radius_exact(sphere_radius, overlap), local_overlap, 0.0)
    return np.maximum(local_overlap, 0.0)



def compute_sphere_plane_contact_linear_exact(
    overlap: float,
    sphere_radius: float,
    law_sphere: LinearPressureLaw,
    law_plane: LinearPressureLaw,
    *,
    normal: np.ndarray | None = None,
    center: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> SpherePlaneContactResult:
    """Closed-form force for linear PFC pressure law under exact sphere geometry.

    For local overlap g(r), the common pressure is p(r)=k_eq g(r). The total force is
    F = π k_eq overlap^2 (R - overlap/3).
    """
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    if center is None:
        center = np.zeros(3)
    if reference is None:
        reference = np.zeros(3)

    a = sphere_plane_contact_radius_exact(sphere_radius, overlap)
    k_eq = law_sphere.stiffness * law_plane.stiffness / (law_sphere.stiffness + law_plane.stiffness)
    peak_pressure = k_eq * overlap
    total_force = math.pi * k_eq * overlap * overlap * (sphere_radius - overlap / 3.0)
    wrench = accumulate_uniform_pressure_wrench(
        pressure=total_force,
        area=1.0,
        normal=normal,
        center=center,
        reference=reference,
    )
    return SpherePlaneContactResult(overlap=overlap, contact_radius=a, peak_pressure=peak_pressure, wrench=wrench)



def compute_sphere_plane_contact_linear_quadrature(
    overlap: float,
    sphere_radius: float,
    law_sphere: LinearPressureLaw,
    law_plane: LinearPressureLaw,
    *,
    normal: np.ndarray | None = None,
    center: np.ndarray | None = None,
    reference: np.ndarray | None = None,
    n_rings: int = 4096,
) -> SpherePlaneContactResult:
    """Numerical axisymmetric validator using one local-normal column solve per ring."""
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    if center is None:
        center = np.zeros(3)
    if reference is None:
        reference = np.zeros(3)

    a = sphere_plane_contact_radius_exact(sphere_radius, overlap)

    def pressure_profile(radii: np.ndarray) -> np.ndarray:
        local_overlaps = sphere_plane_local_overlap(radii, sphere_radius, overlap)
        pressures = np.zeros_like(local_overlaps)
        for idx, g in enumerate(local_overlaps):
            if g > 0.0:
                pressures[idx] = solve_column_equilibrium(float(g), law_sphere, law_plane).pressure
        return pressures

    peak_pressure = float(solve_column_equilibrium(overlap, law_sphere, law_plane).pressure) if overlap > 0 else 0.0
    wrench = accumulate_axisymmetric_pressure_wrench(
        pressure_profile,
        contact_radius=a,
        normal=normal,
        center=center,
        reference=reference,
        n_rings=n_rings,
    )
    return SpherePlaneContactResult(overlap=overlap, contact_radius=a, peak_pressure=peak_pressure, wrench=wrench)
