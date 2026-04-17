from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class PairWrench:
    force: np.ndarray
    torque: np.ndarray


def accumulate_uniform_pressure_wrench(
    pressure: float,
    area: float,
    normal: np.ndarray,
    center: np.ndarray,
    reference: np.ndarray,
) -> PairWrench:
    normal = np.asarray(normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-14:
        raise ValueError("normal must be non-zero")
    unit_normal = normal / normal_norm
    center = np.asarray(center, dtype=float)
    reference = np.asarray(reference, dtype=float)
    force = pressure * area * unit_normal
    torque = np.cross(center - reference, force)
    return PairWrench(force=force, torque=torque)


def accumulate_axisymmetric_pressure_wrench(
    pressure_fn: Callable[[np.ndarray], np.ndarray],
    contact_radius: float,
    *,
    normal: np.ndarray,
    center: np.ndarray,
    reference: np.ndarray,
    n_rings: int = 2048,
) -> PairWrench:
    """Accumulate a normal wrench for an axisymmetric pressure field p(r).

    The pressure field is assumed to act on a circular support centered at ``center``
    with axis ``normal``. For an axisymmetric normal traction, the net torque about
    the center vanishes by symmetry and only the offset moment remains.
    """
    if contact_radius < 0.0:
        raise ValueError("contact_radius must be non-negative")
    if n_rings <= 0:
        raise ValueError("n_rings must be positive")

    normal = np.asarray(normal, dtype=float)
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-14:
        raise ValueError("normal must be non-zero")
    unit_normal = normal / normal_norm
    center = np.asarray(center, dtype=float)
    reference = np.asarray(reference, dtype=float)

    if contact_radius == 0.0:
        return PairWrench(force=np.zeros(3), torque=np.zeros(3))

    edges = np.linspace(0.0, contact_radius, n_rings + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1:] - edges[:-1]
    areas = 2.0 * np.pi * mids * dr
    pressures = np.asarray(pressure_fn(mids), dtype=float)
    total_force_mag = float(np.sum(pressures * areas))
    force = total_force_mag * unit_normal
    torque = np.cross(center - reference, force)
    return PairWrench(force=force, torque=torque)



def shift_wrench_reference(wrench: PairWrench, reference_from: np.ndarray, reference_to: np.ndarray) -> PairWrench:
    """Shift a wrench expressed about reference_from to reference_to."""
    force = np.asarray(wrench.force, dtype=float)
    torque = np.asarray(wrench.torque, dtype=float)
    reference_from = np.asarray(reference_from, dtype=float)
    reference_to = np.asarray(reference_to, dtype=float)
    shifted_torque = torque + np.cross(reference_from - reference_to, force)
    return PairWrench(force=force.copy(), torque=shifted_torque)


def make_point_force_wrench(force: np.ndarray, application_point: np.ndarray, about_point: np.ndarray) -> PairWrench:
    force = np.asarray(force, dtype=float).reshape(3)
    application_point = np.asarray(application_point, dtype=float).reshape(3)
    about_point = np.asarray(about_point, dtype=float).reshape(3)
    torque = np.cross(application_point - about_point, force)
    return PairWrench(force=force, torque=torque)
