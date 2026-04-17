from __future__ import annotations

from dataclasses import dataclass

import numpy as np



def relative_error(computed: float, reference: float, *, atol: float = 1e-14) -> float:
    denom = max(abs(reference), atol)
    return float(abs(computed - reference) / denom)



def timing_error(computed: float, reference: float) -> float:
    return float(abs(computed - reference))



def integrate_impulse(times: np.ndarray, forces: np.ndarray) -> float:
    times = np.asarray(times, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if times.ndim != 1 or forces.ndim != 1 or times.shape != forces.shape:
        raise ValueError("times and forces must be matching 1D arrays")
    if len(times) < 2:
        return 0.0
    return float(np.trapezoid(forces, times))



def kinetic_energy(mass: float, velocity: float) -> float:
    return 0.5 * mass * velocity * velocity



def linear_contact_potential(stiffness: float, gap: float) -> float:
    penetration = max(-gap, 0.0)
    return 0.5 * stiffness * penetration * penetration



def energy_drift(total_energy: np.ndarray) -> float:
    total_energy = np.asarray(total_energy, dtype=float)
    if total_energy.ndim != 1 or total_energy.size == 0:
        raise ValueError("total_energy must be a non-empty 1D array")
    return float(total_energy[-1] - total_energy[0])


@dataclass(frozen=True)
class DynamicErrorSummary:
    state_error: float
    force_error: float
    impulse_error: float
    onset_timing_error: float
    release_timing_error: float = float("nan")
    peak_force_error: float = float("nan")
    max_penetration_error: float = float("nan")
    rebound_velocity_error: float = float("nan")
