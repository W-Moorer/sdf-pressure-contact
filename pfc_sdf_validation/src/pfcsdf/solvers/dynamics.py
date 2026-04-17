from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Step1D:
    z_next: float
    v_next: float
    a_mid: float


def midpoint_step_1d(
    z: float,
    v: float,
    dt: float,
    acceleration: Callable[[float, float], float],
) -> Step1D:
    """Minimal midpoint integrator used as the dynamics hook."""
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    a0 = acceleration(z, v)
    z_mid = z + 0.5 * dt * v
    v_mid = v + 0.5 * dt * a0
    a_mid = acceleration(z_mid, v_mid)
    z_next = z + dt * v_mid
    v_next = v + dt * a_mid
    return Step1D(z_next=z_next, v_next=v_next, a_mid=a_mid)
