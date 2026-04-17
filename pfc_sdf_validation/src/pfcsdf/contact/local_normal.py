from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PressureLaw(Protocol):
    def pressure(self, depth: float | np.ndarray) -> float | np.ndarray:
        ...


@dataclass(frozen=True)
class ColumnEquilibrium:
    overlap: float
    depth_a: float
    depth_b: float
    pressure: float


def solve_column_equilibrium(
    overlap: float,
    law_a: PressureLaw,
    law_b: PressureLaw,
    *,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> ColumnEquilibrium:
    """Solve P_a(d_a) = P_b(d_b), with d_a + d_b = overlap for one local normal column."""
    if overlap < 0.0:
        raise ValueError("overlap must be non-negative.")
    if overlap == 0.0:
        return ColumnEquilibrium(overlap=0.0, depth_a=0.0, depth_b=0.0, pressure=0.0)

    lo, hi = 0.0, overlap
    for _ in range(max_iter):
        d_a = 0.5 * (lo + hi)
        d_b = overlap - d_a
        f = float(law_a.pressure(d_a) - law_b.pressure(d_b))
        if abs(f) < tol or (hi - lo) < tol:
            p = 0.5 * (float(law_a.pressure(d_a)) + float(law_b.pressure(d_b)))
            return ColumnEquilibrium(overlap=overlap, depth_a=d_a, depth_b=d_b, pressure=p)
        if f > 0.0:
            hi = d_a
        else:
            lo = d_a

    d_a = 0.5 * (lo + hi)
    d_b = overlap - d_a
    p = 0.5 * (float(law_a.pressure(d_a)) + float(law_b.pressure(d_b)))
    return ColumnEquilibrium(overlap=overlap, depth_a=d_a, depth_b=d_b, pressure=p)
