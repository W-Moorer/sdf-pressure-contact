from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LinearPressureLaw:
    stiffness: float

    def __post_init__(self) -> None:
        if self.stiffness <= 0.0:
            raise ValueError("stiffness must be positive.")

    def pressure(self, depth: float | np.ndarray) -> float | np.ndarray:
        d = np.asarray(depth, dtype=float)
        return self.stiffness * d
