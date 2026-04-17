from __future__ import annotations

import numpy as np


def depth_from_phi(phi: float | np.ndarray, max_depth: float) -> float | np.ndarray:
    """PFC-SDF interior depth d = clamp(-phi, 0, H)."""
    if max_depth <= 0.0:
        raise ValueError("max_depth must be positive.")
    return np.clip(-np.asarray(phi), 0.0, max_depth)
