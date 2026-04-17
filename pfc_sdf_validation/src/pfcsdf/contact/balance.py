from __future__ import annotations

import numpy as np


def balance_value(p_a: float, p_b: float) -> float:
    return float(p_a - p_b)


def balance_gradient(grad_p_a: np.ndarray, grad_p_b: np.ndarray) -> np.ndarray:
    grad_p_a = np.asarray(grad_p_a, dtype=float)
    grad_p_b = np.asarray(grad_p_b, dtype=float)
    grad_h = grad_p_a - grad_p_b
    norm = np.linalg.norm(grad_h)
    if norm <= 1e-14:
        raise ValueError("Degenerate equal-pressure gradient.")
    return grad_h / norm
