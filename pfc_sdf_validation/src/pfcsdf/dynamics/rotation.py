from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray


def skew(w: ArrayLike) -> ArrayLike:
    w = np.asarray(w, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -w[2], w[1]],
            [w[2], 0.0, -w[0]],
            [-w[1], w[0], 0.0],
        ],
        dtype=float,
    )


def exp_so3(phi: ArrayLike) -> ArrayLike:
    """Exponential map from so(3) to SO(3)."""
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = skew(phi)
    if theta < 1e-10:
        # Second-order accurate series for small angles.
        return np.eye(3) + K + 0.5 * (K @ K)
    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + A * K + B * (K @ K)


def project_to_so3(R: ArrayLike) -> ArrayLike:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0.0:
        U[:, -1] *= -1.0
        R_proj = U @ Vt
    return R_proj


def rotation_angle_error(R_a: ArrayLike, R_b: ArrayLike) -> float:
    R_a = np.asarray(R_a, dtype=float).reshape(3, 3)
    R_b = np.asarray(R_b, dtype=float).reshape(3, 3)
    R_rel = R_a.T @ R_b
    trace_value = float(np.trace(R_rel))
    cos_theta = np.clip(0.5 * (trace_value - 1.0), -1.0, 1.0)
    return float(np.arccos(cos_theta))
