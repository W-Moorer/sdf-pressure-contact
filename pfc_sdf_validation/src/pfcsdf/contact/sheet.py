from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfcsdf.geometry.primitives import BoxFootprint


@dataclass(frozen=True)
class FlatSheetPatch:
    """Minimal zero-thickness sheet representation for a flat contact patch.

    This object carries *geometric semantics* only: center, normal, tangential frame,
    and area. It does not define how forces are integrated; that is delegated to the
    band/coarea layer.
    """

    center: np.ndarray
    normal: np.ndarray
    tangent_u: np.ndarray
    tangent_v: np.ndarray
    area: float



def normalize(vector: np.ndarray) -> np.ndarray:
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= 0.0:
        raise ValueError("vector norm must be positive")
    return vec / norm



def orthonormal_frame(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normalize(normal)
    if abs(n[2]) < 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])
    u = helper - np.dot(helper, n) * n
    u = normalize(u)
    v = np.cross(n, u)
    return n, u, v



def build_flat_sheet_patch(
    footprint: BoxFootprint,
    *,
    center: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> FlatSheetPatch:
    if center is None:
        center = np.zeros(3)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    n, u, v = orthonormal_frame(normal)
    return FlatSheetPatch(
        center=np.asarray(center, dtype=float),
        normal=n,
        tangent_u=u,
        tangent_v=v,
        area=footprint.area,
    )



def uniform_pressure_center(patch: FlatSheetPatch) -> np.ndarray:
    return np.asarray(patch.center, dtype=float)
