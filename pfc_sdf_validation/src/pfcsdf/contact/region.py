from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfcsdf.contact.patch import PlanarPatch
from pfcsdf.geometry.polygon import ConvexPolygon2D, HalfSpace2D, clip_convex_polygon_with_halfspace

ArrayLike = np.ndarray


@dataclass(frozen=True)
class AffineOverlapField2D:
    """Affine overlap model g(x,y) = offset + gradient·[x,y]."""

    offset: float
    gradient: ArrayLike

    def __post_init__(self) -> None:
        gradient = np.asarray(self.gradient, dtype=float)
        if gradient.shape != (2,):
            raise ValueError("gradient must have shape (2,)")
        object.__setattr__(self, "offset", float(self.offset))
        object.__setattr__(self, "gradient", gradient)

    def evaluate(self, local_xy: ArrayLike) -> float:
        local_xy = np.asarray(local_xy, dtype=float)
        return float(self.offset + np.dot(self.gradient, local_xy))

    def halfspace(self) -> HalfSpace2D | None:
        grad_norm = float(np.linalg.norm(self.gradient))
        if grad_norm <= 1e-14:
            return None
        return HalfSpace2D(normal=self.gradient, offset=self.offset)


@dataclass(frozen=True)
class SupportRegionDetectionResult:
    carrier_patch: PlanarPatch
    support_patch: PlanarPatch | None
    field: AffineOverlapField2D



def detect_support_polygon_affine(
    carrier_polygon: ConvexPolygon2D,
    field: AffineOverlapField2D,
    *,
    tol: float = 1e-12,
) -> ConvexPolygon2D | None:
    """Detect the positive-overlap support region for an affine overlap field."""
    halfspace = field.halfspace()
    if halfspace is None:
        if field.offset >= -tol:
            return carrier_polygon
        return None
    return clip_convex_polygon_with_halfspace(carrier_polygon, halfspace, tol=tol)



def detect_support_patch_affine(
    carrier_patch: PlanarPatch,
    field: AffineOverlapField2D,
    *,
    tol: float = 1e-12,
) -> SupportRegionDetectionResult:
    support_polygon = detect_support_polygon_affine(carrier_patch.polygon, field, tol=tol)
    support_patch = None
    if support_polygon is not None:
        support_patch = PlanarPatch(
            polygon=support_polygon,
            center=carrier_patch.center,
            normal=carrier_patch.normal,
            tangent_u=carrier_patch.tangent_u,
            tangent_v=carrier_patch.tangent_v,
        )
    return SupportRegionDetectionResult(carrier_patch=carrier_patch, support_patch=support_patch, field=field)
