from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pfcsdf.contact.local_normal import ColumnEquilibrium, PressureLaw, solve_column_equilibrium
from pfcsdf.contact.sheet import orthonormal_frame
from pfcsdf.contact.wrench import PairWrench
from pfcsdf.geometry.polygon import (
    ConvexPolygon2D,
    HalfSpace2D,
    clip_convex_polygon_with_halfspace,
    triangle_area,
)
from pfcsdf.geometry.primitives import BoxFootprint

ArrayLike = np.ndarray


@dataclass(frozen=True)
class PlanarPatch:
    """Static patch semantics: support polygon plus world-space frame."""

    polygon: ConvexPolygon2D
    center: ArrayLike
    normal: ArrayLike
    tangent_u: ArrayLike
    tangent_v: ArrayLike

    @property
    def area(self) -> float:
        return self.polygon.area

    @property
    def local_centroid(self) -> ArrayLike:
        return self.polygon.centroid

    @property
    def world_centroid(self) -> ArrayLike:
        return self.local_to_world(self.local_centroid)

    def local_to_world(self, local_xy: ArrayLike) -> ArrayLike:
        local_xy = np.asarray(local_xy, dtype=float)
        return self.center + local_xy[..., 0, None] * self.tangent_u + local_xy[..., 1, None] * self.tangent_v


@dataclass(frozen=True)
class PolygonQuadraturePoint:
    local_xy: ArrayLike
    weight: float


@dataclass(frozen=True)
class PatchSample:
    local_xy: ArrayLike
    world_xyz: ArrayLike
    weight: float
    column: ColumnEquilibrium


@dataclass(frozen=True)
class StaticPatchResult:
    patch: PlanarPatch
    wrench: PairWrench
    samples: tuple[PatchSample, ...]

    @property
    def integrated_pressure(self) -> float:
        return float(sum(sample.column.pressure * sample.weight for sample in self.samples))

    @property
    def mean_pressure(self) -> float:
        if self.patch.area <= 0.0:
            return 0.0
        return self.integrated_pressure / self.patch.area


@dataclass(frozen=True)
class StaticPatchTask:
    patch: PlanarPatch
    overlap_fn: Callable[[ArrayLike, ArrayLike], float]


@dataclass(frozen=True)
class MultiPatchStaticResult:
    subresults: tuple[StaticPatchResult, ...]
    wrench: PairWrench

    @property
    def total_area(self) -> float:
        return float(sum(result.patch.area for result in self.subresults))

    @property
    def total_integrated_pressure(self) -> float:
        return float(sum(result.integrated_pressure for result in self.subresults))



def build_rectangular_planar_patch(
    footprint: BoxFootprint,
    *,
    center: ArrayLike | None = None,
    normal: ArrayLike | None = None,
) -> PlanarPatch:
    if center is None:
        center = np.zeros(3)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    n, u, v = orthonormal_frame(normal)
    hx = 0.5 * footprint.lx
    hy = 0.5 * footprint.ly
    polygon = ConvexPolygon2D(np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=float))
    return PlanarPatch(
        polygon=polygon,
        center=np.asarray(center, dtype=float),
        normal=n,
        tangent_u=u,
        tangent_v=v,
    )



def clip_planar_patch_with_halfspace(
    patch: PlanarPatch,
    halfspace: HalfSpace2D,
    *,
    tol: float = 1e-12,
) -> PlanarPatch | None:
    clipped_polygon = clip_convex_polygon_with_halfspace(patch.polygon, halfspace, tol=tol)
    if clipped_polygon is None:
        return None
    return PlanarPatch(
        polygon=clipped_polygon,
        center=patch.center,
        normal=patch.normal,
        tangent_u=patch.tangent_u,
        tangent_v=patch.tangent_v,
    )



def triangle_quadrature_degree2(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> tuple[PolygonQuadraturePoint, ...]:
    """Three-point degree-2 exact rule on a triangle."""
    area = triangle_area(a, b, c)
    bary_sets = (
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
    )
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    return tuple(
        PolygonQuadraturePoint(local_xy=wa * a + wb * b + wc * c, weight=area / 3.0)
        for wa, wb, wc in bary_sets
    )



def polygon_quadrature_degree2(polygon: ConvexPolygon2D) -> tuple[PolygonQuadraturePoint, ...]:
    points: list[PolygonQuadraturePoint] = []
    for tri in polygon.triangles_from_centroid():
        points.extend(triangle_quadrature_degree2(*tri))
    return tuple(points)



def integrate_over_planar_patch(
    patch: PlanarPatch,
    integrand: Callable[[ArrayLike, ArrayLike], ArrayLike],
) -> ArrayLike:
    acc: ArrayLike | None = None
    for qp in polygon_quadrature_degree2(patch.polygon):
        world_xyz = patch.local_to_world(qp.local_xy)
        value = np.asarray(integrand(qp.local_xy, world_xyz), dtype=float)
        contribution = qp.weight * value
        acc = contribution if acc is None else acc + contribution
    if acc is None:
        return np.array(0.0)
    return acc



def evaluate_static_patch(
    patch: PlanarPatch,
    overlap_fn: Callable[[ArrayLike, ArrayLike], float],
    law_a: PressureLaw,
    law_b: PressureLaw,
    *,
    reference: ArrayLike | None = None,
) -> StaticPatchResult:
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    total_force = np.zeros(3, dtype=float)
    total_torque = np.zeros(3, dtype=float)
    samples: list[PatchSample] = []
    for qp in polygon_quadrature_degree2(patch.polygon):
        world_xyz = patch.local_to_world(qp.local_xy)
        overlap = float(overlap_fn(qp.local_xy, world_xyz))
        overlap = max(overlap, 0.0)
        column = solve_column_equilibrium(overlap, law_a, law_b)
        traction = column.pressure * patch.normal
        total_force += qp.weight * traction
        total_torque += qp.weight * np.cross(world_xyz - reference, traction)
        samples.append(PatchSample(local_xy=qp.local_xy, world_xyz=world_xyz, weight=qp.weight, column=column))

    return StaticPatchResult(
        patch=patch,
        wrench=PairWrench(force=total_force, torque=total_torque),
        samples=tuple(samples),
    )



def evaluate_static_patch_collection(
    tasks: list[StaticPatchTask] | tuple[StaticPatchTask, ...],
    law_a: PressureLaw,
    law_b: PressureLaw,
    *,
    reference: ArrayLike | None = None,
) -> MultiPatchStaticResult:
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    subresults = tuple(
        evaluate_static_patch(task.patch, task.overlap_fn, law_a, law_b, reference=reference) for task in tasks
    )
    total_force = np.sum([result.wrench.force for result in subresults], axis=0) if subresults else np.zeros(3)
    total_torque = np.sum([result.wrench.torque for result in subresults], axis=0) if subresults else np.zeros(3)
    return MultiPatchStaticResult(subresults=subresults, wrench=PairWrench(force=total_force, torque=total_torque))
