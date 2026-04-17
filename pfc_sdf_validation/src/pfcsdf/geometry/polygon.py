from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray


@dataclass(frozen=True)
class ConvexPolygon2D:
    """Convex polygon in a local 2D patch frame.

    Vertices may be provided in clockwise or counter-clockwise order. The stored
    representation is normalized to counter-clockwise orientation.
    """

    vertices: ArrayLike

    def __post_init__(self) -> None:
        verts = np.asarray(self.vertices, dtype=float)
        if verts.ndim != 2 or verts.shape[1] != 2:
            raise ValueError("vertices must have shape (N, 2)")
        if len(verts) < 3:
            raise ValueError("polygon must have at least three vertices")
        area2 = signed_area_times_two(verts)
        if abs(area2) <= 1e-14:
            raise ValueError("polygon area must be positive")
        if area2 < 0.0:
            verts = verts[::-1].copy()
        object.__setattr__(self, "vertices", verts)

    @property
    def area(self) -> float:
        return 0.5 * signed_area_times_two(self.vertices)

    @property
    def centroid(self) -> ArrayLike:
        return polygon_centroid(self.vertices)

    def triangles_from_centroid(self) -> list[tuple[ArrayLike, ArrayLike, ArrayLike]]:
        c = self.centroid
        tris: list[tuple[ArrayLike, ArrayLike, ArrayLike]] = []
        for i in range(len(self.vertices)):
            a = self.vertices[i]
            b = self.vertices[(i + 1) % len(self.vertices)]
            tris.append((c, a, b))
        return tris


@dataclass(frozen=True)
class HalfSpace2D:
    """Half-space n·x + offset >= 0 in a local 2D frame."""

    normal: ArrayLike
    offset: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal, dtype=float)
        if normal.shape != (2,):
            raise ValueError("normal must have shape (2,)")
        norm = np.linalg.norm(normal)
        if norm <= 1e-14:
            raise ValueError("normal must be non-zero")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset", float(self.offset))

    def signed_value(self, point: ArrayLike) -> float:
        point = np.asarray(point, dtype=float)
        return float(np.dot(self.normal, point) + self.offset)


def signed_area_times_two(vertices: ArrayLike) -> float:
    x = vertices[:, 0]
    y = vertices[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))



def polygon_centroid(vertices: ArrayLike) -> ArrayLike:
    x = vertices[:, 0]
    y = vertices[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area2 = float(np.sum(cross))
    if abs(area2) <= 1e-14:
        raise ValueError("polygon area must be positive")
    cx = float(np.sum((x + np.roll(x, -1)) * cross) / (3.0 * area2))
    cy = float(np.sum((y + np.roll(y, -1)) * cross) / (3.0 * area2))
    return np.array([cx, cy], dtype=float)



def triangle_area(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    ab = b - a
    ac = c - a
    return 0.5 * abs(ab[0] * ac[1] - ab[1] * ac[0])



def _segment_halfspace_intersection(p0: ArrayLike, p1: ArrayLike, halfspace: HalfSpace2D) -> ArrayLike:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v0 = halfspace.signed_value(p0)
    v1 = halfspace.signed_value(p1)
    denom = v0 - v1
    if abs(denom) <= 1e-14:
        return p0.copy()
    t = v0 / denom
    return p0 + t * (p1 - p0)



def _deduplicate_polygon_vertices(vertices: list[ArrayLike], tol: float = 1e-12) -> list[ArrayLike]:
    if not vertices:
        return []
    deduped: list[ArrayLike] = []
    for v in vertices:
        vv = np.asarray(v, dtype=float)
        if not deduped or np.linalg.norm(vv - deduped[-1]) > tol:
            deduped.append(vv)
    if len(deduped) >= 2 and np.linalg.norm(deduped[0] - deduped[-1]) <= tol:
        deduped.pop()
    return deduped



def clip_convex_polygon_with_halfspace(
    polygon: ConvexPolygon2D,
    halfspace: HalfSpace2D,
    *,
    tol: float = 1e-12,
) -> ConvexPolygon2D | None:
    """Clip a convex polygon by a half-space using Sutherland-Hodgman."""
    input_vertices = [np.asarray(v, dtype=float) for v in polygon.vertices]
    output: list[ArrayLike] = []
    n = len(input_vertices)
    for i in range(n):
        current = input_vertices[i]
        previous = input_vertices[i - 1]
        current_val = halfspace.signed_value(current)
        previous_val = halfspace.signed_value(previous)
        current_inside = current_val >= -tol
        previous_inside = previous_val >= -tol

        if current_inside:
            if not previous_inside:
                output.append(_segment_halfspace_intersection(previous, current, halfspace))
            output.append(current)
        elif previous_inside:
            output.append(_segment_halfspace_intersection(previous, current, halfspace))

    output = _deduplicate_polygon_vertices(output, tol=tol)
    if len(output) < 3:
        return None
    try:
        return ConvexPolygon2D(np.asarray(output, dtype=float))
    except ValueError:
        return None
