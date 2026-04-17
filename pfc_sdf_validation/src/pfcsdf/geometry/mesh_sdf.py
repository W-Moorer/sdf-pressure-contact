from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid_sdf import GridSDFGeometry
from .mesh_io import TriangleMesh
from .mesh_preprocess import MeshValidationReport, inspect_triangle_mesh, validate_triangle_mesh

ArrayLike = np.ndarray


def _as_vector3(value: float | ArrayLike, *, name: str) -> ArrayLike:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or have shape (3,)")
    return arr


def _point_triangle_distance(point: ArrayLike, a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    """Return the Euclidean distance between a point and a triangle."""

    ab = b - a
    ac = c - a
    ap = point - a

    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return float(np.linalg.norm(ap))

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return float(np.linalg.norm(bp))

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        projection = a + v * ab
        return float(np.linalg.norm(point - projection))

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return float(np.linalg.norm(cp))

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        projection = a + w * ac
        return float(np.linalg.norm(point - projection))

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        bc = c - b
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        projection = b + w * bc
        return float(np.linalg.norm(point - projection))

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    projection = a + ab * v + ac * w
    return float(np.linalg.norm(point - projection))


def _triangle_solid_angle(point: ArrayLike, a: ArrayLike, b: ArrayLike, c: ArrayLike) -> float:
    pa = a - point
    pb = b - point
    pc = c - point
    la = float(np.linalg.norm(pa))
    lb = float(np.linalg.norm(pb))
    lc = float(np.linalg.norm(pc))
    if min(la, lb, lc) <= 1e-15:
        return 0.0

    numerator = float(np.dot(pa, np.cross(pb, pc)))
    denominator = (
        la * lb * lc
        + float(np.dot(pa, pb)) * lc
        + float(np.dot(pb, pc)) * la
        + float(np.dot(pc, pa)) * lb
    )
    return 2.0 * float(np.arctan2(numerator, denominator))


def mesh_signed_distance(
    mesh: TriangleMesh,
    point: ArrayLike,
    *,
    surface_tol: float = 1e-12,
) -> float:
    """Evaluate a signed distance from a coherently oriented triangle mesh.

    Sign convention:
    - negative inside
    - positive outside

    Inside/outside is inferred from the absolute winding number computed from
    oriented triangle solid angles. This is suitable for clean, watertight,
    coherently oriented meshes. It is not intended as a robust general mesh
    repair or CAD ingestion path.
    """

    point = np.asarray(point, dtype=float)
    min_distance = np.inf
    total_solid_angle = 0.0

    for face in mesh.faces:
        a, b, c = mesh.vertices[np.asarray(face, dtype=int)]
        min_distance = min(min_distance, _point_triangle_distance(point, a, b, c))
        total_solid_angle += _triangle_solid_angle(point, a, b, c)

    if min_distance <= surface_tol:
        return 0.0

    winding_number = total_solid_angle / (4.0 * np.pi)
    inside = abs(winding_number) > 0.5
    return float(-min_distance if inside else min_distance)


@dataclass(frozen=True)
class MeshGridSDFBuildResult:
    geometry: GridSDFGeometry
    validation: MeshValidationReport
    requested_minimum: ArrayLike
    requested_maximum: ArrayLike


def mesh_to_grid_sdf(
    mesh: TriangleMesh,
    *,
    spacing: float | ArrayLike,
    padding: float | ArrayLike = 0.0,
    validate: bool = True,
    require_watertight: bool = True,
    require_consistent_orientation: bool = True,
    require_non_degenerate_faces: bool = True,
) -> GridSDFGeometry:
    """Sample a watertight local-frame triangle mesh into a regular-grid SDF.

    The grid domain is built from the mesh AABB plus explicit padding. Sampled
    values live on grid nodes because ``GridSDFGeometry`` uses nodal trilinear
    interpolation.
    """

    return build_mesh_grid_sdf(
        mesh,
        spacing=spacing,
        padding=padding,
        validate=validate,
        require_watertight=require_watertight,
        require_consistent_orientation=require_consistent_orientation,
        require_non_degenerate_faces=require_non_degenerate_faces,
    ).geometry


def build_mesh_grid_sdf(
    mesh: TriangleMesh,
    *,
    spacing: float | ArrayLike,
    padding: float | ArrayLike = 0.0,
    validate: bool = True,
    require_watertight: bool = True,
    require_consistent_orientation: bool = True,
    require_non_degenerate_faces: bool = True,
) -> MeshGridSDFBuildResult:
    spacing_vec = _as_vector3(spacing, name="spacing")
    padding_vec = _as_vector3(padding, name="padding")
    if np.any(spacing_vec <= 0.0):
        raise ValueError("spacing must be strictly positive")
    if np.any(padding_vec < 0.0):
        raise ValueError("padding must be non-negative")

    validation = (
        validate_triangle_mesh(
            mesh,
            require_watertight=require_watertight,
            require_consistent_orientation=require_consistent_orientation,
            require_non_degenerate_faces=require_non_degenerate_faces,
        )
        if validate
        else inspect_triangle_mesh(mesh)
    )

    padded_minimum = validation.bounding_box.minimum - padding_vec
    padded_maximum = validation.bounding_box.maximum + padding_vec
    extent = padded_maximum - padded_minimum
    shape = tuple(int(max(2, np.ceil(extent[axis] / spacing_vec[axis]) + 1)) for axis in range(3))

    values = np.empty(shape, dtype=float)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                point = padded_minimum + spacing_vec * np.array([i, j, k], dtype=float)
                values[i, j, k] = mesh_signed_distance(mesh, point)

    return MeshGridSDFBuildResult(
        geometry=GridSDFGeometry(origin=padded_minimum, spacing=spacing_vec, values=values),
        validation=validation,
        requested_minimum=padded_minimum,
        requested_maximum=padded_maximum,
    )
