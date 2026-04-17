from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .base import BoundingBox
from .mesh_io import TriangleMesh

ArrayLike = np.ndarray


@dataclass(frozen=True)
class MeshValidationReport:
    """Summary of explicit assumptions checked before mesh-to-SDF conversion.

    This prototype assumes:
    - local-frame mesh coordinates
    - triangular faces only
    - no implicit normalization or unit conversion
    - watertight, coherently oriented meshes for robust inside/outside signing

    The current checks are combinatorial and lightweight. They do not detect
    self-intersections or repair invalid meshes.
    """

    bounding_box: BoundingBox
    num_vertices: int
    num_faces: int
    num_degenerate_faces: int
    num_boundary_edges: int
    num_nonmanifold_edges: int
    num_inconsistent_edges: int

    @property
    def is_closed_edge_manifold(self) -> bool:
        return self.num_boundary_edges == 0 and self.num_nonmanifold_edges == 0

    @property
    def has_consistent_orientation(self) -> bool:
        return self.num_inconsistent_edges == 0

    @property
    def has_no_degenerate_faces(self) -> bool:
        return self.num_degenerate_faces == 0

    @property
    def is_signed_distance_ready(self) -> bool:
        return self.is_closed_edge_manifold and self.has_consistent_orientation and self.has_no_degenerate_faces


def triangle_mesh_aabb(mesh: TriangleMesh) -> BoundingBox:
    vertices = np.asarray(mesh.vertices, dtype=float)
    return BoundingBox(minimum=np.min(vertices, axis=0), maximum=np.max(vertices, axis=0))


def inspect_triangle_mesh(mesh: TriangleMesh, *, area_tol: float = 1e-12) -> MeshValidationReport:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if area_tol <= 0.0:
        raise ValueError("area_tol must be positive")

    edge_directions: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    degenerate_faces = 0
    for face in faces:
        i, j, k = (int(face[0]), int(face[1]), int(face[2]))
        p0, p1, p2 = vertices[[i, j, k]]
        double_area = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
        if double_area <= area_tol:
            degenerate_faces += 1
        directed_edges = ((i, j), (j, k), (k, i))
        for start, end in directed_edges:
            edge_directions[tuple(sorted((start, end)))].append((start, end))

    boundary_edges = 0
    nonmanifold_edges = 0
    inconsistent_edges = 0
    for directions in edge_directions.values():
        if len(directions) == 1:
            boundary_edges += 1
            continue
        if len(directions) != 2:
            nonmanifold_edges += 1
            continue
        if directions[0] != (directions[1][1], directions[1][0]):
            inconsistent_edges += 1

    return MeshValidationReport(
        bounding_box=triangle_mesh_aabb(mesh),
        num_vertices=mesh.num_vertices,
        num_faces=mesh.num_faces,
        num_degenerate_faces=degenerate_faces,
        num_boundary_edges=boundary_edges,
        num_nonmanifold_edges=nonmanifold_edges,
        num_inconsistent_edges=inconsistent_edges,
    )


def validate_triangle_mesh(
    mesh: TriangleMesh,
    *,
    require_watertight: bool = True,
    require_consistent_orientation: bool = True,
    require_non_degenerate_faces: bool = True,
) -> MeshValidationReport:
    """Validate the minimum assumptions required by the current mesh-to-SDF path."""

    report = inspect_triangle_mesh(mesh)
    if require_watertight and not report.is_closed_edge_manifold:
        raise ValueError(
            "mesh-to-SDF currently requires a closed edge-manifold triangle mesh; "
            f"found {report.num_boundary_edges} boundary edges and {report.num_nonmanifold_edges} non-manifold edges"
        )
    if require_consistent_orientation and not report.has_consistent_orientation:
        raise ValueError(
            "mesh-to-SDF currently requires coherent face orientation for robust inside/outside signing; "
            f"found {report.num_inconsistent_edges} inconsistent shared edges"
        )
    if require_non_degenerate_faces and not report.has_no_degenerate_faces:
        raise ValueError(
            "mesh-to-SDF currently requires non-degenerate triangles; "
            f"found {report.num_degenerate_faces} zero-area faces"
        )
    return report
