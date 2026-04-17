from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from skimage import measure

from pfcsdf.contact.wrench import PairWrench
from pfcsdf.geometry.volume import SampledScalarField3D

ArrayLike = np.ndarray


@dataclass(frozen=True)
class TriangleMeshComponent:
    """Connected triangle-mesh component extracted from a sampled scalar field."""

    component_id: int
    vertices: ArrayLike
    faces: ArrayLike

    @property
    def num_vertices(self) -> int:
        return int(len(self.vertices))

    @property
    def num_faces(self) -> int:
        return int(len(self.faces))

    @property
    def area(self) -> float:
        return float(sum(_triangle_area_and_normal(*self.vertices[face])[0] for face in self.faces))

    @property
    def centroid(self) -> ArrayLike:
        if len(self.faces) == 0:
            if len(self.vertices) == 0:
                return np.zeros(3)
            return np.mean(self.vertices, axis=0)
        weighted = np.zeros(3, dtype=float)
        area_total = 0.0
        for face in self.faces:
            p0, p1, p2 = self.vertices[face]
            area, _ = _triangle_area_and_normal(p0, p1, p2)
            weighted += area * (p0 + p1 + p2) / 3.0
            area_total += area
        return weighted / area_total

    @property
    def mean_normal(self) -> ArrayLike:
        if len(self.faces) == 0:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        acc = np.zeros(3, dtype=float)
        for face in self.faces:
            _, normal = _triangle_area_and_normal(*self.vertices[face])
            acc += normal
        norm = float(np.linalg.norm(acc))
        if norm <= 1e-14:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return acc / norm


@dataclass(frozen=True)
class MarchingCubesReconstruction:
    level: float
    vertices: ArrayLike
    faces: ArrayLike
    normals: ArrayLike
    components: tuple[TriangleMeshComponent, ...]

    @property
    def num_components(self) -> int:
        return len(self.components)


def _triangle_area_and_normal(p0: ArrayLike, p1: ArrayLike, p2: ArrayLike) -> tuple[float, ArrayLike]:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    cross = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(cross))
    if norm <= 1e-14:
        return 0.0, np.array([0.0, 0.0, 1.0], dtype=float)
    return 0.5 * norm, cross / norm


def _face_components(num_vertices: int, faces: ArrayLike) -> list[list[int]]:
    vertex_to_faces: list[list[int]] = [[] for _ in range(num_vertices)]
    for face_idx, face in enumerate(faces):
        for vertex_idx in face:
            vertex_to_faces[int(vertex_idx)].append(face_idx)

    visited = np.zeros(len(faces), dtype=bool)
    components: list[list[int]] = []
    for start_face in range(len(faces)):
        if visited[start_face]:
            continue
        queue: deque[int] = deque([start_face])
        visited[start_face] = True
        component: list[int] = []
        while queue:
            face_idx = queue.popleft()
            component.append(face_idx)
            for vertex_idx in faces[face_idx]:
                for neigh_face in vertex_to_faces[int(vertex_idx)]:
                    if not visited[neigh_face]:
                        visited[neigh_face] = True
                        queue.append(neigh_face)
        components.append(component)
    return components


def reconstruct_sheet_mesh_marching_cubes(
    field: SampledScalarField3D,
    *,
    level: float = 0.0,
    gradient_direction: str = "descent",
    step_size: int = 1,
    allow_degenerate: bool = False,
    mask: ArrayLike | None = None,
) -> MarchingCubesReconstruction:
    """Extract a general 3D zero/level sheet using marching-cubes style triangulation.

    This prototype wraps ``skimage.measure.marching_cubes`` and then splits the mesh
    into connected components. Unlike the earlier graph-style reconstruction, this
    path does not assume one zero crossing per vertical column and can recover closed
    or overhanging sheets such as spheres.
    """

    values = np.asarray(field.values, dtype=np.float32)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("mask shape must match the sampled field shape")
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if not (vmin <= level <= vmax):
        return MarchingCubesReconstruction(
            level=level,
            vertices=np.zeros((0, 3), dtype=float),
            faces=np.zeros((0, 3), dtype=int),
            normals=np.zeros((0, 3), dtype=float),
            components=tuple(),
        )

    verts, faces, normals, _ = measure.marching_cubes(
        values,
        level=level,
        spacing=tuple(float(x) for x in field.grid.spacing),
        gradient_direction=gradient_direction,
        step_size=step_size,
        allow_degenerate=allow_degenerate,
        mask=mask,
    )
    verts = np.asarray(verts, dtype=float) + field.grid.origin
    faces = np.asarray(faces, dtype=int)
    normals = np.asarray(normals, dtype=float)

    components: list[TriangleMeshComponent] = []
    for component_id, face_ids in enumerate(_face_components(len(verts), faces)):
        comp_faces_global = faces[np.asarray(face_ids, dtype=int)]
        vertex_ids = np.unique(comp_faces_global.reshape(-1))
        local_index = {int(gid): lid for lid, gid in enumerate(vertex_ids.tolist())}
        comp_vertices = verts[vertex_ids]
        comp_faces = np.asarray(
            [[local_index[int(v)] for v in face] for face in comp_faces_global],
            dtype=int,
        )
        components.append(
            TriangleMeshComponent(
                component_id=component_id,
                vertices=comp_vertices,
                faces=comp_faces,
            )
        )

    return MarchingCubesReconstruction(
        level=level,
        vertices=verts,
        faces=faces,
        normals=normals,
        components=tuple(components),
    )


def integrate_uniform_pressure_on_triangle_mesh(
    component: TriangleMeshComponent,
    pressure: float,
    *,
    reference: ArrayLike | None = None,
) -> PairWrench:
    if pressure < 0.0:
        raise ValueError("pressure must be non-negative")
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    total_force = np.zeros(3, dtype=float)
    total_torque = np.zeros(3, dtype=float)
    for face in component.faces:
        p0, p1, p2 = component.vertices[face]
        area, normal = _triangle_area_and_normal(p0, p1, p2)
        if area <= 0.0:
            continue
        traction_force = pressure * area * normal
        centroid = (p0 + p1 + p2) / 3.0
        total_force += traction_force
        total_torque += np.cross(centroid - reference, traction_force)
    return PairWrench(force=total_force, torque=total_torque)


def integrate_uniform_pressure_on_mesh_collection(
    components: tuple[TriangleMeshComponent, ...] | list[TriangleMeshComponent],
    pressure: float,
    *,
    reference: ArrayLike | None = None,
) -> PairWrench:
    total_force = np.zeros(3, dtype=float)
    total_torque = np.zeros(3, dtype=float)
    for component in components:
        wrench = integrate_uniform_pressure_on_triangle_mesh(component, pressure, reference=reference)
        total_force += wrench.force
        total_torque += wrench.torque
    return PairWrench(force=total_force, torque=total_torque)
