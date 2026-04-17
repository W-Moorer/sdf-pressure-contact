from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from pfcsdf.contact.wrench import PairWrench
from pfcsdf.geometry.volume import SampledScalarField3D, UniformGrid3D

ArrayLike = np.ndarray


@dataclass(frozen=True)
class CurvedSheetComponent:
    """Graph-style curved sheet component reconstructed from a sampled 3D scalar field.

    The current prototype assumes one zero crossing per vertical (z) column and uses
    the column roots to build a height-field triangulation over connected xy support.
    """

    component_id: int
    vertices: ArrayLike
    faces: ArrayLike
    projected_indices: ArrayLike

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
        norm = np.linalg.norm(acc)
        if norm <= 1e-14:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return acc / norm


@dataclass(frozen=True)
class CurvedSheetReconstruction:
    grid: UniformGrid3D
    root_mask: ArrayLike
    root_points: ArrayLike
    component_labels: ArrayLike
    components: tuple[CurvedSheetComponent, ...]

    @property
    def num_components(self) -> int:
        return len(self.components)



def _column_zero_crossing(z_coords: ArrayLike, values: ArrayLike, *, tol: float = 1e-14) -> float | None:
    for k in range(len(z_coords) - 1):
        z0 = float(z_coords[k])
        z1 = float(z_coords[k + 1])
        v0 = float(values[k])
        v1 = float(values[k + 1])
        if abs(v0) <= tol:
            return z0
        if abs(v1) <= tol:
            return z1
        if v0 * v1 < 0.0:
            t = v0 / (v0 - v1)
            return z0 + t * (z1 - z0)
    return None



def _label_connected_components(mask: ArrayLike) -> ArrayLike:
    mask = np.asarray(mask, dtype=bool)
    labels = -np.ones(mask.shape, dtype=int)
    next_label = 0
    nx, ny = mask.shape
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for i in range(nx):
        for j in range(ny):
            if not mask[i, j] or labels[i, j] >= 0:
                continue
            queue: deque[tuple[int, int]] = deque([(i, j)])
            labels[i, j] = next_label
            while queue:
                ci, cj = queue.popleft()
                for di, dj in neighbors:
                    ni = ci + di
                    nj = cj + dj
                    if 0 <= ni < nx and 0 <= nj < ny and mask[ni, nj] and labels[ni, nj] < 0:
                        labels[ni, nj] = next_label
                        queue.append((ni, nj))
            next_label += 1
    return labels



def _triangle_area_and_normal(p0: ArrayLike, p1: ArrayLike, p2: ArrayLike) -> tuple[float, ArrayLike]:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    cross = np.cross(p1 - p0, p2 - p0)
    norm = float(np.linalg.norm(cross))
    if norm <= 1e-14:
        return 0.0, np.array([0.0, 0.0, 1.0], dtype=float)
    return 0.5 * norm, cross / norm



def reconstruct_curved_sheets_from_sampled_field(
    field: SampledScalarField3D,
    *,
    band_half_width: float | None = None,
) -> CurvedSheetReconstruction:
    """Reconstruct curved sheet components from a sampled scalar field.

    Current prototype assumption:
    - sheet is extracted as one zero crossing per z-column,
    - connected components are detected in the xy support mask,
    - each component is meshed as a height-field triangulation.
    """

    grid = field.grid
    nx, ny, _ = grid.shape
    x_coords = grid.x_coords
    y_coords = grid.y_coords
    z_coords = grid.z_coords

    root_mask = np.zeros((nx, ny), dtype=bool)
    root_points = np.full((nx, ny, 3), np.nan, dtype=float)

    for i in range(nx):
        for j in range(ny):
            column_values = field.values[i, j, :]
            root_z = _column_zero_crossing(z_coords, column_values)
            if root_z is None:
                continue
            if band_half_width is not None:
                if band_half_width <= 0.0:
                    raise ValueError("band_half_width must be positive when provided")
                if np.min(np.abs(column_values)) > band_half_width:
                    continue
            root_mask[i, j] = True
            root_points[i, j] = np.array([x_coords[i], y_coords[j], root_z], dtype=float)

    labels = _label_connected_components(root_mask)
    components: list[CurvedSheetComponent] = []
    num_components = int(labels.max()) + 1 if np.any(labels >= 0) else 0

    for label in range(num_components):
        vertices_map: dict[tuple[int, int], int] = {}
        vertices: list[ArrayLike] = []
        projected_indices: list[tuple[int, int]] = []

        for i in range(nx):
            for j in range(ny):
                if labels[i, j] == label:
                    vertices_map[(i, j)] = len(vertices)
                    vertices.append(root_points[i, j].copy())
                    projected_indices.append((i, j))

        faces: list[tuple[int, int, int]] = []
        for i in range(nx - 1):
            for j in range(ny - 1):
                corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                if any(labels[ii, jj] != label for ii, jj in corners):
                    continue
                v00 = vertices_map[(i, j)]
                v10 = vertices_map[(i + 1, j)]
                v11 = vertices_map[(i + 1, j + 1)]
                v01 = vertices_map[(i, j + 1)]
                faces.append((v00, v10, v11))
                faces.append((v00, v11, v01))

        components.append(
            CurvedSheetComponent(
                component_id=label,
                vertices=np.asarray(vertices, dtype=float),
                faces=np.asarray(faces, dtype=int) if faces else np.zeros((0, 3), dtype=int),
                projected_indices=np.asarray(projected_indices, dtype=int),
            )
        )

    return CurvedSheetReconstruction(
        grid=grid,
        root_mask=root_mask,
        root_points=root_points,
        component_labels=labels,
        components=tuple(components),
    )



def integrate_uniform_pressure_on_curved_sheet(
    component: CurvedSheetComponent,
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



def integrate_uniform_pressure_on_component_collection(
    components: tuple[CurvedSheetComponent, ...] | list[CurvedSheetComponent],
    pressure: float,
    *,
    reference: ArrayLike | None = None,
) -> PairWrench:
    total_force = np.zeros(3, dtype=float)
    total_torque = np.zeros(3, dtype=float)
    for component in components:
        wrench = integrate_uniform_pressure_on_curved_sheet(component, pressure, reference=reference)
        total_force += wrench.force
        total_torque += wrench.torque
    return PairWrench(force=total_force, torque=total_torque)
