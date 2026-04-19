from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class TriangleMesh:
    """Minimal local-frame triangle mesh used by the mesh-to-SDF pipeline.

    The mesh stores vertices in local coordinates without any implicit recentering,
    normalization, or unit conversion. World placement should continue to use
    ``TransformedGeometry`` around the resulting SDF geometry.
    """

    vertices: ArrayLike
    faces: ArrayLike

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=float)
        faces = np.asarray(self.faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (N, 3)")
        if len(vertices) < 3:
            raise ValueError("triangle mesh must contain at least three vertices")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (M, 3)")
        if len(faces) == 0:
            raise ValueError("triangle mesh must contain at least one face")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("face indices must refer to valid vertices")
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "faces", faces)

    @property
    def num_vertices(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def num_faces(self) -> int:
        return int(self.faces.shape[0])


def _parse_obj_vertex(parts: list[str], line_number: int) -> ArrayLike:
    if len(parts) < 4:
        raise ValueError(f"OBJ vertex line {line_number} must provide x y z coordinates")
    return np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float)


def _parse_obj_face_vertex(token: str, num_vertices: int, line_number: int) -> int:
    vertex_token = token.split("/")[0]
    if vertex_token == "":
        raise ValueError(f"OBJ face line {line_number} is missing a vertex index")

    raw_index = int(vertex_token)
    if raw_index > 0:
        index = raw_index - 1
    else:
        index = num_vertices + raw_index

    if index < 0 or index >= num_vertices:
        raise ValueError(f"OBJ face line {line_number} references vertex {raw_index} out of range")
    return index


def load_obj_triangle_mesh(path: str | Path) -> TriangleMesh:
    """Load a local-frame triangle mesh from an OBJ file.

    Supported subset:
    - ``v x y z`` vertex records
    - ``f i j k`` triangle faces
    - face tokens with ``v/vt/vn``-style suffixes
    - positive and negative OBJ indices

    This loader intentionally does not triangulate polygons. Non-triangular faces
    raise ``ValueError`` so mesh assumptions stay explicit.
    """

    path = Path(path)
    vertices: list[ArrayLike] = []
    faces: list[tuple[int, int, int]] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        parts = stripped.split()
        record = parts[0]
        if record == "v":
            vertices.append(_parse_obj_vertex(parts, line_number))
            continue
        if record == "f":
            if len(parts) != 4:
                raise ValueError(f"OBJ face line {line_number} must be triangular; got {len(parts) - 1} vertices")
            face = tuple(_parse_obj_face_vertex(token, len(vertices), line_number) for token in parts[1:])
            faces.append(face)
            continue
        # Ignore normals, texcoords, groups, materials, and other non-geometry records.

    return TriangleMesh(
        vertices=np.asarray(vertices, dtype=float),
        faces=np.asarray(faces, dtype=int),
    )


def build_uv_sphere_triangle_mesh(
    *,
    radius: float,
    segments: int,
    stacks: int,
) -> TriangleMesh:
    """Build a small deterministic UV-sphere triangle mesh in the local frame.

    This helper is intended for validation and sensitivity studies where a tiny,
    reproducible sphere-like mesh is preferable to committing multiple static
    OBJ assets. Face orientation is adjusted so outward normals are consistent.
    """

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    if segments < 3:
        raise ValueError("segments must be at least 3")
    if stacks < 3:
        raise ValueError("stacks must be at least 3")

    vertices: list[list[float]] = [[0.0, 0.0, radius]]
    for i in range(1, stacks):
        theta = np.pi * i / stacks
        z = radius * float(np.cos(theta))
        rxy = radius * float(np.sin(theta))
        for j in range(segments):
            phi = 2.0 * np.pi * j / segments
            vertices.append([rxy * float(np.cos(phi)), rxy * float(np.sin(phi)), z])
    vertices.append([0.0, 0.0, -radius])

    top = 0
    bottom = len(vertices) - 1

    def ring_start(i: int) -> int:
        return 1 + (i - 1) * segments

    faces: list[list[int]] = []
    for j in range(segments):
        a = ring_start(1) + j
        b = ring_start(1) + (j + 1) % segments
        faces.append([top, a, b])
    for i in range(1, stacks - 1):
        rs = ring_start(i)
        ns = ring_start(i + 1)
        for j in range(segments):
            a = rs + j
            b = rs + (j + 1) % segments
            c = ns + j
            d = ns + (j + 1) % segments
            faces.append([a, c, d])
            faces.append([a, d, b])
    last = ring_start(stacks - 1)
    for j in range(segments):
        a = last + j
        b = last + (j + 1) % segments
        faces.append([a, b, bottom])

    vertices_arr = np.asarray(vertices, dtype=float)
    faces_arr = np.asarray(faces, dtype=int)
    oriented_faces: list[list[int]] = []
    for face in faces_arr:
        p0, p1, p2 = vertices_arr[face]
        normal = np.cross(p1 - p0, p2 - p0)
        centroid = (p0 + p1 + p2) / 3.0
        if float(np.dot(normal, centroid)) < 0.0:
            oriented_faces.append([int(face[0]), int(face[2]), int(face[1])])
        else:
            oriented_faces.append([int(face[0]), int(face[1]), int(face[2])])

    return TriangleMesh(
        vertices=vertices_arr,
        faces=np.asarray(oriented_faces, dtype=int),
    )
