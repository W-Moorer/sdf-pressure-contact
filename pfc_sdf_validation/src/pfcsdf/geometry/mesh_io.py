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
