from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import SignedDistanceGeometry
from .grid_sdf import GridSDFGeometry
from .mesh_io import TriangleMesh, load_obj_triangle_mesh
from .mesh_preprocess import MeshValidationReport
from .mesh_sdf import MeshGridSDFBuildResult, build_mesh_grid_sdf
from .transforms import TransformedGeometry

ArrayLike = np.ndarray


@dataclass(frozen=True)
class MeshAssetGeometryBuildResult:
    """Thin asset-to-geometry factory result for mesh-backed SDF validation paths.

    ``local_geometry`` remains in the mesh asset's local frame.
    ``geometry`` is either that same local geometry or a world-placed
    ``TransformedGeometry`` wrapper when rotation/translation are supplied.
    """

    asset_path: Path
    mesh: TriangleMesh
    validation: MeshValidationReport
    local_geometry: GridSDFGeometry
    geometry: SignedDistanceGeometry


def build_mesh_asset_sdf_geometry(
    asset_path: str | Path,
    *,
    spacing: float | ArrayLike,
    padding: float | ArrayLike = 0.0,
    rotation: ArrayLike | None = None,
    translation: ArrayLike | None = None,
    validate: bool = True,
    require_watertight: bool = True,
    require_consistent_orientation: bool = True,
    require_non_degenerate_faces: bool = True,
) -> MeshAssetGeometryBuildResult:
    """Load an OBJ asset and convert it into a mesh-backed signed-distance geometry.

    Important frame convention:
    - the sampled ``GridSDFGeometry`` stays in the mesh asset's local frame
    - world placement is explicit and only applied if ``rotation`` or ``translation``
      are provided
    """

    asset_path = Path(asset_path)
    mesh = load_obj_triangle_mesh(asset_path)
    sdf_result: MeshGridSDFBuildResult = build_mesh_grid_sdf(
        mesh,
        spacing=spacing,
        padding=padding,
        validate=validate,
        require_watertight=require_watertight,
        require_consistent_orientation=require_consistent_orientation,
        require_non_degenerate_faces=require_non_degenerate_faces,
    )

    geometry: SignedDistanceGeometry = sdf_result.geometry
    if rotation is not None or translation is not None:
        if rotation is None:
            rotation = np.eye(3, dtype=float)
        if translation is None:
            translation = np.zeros(3, dtype=float)
        geometry = TransformedGeometry(geometry=sdf_result.geometry, rotation=rotation, translation=translation)

    return MeshAssetGeometryBuildResult(
        asset_path=asset_path,
        mesh=mesh,
        validation=sdf_result.validation,
        local_geometry=sdf_result.geometry,
        geometry=geometry,
    )
