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


def _as_vector3(value: float | ArrayLike, *, name: str) -> ArrayLike:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or have shape (3,)")
    return arr


def recommend_mesh_sdf_spacing_from_native_band(
    native_band_spacing: float | ArrayLike,
    *,
    relative_scale: float = 0.9,
) -> float:
    """Recommend an isotropic mesh-SDF spacing for the current native-band sphere path.

    The current mesh-backed sphere validation uses an isotropic ``GridSDFGeometry``
    against an anisotropic native-band sampling grid. Recent sensitivity runs showed
    that the old ``0.12`` mesh-SDF spacing was too coarse for the current
    ``(0.1, 0.1, 0.05)`` native-band grid, but forcing the mesh SDF all the way down
    to the smallest native-band spacing made the pipeline much more expensive than
    necessary for a validation default. The practical default is therefore:

    ``recommended_spacing = relative_scale * max(native_band_spacing)``

    with ``relative_scale=0.9`` by default, so the isotropic mesh SDF is slightly
    finer than the coarsest native-band direction while remaining much cheaper than
    matching the finest direction exactly. This is a validation-oriented default,
    not a universal mesh-to-SDF policy.
    """

    spacing_vec = _as_vector3(native_band_spacing, name="native_band_spacing")
    if relative_scale <= 0.0:
        raise ValueError("relative_scale must be positive")
    return float(relative_scale * np.max(spacing_vec))


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
    sdf_spacing: ArrayLike
    recommended_sdf_spacing: float | None
    used_recommended_spacing_policy: bool


def build_mesh_asset_sdf_geometry(
    asset_path: str | Path,
    *,
    spacing: float | ArrayLike | None = None,
    native_band_spacing: float | ArrayLike | None = None,
    recommended_spacing_scale: float = 0.9,
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

    Resolution policy:
    - pass ``spacing=...`` to override the mesh-SDF spacing explicitly
    - or pass ``native_band_spacing=...`` with ``spacing=None`` to use the current
      recommended sphere-validation policy from
      ``recommend_mesh_sdf_spacing_from_native_band(...)``
    """

    asset_path = Path(asset_path)
    recommended_spacing = None
    used_recommended_spacing_policy = False
    if native_band_spacing is not None:
        recommended_spacing = recommend_mesh_sdf_spacing_from_native_band(
            native_band_spacing,
            relative_scale=recommended_spacing_scale,
        )
    if spacing is None:
        if recommended_spacing is None:
            raise ValueError("spacing must be provided unless native_band_spacing is supplied for the recommended policy")
        spacing = recommended_spacing
        used_recommended_spacing_policy = True

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
        sdf_spacing=np.asarray(sdf_result.geometry.spacing, dtype=float),
        recommended_sdf_spacing=recommended_spacing,
        used_recommended_spacing_policy=used_recommended_spacing_policy,
    )
