from __future__ import annotations

import numpy as np

from pfcsdf.contact.sheet import build_flat_sheet_patch, uniform_pressure_center
from pfcsdf.geometry.primitives import BoxFootprint



def test_flat_sheet_patch_preserves_area_and_center() -> None:
    footprint = BoxFootprint(lx=2.0, ly=3.0)
    center = np.array([1.0, -2.0, 0.5])
    patch = build_flat_sheet_patch(footprint, center=center)

    assert patch.area == 6.0
    assert np.allclose(patch.center, center)
    assert np.allclose(uniform_pressure_center(patch), center)



def test_flat_sheet_patch_builds_orthonormal_frame() -> None:
    footprint = BoxFootprint(lx=1.0, ly=1.0)
    normal = np.array([0.0, 1.0, 1.0])
    patch = build_flat_sheet_patch(footprint, normal=normal)

    assert np.isclose(np.linalg.norm(patch.normal), 1.0)
    assert np.isclose(np.linalg.norm(patch.tangent_u), 1.0)
    assert np.isclose(np.linalg.norm(patch.tangent_v), 1.0)
    assert np.isclose(np.dot(patch.normal, patch.tangent_u), 0.0, atol=1e-12)
    assert np.isclose(np.dot(patch.normal, patch.tangent_v), 0.0, atol=1e-12)
    assert np.isclose(np.dot(patch.tangent_u, patch.tangent_v), 0.0, atol=1e-12)
