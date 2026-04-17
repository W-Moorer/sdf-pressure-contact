from __future__ import annotations

import numpy as np

from pfcsdf.contact.patch import build_rectangular_planar_patch, clip_planar_patch_with_halfspace
from pfcsdf.contact.region import AffineOverlapField2D, detect_support_patch_affine
from pfcsdf.geometry.polygon import HalfSpace2D
from pfcsdf.geometry.primitives import BoxFootprint


def test_patch_clipping_by_halfspace_recovers_expected_half_rectangle() -> None:
    patch = build_rectangular_planar_patch(BoxFootprint(lx=2.0, ly=1.0))
    clipped = clip_planar_patch_with_halfspace(patch, HalfSpace2D(normal=np.array([1.0, 0.0]), offset=0.0))

    assert clipped is not None
    assert np.isclose(clipped.area, 1.0)
    assert np.allclose(clipped.local_centroid, np.array([0.5, 0.0]))
    assert np.allclose(clipped.world_centroid, np.array([0.5, 0.0, 0.0]))


def test_affine_support_region_detection_clips_patch_consistently() -> None:
    patch = build_rectangular_planar_patch(BoxFootprint(lx=2.0, ly=1.0))
    field = AffineOverlapField2D(offset=0.25, gradient=np.array([1.0, 0.0]))

    detected = detect_support_patch_affine(patch, field)
    assert detected.support_patch is not None
    support_patch = detected.support_patch

    assert np.isclose(support_patch.area, 1.25)
    assert np.allclose(support_patch.local_centroid, np.array([0.375, 0.0]))



def test_affine_support_region_detection_returns_none_for_negative_constant_overlap() -> None:
    patch = build_rectangular_planar_patch(BoxFootprint(lx=2.0, ly=1.0))
    field = AffineOverlapField2D(offset=-0.1, gradient=np.array([0.0, 0.0]))

    detected = detect_support_patch_affine(patch, field)
    assert detected.support_patch is None
