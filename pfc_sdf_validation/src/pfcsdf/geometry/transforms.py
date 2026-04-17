from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox, BoundedSignedDistanceGeometry, SignedDistanceGeometry, signed_distance_gradient

ArrayLike = np.ndarray


def _as_rotation_matrix(rotation: ArrayLike) -> ArrayLike:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("rotation must have shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
        raise ValueError("rotation must have determinant +1")
    return rotation


@dataclass(frozen=True)
class TransformedGeometry:
    """Rigid world-space wrapper around a local-frame signed-distance geometry.

    Base geometry objects remain local-frame descriptions. This wrapper accepts world-space
    query points, maps them into the wrapped geometry's local frame, and maps gradients back
    into world coordinates.

    If the wrapped geometry supports ``bounding_box()``, this wrapper returns a world-space
    axis-aligned bounding box that encloses the transformed local bounding-box corners.
    """

    geometry: SignedDistanceGeometry
    rotation: ArrayLike
    translation: ArrayLike

    def __post_init__(self) -> None:
        object.__setattr__(self, "rotation", _as_rotation_matrix(self.rotation))
        translation = np.asarray(self.translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError("translation must have shape (3,)")
        object.__setattr__(self, "translation", translation)

    @classmethod
    def identity(cls, geometry: SignedDistanceGeometry) -> "TransformedGeometry":
        return cls(geometry=geometry, rotation=np.eye(3), translation=np.zeros(3))

    @classmethod
    def from_translation(cls, geometry: SignedDistanceGeometry, translation: ArrayLike) -> "TransformedGeometry":
        return cls(geometry=geometry, rotation=np.eye(3), translation=translation)

    def world_to_local_point(self, point_world: ArrayLike) -> ArrayLike:
        point_world = np.asarray(point_world, dtype=float)
        if point_world.shape != (3,):
            raise ValueError("point_world must have shape (3,)")
        return self.rotation.T @ (point_world - self.translation)

    def local_to_world_point(self, point_local: ArrayLike) -> ArrayLike:
        point_local = np.asarray(point_local, dtype=float)
        if point_local.shape != (3,):
            raise ValueError("point_local must have shape (3,)")
        return self.rotation @ point_local + self.translation

    def local_to_world_vector(self, vector_local: ArrayLike) -> ArrayLike:
        vector_local = np.asarray(vector_local, dtype=float)
        if vector_local.shape != (3,):
            raise ValueError("vector_local must have shape (3,)")
        return self.rotation @ vector_local

    def signed_distance(self, x: ArrayLike) -> float:
        return float(self.geometry.signed_distance(self.world_to_local_point(x)))

    def gradient(self, x: ArrayLike) -> ArrayLike:
        local_point = self.world_to_local_point(x)
        local_grad = signed_distance_gradient(self.geometry, local_point)
        return self.local_to_world_vector(local_grad)

    def bounding_box(self) -> BoundingBox:
        if not isinstance(self.geometry, BoundedSignedDistanceGeometry):
            raise TypeError("wrapped geometry does not provide bounding_box()")

        local_bbox = self.geometry.bounding_box()
        corners_local = np.array(
            [
                [local_bbox.minimum[0], local_bbox.minimum[1], local_bbox.minimum[2]],
                [local_bbox.minimum[0], local_bbox.minimum[1], local_bbox.maximum[2]],
                [local_bbox.minimum[0], local_bbox.maximum[1], local_bbox.minimum[2]],
                [local_bbox.minimum[0], local_bbox.maximum[1], local_bbox.maximum[2]],
                [local_bbox.maximum[0], local_bbox.minimum[1], local_bbox.minimum[2]],
                [local_bbox.maximum[0], local_bbox.minimum[1], local_bbox.maximum[2]],
                [local_bbox.maximum[0], local_bbox.maximum[1], local_bbox.minimum[2]],
                [local_bbox.maximum[0], local_bbox.maximum[1], local_bbox.maximum[2]],
            ],
            dtype=float,
        )
        corners_world = (self.rotation @ corners_local.T).T + self.translation[None, :]
        return BoundingBox(
            minimum=np.min(corners_world, axis=0),
            maximum=np.max(corners_world, axis=0),
        )
