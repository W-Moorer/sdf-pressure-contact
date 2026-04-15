from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol
import math
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if n <= 1.0e-15 else q / n


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rv)
    if theta <= 1.0e-15:
        return quat_normalize(np.array([1.0, 0.5 * rv[0], 0.5 * rv[1], 0.5 * rv[2]], dtype=float))
    axis = rv / theta
    h = 0.5 * theta
    return np.array([math.cos(h), *(math.sin(h) * axis)], dtype=float)


def integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    return quat_normalize(quat_mul(quat_from_rotvec(dt * omega), q))


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q) @ v


def orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / max(np.linalg.norm(n), 1.0e-15)
    a = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, a)
    t1 /= max(np.linalg.norm(t1), 1.0e-15)
    t2 = np.cross(n, t1)
    t2 /= max(np.linalg.norm(t2), 1.0e-15)
    return t1, t2


@dataclass
class Marker:
    name: str
    local_position: np.ndarray


@dataclass
class Pose6D:
    position: np.ndarray
    orientation: np.ndarray


@dataclass
class BodyState6D:
    pose: Pose6D
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def copy(self) -> 'BodyState6D':
        return BodyState6D(
            Pose6D(self.pose.position.copy(), self.pose.orientation.copy()),
            self.linear_velocity.copy(),
            self.angular_velocity.copy(),
        )


@dataclass
class SpatialInertia:
    mass: float
    inertia_body: np.ndarray

    def inertia_world(self, q: np.ndarray) -> np.ndarray:
        R = quat_to_rotmat(q)
        return R @ self.inertia_body @ R.T


class Geometry(Protocol):
    def phi_world(self, x: float, y: float, z: float, pose: Pose6D) -> float: ...
    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray: ...
    def bounding_radius(self) -> float: ...


@dataclass
class RigidBody6D:
    name: str
    inertia: SpatialInertia
    geometry: Geometry
    state: BodyState6D
    markers: list[Marker] = field(default_factory=list)
    is_static: bool = False
    linear_damping: float = 0.0
    angular_damping: float = 0.0

    def world_markers(self) -> dict[str, np.ndarray]:
        return {
            m.name: self.state.pose.position + rotate_vec(self.state.pose.orientation, m.local_position)
            for m in self.markers
        }

    def clone_with_state(self, state: BodyState6D) -> 'RigidBody6D':
        return RigidBody6D(
            self.name,
            self.inertia,
            self.geometry,
            state.copy(),
            self.markers,
            self.is_static,
            self.linear_damping,
            self.angular_damping,
        )


@dataclass
class DomainSpec:
    cube_size: float
    cube_height: float
    top_y: float = 0.0


@dataclass
class World:
    domain: DomainSpec
    gravity: np.ndarray
    bodies: list[RigidBody6D]
    domain_sources: list['SDFGeometryDomainSource'] = field(default_factory=list)


class SDFSource(Protocol):
    name: str
    is_dynamic: bool

    def phi_world(self, x: float, y: float, z: float) -> float: ...
    def normal_world(self, x: float, y: float, z: float, h: float = 1.0e-6) -> np.ndarray: ...
    def velocity_world(self, point: np.ndarray) -> np.ndarray: ...
    def reference_center_world(self) -> np.ndarray: ...
    def patch_hint_radius(self) -> float: ...
    def wrench_ref_point(self) -> np.ndarray: ...


@dataclass
class BodySource:
    body_index: int
    body: RigidBody6D
    name: str = ''
    is_dynamic: bool = True

    def __post_init__(self) -> None:
        self.name = self.body.name
        self.is_dynamic = not self.body.is_static

    def phi_world(self, x: float, y: float, z: float) -> float:
        return float(self.body.geometry.phi_world(x, y, z, self.body.state.pose))

    def normal_world(self, x: float, y: float, z: float, h: float = 1.0e-6) -> np.ndarray:
        return np.asarray(self.body.geometry.normal_world(x, y, z, self.body.state.pose), dtype=float)

    def velocity_world(self, point: np.ndarray) -> np.ndarray:
        r = point - self.body.state.pose.position
        return self.body.state.linear_velocity + np.cross(self.body.state.angular_velocity, r)

    def reference_center_world(self) -> np.ndarray:
        return self.body.state.pose.position.copy()

    def patch_hint_radius(self) -> float:
        return float(self.body.geometry.bounding_radius())

    def wrench_ref_point(self) -> np.ndarray:
        return self.body.state.pose.position.copy()


@dataclass
class SDFGeometryDomainSource:
    geometry: Geometry
    pose: Pose6D
    name: str = 'domain_sdf'
    hint_radius: float = 0.5
    reference_center: Optional[np.ndarray] = None
    is_dynamic: bool = False

    def phi_world(self, x: float, y: float, z: float) -> float:
        return float(self.geometry.phi_world(x, y, z, self.pose))

    def normal_world(self, x: float, y: float, z: float, h: float = 1.0e-6) -> np.ndarray:
        n = np.asarray(self.geometry.normal_world(x, y, z, self.pose), dtype=float)
        return n / max(np.linalg.norm(n), 1.0e-15)

    def velocity_world(self, point: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def reference_center_world(self) -> np.ndarray:
        return self.pose.position.copy() if self.reference_center is None else self.reference_center.copy()

    def patch_hint_radius(self) -> float:
        return float(self.hint_radius)

    def wrench_ref_point(self) -> np.ndarray:
        return self.reference_center_world()


@dataclass
class PairPatchSample:
    uv: np.ndarray
    world_seed: np.ndarray
    projected_weight: float


@dataclass
class PairContactPatches:
    samples: list[PairPatchSample]
    center: np.ndarray
    normal0: np.ndarray
    tangent1: np.ndarray
    tangent2: np.ndarray
    radius: float
    metadata: dict = field(default_factory=dict)


@dataclass
class PairSheetPoint:
    midpoint: np.ndarray
    point_a: np.ndarray
    point_b: np.ndarray
    normal_a: np.ndarray
    normal_b: np.ndarray
    sheet_normal: np.ndarray
    projected_weight: float
    surface_weight: float
    source_index: int = -1


@dataclass
class PairSheet:
    samples: list[PairSheetPoint]
    metadata: dict = field(default_factory=dict)


@dataclass
class PairTractionSample:
    midpoint: np.ndarray
    sheet_normal: np.ndarray
    pressure: float
    area_weight: float
    force_on_a: np.ndarray
    force_on_b: np.ndarray
    moment_a: np.ndarray
    moment_b: np.ndarray


@dataclass
class PairTractionField:
    samples: list[PairTractionSample]
    metadata: dict = field(default_factory=dict)


@dataclass
class Wrench:
    force: np.ndarray
    moment: np.ndarray


@dataclass
class PairWrenchContribution:
    pair_kind: str
    source_name: str
    force: np.ndarray
    moment: np.ndarray
    meta: dict = field(default_factory=dict)


@dataclass
class PairRecord:
    pair_kind: str
    source_a_name: str
    source_b_name: str
    contribution_a: PairWrenchContribution
    contribution_b: PairWrenchContribution
    meta: dict = field(default_factory=dict)


@dataclass
class AggregatedContact:
    total_force: np.ndarray
    total_moment: np.ndarray
    pair_records: list[PairRecord]
    num_pair_patch_points: int = 0
    num_pair_sheet_points: int = 0
    num_pair_tractions: int = 0


@dataclass
class ForceAssemblyResult:
    total_force: np.ndarray
    total_moment: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    contact_meta: dict


@dataclass
class SimulationLogEntry:
    time: float
    body_name: str
    position: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    contact_force: np.ndarray
    num_pairs: int
    num_pair_patch_points: int
    num_pair_sheet_points: int
    num_pair_tractions: int
    marker_positions: dict[str, np.ndarray]
