from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
import trimesh

from .core import (
    Marker,
    Pose6D,
    BodyState6D,
    SpatialInertia,
    RigidBody6D,
    DomainSpec,
    World,
    SDFGeometryDomainSource,
    quat_to_rotmat,
)


class SphereGeometry:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def bounding_radius(self) -> float:
        return self.radius

    def phi_world(self, x: float, y: float, z: float, pose: Pose6D) -> float:
        c = pose.position
        return float(np.linalg.norm(np.array([x, y, z], dtype=float) - c) - self.radius)

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        d = np.array([x, y, z], dtype=float) - pose.position
        n = np.linalg.norm(d)
        return np.array([0.0, 1.0, 0.0], dtype=float) if n <= 1.0e-15 else d / n




class BoxGeometry:
    def __init__(self, extents):
        self.extents = np.asarray(extents, dtype=float)
        if self.extents.shape != (3,):
            raise ValueError('extents must be length-3')
        self.half = 0.5 * self.extents

    def bounding_radius(self) -> float:
        return float(np.linalg.norm(self.half))

    def phi_world(self, x: float, y: float, z: float, pose: Pose6D) -> float:
        p_world = np.array([x, y, z], dtype=float)
        R = quat_to_rotmat(pose.orientation)
        p_local = R.T @ (p_world - pose.position)
        q = np.abs(p_local) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0))
        inside = min(max(q[0], max(q[1], q[2])), 0.0)
        return float(outside + inside)

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        p_world = np.array([x, y, z], dtype=float)
        R = quat_to_rotmat(pose.orientation)
        p_local = R.T @ (p_world - pose.position)
        q = np.abs(p_local) - self.half
        if np.any(q > 0.0):
            clamped = np.clip(p_local, -self.half, self.half)
            n_local = p_local - clamped
            nn = np.linalg.norm(n_local)
            if nn <= 1.0e-15:
                n_local = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                n_local = n_local / nn
        else:
            idx = int(np.argmax(q))
            n_local = np.zeros(3, dtype=float)
            n_local[idx] = 1.0 if p_local[idx] >= 0.0 else -1.0
        n_world = R @ n_local
        return n_world / max(np.linalg.norm(n_world), 1.0e-15)


class PlaneGeometry:
    def __init__(self, normal=(0.0, 1.0, 0.0), offset: float = 0.0):
        n = np.asarray(normal, dtype=float)
        self.n = n / max(np.linalg.norm(n), 1.0e-15)
        self.offset = float(offset)

    def bounding_radius(self) -> float:
        return 1.0

    def phi_world(self, x: float, y: float, z: float, pose: Pose6D) -> float:
        p = np.array([x, y, z], dtype=float)
        return float(np.dot(self.n, p) - self.offset)

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        return self.n.copy()


@dataclass
class MeshGeometryFactoryConfig:
    recenter_mode: str = 'center_mass'
    normal_eps: float = 1.0e-5
    ray_eps: float = 1.0e-9
    inside_test_direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.123, 0.456], dtype=float))


class MeshSDFGeometry:
    """
    Self-contained brute-force mesh SDF backend.

    Intended as a correctness-oriented reference backend for small / medium benchmarks.
    For large production scenes, replace with a sparse VDB-style SDF backend.
    """

    def __init__(self, mesh_local: trimesh.Trimesh, cfg: MeshGeometryFactoryConfig | None = None):
        self.mesh = mesh_local.copy()
        self.cfg = cfg or MeshGeometryFactoryConfig()
        self.vertices = np.asarray(self.mesh.vertices, dtype=float)
        self.faces = np.asarray(self.mesh.faces, dtype=int)
        self.triangles = self.vertices[self.faces]
        self.face_normals = np.asarray(self.mesh.face_normals, dtype=float)
        self._bounding_radius = float(np.max(np.linalg.norm(self.vertices, axis=1))) if len(self.vertices) else 0.0
        d = np.asarray(self.cfg.inside_test_direction, dtype=float)
        self._ray_dir = d / max(np.linalg.norm(d), 1.0e-15)

    @classmethod
    def from_trimesh(cls, mesh_local: trimesh.Trimesh, cfg: MeshGeometryFactoryConfig | None = None) -> 'MeshSDFGeometry':
        return cls(mesh_local=mesh_local, cfg=cfg)

    def bounding_radius(self) -> float:
        return self._bounding_radius

    def _world_to_local(self, p_world: np.ndarray, pose: Pose6D) -> np.ndarray:
        R = quat_to_rotmat(pose.orientation)
        return R.T @ (p_world - pose.position)

    def _local_to_world_dir(self, v_local: np.ndarray, pose: Pose6D) -> np.ndarray:
        R = quat_to_rotmat(pose.orientation)
        return R @ v_local

    @staticmethod
    def _closest_point_on_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        ab = b - a
        ac = c - a
        ap = p - a
        d1 = np.dot(ab, ap)
        d2 = np.dot(ac, ap)
        if d1 <= 0.0 and d2 <= 0.0:
            return a

        bp = p - b
        d3 = np.dot(ab, bp)
        d4 = np.dot(ac, bp)
        if d3 >= 0.0 and d4 <= d3:
            return b

        vc = d1 * d4 - d3 * d2
        if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
            v = d1 / max(d1 - d3, 1.0e-15)
            return a + v * ab

        cp = p - c
        d5 = np.dot(ab, cp)
        d6 = np.dot(ac, cp)
        if d6 >= 0.0 and d5 <= d6:
            return c

        vb = d5 * d2 - d1 * d6
        if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
            w = d2 / max(d2 - d6, 1.0e-15)
            return a + w * ac

        va = d3 * d6 - d5 * d4
        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
            bc = c - b
            w = (d4 - d3) / max((d4 - d3) + (d5 - d6), 1.0e-15)
            return b + w * bc

        denom = max(va + vb + vc, 1.0e-15)
        v = vb / denom
        w = vc / denom
        return a + ab * v + ac * w

    def _closest_surface_data_local(self, p_local: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        best_d2 = math.inf
        best_cp = np.zeros(3, dtype=float)
        best_n = np.array([0.0, 1.0, 0.0], dtype=float)
        for tri, n in zip(self.triangles, self.face_normals):
            cp = self._closest_point_on_triangle(p_local, tri[0], tri[1], tri[2])
            d2 = float(np.dot(p_local - cp, p_local - cp))
            if d2 < best_d2:
                best_d2 = d2
                best_cp = cp
                best_n = n
        return math.sqrt(best_d2), best_cp, best_n

    @staticmethod
    def _ray_intersects_triangle(orig: np.ndarray, direction: np.ndarray, tri: np.ndarray, eps: float) -> bool:
        v0, v1, v2 = tri
        e1 = v1 - v0
        e2 = v2 - v0
        h = np.cross(direction, e2)
        a = float(np.dot(e1, h))
        if -eps < a < eps:
            return False
        f = 1.0 / a
        s = orig - v0
        u = f * float(np.dot(s, h))
        if u < 0.0 or u > 1.0:
            return False
        q = np.cross(s, e1)
        v = f * float(np.dot(direction, q))
        if v < 0.0 or (u + v) > 1.0:
            return False
        t = f * float(np.dot(e2, q))
        return t > eps

    def _is_inside_local(self, p_local: np.ndarray) -> bool:
        hits = 0
        for tri in self.triangles:
            if self._ray_intersects_triangle(p_local, self._ray_dir, tri, self.cfg.ray_eps):
                hits += 1
        return (hits % 2) == 1

    def phi_world(self, x: float, y: float, z: float, pose: Pose6D) -> float:
        p_world = np.array([x, y, z], dtype=float)
        p_local = self._world_to_local(p_world, pose)
        dist, _, _ = self._closest_surface_data_local(p_local)
        if dist <= self.cfg.normal_eps:
            return 0.0
        return -dist if self._is_inside_local(p_local) else dist

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        p_world = np.array([x, y, z], dtype=float)
        p_local = self._world_to_local(p_world, pose)
        dist, cp, n_local = self._closest_surface_data_local(p_local)
        if dist > self.cfg.normal_eps:
            delta = p_local - cp
            dn = np.linalg.norm(delta)
            if dn > 1.0e-15:
                n_local = delta / dn
        n_world = self._local_to_world_dir(n_local, pose)
        return n_world / max(np.linalg.norm(n_world), 1.0e-15)


def _load_trimesh_any(mesh_or_path) -> trimesh.Trimesh:
    if isinstance(mesh_or_path, trimesh.Trimesh):
        mesh = mesh_or_path.copy()
    else:
        mesh = trimesh.load_mesh(str(mesh_or_path), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError('Expected a trimesh.Trimesh or a mesh file path.')
    return mesh


def _recenter_mesh(mesh: trimesh.Trimesh, mode: str) -> tuple[trimesh.Trimesh, np.ndarray]:
    mesh = mesh.copy()
    if mode == 'none':
        offset = np.zeros(3, dtype=float)
    elif mode == 'bbox_center':
        offset = np.asarray(mesh.bounds.mean(axis=0), dtype=float)
    elif mode == 'center_mass':
        try:
            offset = np.asarray(mesh.center_mass, dtype=float)
        except Exception:
            offset = np.asarray(mesh.bounds.mean(axis=0), dtype=float)
    else:
        raise ValueError(f'Unknown recenter_mode: {mode}')
    mesh.vertices = np.asarray(mesh.vertices, dtype=float) - offset
    return mesh, offset


def _compute_spatial_inertia_from_mesh(mesh_local: trimesh.Trimesh, mass: float) -> SpatialInertia:
    props = mesh_local.mass_properties
    try:
        mesh_mass = float(props.mass)
        inertia_body = np.asarray(props.inertia, dtype=float)
    except AttributeError:
        mesh_mass = float(props['mass'])
        inertia_body = np.asarray(props['inertia'], dtype=float)

    if mesh_mass <= 1.0e-15:
        verts = np.asarray(mesh_local.vertices, dtype=float)
        r = float(np.max(np.linalg.norm(verts, axis=1))) if len(verts) else 1.0
        I = (2.0 / 5.0) * mass * r * r
        inertia_body = np.diag([I, I, I])
    else:
        inertia_body = inertia_body * (mass / mesh_mass)
    return SpatialInertia(mass=float(mass), inertia_body=inertia_body)


def make_mesh_rigidbody(*, name, mesh_or_path, mass, position, orientation=None, linear_velocity=None, angular_velocity=None, markers=None, linear_damping=0.0, angular_damping=0.0, cfg=None) -> RigidBody6D:
    cfg = cfg or MeshGeometryFactoryConfig()
    mesh = _load_trimesh_any(mesh_or_path)
    mesh_local, _ = _recenter_mesh(mesh, cfg.recenter_mode)
    geom = MeshSDFGeometry.from_trimesh(mesh_local, cfg=cfg)
    inertia = _compute_spatial_inertia_from_mesh(mesh_local, mass)
    return RigidBody6D(
        name=name,
        inertia=inertia,
        geometry=geom,
        state=BodyState6D(
            pose=Pose6D(
                position=np.asarray(position, dtype=float).copy(),
                orientation=np.asarray(orientation if orientation is not None else [1.0, 0.0, 0.0, 0.0], dtype=float).copy(),
            ),
            linear_velocity=np.asarray(linear_velocity if linear_velocity is not None else [0.0, 0.0, 0.0], dtype=float).copy(),
            angular_velocity=np.asarray(angular_velocity if angular_velocity is not None else [0.0, 0.0, 0.0], dtype=float).copy(),
        ),
        markers=list(markers) if markers is not None else [],
        linear_damping=float(linear_damping),
        angular_damping=float(angular_damping),
    )


def make_mesh_domain_source(*, name, mesh_or_path, position, orientation=None, hint_radius=None, reference_center=None, cfg=None, is_dynamic=False) -> SDFGeometryDomainSource:
    cfg = cfg or MeshGeometryFactoryConfig(recenter_mode='bbox_center')
    mesh = _load_trimesh_any(mesh_or_path)
    mesh_local, _ = _recenter_mesh(mesh, cfg.recenter_mode)
    geom = MeshSDFGeometry.from_trimesh(mesh_local, cfg=cfg)
    if hint_radius is None:
        hint_radius = float(geom.bounding_radius())
    if reference_center is None:
        reference_center = np.asarray(position, dtype=float)
    return SDFGeometryDomainSource(
        geometry=geom,
        pose=Pose6D(
            position=np.asarray(position, dtype=float).copy(),
            orientation=np.asarray(orientation if orientation is not None else [1.0, 0.0, 0.0, 0.0], dtype=float).copy(),
        ),
        name=name,
        hint_radius=float(hint_radius),
        reference_center=np.asarray(reference_center, dtype=float).copy(),
        is_dynamic=bool(is_dynamic),
    )


def make_world(*, bodies, domain_sources=None, gravity=(0.0, -9.81, 0.0), domain_spec=None) -> World:
    return World(
        domain=domain_spec if domain_spec is not None else DomainSpec(cube_size=1.0, cube_height=1.0, top_y=0.0),
        gravity=np.asarray(gravity, dtype=float),
        bodies=list(bodies),
        domain_sources=list(domain_sources) if domain_sources is not None else [],
    )
