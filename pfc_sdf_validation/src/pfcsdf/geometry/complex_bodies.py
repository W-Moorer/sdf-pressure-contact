from __future__ import annotations

"""Benchmark-specific complex-body helpers.

These 2D signed-distance samplers use scalar ``(x, z)`` queries to build support profiles for
the paper benchmarks. They are intentionally kept separate from the shared 3D
``SignedDistanceGeometry`` contract in ``pfcsdf.geometry.base`` so this refactor does not force
an awkward abstraction onto the specialized profile/cloud benchmark path.
"""

from dataclasses import dataclass
import math
import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class CircleSDF2D:
    center: tuple[float, float]
    radius: float

    def signed_distance(self, x: float, z: float) -> float:
        cx, cz = self.center
        return float(math.hypot(x - cx, z - cz) - self.radius)


@dataclass(frozen=True)
class BoxSDF2D:
    center: tuple[float, float]
    half_extents: tuple[float, float]

    def signed_distance(self, x: float, z: float) -> float:
        cx, cz = self.center
        hx, hz = self.half_extents
        qx = abs(x - cx) - hx
        qz = abs(z - cz) - hz
        ox = max(qx, 0.0)
        oz = max(qz, 0.0)
        outside = math.hypot(ox, oz)
        inside = min(max(qx, qz), 0.0)
        return float(outside + inside)


@dataclass(frozen=True)
class UnionSDF2D:
    parts: tuple[object, ...]

    def signed_distance(self, x: float, z: float) -> float:
        return float(min(part.signed_distance(x, z) for part in self.parts))


@dataclass(frozen=True)
class SupportProfile2D:
    xs: ArrayLike
    zs: ArrayLike
    weights: ArrayLike
    width: float

    def __post_init__(self) -> None:
        xs = np.asarray(self.xs, dtype=float)
        zs = np.asarray(self.zs, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if xs.ndim != 1 or zs.ndim != 1 or weights.ndim != 1:
            raise ValueError('SupportProfile2D arrays must be 1D')
        if not (xs.size == zs.size == weights.size):
            raise ValueError('SupportProfile2D arrays must have same length')
        object.__setattr__(self, 'xs', xs)
        object.__setattr__(self, 'zs', zs)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'width', float(self.width))

    @property
    def n(self) -> int:
        return int(self.xs.size)

    @property
    def mask_shape(self) -> tuple[int, int, int]:
        return (self.n, 1, 1)

    @property
    def x_extent(self) -> float:
        return float(max(abs(self.xs.min()), abs(self.xs.max())))


def _refine_root(shape: UnionSDF2D, x: float, z_lo: float, z_hi: float, *, iters: int = 40) -> float:
    f_lo = shape.signed_distance(x, z_lo)
    f_hi = shape.signed_distance(x, z_hi)
    if not (f_lo > 0.0 and f_hi <= 0.0):
        return float(z_hi)
    lo = z_lo
    hi = z_hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f_mid = shape.signed_distance(x, mid)
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def sample_support_profile(
    shape: UnionSDF2D,
    *,
    x_min: float,
    x_max: float,
    nx: int,
    z_min: float,
    z_max: float,
    nz: int,
    width: float = 0.6,
) -> SupportProfile2D:
    xs = np.linspace(float(x_min), float(x_max), int(nx))
    z_grid = np.linspace(float(z_min), float(z_max), int(nz))
    support_x = []
    support_z = []
    for x in xs:
        vals = np.array([shape.signed_distance(float(x), float(z)) for z in z_grid], dtype=float)
        inside = np.flatnonzero(vals <= 0.0)
        if inside.size == 0:
            continue
        idx = int(inside[0])
        if idx == 0:
            z_b = float(z_grid[0])
        else:
            z_b = _refine_root(shape, float(x), float(z_grid[idx - 1]), float(z_grid[idx]))
        support_x.append(float(x))
        support_z.append(float(z_b))
    support_x = np.asarray(support_x, dtype=float)
    support_z = np.asarray(support_z, dtype=float)
    if support_x.size < 3:
        raise ValueError('support profile sampling produced too few support points')
    dx = np.empty_like(support_x)
    dx[1:-1] = 0.5 * (support_x[2:] - support_x[:-2])
    dx[0] = support_x[1] - support_x[0]
    dx[-1] = support_x[-1] - support_x[-2]
    weights = np.maximum(dx * float(width), 0.0)
    return SupportProfile2D(xs=support_x, zs=support_z, weights=weights, width=float(width))


def build_capsule_flat_edge_body_profile(*, nx: int = 81, width: float = 0.6) -> SupportProfile2D:
    shape = UnionSDF2D(
        parts=(
            CircleSDF2D(center=(-0.12, 0.20), radius=0.18),
            CircleSDF2D(center=(0.12, 0.20), radius=0.18),
            BoxSDF2D(center=(0.0, -0.02), half_extents=(0.15, 0.03)),
            CircleSDF2D(center=(0.22, 0.03), radius=0.07),
        )
    )
    return sample_support_profile(shape, x_min=-0.34, x_max=0.34, nx=nx, z_min=-0.10, z_max=0.50, nz=600, width=width)



@dataclass(frozen=True)
class SupportCloud3D:
    body_points: ArrayLike
    weights: ArrayLike
    grid_shape: tuple[int, int]

    def __post_init__(self) -> None:
        points = np.asarray(self.body_points, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('body_points must have shape (n, 3)')
        if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
            raise ValueError('weights must be 1D and match body_points')
        if len(self.grid_shape) != 2:
            raise ValueError('grid_shape must be (nx, ny)')
        nx, ny = int(self.grid_shape[0]), int(self.grid_shape[1])
        if nx * ny != points.shape[0]:
            raise ValueError('grid_shape must match point count')
        object.__setattr__(self, 'body_points', points)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'grid_shape', (nx, ny))

    @property
    def n(self) -> int:
        return int(self.body_points.shape[0])

    @property
    def mask_shape(self) -> tuple[int, int, int]:
        nx, ny = self.grid_shape
        return (nx, ny, 1)


def _ellipsoid_bottom(x: float, y: float, *, center: tuple[float, float, float], radii: tuple[float, float, float]) -> float | None:
    cx, cy, cz = center
    rx, ry, rz = radii
    q = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2
    if q > 1.0:
        return None
    return float(cz - rz * math.sqrt(max(1.0 - q, 0.0)))


def _sphere_bottom(x: float, y: float, *, center: tuple[float, float, float], radius: float) -> float | None:
    cx, cy, cz = center
    q = (x - cx) ** 2 + (y - cy) ** 2
    if q > radius ** 2:
        return None
    return float(cz - math.sqrt(max(radius ** 2 - q, 0.0)))


def _flat_patch_bottom(x: float, y: float, *, x_range: tuple[float, float], y_range: tuple[float, float], z: float) -> float | None:
    if x_range[0] <= x <= x_range[1] and y_range[0] <= y <= y_range[1]:
        return float(z)
    return None


def build_capsule_flat_edge_body_cloud(*, nx: int = 29, ny: int = 17) -> SupportCloud3D:
    xs = np.linspace(-0.34, 0.34, int(nx))
    ys = np.linspace(-0.20, 0.20, int(ny))
    dx = float(xs[1] - xs[0]) if xs.size > 1 else 0.68
    dy = float(ys[1] - ys[0]) if ys.size > 1 else 0.40
    points = []
    weights = []
    for x in xs:
        for y in ys:
            candidates = [
                _ellipsoid_bottom(x, y, center=(0.0, 0.0, 0.23), radii=(0.29, 0.18, 0.22)),
                _flat_patch_bottom(x, y, x_range=(-0.14, 0.07), y_range=(-0.09, 0.09), z=-0.025),
                _sphere_bottom(x, y, center=(0.18, 0.06, 0.04), radius=0.08),
                _sphere_bottom(x, y, center=(-0.20, -0.05, 0.10), radius=0.06),
            ]
            candidates = [v for v in candidates if v is not None]
            if not candidates:
                z = 0.12
            else:
                z = min(candidates)
            points.append([float(x), float(y), float(z)])
            weights.append(dx * dy)
    return SupportCloud3D(body_points=np.asarray(points, dtype=float), weights=np.asarray(weights, dtype=float), grid_shape=(xs.size, ys.size))
