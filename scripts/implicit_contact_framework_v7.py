from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol
import math
import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    if n <= 1.0e-15:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q) @ v


def orthonormal_basis_from_normal(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = n / max(np.linalg.norm(n), 1.0e-15)
    a = np.array([0.0, 0.0, 1.0], dtype=float) if abs(n[2]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
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


@dataclass
class SpatialInertia:
    mass: float
    inertia_body: np.ndarray


class Geometry(Protocol):
    def phi_world(self, x, y, z, pose: Pose6D):
        ...

    def bounding_radius(self) -> float:
        ...


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
        return {m.name: self.state.pose.position + rotate_vec(self.state.pose.orientation, m.local_position) for m in self.markers}


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


class SphereGeometry:
    def __init__(self, radius: float):
        self.radius = float(radius)

    def bounding_radius(self) -> float:
        return self.radius

    def phi_world(self, x, y, z, pose: Pose6D):
        c = pose.position
        return np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - self.radius

    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        c = pose.position
        d = np.array([x, y, z], dtype=float) - c
        n = np.linalg.norm(d)
        if n <= 1.0e-15:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return d / n


class SDFSource(Protocol):
    name: str
    is_dynamic: bool

    def phi_world(self, x: float, y: float, z: float) -> float:
        ...

    def normal_world(self, x: float, y: float, z: float, h: float = 1.0e-6) -> np.ndarray:
        ...

    def velocity_world(self, point: np.ndarray) -> np.ndarray:
        ...

    def reference_center_world(self) -> np.ndarray:
        ...

    def patch_hint_radius(self) -> float:
        ...

    def wrench_ref_point(self) -> np.ndarray:
        ...


@dataclass
class BodySource:
    body_index: int
    body: RigidBody6D
    name: str = ""
    is_dynamic: bool = True

    def __post_init__(self):
        self.name = self.body.name
        self.is_dynamic = not self.body.is_static

    def phi_world(self, x: float, y: float, z: float) -> float:
        return float(self.body.geometry.phi_world(x, y, z, self.body.state.pose))

    def normal_world(self, x: float, y: float, z: float, h: float = 1.0e-6) -> np.ndarray:
        geom = self.body.geometry
        if hasattr(geom, "normal_world"):
            n = np.asarray(geom.normal_world(x, y, z, self.body.state.pose), dtype=float)
            nn = np.linalg.norm(n)
            if nn > 1.0e-15:
                return n / nn
        px = (self.phi_world(x + h, y, z) - self.phi_world(x - h, y, z)) / (2.0 * h)
        py = (self.phi_world(x, y + h, z) - self.phi_world(x, y - h, z)) / (2.0 * h)
        pz = (self.phi_world(x, y, z + h) - self.phi_world(x, y, z - h)) / (2.0 * h)
        g = np.array([px, py, pz], dtype=float)
        ng = np.linalg.norm(g)
        return g / max(ng, 1.0e-15)

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
class TopPlaneDomainSource:
    domain: DomainSpec
    name: str = "domain_top"
    is_dynamic: bool = False

    def phi_world(self, x: float, y: float, z: float) -> float:
        return float(y - self.domain.top_y)

    def normal_world(self, x: float, y: float, z: float, h: float = 1.0e-6) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0], dtype=float)

    def velocity_world(self, point: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def reference_center_world(self) -> np.ndarray:
        return np.array([0.0, self.domain.top_y, 0.0], dtype=float)

    def patch_hint_radius(self) -> float:
        return self.domain.cube_size / 2.0

    def wrench_ref_point(self) -> np.ndarray:
        return np.array([0.0, self.domain.top_y, 0.0], dtype=float)


@dataclass
class UnifiedPairPatchConfig:
    Nuv: int = 12
    quad_order: int = 3
    radius_scale: float = 1.25
    min_patch_radius: float = 0.01
    max_patch_radius: float = 0.2
    ray_span_scale: float = 1.2


@dataclass
class SheetExtractConfig:
    bisection_steps: int = 22
    normal_step: float = 1.0e-6


@dataclass
class ContactModelConfig:
    stiffness_k: float = 15000.0
    damping_c: float = 120.0


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


def _root_from_inside(source: SDFSource, seed: np.ndarray, direction: np.ndarray, smax: float, bisection_steps: int) -> np.ndarray | None:
    f0 = source.phi_world(seed[0], seed[1], seed[2])
    if f0 > 0.0:
        return None
    p1 = seed + smax * direction
    f1 = source.phi_world(p1[0], p1[1], p1[2])
    if f1 < 0.0:
        return None
    sl, sr = 0.0, smax
    for _ in range(bisection_steps):
        sm = 0.5 * (sl + sr)
        pm = seed + sm * direction
        fm = source.phi_world(pm[0], pm[1], pm[2])
        if fm > 0.0:
            sr = sm
        else:
            sl = sm
    return seed + 0.5 * (sl + sr) * direction


def estimate_initial_pair_frame(source_a: SDFSource, source_b: SDFSource, cfg: UnifiedPairPatchConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | None:
    # special handling for plane-domain as source_a
    if isinstance(source_a, TopPlaneDomainSource):
        n0 = np.array([0.0, 1.0, 0.0], dtype=float)
        cb = source_b.reference_center_world()
        rb = source_b.patch_hint_radius()
        root_b = _root_from_inside(source_b, cb, -n0, 2.5 * rb, 24)
        if root_b is None:
            root_b = cb - rb * n0
        pa = np.array([root_b[0], source_a.domain.top_y, root_b[2]], dtype=float)
        depth = max(0.0, source_a.domain.top_y - root_b[1])
        center0 = 0.5 * (pa + root_b)
        radius = cfg.radius_scale * math.sqrt(max(0.0, 2.0 * rb * depth))
        radius = min(cfg.max_patch_radius, max(cfg.min_patch_radius, radius))
        ray_span = cfg.ray_span_scale * (rb + radius + max(depth, 0.01))
        t1, t2 = orthonormal_basis_from_normal(n0)
        return center0, n0, t1, t2, radius, ray_span

    if isinstance(source_b, TopPlaneDomainSource):
        # orient normal from a to b (toward plane) by swapping and flipping
        geom = estimate_initial_pair_frame(source_b, source_a, cfg)
        if geom is None:
            return None
        center0, n0, t1, t2, radius, ray_span = geom
        return center0, -n0, t1, -t2, radius, ray_span

    ca = source_a.reference_center_world()
    cb = source_b.reference_center_world()
    d = cb - ca
    dist = np.linalg.norm(d)
    if dist <= 1.0e-15:
        n0 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        n0 = d / dist

    ra = source_a.patch_hint_radius()
    rb = source_b.patch_hint_radius()
    xa = _root_from_inside(source_a, ca, +n0, 2.5 * ra, 24)
    xb = _root_from_inside(source_b, cb, -n0, 2.5 * rb, 24)
    if xa is None:
        xa = ca + ra * n0
    if xb is None:
        xb = cb - rb * n0

    gap = float(np.dot(xb - xa, n0))
    depth = max(0.0, -gap)
    if depth <= 0.0:
        return None

    center0 = 0.5 * (xa + xb)
    reff = (ra * rb) / max(ra + rb, 1.0e-15)
    radius = cfg.radius_scale * math.sqrt(max(0.0, 2.0 * reff * depth))
    radius = min(cfg.max_patch_radius, max(cfg.min_patch_radius, radius))
    ray_span = cfg.ray_span_scale * (ra + rb + radius)
    t1, t2 = orthonormal_basis_from_normal(n0)
    return center0, n0, t1, t2, radius, ray_span


def build_local_contact_patches(source_a: SDFSource, source_b: SDFSource, cfg: UnifiedPairPatchConfig) -> PairContactPatches | None:
    geom = estimate_initial_pair_frame(source_a, source_b, cfg)
    if geom is None:
        return None
    center0, n0, t1, t2, radius, ray_span = geom

    xi, wi = np.polynomial.legendre.leggauss(cfg.quad_order)
    du = 2.0 * radius / cfg.Nuv
    dv = du
    samples: list[PairPatchSample] = []

    for iu in range(cfg.Nuv):
        ul = -radius + iu * du
        ur = ul + du
        for iv in range(cfg.Nuv):
            vl = -radius + iv * dv
            vr = vl + dv
            for a, wa in zip(xi, wi):
                u = 0.5 * (ur - ul) * a + 0.5 * (ur + ul)
                for b, wb in zip(xi, wi):
                    v = 0.5 * (vr - vl) * b + 0.5 * (vr + vl)
                    seed = center0 + u * t1 + v * t2
                    w_proj = 0.25 * (ur - ul) * (vr - vl) * wa * wb
                    if source_a.phi_world(seed[0], seed[1], seed[2]) < 0.0 and source_b.phi_world(seed[0], seed[1], seed[2]) < 0.0:
                        samples.append(PairPatchSample(np.array([u, v], dtype=float), seed.astype(float), float(w_proj)))

    if not samples:
        return None
    return PairContactPatches(samples=samples, center=center0, normal0=n0, tangent1=t1, tangent2=t2, radius=radius, metadata={"num_pair_patch_points": len(samples), "ray_span": ray_span})


def extract_local_sheet(source_a: SDFSource, source_b: SDFSource, patches: PairContactPatches, cfg: SheetExtractConfig) -> PairSheet:
    samples: list[PairSheetPoint] = []
    n0 = patches.normal0
    smax = float(patches.metadata["ray_span"])

    for idx, pp in enumerate(patches.samples):
        xa = _root_from_inside(source_a, pp.world_seed, +n0, smax, cfg.bisection_steps)
        xb = _root_from_inside(source_b, pp.world_seed, -n0, smax, cfg.bisection_steps)
        if xa is None or xb is None:
            continue

        na = source_a.normal_world(xa[0], xa[1], xa[2], cfg.normal_step)
        nb = source_b.normal_world(xb[0], xb[1], xb[2], cfg.normal_step)
        ns = na - nb
        nsn = np.linalg.norm(ns)
        ns = patches.normal0.copy() if nsn <= 1.0e-12 else ns / nsn

        cos_a = abs(float(np.dot(na, n0)))
        cos_b = abs(float(np.dot(nb, n0)))
        if cos_a < 1.0e-12 or cos_b < 1.0e-12:
            continue

        dA = 0.5 * (pp.projected_weight / cos_a + pp.projected_weight / cos_b)
        mid = 0.5 * (xa + xb)
        samples.append(PairSheetPoint(mid.astype(float), xa.astype(float), xb.astype(float), na.astype(float), nb.astype(float), ns.astype(float), pp.projected_weight, float(dA), idx))

    return PairSheet(samples=samples, metadata={"num_pair_sheet_points": len(samples)})


def integrate_local_tractions(source_a: SDFSource, source_b: SDFSource, sheet: PairSheet, cfg: ContactModelConfig) -> PairTractionField:
    samples: list[PairTractionSample] = []
    for sp in sheet.samples:
        n = sp.sheet_normal
        gap = float(np.dot(sp.point_b - sp.point_a, n))
        depth = max(0.0, -gap)
        va = source_a.velocity_world(sp.point_a)
        vb = source_b.velocity_world(sp.point_b)
        vrel_n = float(np.dot(vb - va, n))
        q = cfg.stiffness_k * depth + cfg.damping_c * max(0.0, -vrel_n)
        fa = -q * n * sp.surface_weight
        fb = +q * n * sp.surface_weight
        ma = np.cross(sp.point_a - source_a.wrench_ref_point(), fa)
        mb = np.cross(sp.point_b - source_b.wrench_ref_point(), fb)
        samples.append(PairTractionSample(sp.midpoint.copy(), n.copy(), float(q), float(sp.surface_weight), fa.astype(float), fb.astype(float), ma.astype(float), mb.astype(float)))
    return PairTractionField(samples=samples, metadata={"num_pair_tractions": len(samples)})


def assemble_pair_wrench(source_a: SDFSource, source_b: SDFSource, tractions: PairTractionField) -> tuple[Wrench, Wrench]:
    Fa = np.zeros(3, dtype=float)
    Fb = np.zeros(3, dtype=float)
    Ma = np.zeros(3, dtype=float)
    Mb = np.zeros(3, dtype=float)
    for ts in tractions.samples:
        Fa += ts.force_on_a
        Fb += ts.force_on_b
        Ma += ts.moment_a
        Mb += ts.moment_b
    return Wrench(Fa, Ma), Wrench(Fb, Mb)


class UnifiedContactManager:
    """
    Unified source-source contact manager.

    Both body-domain and body-body are dispatched into the same pipeline:
        build_local_contact_patches
        -> extract_local_sheet
        -> integrate_local_tractions
        -> assemble_pair_wrench
    """
    def __init__(self, pair_patch_cfg: UnifiedPairPatchConfig, sheet_cfg: SheetExtractConfig, contact_cfg: ContactModelConfig):
        self.pair_patch_cfg = pair_patch_cfg
        self.sheet_cfg = sheet_cfg
        self.contact_cfg = contact_cfg

    def _compute_source_pair(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str) -> PairRecord | None:
        patches = build_local_contact_patches(source_a, source_b, self.pair_patch_cfg)
        if patches is None:
            return None
        sheet = extract_local_sheet(source_a, source_b, patches, self.sheet_cfg)
        if len(sheet.samples) == 0:
            return None
        tr = integrate_local_tractions(source_a, source_b, sheet, self.contact_cfg)
        if len(tr.samples) == 0:
            return None
        wa, wb = assemble_pair_wrench(source_a, source_b, tr)
        meta = {
            "num_pair_patch_points": patches.metadata["num_pair_patch_points"],
            "num_pair_sheet_points": sheet.metadata["num_pair_sheet_points"],
            "num_pair_tractions": tr.metadata["num_pair_tractions"],
            "patch_radius": patches.radius,
        }
        return PairRecord(
            pair_kind=pair_kind,
            source_a_name=source_a.name,
            source_b_name=source_b.name,
            contribution_a=PairWrenchContribution(pair_kind, source_a.name, wa.force.copy(), wa.moment.copy(), meta.copy()),
            contribution_b=PairWrenchContribution(pair_kind, source_b.name, wb.force.copy(), wb.moment.copy(), meta.copy()),
            meta=meta,
        )

    def compute_all_contacts(self, world: World) -> dict[str, AggregatedContact]:
        domain_source = TopPlaneDomainSource(world.domain)
        body_sources = [BodySource(i, b) for i, b in enumerate(world.bodies)]

        names = [s.name for s in body_sources] + [domain_source.name]
        out = {name: AggregatedContact(np.zeros(3, dtype=float), np.zeros(3, dtype=float), []) for name in names}

        # body-domain via unified source-source pipeline
        for src in body_sources:
            if not src.is_dynamic:
                continue
            rec = self._compute_source_pair(domain_source, src, "source_source")
            if rec is None:
                continue
            out[domain_source.name].pair_records.append(rec)
            out[src.name].pair_records.append(rec)
            out[domain_source.name].total_force += rec.contribution_a.force
            out[domain_source.name].total_moment += rec.contribution_a.moment
            out[src.name].total_force += rec.contribution_b.force
            out[src.name].total_moment += rec.contribution_b.moment
            for agg, contrib in [(out[domain_source.name], rec.contribution_a), (out[src.name], rec.contribution_b)]:
                agg.num_pair_patch_points += contrib.meta["num_pair_patch_points"]
                agg.num_pair_sheet_points += contrib.meta["num_pair_sheet_points"]
                agg.num_pair_tractions += contrib.meta["num_pair_tractions"]

        # body-body via same unified pipeline
        for i in range(len(body_sources)):
            for j in range(i + 1, len(body_sources)):
                rec = self._compute_source_pair(body_sources[i], body_sources[j], "source_source")
                if rec is None:
                    continue
                sa = body_sources[i].name
                sb = body_sources[j].name
                out[sa].pair_records.append(rec)
                out[sb].pair_records.append(rec)
                out[sa].total_force += rec.contribution_a.force
                out[sa].total_moment += rec.contribution_a.moment
                out[sb].total_force += rec.contribution_b.force
                out[sb].total_moment += rec.contribution_b.moment
                for agg, contrib in [(out[sa], rec.contribution_a), (out[sb], rec.contribution_b)]:
                    agg.num_pair_patch_points += contrib.meta["num_pair_patch_points"]
                    agg.num_pair_sheet_points += contrib.meta["num_pair_sheet_points"]
                    agg.num_pair_tractions += contrib.meta["num_pair_tractions"]

        return out
