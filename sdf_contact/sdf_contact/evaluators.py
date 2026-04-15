from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from shapely.geometry import LineString, Polygon, box
from shapely.ops import polygonize, triangulate, unary_union
from skimage import measure

from .core import (
    SDFSource,
    PairWrenchContribution,
    PairRecord,
    PairPatchSample,
    PairContactPatches,
    PairSheetPoint,
    PairSheet,
    PairTractionSample,
    PairTractionField,
    BandCellSample,
    BandMechanicsResult,
    SheetPatchGeometry,
    SheetRepresentation,
    Wrench,
    orthonormal_basis_from_normal,
)


@dataclass
class BaselinePatchConfig:
    Nuv: int = 10
    quad_order: int = 2
    radius_scale: float = 1.2
    support_radius_floor_scale: float = 0.90
    min_patch_radius: float = 0.01
    max_patch_radius: float = 0.2
    ray_span_scale: float = 1.2


@dataclass
class PolygonPatchConfig:
    raster_cells: int = 12
    contour_padding_fraction: float = 0.0
    radius_scale: float = 1.2
    support_radius_floor_scale: float = 0.90
    min_patch_radius: float = 0.01
    max_patch_radius: float = 0.18
    ray_span_scale: float = 1.1
    min_polygon_area: float = 1.0e-6
    triangle_rule: str = 'three_point'


@dataclass
class SheetExtractConfig:
    bisection_steps: int = 16
    normal_step: float = 1.0e-6


@dataclass
class ContactModelConfig:
    stiffness_k: float = 10000.0
    damping_c: float = 100.0


@dataclass
class FormalPressureFieldConfig:
    stiffness_default: float = 10000.0
    damping_gamma: float = 0.0
    thickness_default: float = 1.0e9
    ray_scan_steps: int = 64
    source_stiffness: dict[str, float] | None = None
    source_thickness: dict[str, float] | None = None

    def stiffness_for(self, source: SDFSource) -> float:
        if self.source_stiffness is not None and getattr(source, 'name', None) in self.source_stiffness:
            return float(self.source_stiffness[getattr(source, 'name')])
        return float(self.stiffness_default)

    def thickness_for(self, source: SDFSource) -> float:
        if self.source_thickness is not None and getattr(source, 'name', None) in self.source_thickness:
            return float(self.source_thickness[getattr(source, 'name')])
        return float(self.thickness_default)


@dataclass
class SheetRepresentationConfig:
    distance_scale: float = 1.75
    normal_cos_min: float = 0.80


def _root_from_inside(source: SDFSource, seed: np.ndarray, direction: np.ndarray, smax: float, bisection_steps: int):
    f0 = source.phi_world(*seed)
    if f0 > 0.0:
        return None
    p1 = seed + smax * direction
    f1 = source.phi_world(*p1)
    if f1 < 0.0:
        return None
    sl, sr = 0.0, smax
    for _ in range(bisection_steps):
        sm = 0.5 * (sl + sr)
        pm = seed + sm * direction
        fm = source.phi_world(*pm)
        if fm > 0.0:
            sr = sm
        else:
            sl = sm
    return seed + 0.5 * (sl + sr) * direction


def _root_along_ray(source: SDFSource, seed: np.ndarray, direction: np.ndarray, smax: float, scan_steps: int, bisection_steps: int):
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1.0e-15)
    ss = np.linspace(0.0, float(smax), int(max(2, scan_steps)) + 1)
    prev_s = float(ss[0])
    prev_f = float(source.phi_world(*(seed + prev_s * direction)))
    if abs(prev_f) <= 1.0e-12:
        return seed.copy()
    for s in ss[1:]:
        s = float(s)
        curr_p = seed + s * direction
        curr_f = float(source.phi_world(*curr_p))
        if abs(curr_f) <= 1.0e-12:
            return curr_p.astype(float)
        if prev_f * curr_f < 0.0:
            sl, sr = prev_s, s
            fl, fr = prev_f, curr_f
            for _ in range(bisection_steps):
                sm = 0.5 * (sl + sr)
                fm = float(source.phi_world(*(seed + sm * direction)))
                if abs(fm) <= 1.0e-14:
                    sl = sr = sm
                    break
                if fl * fm <= 0.0:
                    sr = sm
                    fr = fm
                else:
                    sl = sm
                    fl = fm
            return (seed + 0.5 * (sl + sr) * direction).astype(float)
        prev_s, prev_f = s, curr_f
    return None


def _bisect_root_on_line(source: SDFSource, seed: np.ndarray, direction: np.ndarray, sl: float, sr: float, bisection_steps: int) -> float:
    fl = float(source.phi_world(*(seed + sl * direction)))
    fr = float(source.phi_world(*(seed + sr * direction)))
    if abs(fl) <= 1.0e-14:
        return float(sl)
    if abs(fr) <= 1.0e-14:
        return float(sr)
    for _ in range(bisection_steps):
        sm = 0.5 * (sl + sr)
        fm = float(source.phi_world(*(seed + sm * direction)))
        if abs(fm) <= 1.0e-14:
            return float(sm)
        if fl * fm <= 0.0:
            sr = sm
            fr = fm
        else:
            sl = sm
            fl = fm
    return float(0.5 * (sl + sr))


def _inside_intervals_along_line(source: SDFSource, seed: np.ndarray, direction: np.ndarray, smax: float, scan_steps: int, bisection_steps: int):
    direction = np.asarray(direction, dtype=float)
    direction = direction / max(np.linalg.norm(direction), 1.0e-15)
    ss = np.linspace(-float(smax), float(smax), int(max(8, scan_steps)) + 1)
    phis = [float(source.phi_world(*(seed + float(s) * direction))) for s in ss]

    roots = []
    for i in range(len(ss) - 1):
        s0 = float(ss[i]); s1 = float(ss[i + 1])
        f0 = float(phis[i]); f1 = float(phis[i + 1])
        if abs(f0) <= 1.0e-12:
            roots.append(s0)
        if f0 * f1 < 0.0:
            roots.append(_bisect_root_on_line(source, seed, direction, s0, s1, bisection_steps))
    if abs(float(phis[-1])) <= 1.0e-12:
        roots.append(float(ss[-1]))

    bounds = [-float(smax)] + sorted(set(round(r, 12) for r in roots)) + [float(smax)]
    intervals = []
    for bl, br in zip(bounds[:-1], bounds[1:]):
        if br - bl <= 1.0e-12:
            continue
        sm = 0.5 * (bl + br)
        fm = float(source.phi_world(*(seed + sm * direction)))
        if fm < 0.0:
            intervals.append((float(bl), float(br)))
    return intervals


def _project_point_to_surface(source: SDFSource, point: np.ndarray, n_hint: np.ndarray | None = None, steps: int = 2):
    q = np.asarray(point, dtype=float).copy()
    n = np.array([0.0, 1.0, 0.0], dtype=float)
    for _ in range(max(1, steps)):
        n = np.asarray(source.normal_world(*q), dtype=float)
        nn = np.linalg.norm(n)
        if nn <= 1.0e-15:
            n = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            n = n / nn
        if n_hint is not None and float(np.dot(n, n_hint)) < 0.0:
            n = -n
        phi = float(source.phi_world(*q))
        q = q - phi * n
    n = np.asarray(source.normal_world(*q), dtype=float)
    nn = np.linalg.norm(n)
    if nn <= 1.0e-15:
        n = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        n = n / nn
    if n_hint is not None and float(np.dot(n, n_hint)) < 0.0:
        n = -n
    return q.astype(float), n.astype(float)



def _candidate_pair_frame_from_seed_a(source_a: SDFSource, source_b: SDFSource, seed_a: np.ndarray):
    qb, nb = _project_point_to_surface(source_b, seed_a)
    qa, na = _project_point_to_surface(source_a, qb, n_hint=-nb)

    for _ in range(2):
        mid = 0.5 * (qa + qb)
        qa, na = _project_point_to_surface(source_a, mid, n_hint=na)
        qb, nb = _project_point_to_surface(source_b, mid, n_hint=nb)

    n0 = na - nb
    n0n = np.linalg.norm(n0)
    if n0n <= 1.0e-12:
        d = qb - qa
        dn = np.linalg.norm(d)
        if dn <= 1.0e-12:
            return None
        n0 = d / dn
    else:
        n0 = n0 / n0n

    gap = float(np.dot(qb - qa, n0))
    depth = max(0.0, -gap)
    if depth <= 0.0:
        return None

    return {
        'point_a': qa,
        'point_b': qb,
        'normal0': n0,
        'depth': depth,
        'seed_mode': 'from_a',
    }



def _candidate_pair_frame_from_seed_b(source_a: SDFSource, source_b: SDFSource, seed_b: np.ndarray):
    qa, na = _project_point_to_surface(source_a, seed_b)
    qb, nb = _project_point_to_surface(source_b, qa, n_hint=-na)

    for _ in range(2):
        mid = 0.5 * (qa + qb)
        qa, na = _project_point_to_surface(source_a, mid, n_hint=na)
        qb, nb = _project_point_to_surface(source_b, mid, n_hint=nb)

    n0 = na - nb
    n0n = np.linalg.norm(n0)
    if n0n <= 1.0e-12:
        d = qb - qa
        dn = np.linalg.norm(d)
        if dn <= 1.0e-12:
            return None
        n0 = d / dn
    else:
        n0 = n0 / n0n

    gap = float(np.dot(qb - qa, n0))
    depth = max(0.0, -gap)
    if depth <= 0.0:
        return None

    return {
        'point_a': qa,
        'point_b': qb,
        'normal0': n0,
        'depth': depth,
        'seed_mode': 'from_b',
    }



def _estimate_initial_pair_frame(source_a: SDFSource, source_b: SDFSource, radius_scale: float, support_radius_floor_scale: float, min_patch_radius: float, max_patch_radius: float, ray_span_scale: float):
    ra = source_a.patch_hint_radius()
    rb = source_b.patch_hint_radius()
    seed_a = np.asarray(source_a.reference_center_world(), dtype=float)
    seed_b = np.asarray(source_b.reference_center_world(), dtype=float)

    if bool(getattr(source_b, 'is_dynamic', False)) and not bool(getattr(source_a, 'is_dynamic', False)):
        ordered = [
            _candidate_pair_frame_from_seed_b(source_a, source_b, seed_b),
            _candidate_pair_frame_from_seed_a(source_a, source_b, seed_a),
        ]
    elif bool(getattr(source_a, 'is_dynamic', False)) and not bool(getattr(source_b, 'is_dynamic', False)):
        ordered = [
            _candidate_pair_frame_from_seed_a(source_a, source_b, seed_a),
            _candidate_pair_frame_from_seed_b(source_a, source_b, seed_b),
        ]
    elif rb <= ra:
        ordered = [
            _candidate_pair_frame_from_seed_b(source_a, source_b, seed_b),
            _candidate_pair_frame_from_seed_a(source_a, source_b, seed_a),
        ]
    else:
        ordered = [
            _candidate_pair_frame_from_seed_a(source_a, source_b, seed_a),
            _candidate_pair_frame_from_seed_b(source_a, source_b, seed_b),
        ]

    best = None
    for cand in ordered:
        if cand is None:
            continue
        if best is None or cand['depth'] > best['depth']:
            best = cand
    if best is None:
        return None

    center0 = 0.5 * (best['point_a'] + best['point_b'])
    n0 = best['normal0']
    depth = best['depth']
    reff = (ra * rb) / max(ra + rb, 1.0e-15)
    radius = radius_scale * math.sqrt(max(0.0, 2.0 * reff * depth))
    radius = max(radius, support_radius_floor_scale * min(ra, rb))
    radius = min(max_patch_radius, max(min_patch_radius, radius))
    ray_span = ray_span_scale * (max(depth, 0.0) + radius + min(ra, rb) + 1.0e-6)
    t1, t2 = orthonormal_basis_from_normal(n0)
    return center0, n0, t1, t2, radius, ray_span


def _uv_to_world(center0: np.ndarray, t1: np.ndarray, t2: np.ndarray, u: float, v: float) -> np.ndarray:
    return center0 + u * t1 + v * t2


def _extract_overlap_polygons(source_a: SDFSource, source_b: SDFSource, center0, t1, t2, radius: float, cfg: PolygonPatchConfig):
    N = cfg.raster_cells
    us = np.linspace(-radius, radius, N + 1)
    vs = np.linspace(-radius, radius, N + 1)
    g = np.zeros((N + 1, N + 1), dtype=float)

    for i, v in enumerate(vs):
        for j, u in enumerate(us):
            p = _uv_to_world(center0, t1, t2, float(u), float(v))
            pa = source_a.phi_world(*p)
            pb = source_b.phi_world(*p)
            g[i, j] = -max(pa, pb)

    contours = measure.find_contours(g, level=0.0)
    lines: list[LineString] = []
    for c in contours:
        if len(c) < 3:
            continue
        pts = []
        for row, col in c:
            u = -radius + (col / N) * 2.0 * radius
            v = -radius + (row / N) * 2.0 * radius
            pts.append((float(u), float(v)))
        try:
            line = LineString(pts)
        except Exception:
            continue
        if line.length > 1.0e-12:
            lines.append(line)

    polys = []
    if lines:
        raw_polys = list(polygonize(lines))
        clip_box = box(-radius, -radius, radius, radius)
        for poly in raw_polys:
            if poly.is_empty:
                continue
            poly = poly.intersection(clip_box)
            if poly.is_empty:
                continue
            if poly.geom_type == 'Polygon':
                candidates = [poly]
            else:
                candidates = [gg for gg in getattr(poly, 'geoms', []) if gg.geom_type == 'Polygon']
            for cand in candidates:
                if cand.area <= cfg.min_polygon_area:
                    continue
                cu, cv = cand.representative_point().x, cand.representative_point().y
                pw = _uv_to_world(center0, t1, t2, cu, cv)
                if source_a.phi_world(*pw) < 0.0 and source_b.phi_world(*pw) < 0.0:
                    polys.append(cand)

    if not polys:
        cell_polys = []
        for i in range(N):
            vc = 0.5 * (vs[i] + vs[i + 1])
            for j in range(N):
                uc = 0.5 * (us[j] + us[j + 1])
                p = _uv_to_world(center0, t1, t2, float(uc), float(vc))
                if source_a.phi_world(*p) < 0.0 and source_b.phi_world(*p) < 0.0:
                    cell_polys.append(box(us[j], vs[i], us[j + 1], vs[i + 1]))
        if cell_polys:
            merged = unary_union(cell_polys)
            if merged.geom_type == 'Polygon':
                polys = [merged]
            else:
                polys = [gg for gg in getattr(merged, 'geoms', []) if gg.geom_type == 'Polygon']
    return polys


def _triangle_quadrature_points(poly_tri: Polygon, rule: str):
    coords = list(poly_tri.exterior.coords)[:-1]
    if len(coords) != 3:
        return []
    p0 = np.array(coords[0], dtype=float)
    p1 = np.array(coords[1], dtype=float)
    p2 = np.array(coords[2], dtype=float)
    area = float(poly_tri.area)
    if area <= 1.0e-16:
        return []
    if rule == 'centroid':
        return [((p0 + p1 + p2) / 3.0, area)]
    bary = [
        (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    ]
    out = []
    for l0, l1, l2 in bary:
        uv = l0 * p0 + l1 * p1 + l2 * p2
        out.append((uv, area / 3.0))
    return out


class _SharedTractionOps:
    sheet_cfg: SheetExtractConfig
    contact_cfg: ContactModelConfig

    def extract_local_sheet(self, source_a: SDFSource, source_b: SDFSource, patches: PairContactPatches) -> PairSheet:
        samples: list[PairSheetPoint] = []
        n0 = patches.normal0
        smax = float(patches.metadata['ray_span'])
        for idx, pp in enumerate(patches.samples):
            xa = _root_from_inside(source_a, pp.world_seed, +n0, smax, self.sheet_cfg.bisection_steps)
            xb = _root_from_inside(source_b, pp.world_seed, -n0, smax, self.sheet_cfg.bisection_steps)
            if xa is None or xb is None:
                continue
            na = source_a.normal_world(*xa, self.sheet_cfg.normal_step)
            nb = source_b.normal_world(*xb, self.sheet_cfg.normal_step)
            ns = na - nb
            nsn = np.linalg.norm(ns)
            ns = patches.normal0.copy() if nsn <= 1.0e-12 else ns / nsn
            cos_a = abs(float(np.dot(na, n0)))
            cos_b = abs(float(np.dot(nb, n0)))
            if cos_a < 1.0e-12 or cos_b < 1.0e-12:
                continue
            dA = 0.5 * (pp.projected_weight / cos_a + pp.projected_weight / cos_b)
            mid = 0.5 * (xa + xb)
            samples.append(
                PairSheetPoint(
                    midpoint=mid.astype(float),
                    point_a=xa.astype(float),
                    point_b=xb.astype(float),
                    normal_a=na.astype(float),
                    normal_b=nb.astype(float),
                    sheet_normal=ns.astype(float),
                    projected_weight=pp.projected_weight,
                    surface_weight=float(dA),
                    source_index=idx,
                )
            )
        return PairSheet(samples=samples, metadata={'num_pair_sheet_points': len(samples)})

    def integrate_local_tractions(self, source_a: SDFSource, source_b: SDFSource, sheet: PairSheet) -> PairTractionField:
        samples: list[PairTractionSample] = []
        for sp in sheet.samples:
            n = sp.sheet_normal
            gap = float(np.dot(sp.point_b - sp.point_a, n))
            depth = max(0.0, -gap)
            va = source_a.velocity_world(sp.point_a)
            vb = source_b.velocity_world(sp.point_b)
            vrel_n = float(np.dot(vb - va, n))
            q = self.contact_cfg.stiffness_k * depth + self.contact_cfg.damping_c * max(0.0, -vrel_n)
            fa = -q * n * sp.surface_weight
            fb = +q * n * sp.surface_weight
            ma = np.cross(sp.point_a - source_a.wrench_ref_point(), fa)
            mb = np.cross(sp.point_b - source_b.wrench_ref_point(), fb)
            samples.append(
                PairTractionSample(
                    midpoint=sp.midpoint.copy(),
                    sheet_normal=n.copy(),
                    pressure=float(q),
                    area_weight=float(sp.surface_weight),
                    force_on_a=fa.astype(float),
                    force_on_b=fb.astype(float),
                    moment_a=ma.astype(float),
                    moment_b=mb.astype(float),
                )
            )
        return PairTractionField(samples=samples, metadata={'num_pair_tractions': len(samples)})

    @staticmethod
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


class BaselineGridLocalEvaluator(_SharedTractionOps):
    def __init__(self, patch_cfg: BaselinePatchConfig | None = None, sheet_cfg: SheetExtractConfig | None = None, contact_cfg: ContactModelConfig | None = None):
        self.patch_cfg = patch_cfg or BaselinePatchConfig()
        self.sheet_cfg = sheet_cfg or SheetExtractConfig()
        self.contact_cfg = contact_cfg or ContactModelConfig()

    def build_local_contact_patches(self, source_a: SDFSource, source_b: SDFSource):
        geom = _estimate_initial_pair_frame(
            source_a,
            source_b,
            self.patch_cfg.radius_scale,
            self.patch_cfg.support_radius_floor_scale,
            self.patch_cfg.min_patch_radius,
            self.patch_cfg.max_patch_radius,
            self.patch_cfg.ray_span_scale,
        )
        if geom is None:
            return None
        center0, n0, t1, t2, radius, ray_span = geom
        xi, wi = np.polynomial.legendre.leggauss(self.patch_cfg.quad_order)
        du = 2.0 * radius / self.patch_cfg.Nuv
        dv = du
        samples = []
        for iu in range(self.patch_cfg.Nuv):
            ul = -radius + iu * du
            ur = ul + du
            for iv in range(self.patch_cfg.Nuv):
                vl = -radius + iv * dv
                vr = vl + dv
                for a, wa in zip(xi, wi):
                    u = 0.5 * (ur - ul) * a + 0.5 * (ur + ul)
                    for b, wb in zip(xi, wi):
                        v = 0.5 * (vr - vl) * b + 0.5 * (vr + vl)
                        seed = center0 + u * t1 + v * t2
                        w_proj = 0.25 * (ur - ul) * (vr - vl) * wa * wb
                        if source_a.phi_world(*seed) < 0.0 and source_b.phi_world(*seed) < 0.0:
                            samples.append(PairPatchSample(np.array([u, v], dtype=float), seed.astype(float), float(w_proj)))
        if not samples:
            return None
        return PairContactPatches(samples, center0, n0, t1, t2, radius, {'num_pair_patch_points': len(samples), 'ray_span': ray_span})

    def compute_source_pair(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str = 'baseline_grid'):
        patches = self.build_local_contact_patches(source_a, source_b)
        if patches is None:
            return None
        sheet = self.extract_local_sheet(source_a, source_b, patches)
        if len(sheet.samples) == 0:
            return None
        tr = self.integrate_local_tractions(source_a, source_b, sheet)
        if len(tr.samples) == 0:
            return None
        wa, wb = self.assemble_pair_wrench(source_a, source_b, tr)
        meta = {
            'num_pair_patch_points': patches.metadata['num_pair_patch_points'],
            'num_pair_sheet_points': sheet.metadata['num_pair_sheet_points'],
            'num_pair_tractions': tr.metadata['num_pair_tractions'],
            'patch_radius': patches.radius,
        }
        return PairRecord(
            pair_kind,
            source_a.name,
            source_b.name,
            PairWrenchContribution(pair_kind, source_a.name, wa.force.copy(), wa.moment.copy(), meta.copy()),
            PairWrenchContribution(pair_kind, source_b.name, wb.force.copy(), wb.moment.copy(), meta.copy()),
            meta,
        )


class PolygonHighAccuracyLocalEvaluator(_SharedTractionOps):
    def __init__(self, patch_cfg: PolygonPatchConfig | None = None, sheet_cfg: SheetExtractConfig | None = None, contact_cfg: ContactModelConfig | None = None):
        self.patch_cfg = patch_cfg or PolygonPatchConfig()
        self.sheet_cfg = sheet_cfg or SheetExtractConfig()
        self.contact_cfg = contact_cfg or ContactModelConfig()

    def build_local_contact_patches(self, source_a: SDFSource, source_b: SDFSource):
        geom = _estimate_initial_pair_frame(
            source_a,
            source_b,
            self.patch_cfg.radius_scale,
            self.patch_cfg.support_radius_floor_scale,
            self.patch_cfg.min_patch_radius,
            self.patch_cfg.max_patch_radius,
            self.patch_cfg.ray_span_scale,
        )
        if geom is None:
            return None
        center0, n0, t1, t2, radius, ray_span = geom
        radius *= (1.0 + self.patch_cfg.contour_padding_fraction)
        polys = _extract_overlap_polygons(source_a, source_b, center0, t1, t2, radius, self.patch_cfg)
        if not polys:
            return None
        patch_samples: list[PairPatchSample] = []
        num_triangles = 0
        for poly in polys:
            tris = triangulate(poly)
            for tri in tris:
                tri_clip = tri.intersection(poly)
                if tri_clip.is_empty:
                    continue
                tri_list = [tri_clip] if tri_clip.geom_type == 'Polygon' else [gg for gg in getattr(tri_clip, 'geoms', []) if gg.geom_type == 'Polygon']
                for tr in tri_list:
                    if tr.area <= self.patch_cfg.min_polygon_area:
                        continue
                    num_triangles += 1
                    for uv, w_proj in _triangle_quadrature_points(tr, self.patch_cfg.triangle_rule):
                        seed = _uv_to_world(center0, t1, t2, float(uv[0]), float(uv[1]))
                        patch_samples.append(PairPatchSample(np.asarray(uv, dtype=float), seed.astype(float), float(w_proj)))
        if not patch_samples:
            return None
        return PairContactPatches(
            samples=patch_samples,
            center=center0,
            normal0=n0,
            tangent1=t1,
            tangent2=t2,
            radius=float(radius),
            metadata={
                'num_pair_patch_points': len(patch_samples),
                'num_pair_polygons': len(polys),
                'num_pair_triangles': num_triangles,
                'ray_span': float(ray_span),
            },
        )

    def compute_source_pair(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str = 'polygon_high_accuracy'):
        patches = self.build_local_contact_patches(source_a, source_b)
        if patches is None:
            return None
        sheet = self.extract_local_sheet(source_a, source_b, patches)
        if len(sheet.samples) == 0:
            return None
        tr = self.integrate_local_tractions(source_a, source_b, sheet)
        if len(tr.samples) == 0:
            return None
        wa, wb = self.assemble_pair_wrench(source_a, source_b, tr)
        meta = {
            'num_pair_patch_points': patches.metadata['num_pair_patch_points'],
            'num_pair_sheet_points': sheet.metadata['num_pair_sheet_points'],
            'num_pair_tractions': tr.metadata['num_pair_tractions'],
            'num_pair_polygons': patches.metadata.get('num_pair_polygons', 0),
            'num_pair_triangles': patches.metadata.get('num_pair_triangles', 0),
            'patch_radius': patches.radius,
        }
        return PairRecord(
            pair_kind,
            source_a.name,
            source_b.name,
            PairWrenchContribution(pair_kind, source_a.name, wa.force.copy(), wa.moment.copy(), meta.copy()),
            PairWrenchContribution(pair_kind, source_b.name, wb.force.copy(), wb.moment.copy(), meta.copy()),
            meta,
        )


class FormalPressureFieldLocalEvaluator(PolygonHighAccuracyLocalEvaluator):
    """
    Closer to the formal linear-pressure theory than the legacy spring-gap traction.

    It keeps the same high-accuracy outer architecture
    (patch frame -> polygonized support -> quadrature -> per-column local solve),
    but changes the constitutive and sheet layers to the formal linear-pressure version.
    """

    def __init__(self, patch_cfg: PolygonPatchConfig | None = None, sheet_cfg: SheetExtractConfig | None = None, pressure_cfg: FormalPressureFieldConfig | None = None):
        super().__init__(patch_cfg=patch_cfg, sheet_cfg=sheet_cfg, contact_cfg=ContactModelConfig(stiffness_k=1.0, damping_c=0.0))
        self.pressure_cfg = pressure_cfg or FormalPressureFieldConfig()

    def _column_surfaces(self, source_a: SDFSource, source_b: SDFSource, seed: np.ndarray, n0: np.ndarray, smax: float):
        ia = _inside_intervals_along_line(source_a, seed, n0, smax, self.pressure_cfg.ray_scan_steps, self.sheet_cfg.bisection_steps)
        ib = _inside_intervals_along_line(source_b, seed, n0, smax, self.pressure_cfg.ray_scan_steps, self.sheet_cfg.bisection_steps)
        if not ia or not ib:
            return None
        best = None
        for a0, a1 in ia:
            for b0, b1 in ib:
                lo = max(a0, b0)
                hi = min(a1, b1)
                if hi - lo <= 1.0e-14:
                    continue
                score = hi - lo
                if best is None or score > best['closure']:
                    lower_from = 'a' if a0 >= b0 else 'b'
                    upper_from = 'a' if a1 <= b1 else 'b'
                    best = {
                        'closure': float(score),
                        's_lo': float(lo),
                        's_hi': float(hi),
                        'lower_from': lower_from,
                        'upper_from': upper_from,
                    }
        if best is None:
            return None
        if best['lower_from'] == best['upper_from']:
            return None
        x_lo = seed + best['s_lo'] * n0
        x_hi = seed + best['s_hi'] * n0
        if best['upper_from'] == 'a' and best['lower_from'] == 'b':
            sA, sB = best['s_hi'], best['s_lo']
            xa, xb = x_hi, x_lo
        elif best['upper_from'] == 'b' and best['lower_from'] == 'a':
            sA, sB = best['s_lo'], best['s_hi']
            xa, xb = x_lo, x_hi
        else:
            return None
        return {
            'xa': xa.astype(float),
            'xb': xb.astype(float),
            'sA': float(sA),
            'sB': float(sB),
            'closure': abs(float(sA - sB)),
        }

    def build_local_contact_patches(self, source_a: SDFSource, source_b: SDFSource):
        geom = _estimate_initial_pair_frame(
            source_a,
            source_b,
            self.patch_cfg.radius_scale,
            self.patch_cfg.support_radius_floor_scale,
            self.patch_cfg.min_patch_radius,
            self.patch_cfg.max_patch_radius,
            self.patch_cfg.ray_span_scale,
        )
        if geom is None:
            return None
        center0, n0, t1, t2, radius, ray_span = geom
        radius *= (1.0 + self.patch_cfg.contour_padding_fraction)

        N = self.patch_cfg.raster_cells
        us = np.linspace(-radius, radius, N + 1)
        vs = np.linspace(-radius, radius, N + 1)
        active_cells = []
        for i in range(N):
            vc = 0.5 * (vs[i] + vs[i + 1])
            for j in range(N):
                uc = 0.5 * (us[j] + us[j + 1])
                seed = _uv_to_world(center0, t1, t2, float(uc), float(vc))
                hit = self._column_surfaces(source_a, source_b, seed, n0, ray_span)
                if hit is not None:
                    active_cells.append(box(us[j], vs[i], us[j + 1], vs[i + 1]))

        if not active_cells:
            return None
        merged = unary_union(active_cells)
        polys = [merged] if merged.geom_type == 'Polygon' else [gg for gg in getattr(merged, 'geoms', []) if gg.geom_type == 'Polygon']

        patch_samples: list[PairPatchSample] = []
        num_triangles = 0
        for poly in polys:
            if poly.area <= self.patch_cfg.min_polygon_area:
                continue
            tris = triangulate(poly)
            for tri in tris:
                tri_clip = tri.intersection(poly)
                if tri_clip.is_empty:
                    continue
                tri_list = [tri_clip] if tri_clip.geom_type == 'Polygon' else [gg for gg in getattr(tri_clip, 'geoms', []) if gg.geom_type == 'Polygon']
                for tr in tri_list:
                    if tr.area <= self.patch_cfg.min_polygon_area:
                        continue
                    num_triangles += 1
                    for uv, w_proj in _triangle_quadrature_points(tr, self.patch_cfg.triangle_rule):
                        seed = _uv_to_world(center0, t1, t2, float(uv[0]), float(uv[1]))
                        patch_samples.append(PairPatchSample(np.asarray(uv, dtype=float), seed.astype(float), float(w_proj)))

        if not patch_samples:
            return None
        return PairContactPatches(
            samples=patch_samples,
            center=center0,
            normal0=n0,
            tangent1=t1,
            tangent2=t2,
            radius=float(radius),
            metadata={
                'num_pair_patch_points': len(patch_samples),
                'num_pair_polygons': len(polys),
                'num_pair_triangles': num_triangles,
                'ray_span': float(ray_span),
            },
        )

    def extract_local_sheet(self, source_a: SDFSource, source_b: SDFSource, patches: PairContactPatches) -> PairSheet:
        samples: list[PairSheetPoint] = []
        n0 = patches.normal0
        smax = float(patches.metadata['ray_span'])
        kA = self.pressure_cfg.stiffness_for(source_a)
        kB = self.pressure_cfg.stiffness_for(source_b)
        HA = self.pressure_cfg.thickness_for(source_a)
        HB = self.pressure_cfg.thickness_for(source_b)
        keq = (kA * kB) / max(kA + kB, 1.0e-15)

        for idx, pp in enumerate(patches.samples):
            hit = self._column_surfaces(source_a, source_b, pp.world_seed, n0, smax)
            if hit is None:
                continue
            xa = hit['xa']
            xb = hit['xb']
            sA = hit['sA']
            sB = hit['sB']
            closure = hit['closure']
            dA = min(HA, (kB / max(kA + kB, 1.0e-15)) * closure)
            dB = min(HB, (kA / max(kA + kB, 1.0e-15)) * closure)
            s_star = (kA * sA + kB * sB) / max(kA + kB, 1.0e-15)
            xs = pp.world_seed + s_star * n0

            na = np.asarray(source_a.normal_world(*xs, self.sheet_cfg.normal_step), dtype=float)
            nb = np.asarray(source_b.normal_world(*xs, self.sheet_cfg.normal_step), dtype=float)
            gh = -kA * na + kB * nb
            ghn = np.linalg.norm(gh)
            ns = n0.copy() if ghn <= 1.0e-12 else gh / ghn
            if float(np.dot(ns, n0)) < 0.0:
                ns = -ns
            cos_sheet = abs(float(np.dot(ns, n0)))
            if cos_sheet <= 1.0e-12:
                continue
            dA_sheet = pp.projected_weight / cos_sheet
            pressure = keq * closure
            samples.append(
                PairSheetPoint(
                    midpoint=xs.astype(float),
                    point_a=xa.astype(float),
                    point_b=xb.astype(float),
                    normal_a=na.astype(float),
                    normal_b=nb.astype(float),
                    sheet_normal=ns.astype(float),
                    projected_weight=float(pp.projected_weight),
                    surface_weight=float(dA_sheet),
                    source_index=idx,
                    closure=float(closure),
                    equilibrium_pressure=float(pressure),
                    column_normal=n0.astype(float).copy(),
                )
            )
        return PairSheet(samples=samples, metadata={'num_pair_sheet_points': len(samples)})

    def integrate_local_tractions(self, source_a: SDFSource, source_b: SDFSource, sheet: PairSheet) -> PairTractionField:
        samples: list[PairTractionSample] = []
        gamma = float(self.pressure_cfg.damping_gamma)
        for sp in sheet.samples:
            n = sp.sheet_normal
            pbar = float(sp.equilibrium_pressure)
            vn = 0.0
            if gamma > 0.0:
                va = source_a.velocity_world(sp.midpoint)
                vb = source_b.velocity_world(sp.midpoint)
                vn = float(np.dot(vb - va, n))
            q = pbar * (1.0 + gamma * max(0.0, -vn))
            fa = -q * n * sp.surface_weight
            fb = +q * n * sp.surface_weight
            ma = np.cross(sp.midpoint - source_a.wrench_ref_point(), fa)
            mb = np.cross(sp.midpoint - source_b.wrench_ref_point(), fb)
            samples.append(
                PairTractionSample(
                    midpoint=sp.midpoint.copy(),
                    sheet_normal=n.copy(),
                    pressure=float(q),
                    area_weight=float(sp.surface_weight),
                    force_on_a=fa.astype(float),
                    force_on_b=fb.astype(float),
                    moment_a=ma.astype(float),
                    moment_b=mb.astype(float),
                )
            )
        return PairTractionField(samples=samples, metadata={'num_pair_tractions': len(samples)})

    def compute_source_pair(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str = 'formal_pressure_field'):
        patches = self.build_local_contact_patches(source_a, source_b)
        if patches is None:
            return None
        sheet = self.extract_local_sheet(source_a, source_b, patches)
        if len(sheet.samples) == 0:
            return None
        tr = self.integrate_local_tractions(source_a, source_b, sheet)
        if len(tr.samples) == 0:
            return None
        wa, wb = self.assemble_pair_wrench(source_a, source_b, tr)
        meta = {
            'num_pair_patch_points': patches.metadata['num_pair_patch_points'],
            'num_pair_sheet_points': sheet.metadata['num_pair_sheet_points'],
            'num_pair_tractions': tr.metadata['num_pair_tractions'],
            'num_pair_polygons': patches.metadata.get('num_pair_polygons', 0),
            'num_pair_triangles': patches.metadata.get('num_pair_triangles', 0),
            'patch_radius': patches.radius,
            'formal_stiffness_a': self.pressure_cfg.stiffness_for(source_a),
            'formal_stiffness_b': self.pressure_cfg.stiffness_for(source_b),
        }
        return PairRecord(
            pair_kind,
            source_a.name,
            source_b.name,
            PairWrenchContribution(pair_kind, source_a.name, wa.force.copy(), wa.moment.copy(), meta.copy()),
            PairWrenchContribution(pair_kind, source_b.name, wb.force.copy(), wb.moment.copy(), meta.copy()),
            meta,
        )



def _cluster_band_cells_to_patches(band: BandMechanicsResult, cfg: SheetRepresentationConfig) -> SheetRepresentation:
    cells = list(band.cells)
    if not cells:
        return SheetRepresentation(patches=[], metadata={'num_sheet_patches': 0, 'num_sheet_cells': 0})

    pts = [c.midpoint for c in cells]
    normals = [c.sheet_normal / max(np.linalg.norm(c.sheet_normal), 1.0e-15) for c in cells]
    scales = [math.sqrt(max(c.sheet_area, 1.0e-15)) for c in cells]
    n = len(cells)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(pts[i] - pts[j]))
            lim = cfg.distance_scale * 0.5 * (scales[i] + scales[j])
            if dist <= lim and float(np.dot(normals[i], normals[j])) >= cfg.normal_cos_min:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    patches: list[SheetPatchGeometry] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        comp = []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in adj[k]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)

        area = float(sum(cells[k].sheet_area for k in comp))
        weights = np.array([cells[k].sheet_area for k in comp], dtype=float)
        centers = np.array([cells[k].midpoint for k in comp], dtype=float)
        centroid = np.average(centers, axis=0, weights=weights)
        nsum = np.sum([cells[k].sheet_area * cells[k].sheet_normal for k in comp], axis=0)
        nrm = np.linalg.norm(nsum)
        normal = np.array([0.0, 1.0, 0.0], dtype=float) if nrm <= 1.0e-15 else nsum / nrm
        bbox_min = np.min(centers, axis=0)
        bbox_max = np.max(centers, axis=0)
        pweights = np.array([cells[k].equilibrium_pressure * cells[k].sheet_area for k in comp], dtype=float)
        if float(np.sum(np.abs(pweights))) <= 1.0e-15:
            pressure_center = centroid.copy()
        else:
            pressure_center = np.average(centers, axis=0, weights=pweights)
        patches.append(SheetPatchGeometry(
            area=area,
            centroid=centroid.astype(float),
            normal=normal.astype(float),
            bbox_min=bbox_min.astype(float),
            bbox_max=bbox_max.astype(float),
            pressure_center=pressure_center.astype(float),
            num_cells=len(comp),
        ))

    return SheetRepresentation(
        patches=patches,
        metadata={
            'num_sheet_patches': len(patches),
            'num_sheet_cells': len(cells),
            'total_sheet_area': float(sum(p.area for p in patches)),
        },
    )


class FormalEndpointBandSheetEvaluator(FormalPressureFieldLocalEvaluator):
    """
    A more complete endpoint-style evaluator.

    It keeps the shared high-accuracy footprint pipeline, but explicitly separates:
      1) band mechanics / local-normal accumulator cells,
      2) zero-thickness sheet representation recovery,
      3) pair wrench assembly from sheet traction points.
    """

    def __init__(self, patch_cfg: PolygonPatchConfig | None = None, sheet_cfg: SheetExtractConfig | None = None,
                 pressure_cfg: FormalPressureFieldConfig | None = None,
                 representation_cfg: SheetRepresentationConfig | None = None):
        super().__init__(patch_cfg=patch_cfg, sheet_cfg=sheet_cfg, pressure_cfg=pressure_cfg)
        self.representation_cfg = representation_cfg or SheetRepresentationConfig()

    def compute_band_mechanics(self, source_a: SDFSource, source_b: SDFSource, patches: PairContactPatches) -> BandMechanicsResult:
        cells: list[BandCellSample] = []
        n0 = patches.normal0
        smax = float(patches.metadata['ray_span'])
        kA = self.pressure_cfg.stiffness_for(source_a)
        kB = self.pressure_cfg.stiffness_for(source_b)
        HA = self.pressure_cfg.thickness_for(source_a)
        HB = self.pressure_cfg.thickness_for(source_b)
        for pp in patches.samples:
            hit = self._column_surfaces(source_a, source_b, pp.world_seed, n0, smax)
            if hit is None:
                continue
            closure = float(hit['closure'])
            if closure <= 0.0:
                continue
            sA = float(hit['sA'])
            sB = float(hit['sB'])
            s_star = float((kA * sA + kB * sB) / max(kA + kB, 1.0e-15))
            xs = pp.world_seed + s_star * n0
            na = np.asarray(source_a.normal_world(*xs, self.sheet_cfg.normal_step), dtype=float)
            nb = np.asarray(source_b.normal_world(*xs, self.sheet_cfg.normal_step), dtype=float)
            gh = -kA * na + kB * nb
            ghn = np.linalg.norm(gh)
            ns = n0.copy() if ghn <= 1.0e-12 else gh / ghn
            if float(np.dot(ns, n0)) < 0.0:
                ns = -ns
            cos_sheet = abs(float(np.dot(ns, n0)))
            if cos_sheet <= 1.0e-12:
                continue
            dA = float(pp.projected_weight / cos_sheet)
            dA_eff = float(min(dA, pp.projected_weight * (1.0 / cos_sheet)))
            dAi = min(HA, (kB / max(kA + kB, 1.0e-15)) * closure)
            dBi = min(HB, (kA / max(kA + kB, 1.0e-15)) * closure)
            pA = kA * dAi
            pB = kB * dBi
            pbar = 0.5 * (pA + pB)
            I_force = pbar / cos_sheet
            I_area = 1.0 / cos_sheet
            I_s = s_star / cos_sheet
            q = pbar
            fa = -q * ns * dA_eff
            fb = +q * ns * dA_eff
            ma = np.cross(xs - source_a.wrench_ref_point(), fa)
            mb = np.cross(xs - source_b.wrench_ref_point(), fb)
            cells.append(BandCellSample(
                midpoint=xs.astype(float),
                sheet_normal=ns.astype(float),
                projected_weight=float(pp.projected_weight),
                sheet_area=dA_eff,
                closure=closure,
                equilibrium_pressure=float(pbar),
                s_star=s_star,
                I_force=float(I_force),
                I_area=float(I_area),
                I_s=float(I_s),
                force_on_a=fa.astype(float),
                force_on_b=fb.astype(float),
                moment_a=ma.astype(float),
                moment_b=mb.astype(float),
            ))
        return BandMechanicsResult(cells=cells, metadata={'num_band_cells': len(cells)})

    def recover_sheet_representation(self, band: BandMechanicsResult) -> SheetRepresentation:
        return _cluster_band_cells_to_patches(band, self.representation_cfg)

    def tractions_from_band(self, band: BandMechanicsResult) -> PairTractionField:
        samples: list[PairTractionSample] = []
        for cell in band.cells:
            samples.append(PairTractionSample(
                midpoint=cell.midpoint.copy(),
                sheet_normal=cell.sheet_normal.copy(),
                pressure=float(cell.equilibrium_pressure),
                area_weight=float(cell.sheet_area),
                force_on_a=cell.force_on_a.copy(),
                force_on_b=cell.force_on_b.copy(),
                moment_a=cell.moment_a.copy(),
                moment_b=cell.moment_b.copy(),
            ))
        return PairTractionField(samples=samples, metadata={'num_pair_tractions': len(samples)})

    def compute_source_pair_bundle(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str = 'formal_endpoint_band_sheet'):
        patches = self.build_local_contact_patches(source_a, source_b)
        if patches is None:
            return None
        band = self.compute_band_mechanics(source_a, source_b, patches)
        if len(band.cells) == 0:
            return None
        sheet_repr = self.recover_sheet_representation(band)
        tr = self.tractions_from_band(band)
        wa, wb = self.assemble_pair_wrench(source_a, source_b, tr)
        meta = {
            'num_pair_patch_points': patches.metadata['num_pair_patch_points'],
            'num_pair_sheet_points': len(band.cells),
            'num_pair_tractions': tr.metadata['num_pair_tractions'],
            'num_pair_polygons': patches.metadata.get('num_pair_polygons', 0),
            'num_pair_triangles': patches.metadata.get('num_pair_triangles', 0),
            'num_band_cells': band.metadata['num_band_cells'],
            'num_sheet_patches': sheet_repr.metadata.get('num_sheet_patches', 0),
            'total_sheet_area': sheet_repr.metadata.get('total_sheet_area', 0.0),
            'patch_radius': patches.radius,
            'formal_stiffness_a': self.pressure_cfg.stiffness_for(source_a),
            'formal_stiffness_b': self.pressure_cfg.stiffness_for(source_b),
        }
        rec = PairRecord(
            pair_kind,
            source_a.name,
            source_b.name,
            PairWrenchContribution(pair_kind, source_a.name, wa.force.copy(), wa.moment.copy(), meta.copy()),
            PairWrenchContribution(pair_kind, source_b.name, wb.force.copy(), wb.moment.copy(), meta.copy()),
            meta,
        )
        return {
            'patches': patches,
            'band': band,
            'sheet_representation': sheet_repr,
            'tractions': tr,
            'pair_record': rec,
        }

    def compute_source_pair(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str = 'formal_endpoint_band_sheet'):
        bundle = self.compute_source_pair_bundle(source_a, source_b, pair_kind=pair_kind)
        return None if bundle is None else bundle['pair_record']
