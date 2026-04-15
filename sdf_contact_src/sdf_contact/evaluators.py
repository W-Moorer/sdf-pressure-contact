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
    Wrench,
    orthonormal_basis_from_normal,
)


@dataclass
class BaselinePatchConfig:
    Nuv: int = 10
    quad_order: int = 2
    radius_scale: float = 1.2
    min_patch_radius: float = 0.01
    max_patch_radius: float = 0.2
    ray_span_scale: float = 1.2


@dataclass
class PolygonPatchConfig:
    raster_cells: int = 12
    contour_padding_fraction: float = 0.0
    radius_scale: float = 1.2
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



def _estimate_initial_pair_frame(source_a: SDFSource, source_b: SDFSource, radius_scale: float, min_patch_radius: float, max_patch_radius: float, ray_span_scale: float):
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
