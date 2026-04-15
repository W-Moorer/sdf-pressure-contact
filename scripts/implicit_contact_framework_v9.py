
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional
import math
import numpy as np

# ---------------- math ----------------
def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return np.array([1.0,0.0,0.0,0.0],dtype=float) if n <= 1e-15 else q/n

def quat_mul(q1,q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2
    ],dtype=float)

def quat_from_rotvec(rv: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rv)
    if theta <= 1e-15:
        return quat_normalize(np.array([1.0,0.5*rv[0],0.5*rv[1],0.5*rv[2]],dtype=float))
    axis = rv/theta
    h = 0.5*theta
    return np.array([math.cos(h), *(math.sin(h)*axis)], dtype=float)

def integrate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    return quat_normalize(quat_mul(quat_from_rotvec(dt*omega), q))

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q); w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ], dtype=float)

def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_to_rotmat(q) @ v

def orthonormal_basis_from_normal(n: np.ndarray):
    n = n / max(np.linalg.norm(n), 1e-15)
    a = np.array([0.,0.,1.]) if abs(n[2]) < 0.9 else np.array([0.,1.,0.])
    t1 = np.cross(n,a); t1 /= max(np.linalg.norm(t1),1e-15)
    t2 = np.cross(n,t1); t2 /= max(np.linalg.norm(t2),1e-15)
    return t1,t2

# ---------------- core data ----------------
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
    def copy(self):
        return BodyState6D(Pose6D(self.pose.position.copy(), self.pose.orientation.copy()),
                           self.linear_velocity.copy(), self.angular_velocity.copy())

@dataclass
class SpatialInertia:
    mass: float
    inertia_body: np.ndarray
    def inertia_world(self, q: np.ndarray) -> np.ndarray:
        R = quat_to_rotmat(q)
        return R @ self.inertia_body @ R.T

class Geometry(Protocol):
    def phi_world(self, x, y, z, pose: Pose6D): ...
    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray: ...
    def bounding_radius(self) -> float: ...

class SphereGeometry:
    def __init__(self, radius: float): self.radius = float(radius)
    def bounding_radius(self) -> float: return self.radius
    def phi_world(self, x, y, z, pose: Pose6D):
        c = pose.position
        return np.sqrt((x-c[0])**2 + (y-c[1])**2 + (z-c[2])**2) - self.radius
    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        d = np.array([x,y,z],dtype=float) - pose.position
        n = np.linalg.norm(d)
        return np.array([0.,1.,0.]) if n <= 1e-15 else d/n

class PlaneGeometry:
    def __init__(self, normal=(0,1,0), offset=0.0):
        n = np.array(normal,dtype=float); self.n = n/np.linalg.norm(n); self.offset = float(offset)
    def bounding_radius(self) -> float: return 1.0
    def phi_world(self, x,y,z, pose: Pose6D):
        p = np.array([x,y,z],dtype=float)
        return float(np.dot(self.n,p) - self.offset)
    def normal_world(self, x: float, y: float, z: float, pose: Pose6D) -> np.ndarray:
        return self.n.copy()

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
    def world_markers(self):
        return {m.name: self.state.pose.position + rotate_vec(self.state.pose.orientation, m.local_position) for m in self.markers}
    def clone_with_state(self, state: BodyState6D):
        return RigidBody6D(self.name,self.inertia,self.geometry,state.copy(),self.markers,self.is_static,self.linear_damping,self.angular_damping)

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
    domain_sources: list["SDFGeometryDomainSource"] = field(default_factory=list)

# ---------------- source abstraction ----------------
class SDFSource(Protocol):
    name: str
    is_dynamic: bool
    def phi_world(self, x: float, y: float, z: float) -> float: ...
    def normal_world(self, x: float, y: float, z: float, h: float = 1e-6) -> np.ndarray: ...
    def velocity_world(self, point: np.ndarray) -> np.ndarray: ...
    def reference_center_world(self) -> np.ndarray: ...
    def patch_hint_radius(self) -> float: ...
    def wrench_ref_point(self) -> np.ndarray: ...

@dataclass
class BodySource:
    body_index: int
    body: RigidBody6D
    name: str = ""
    is_dynamic: bool = True
    def __post_init__(self):
        self.name = self.body.name
        self.is_dynamic = not self.body.is_static
    def phi_world(self,x,y,z): return float(self.body.geometry.phi_world(x,y,z,self.body.state.pose))
    def normal_world(self,x,y,z,h=1e-6):
        return np.asarray(self.body.geometry.normal_world(x,y,z,self.body.state.pose),dtype=float)
    def velocity_world(self, point): 
        r = point - self.body.state.pose.position
        return self.body.state.linear_velocity + np.cross(self.body.state.angular_velocity, r)
    def reference_center_world(self): return self.body.state.pose.position.copy()
    def patch_hint_radius(self): return float(self.body.geometry.bounding_radius())
    def wrench_ref_point(self): return self.body.state.pose.position.copy()

@dataclass
class SDFGeometryDomainSource:
    geometry: Geometry
    pose: Pose6D
    name: str = "domain_sdf"
    hint_radius: float = 0.5
    reference_center: Optional[np.ndarray] = None
    is_dynamic: bool = False
    def phi_world(self,x,y,z): return float(self.geometry.phi_world(x,y,z,self.pose))
    def normal_world(self,x,y,z,h=1e-6):
        n = np.asarray(self.geometry.normal_world(x,y,z,self.pose),dtype=float)
        nn = np.linalg.norm(n)
        return n/max(nn,1e-15)
    def velocity_world(self, point): return np.zeros(3,dtype=float)
    def reference_center_world(self):
        return self.pose.position.copy() if self.reference_center is None else self.reference_center.copy()
    def patch_hint_radius(self): return float(self.hint_radius)
    def wrench_ref_point(self): return self.reference_center_world()

# ---------------- configs and contact data ----------------
@dataclass
class UnifiedPairPatchConfig:
    Nuv: int = 10
    quad_order: int = 2
    radius_scale: float = 1.2
    min_patch_radius: float = 0.01
    max_patch_radius: float = 0.2
    ray_span_scale: float = 1.2

@dataclass
class SheetExtractConfig:
    bisection_steps: int = 18
    normal_step: float = 1e-6

@dataclass
class ContactModelConfig:
    stiffness_k: float = 10000.0
    damping_c: float = 100.0

@dataclass
class IntegratorConfig:
    dt: float = 0.02
    newton_max_iter: int = 6
    newton_tol: float = 1e-8
    fd_eps: float = 1e-5
    line_search_factors: tuple[float,...] = (1.0, 0.5, 0.25)

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

# ---------------- unified pipeline ----------------
def _root_from_inside(source: SDFSource, seed: np.ndarray, direction: np.ndarray, smax: float, bisection_steps: int):
    f0 = source.phi_world(*seed)
    if f0 > 0.0: return None
    p1 = seed + smax*direction
    f1 = source.phi_world(*p1)
    if f1 < 0.0: return None
    sl,sr = 0.0,smax
    for _ in range(bisection_steps):
        sm = 0.5*(sl+sr)
        pm = seed + sm*direction
        fm = source.phi_world(*pm)
        if fm > 0.0: sr = sm
        else: sl = sm
    return seed + 0.5*(sl+sr)*direction

def estimate_initial_pair_frame(source_a: SDFSource, source_b: SDFSource, cfg: UnifiedPairPatchConfig):
    ca = source_a.reference_center_world()
    cb = source_b.reference_center_world()
    d = cb-ca
    dist = np.linalg.norm(d)
    n0 = np.array([0.,1.,0.]) if dist <= 1e-15 else d/dist
    ra = source_a.patch_hint_radius()
    rb = source_b.patch_hint_radius()
    xa = _root_from_inside(source_a, ca, +n0, 2.5*ra, 24)
    xb = _root_from_inside(source_b, cb, -n0, 2.5*rb, 24)
    if xa is None: xa = ca + ra*n0
    if xb is None: xb = cb - rb*n0
    gap = float(np.dot(xb-xa, n0))
    depth = max(0.0, -gap)
    if depth <= 0.0: return None
    center0 = 0.5*(xa+xb)
    reff = (ra*rb)/max(ra+rb,1e-15)
    radius = cfg.radius_scale * math.sqrt(max(0.0, 2.0*reff*depth))
    radius = min(cfg.max_patch_radius, max(cfg.min_patch_radius, radius))
    ray_span = cfg.ray_span_scale * (ra + rb + radius)
    t1,t2 = orthonormal_basis_from_normal(n0)
    return center0, n0, t1, t2, radius, ray_span

def build_local_contact_patches(source_a: SDFSource, source_b: SDFSource, cfg: UnifiedPairPatchConfig):
    geom = estimate_initial_pair_frame(source_a, source_b, cfg)
    if geom is None: return None
    center0,n0,t1,t2,radius,ray_span = geom
    xi, wi = np.polynomial.legendre.leggauss(cfg.quad_order)
    du = 2.0*radius/cfg.Nuv
    dv = du
    samples=[]
    for iu in range(cfg.Nuv):
        ul=-radius+iu*du; ur=ul+du
        for iv in range(cfg.Nuv):
            vl=-radius+iv*dv; vr=vl+dv
            for a,wa in zip(xi,wi):
                u=0.5*(ur-ul)*a+0.5*(ur+ul)
                for b,wb in zip(xi,wi):
                    v=0.5*(vr-vl)*b+0.5*(vr+vl)
                    seed = center0 + u*t1 + v*t2
                    w_proj = 0.25*(ur-ul)*(vr-vl)*wa*wb
                    if source_a.phi_world(*seed) < 0.0 and source_b.phi_world(*seed) < 0.0:
                        samples.append(PairPatchSample(np.array([u,v],dtype=float), seed.astype(float), float(w_proj)))
    if not samples: return None
    return PairContactPatches(samples, center0, n0, t1, t2, radius, {"num_pair_patch_points": len(samples), "ray_span": ray_span})

def extract_local_sheet(source_a: SDFSource, source_b: SDFSource, patches: PairContactPatches, cfg: SheetExtractConfig):
    samples=[]; n0=patches.normal0; smax=float(patches.metadata["ray_span"])
    for idx,pp in enumerate(patches.samples):
        xa = _root_from_inside(source_a, pp.world_seed, +n0, smax, cfg.bisection_steps)
        xb = _root_from_inside(source_b, pp.world_seed, -n0, smax, cfg.bisection_steps)
        if xa is None or xb is None: continue
        na = source_a.normal_world(*xa, cfg.normal_step)
        nb = source_b.normal_world(*xb, cfg.normal_step)
        ns = na - nb; nsn = np.linalg.norm(ns)
        ns = patches.normal0.copy() if nsn <= 1e-12 else ns/nsn
        cos_a = abs(float(np.dot(na,n0))); cos_b = abs(float(np.dot(nb,n0)))
        if cos_a < 1e-12 or cos_b < 1e-12: continue
        dA = 0.5*(pp.projected_weight/cos_a + pp.projected_weight/cos_b)
        mid = 0.5*(xa+xb)
        samples.append(PairSheetPoint(mid.astype(float), xa.astype(float), xb.astype(float), na.astype(float), nb.astype(float), ns.astype(float), pp.projected_weight, float(dA), idx))
    return PairSheet(samples, {"num_pair_sheet_points": len(samples)})

def integrate_local_tractions(source_a: SDFSource, source_b: SDFSource, sheet: PairSheet, cfg: ContactModelConfig):
    samples=[]
    for sp in sheet.samples:
        n = sp.sheet_normal
        gap = float(np.dot(sp.point_b - sp.point_a, n))
        depth = max(0.0, -gap)
        va = source_a.velocity_world(sp.point_a)
        vb = source_b.velocity_world(sp.point_b)
        vrel_n = float(np.dot(vb-va, n))
        q = cfg.stiffness_k*depth + cfg.damping_c*max(0.0, -vrel_n)
        fa = -q*n*sp.surface_weight
        fb = +q*n*sp.surface_weight
        ma = np.cross(sp.point_a - source_a.wrench_ref_point(), fa)
        mb = np.cross(sp.point_b - source_b.wrench_ref_point(), fb)
        samples.append(PairTractionSample(sp.midpoint.copy(), n.copy(), float(q), float(sp.surface_weight), fa.astype(float), fb.astype(float), ma.astype(float), mb.astype(float)))
    return PairTractionField(samples, {"num_pair_tractions": len(samples)})

def assemble_pair_wrench(source_a: SDFSource, source_b: SDFSource, tractions: PairTractionField):
    Fa=Fb=Ma=Mb=None
    Fa=np.zeros(3); Fb=np.zeros(3); Ma=np.zeros(3); Mb=np.zeros(3)
    for ts in tractions.samples:
        Fa += ts.force_on_a; Fb += ts.force_on_b
        Ma += ts.moment_a; Mb += ts.moment_b
    return Wrench(Fa,Ma), Wrench(Fb,Mb)

class UnifiedContactManager:
    def __init__(self, pair_patch_cfg: UnifiedPairPatchConfig, sheet_cfg: SheetExtractConfig, contact_cfg: ContactModelConfig):
        self.pair_patch_cfg=pair_patch_cfg; self.sheet_cfg=sheet_cfg; self.contact_cfg=contact_cfg
    def _compute_source_pair(self, source_a: SDFSource, source_b: SDFSource, pair_kind: str):
        patches = build_local_contact_patches(source_a, source_b, self.pair_patch_cfg)
        if patches is None: return None
        sheet = extract_local_sheet(source_a, source_b, patches, self.sheet_cfg)
        if len(sheet.samples)==0: return None
        tr = integrate_local_tractions(source_a, source_b, sheet, self.contact_cfg)
        if len(tr.samples)==0: return None
        wa,wb = assemble_pair_wrench(source_a, source_b, tr)
        meta = {
            "num_pair_patch_points": patches.metadata["num_pair_patch_points"],
            "num_pair_sheet_points": sheet.metadata["num_pair_sheet_points"],
            "num_pair_tractions": tr.metadata["num_pair_tractions"],
            "patch_radius": patches.radius,
        }
        return PairRecord(pair_kind, source_a.name, source_b.name,
                          PairWrenchContribution(pair_kind, source_a.name, wa.force.copy(), wa.moment.copy(), meta.copy()),
                          PairWrenchContribution(pair_kind, source_b.name, wb.force.copy(), wb.moment.copy(), meta.copy()),
                          meta)
    def compute_all_contacts(self, world: World):
        body_sources = [BodySource(i,b) for i,b in enumerate(world.bodies)]
        domain_sources = list(world.domain_sources)
        names = [s.name for s in body_sources] + [s.name for s in domain_sources]
        out = {name: AggregatedContact(np.zeros(3), np.zeros(3), []) for name in names}
        # domain-body
        for ds in domain_sources:
            for bs in body_sources:
                if not bs.is_dynamic: continue
                rec = self._compute_source_pair(ds, bs, "source_source")
                if rec is None: continue
                out[ds.name].pair_records.append(rec); out[bs.name].pair_records.append(rec)
                out[ds.name].total_force += rec.contribution_a.force; out[ds.name].total_moment += rec.contribution_a.moment
                out[bs.name].total_force += rec.contribution_b.force; out[bs.name].total_moment += rec.contribution_b.moment
                for agg, contrib in ((out[ds.name], rec.contribution_a),(out[bs.name], rec.contribution_b)):
                    agg.num_pair_patch_points += contrib.meta["num_pair_patch_points"]
                    agg.num_pair_sheet_points += contrib.meta["num_pair_sheet_points"]
                    agg.num_pair_tractions += contrib.meta["num_pair_tractions"]
        # body-body
        for i in range(len(body_sources)):
            for j in range(i+1,len(body_sources)):
                rec = self._compute_source_pair(body_sources[i], body_sources[j], "source_source")
                if rec is None: continue
                for name, contrib in ((body_sources[i].name, rec.contribution_a),(body_sources[j].name, rec.contribution_b)):
                    out[name].pair_records.append(rec)
                    out[name].total_force += contrib.force; out[name].total_moment += contrib.moment
                    out[name].num_pair_patch_points += contrib.meta["num_pair_patch_points"]
                    out[name].num_pair_sheet_points += contrib.meta["num_pair_sheet_points"]
                    out[name].num_pair_tractions += contrib.meta["num_pair_tractions"]
        return out

# ---------------- global solver ----------------
@dataclass
class ForceAssemblyResult:
    total_force: np.ndarray
    total_moment: np.ndarray
    contact_force: np.ndarray
    contact_moment: np.ndarray
    contact_meta: dict

def accumulate_all_body_forces(world: World, cm: UnifiedContactManager):
    contacts = cm.compute_all_contacts(world)
    out=[]
    for body in world.bodies:
        agg = contacts[body.name]
        F_ext = body.inertia.mass * world.gravity - body.linear_damping * body.state.linear_velocity
        M_ext = -body.angular_damping * body.state.angular_velocity
        out.append(ForceAssemblyResult(
            total_force=F_ext + agg.total_force,
            total_moment=M_ext + agg.total_moment,
            contact_force=agg.total_force.copy(),
            contact_moment=agg.total_moment.copy(),
            contact_meta={
                "num_pairs": len(agg.pair_records),
                "num_pair_patch_points": agg.num_pair_patch_points,
                "num_pair_sheet_points": agg.num_pair_sheet_points,
                "num_pair_tractions": agg.num_pair_tractions,
            }
        ))
    return out

class GlobalImplicitSystemSolver6D:
    def __init__(self, contact_manager: UnifiedContactManager, integ_cfg: IntegratorConfig):
        self.contact_manager = contact_manager
        self.cfg = integ_cfg
    def _dynamic_indices(self, world: World): return [i for i,b in enumerate(world.bodies) if not b.is_static]
    def _pack_unknowns(self, world: World, dyn):
        return np.concatenate([np.concatenate([world.bodies[i].state.linear_velocity, world.bodies[i].state.angular_velocity]) for i in dyn]) if dyn else np.zeros(0)
    def _make_trial_world(self, world: World, dyn, U):
        dt=self.cfg.dt; offs={bi:6*k for k,bi in enumerate(dyn)}; new_bodies=[]
        for i,b in enumerate(world.bodies):
            if i not in offs:
                new_bodies.append(b); continue
            off=offs[i]; v=U[off:off+3]; w=U[off+3:off+6]
            x = b.state.pose.position + dt*v
            q = integrate_quaternion(b.state.pose.orientation, w, dt)
            ns=BodyState6D(Pose6D(x.copy(), q.copy()), v.copy(), w.copy())
            new_bodies.append(b.clone_with_state(ns))
        return World(domain=world.domain, gravity=world.gravity.copy(), bodies=new_bodies, domain_sources=world.domain_sources)
    def _eval_global_residual(self, world: World, dyn, U):
        tw = self._make_trial_world(world,dyn,U)
        forces = accumulate_all_body_forces(tw, self.contact_manager)
        dt=self.cfg.dt; blocks=[]
        for bi in dyn:
            b0=world.bodies[bi]; bt=tw.bodies[bi]; f=forces[bi]
            v0=b0.state.linear_velocity; w0=b0.state.angular_velocity
            v=bt.state.linear_velocity; w=bt.state.angular_velocity
            Iw = bt.inertia.inertia_world(bt.state.pose.orientation)
            Rv = bt.inertia.mass*(v-v0) - dt*f.total_force
            Rw = Iw @ (w-w0) + dt*np.cross(w, Iw@w) - dt*f.total_moment
            blocks.append(np.concatenate([Rv,Rw]))
        return (np.concatenate(blocks) if blocks else np.zeros(0)), tw, forces
    def step_world(self, world: World):
        dyn=self._dynamic_indices(world)
        if not dyn: return []
        U=self._pack_unknowns(world,dyn)
        for _ in range(self.cfg.newton_max_iter):
            R,_,_=self._eval_global_residual(world,dyn,U)
            if np.linalg.norm(R) < self.cfg.newton_tol: break
            ndof=len(U); J=np.zeros((ndof,ndof)); eps=self.cfg.fd_eps
            for k in range(ndof):
                Up=U.copy(); Up[k]+=eps
                Rp,_,_=self._eval_global_residual(world,dyn,Up)
                J[:,k]=(Rp-R)/eps
            try: dU=np.linalg.solve(J,-R)
            except np.linalg.LinAlgError: dU=-R
            curr=np.linalg.norm(R); accepted=False
            for alpha in self.cfg.line_search_factors:
                Un=U+alpha*dU
                Rn,_,_=self._eval_global_residual(world,dyn,Un)
                if np.linalg.norm(Rn) < curr:
                    U=Un; accepted=True; break
            if not accepted: U=U-0.2*R
        fw=self._make_trial_world(world,dyn,U)
        world.bodies = fw.bodies
        final_forces=accumulate_all_body_forces(world,self.contact_manager)
        infos=[]
        for i,b in enumerate(world.bodies):
            if b.is_static: continue
            f=final_forces[i]
            infos.append({"body_index":i,"contact_force":f.contact_force.copy(),"contact_moment":f.contact_moment.copy(),**f.contact_meta})
        return infos

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
    marker_positions: dict

class Simulator:
    def __init__(self, world: World, solver: GlobalImplicitSystemSolver6D):
        self.world=world; self.solver=solver; self.time=0.0; self.log=[]
    def step(self):
        dt=self.solver.cfg.dt; infos=self.solver.step_world(self.world); im={x["body_index"]:x for x in infos}
        for i,b in enumerate(self.world.bodies):
            if b.is_static: continue
            info=im[i]
            self.log.append(SimulationLogEntry(
                time=self.time+dt, body_name=b.name, position=b.state.pose.position.copy(),
                linear_velocity=b.state.linear_velocity.copy(), angular_velocity=b.state.angular_velocity.copy(),
                contact_force=info["contact_force"].copy(), num_pairs=int(info["num_pairs"]),
                num_pair_patch_points=int(info["num_pair_patch_points"]),
                num_pair_sheet_points=int(info["num_pair_sheet_points"]),
                num_pair_tractions=int(info["num_pair_tractions"]),
                marker_positions=b.world_markers()
            ))
        self.time += dt
    def run(self,total_time: float):
        nsteps=int(round(total_time/self.solver.cfg.dt))
        for _ in range(nsteps): self.step()
        return self.log
