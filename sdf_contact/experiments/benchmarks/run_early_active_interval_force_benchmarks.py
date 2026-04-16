from __future__ import annotations
from pathlib import Path
import math, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sdf_contact import (
    SpatialInertia, RigidBody6D, BodyState6D, Pose6D, SDFGeometryDomainSource,
    SphereGeometry, BoxGeometry, PlaneGeometry,
    PolygonPatchConfig, SheetExtractConfig, FormalPressureFieldConfig,
    FormalEndpointBandSheetEvaluator, ContactManager, GlobalImplicitSystemSolver6D,
    IntegratorConfig, make_world, cap_volume,
)

OUT = ROOT / 'results' / 'early_active_interval_force_benchmarks'
OUT.mkdir(parents=True, exist_ok=True)
PATCH_CFG = PolygonPatchConfig(raster_cells=4, max_patch_radius=0.50, support_radius_floor_scale=0.90)
SHEET_CFG = SheetExtractConfig(bisection_steps=8)
K_EQUAL = 40000.0
MASS = 0.05
DT = 0.005
GRAVITY = 9.81
K_EQ = K_EQUAL * K_EQUAL / (K_EQUAL + K_EQUAL)
EVALUATOR = FormalEndpointBandSheetEvaluator(
    patch_cfg=PATCH_CFG,
    sheet_cfg=SHEET_CFG,
    pressure_cfg=FormalPressureFieldConfig(stiffness_default=K_EQUAL, damping_gamma=0.0),
)
CONTACT_MANAGER = ContactManager(EVALUATOR)

def make_solver(report_force: bool):
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, IntegratorConfig(
        dt=DT,
        collect_diagnostics=True,
        adaptive_substepping=False,
        newton_max_iter=4,
        onset_focus_enabled=True,
        onset_refine_substeps=2,
        onset_bisect_iters=3,
        onset_local_refine_steps=0,
        early_active_interval_report_force=report_force,
        early_active_interval_enabled=True,
        early_active_interval_samples=2,
    ))

def _plane_source(name='ground'):
    return SDFGeometryDomainSource(
        geometry=PlaneGeometry((0.0,1.0,0.0),0.0),
        pose=Pose6D(np.zeros(3), np.array([1.0,0.0,0.0,0.0])),
        name=name,
        hint_radius=1.0,
        reference_center=np.zeros(3),
    )

def _sphere_inertia(m,R):
    I=(2.0/5.0)*m*R*R
    return np.diag([I,I,I])

def _box_inertia(m,ext):
    x,y,z=[float(v) for v in ext]
    return (m/12.0)*np.diag([y*y+z*z,x*x+z*z,x*x+y*y])

def _sphere_world(R,h0):
    return make_world(bodies=[RigidBody6D('body', SpatialInertia(MASS,_sphere_inertia(MASS,R)), SphereGeometry(R), BodyState6D(Pose6D(np.array([0.0,R+h0,0.0]), np.array([1.0,0.0,0.0,0.0])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)], domain_sources=[_plane_source()], gravity=(0.0,-GRAVITY,0.0))

def _box_world(ext,h0):
    return make_world(bodies=[RigidBody6D('body', SpatialInertia(MASS,_box_inertia(MASS,ext)), BoxGeometry(ext), BodyState6D(Pose6D(np.array([0.0,0.5*ext[1]+h0,0.0]), np.array([1.0,0.0,0.0,0.0])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)], domain_sources=[_plane_source()], gravity=(0.0,-GRAVITY,0.0))

def _run_case(world, solver, delta_fn, static_force_fn, total_time: float):
    rows=[]
    nsteps = int(round(total_time / DT))
    for step in range(nsteps):
        infos = solver.step_world(world)
        body = world.bodies[0]
        info_map = {int(x['body_index']): x for x in infos}
        Fy = float(np.asarray(info_map.get(0, {}).get('contact_force', np.zeros(3)), dtype=float)[1])
        t = (step + 1) * DT
        y = float(body.state.pose.position[1])
        delta = max(0.0, float(delta_fn(y)))
        static_F = float(static_force_fn(delta))
        rows.append({
            'step': step + 1,
            'time': t,
            'y': y,
            'delta': delta,
            'Fy': Fy,
            'static_Fy': static_F,
            'contact_active': int(Fy > 1.0e-8),
            'rel_err': abs(Fy - static_F) / max(abs(static_F), 1.0e-15) if static_F > 1.0e-15 else 0.0,
            'solver_substeps': int((solver.last_step_diagnostics or {}).get('substeps', 1)),
        })
    return pd.DataFrame(rows)

def _summarize(df: pd.DataFrame):
    active = df[(df['contact_active'] > 0) & (df['static_Fy'] > 1.0e-8)].copy()
    out = {
        'n_steps': int(len(df)),
        'n_active': int(len(active)),
        'max_substeps': int(df['solver_substeps'].max()) if len(df) else 0,
    }
    if len(active):
        out.update({
            'mean_rel_err_active': float(active['rel_err'].mean()),
            'max_rel_err_active': float(active['rel_err'].max()),
            'first_active_time': float(active.iloc[0]['time']),
            'first_active_rel_err': float(active.iloc[0]['rel_err']),
        })
    else:
        out.update({
            'mean_rel_err_active': float('nan'),
            'max_rel_err_active': float('nan'),
            'first_active_time': float('nan'),
            'first_active_rel_err': float('nan'),
        })
    return out

def _plot(df_off, df_on, title, out_png):
    plt.figure(figsize=(7.4,4.6))
    plt.plot(df_off['time'], df_off['Fy'], 'o-', label='report_force off')
    plt.plot(df_on['time'], df_on['Fy'], 's-', label='report_force on')
    plt.plot(df_off['time'], df_off['static_Fy'], '--', label='static theory')
    plt.xlabel('time [s]'); plt.ylabel('Fy'); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def main():
    rows=[]
    # sphere short-horizon
    R=0.12; h0=0.02; total_time=0.075
    df_off = _run_case(_sphere_world(R,h0), make_solver(False), lambda y:R-y, lambda d:K_EQ*cap_volume(R,d), total_time)
    df_on = _run_case(_sphere_world(R,h0), make_solver(True), lambda y:R-y, lambda d:K_EQ*cap_volume(R,d), total_time)
    df_off.to_csv(OUT/'sphere_off.csv', index=False)
    df_on.to_csv(OUT/'sphere_on.csv', index=False)
    _plot(df_off, df_on, 'Sphere early-active interval force reconstruction', OUT/'sphere_plot.png')
    rows.append({'case':'sphere_centered','mode':'report_force_off', **_summarize(df_off)})
    rows.append({'case':'sphere_centered','mode':'report_force_on', **_summarize(df_on)})
    # flat punch short-horizon
    ext=np.array([0.2,0.1,0.15],dtype=float); area=float(ext[0]*ext[2]); h0=0.01; total_time=0.060
    df_off = _run_case(_box_world(ext,h0), make_solver(False), lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d, total_time)
    df_on = _run_case(_box_world(ext,h0), make_solver(True), lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d, total_time)
    df_off.to_csv(OUT/'flat_off.csv', index=False)
    df_on.to_csv(OUT/'flat_on.csv', index=False)
    _plot(df_off, df_on, 'Flat-punch early-active interval force reconstruction', OUT/'flat_plot.png')
    rows.append({'case':'flat_punch_centered','mode':'report_force_off', **_summarize(df_off)})
    rows.append({'case':'flat_punch_centered','mode':'report_force_on', **_summarize(df_on)})
    overview=pd.DataFrame(rows)
    overview.to_csv(OUT/'comparison.csv', index=False)
    lines=['# Early-active interval force reconstruction benchmarks','']
    lines.append('This benchmark keeps the same dynamic trajectory integration scheme and only changes how the first-contact / early-active step reports contact force:')
    lines.append('- report_force_off: solver reports end-of-step contact force')
    lines.append('- report_force_on: solver reports onset-aware active-interval average force on onset step')
    lines.append('')
    for case in ['sphere_centered','flat_punch_centered']:
        lines.append(f'## {case}')
        sub=overview[overview['case']==case]
        for _,row in sub.iterrows():
            lines.append(f"### {row['mode']}")
            lines.append(f"- n_active = {int(row['n_active'])}")
            lines.append(f"- mean_rel_err_active = {row['mean_rel_err_active']:.6f}")
            lines.append(f"- max_rel_err_active = {row['max_rel_err_active']:.6f}")
            lines.append(f"- first_active_time = {row['first_active_time']:.6f}")
            lines.append(f"- first_active_rel_err = {row['first_active_rel_err']:.6f}")
            lines.append(f"- max_substeps = {int(row['max_substeps'])}")
            lines.append('')
    (OUT/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT/'comparison.csv')
    print('saved', OUT/'summary.md')

if __name__ == '__main__':
    main()
