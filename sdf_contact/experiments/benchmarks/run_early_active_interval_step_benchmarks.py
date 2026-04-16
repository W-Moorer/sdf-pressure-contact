from __future__ import annotations
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sdf_contact import *
from sdf_contact.benchmarks import cap_volume

OUT = ROOT / 'results' / 'early_active_interval_step_benchmarks'
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

def _cfg(enabled: bool, report: bool):
    return IntegratorConfig(
        dt=DT, collect_diagnostics=True,
        adaptive_substepping=False,
        newton_max_iter=4,
        onset_focus_enabled=True,
        onset_refine_substeps=2,
        onset_bisect_iters=3,
        onset_local_refine_steps=0,
        early_active_interval_enabled=enabled,
        early_active_interval_report_force=report,
        early_active_interval_samples=2,
    )

def _solver(enabled: bool, report: bool):
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, _cfg(enabled, report))

def _plane_source():
    return SDFGeometryDomainSource(geometry=PlaneGeometry((0.,1.,0.),0.), pose=Pose6D(np.zeros(3), np.array([1.,0.,0.,0.])), name='ground', hint_radius=1.0, reference_center=np.zeros(3))

def _sphere_inertia(m,R):
    I=(2./5.)*m*R*R
    return np.diag([I,I,I])

def _box_inertia(m,ext):
    x,y,z=[float(v) for v in ext]
    return (m/12.)*np.diag([y*y+z*z, x*x+z*z, x*x+y*y])

def _sphere_world(R,h0):
    body = RigidBody6D('body', SpatialInertia(MASS,_sphere_inertia(MASS,R)), SphereGeometry(R), BodyState6D(Pose6D(np.array([0.,R+h0,0.]), np.array([1.,0.,0.,0.])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)
    return make_world(bodies=[body], domain_sources=[_plane_source()], gravity=(0.,-GRAVITY,0.))

def _box_world(ext,h0):
    body = RigidBody6D('body', SpatialInertia(MASS,_box_inertia(MASS,ext)), BoxGeometry(ext), BodyState6D(Pose6D(np.array([0.,0.5*ext[1]+h0,0.]), np.array([1.,0.,0.,0.])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)
    return make_world(bodies=[body], domain_sources=[_plane_source()], gravity=(0.,-GRAVITY,0.))

def _run(world, solver, nsteps, delta_fn, static_fn):
    rows=[]
    for step in range(nsteps):
        infos = solver.step_world(world)
        body=world.bodies[0]
        fy=float(np.asarray({int(x['body_index']):x for x in infos}.get(0,{}).get('contact_force',np.zeros(3)),dtype=float)[1])
        y=float(body.state.pose.position[1]); delta=max(0.0,float(delta_fn(y))); static=float(static_fn(delta))
        rows.append({'step':step+1,'time':(step+1)*DT,'y':y,'delta':delta,'Fy':fy,'static_Fy':static,'rel_err':abs(fy-static)/max(abs(static),1e-15) if static>1e-15 else 0.0})
    return pd.DataFrame(rows)

def _plot(df_a, df_b, title, out_png):
    plt.figure(figsize=(7.4,4.6))
    plt.plot(df_a['time'], df_a['Fy'], 'o-', label='baseline step force')
    plt.plot(df_b['time'], df_b['Fy'], 's-', label='active-interval report force')
    plt.plot(df_a['time'], df_a['static_Fy'], '--', label='static theory')
    plt.xlabel('time [s]'); plt.ylabel('Fy'); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def main():
    rows=[]
    # sphere: compare at first few active steps
    R=0.12
    df_off = _run(_sphere_world(R,0.02), _solver(False, False), 17, lambda y:R-y, lambda d:K_EQ*cap_volume(R,d))
    df_on  = _run(_sphere_world(R,0.02), _solver(True, True), 17, lambda y:R-y, lambda d:K_EQ*cap_volume(R,d))
    df_off.to_csv(OUT/'sphere_baseline.csv', index=False)
    df_on.to_csv(OUT/'sphere_active_interval.csv', index=False)
    _plot(df_off, df_on, 'Sphere onset/early-active force', OUT/'sphere_plot.png')
    for mode, df in [('baseline', df_off), ('active_interval', df_on)]:
        active=df[df['Fy']>1e-8]
        rows.append({'case':'sphere_centered','mode':mode,'first_active_time':float(active.iloc[0]['time']) if len(active) else float('nan'),'first_active_rel_err':float(active.iloc[0]['rel_err']) if len(active) else float('nan'),'mean_rel_err_active':float(active['rel_err'].mean()) if len(active) else float('nan'),'max_rel_err_active':float(active['rel_err'].max()) if len(active) else float('nan')})
    # flat: compare onset step only (step 10 is first active and currently the tractable early-active window)
    ext=np.array([0.2,0.1,0.15],dtype=float); area=float(ext[0]*ext[2])
    df_off = _run(_box_world(ext,0.01), _solver(False, False), 10, lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d)
    df_on  = _run(_box_world(ext,0.01), _solver(True, True), 10, lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d)
    df_off.to_csv(OUT/'flat_baseline.csv', index=False)
    df_on.to_csv(OUT/'flat_active_interval.csv', index=False)
    _plot(df_off, df_on, 'Flat punch onset/early-active force', OUT/'flat_plot.png')
    for mode, df in [('baseline', df_off), ('active_interval', df_on)]:
        active=df[df['Fy']>1e-8]
        rows.append({'case':'flat_punch_centered','mode':mode,'first_active_time':float(active.iloc[0]['time']) if len(active) else float('nan'),'first_active_rel_err':float(active.iloc[0]['rel_err']) if len(active) else float('nan'),'mean_rel_err_active':float(active['rel_err'].mean()) if len(active) else float('nan'),'max_rel_err_active':float(active['rel_err'].max()) if len(active) else float('nan')})
    overview=pd.DataFrame(rows)
    overview.to_csv(OUT/'comparison.csv', index=False)
    lines=['# Early-active interval force reconstruction (step benchmark)','',
           'Baseline = normal step-end reported contact force.',
           'Active_interval = onset-aware active-interval averaged reported force on the onset step (same dynamics/state update, different reported force reconstruction).','']
    for case in ['sphere_centered','flat_punch_centered']:
        lines.append(f'## {case}')
        sub=overview[overview['case']==case]
        for _,row in sub.iterrows():
            lines.append(f"### {row['mode']}")
            lines.append(f"- first_active_time = {row['first_active_time']:.6f}")
            lines.append(f"- first_active_rel_err = {row['first_active_rel_err']:.6f}")
            lines.append(f"- mean_rel_err_active = {row['mean_rel_err_active']:.6f}")
            lines.append(f"- max_rel_err_active = {row['max_rel_err_active']:.6f}")
            lines.append('')
    (OUT/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT/'comparison.csv')
    print('saved', OUT/'summary.md')

if __name__ == '__main__':
    main()
