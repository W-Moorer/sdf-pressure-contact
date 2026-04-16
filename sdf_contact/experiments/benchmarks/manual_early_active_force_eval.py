from __future__ import annotations
from pathlib import Path
import sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sdf_contact import *
from sdf_contact.benchmarks import cap_volume

OUT = ROOT / 'results' / 'early_active_interval_manual'
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

def solver(enabled, report):
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, IntegratorConfig(
        dt=DT, collect_diagnostics=True, adaptive_substepping=False, newton_max_iter=4,
        onset_focus_enabled=True, onset_refine_substeps=2, onset_bisect_iters=3,
        onset_local_refine_steps=0, early_active_interval_enabled=enabled,
        early_active_interval_report_force=report, early_active_interval_samples=2,
    ))

def plane():
    return SDFGeometryDomainSource(geometry=PlaneGeometry((0.,1.,0.),0.), pose=Pose6D(np.zeros(3), np.array([1.,0.,0.,0.])), name='ground', hint_radius=1.0, reference_center=np.zeros(3))

def sphere_world(R,h0):
    I=(2./5.)*MASS*R*R
    body = RigidBody6D('body', SpatialInertia(MASS,np.diag([I,I,I])), SphereGeometry(R), BodyState6D(Pose6D(np.array([0.,R+h0,0.]), np.array([1.,0.,0.,0.])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)
    return make_world(bodies=[body], domain_sources=[plane()], gravity=(0.,-GRAVITY,0.))

def box_world(ext,h0):
    x,y,z=[float(v) for v in ext]
    I=(MASS/12.)*np.diag([y*y+z*z,x*x+z*z,x*x+y*y])
    body = RigidBody6D('body', SpatialInertia(MASS,I), BoxGeometry(ext), BodyState6D(Pose6D(np.array([0.,0.5*ext[1]+h0,0.]), np.array([1.,0.,0.,0.])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)
    return make_world(bodies=[body], domain_sources=[plane()], gravity=(0.,-GRAVITY,0.))

def run(world, s, nsteps, delta_fn, static_fn):
    rows=[]
    for step in range(nsteps):
        t0=time.time(); infos=s.step_world(world); elapsed=time.time()-t0
        body=world.bodies[0]
        fy=float(np.asarray({int(x['body_index']):x for x in infos}.get(0,{}).get('contact_force',np.zeros(3)),dtype=float)[1])
        y=float(body.state.pose.position[1]); delta=max(0.0,float(delta_fn(y))); static=float(static_fn(delta))
        rows.append({'step':step+1,'time':(step+1)*DT,'y':y,'delta':delta,'Fy':fy,'static_Fy':static,'rel_err':abs(fy-static)/max(abs(static),1e-15) if static>1e-15 else 0.0,'elapsed':elapsed})
        print('step',step+1,'Fy',fy,'rel',rows[-1]['rel_err'],'elapsed',elapsed, flush=True)
    return pd.DataFrame(rows)

def plot(df0, df1, title, path):
    plt.figure(figsize=(7.4,4.6))
    plt.plot(df0['time'], df0['Fy'], 'o-', label='baseline step-end')
    plt.plot(df1['time'], df1['Fy'], 's-', label='active-interval report')
    plt.plot(df0['time'], df0['static_Fy'], '--', label='static theory')
    plt.xlabel('time [s]'); plt.ylabel('Fy'); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()

def main():
    rows=[]
    print('sphere baseline', flush=True)
    df_sb = run(sphere_world(0.12,0.02), solver(False,False), 17, lambda y:0.12-y, lambda d:K_EQ*cap_volume(0.12,d))
    print('sphere active interval', flush=True)
    df_sa = run(sphere_world(0.12,0.02), solver(True,True), 17, lambda y:0.12-y, lambda d:K_EQ*cap_volume(0.12,d))
    df_sb.to_csv(OUT/'sphere_baseline.csv', index=False); df_sa.to_csv(OUT/'sphere_active_interval.csv', index=False)
    plot(df_sb, df_sa, 'Sphere early-active interval report force', OUT/'sphere_plot.png')
    for mode, df in [('baseline', df_sb), ('active_interval', df_sa)]:
        active=df[df['Fy']>1e-8]
        rows.append({'case':'sphere_centered','mode':mode,'first_active_time':float(active.iloc[0]['time']) if len(active) else float('nan'),'first_active_rel_err':float(active.iloc[0]['rel_err']) if len(active) else float('nan'),'mean_rel_err_active':float(active['rel_err'].mean()) if len(active) else float('nan'),'max_rel_err_active':float(active['rel_err'].max()) if len(active) else float('nan')})
    ext=np.array([0.2,0.1,0.15],dtype=float); area=float(ext[0]*ext[2])
    print('flat baseline', flush=True)
    df_fb = run(box_world(ext,0.01), solver(False,False), 10, lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d)
    print('flat active interval', flush=True)
    df_fa = run(box_world(ext,0.01), solver(True,True), 10, lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d)
    df_fb.to_csv(OUT/'flat_baseline.csv', index=False); df_fa.to_csv(OUT/'flat_active_interval.csv', index=False)
    plot(df_fb, df_fa, 'Flat early-active interval report force', OUT/'flat_plot.png')
    for mode, df in [('baseline', df_fb), ('active_interval', df_fa)]:
        active=df[df['Fy']>1e-8]
        rows.append({'case':'flat_punch_centered','mode':mode,'first_active_time':float(active.iloc[0]['time']) if len(active) else float('nan'),'first_active_rel_err':float(active.iloc[0]['rel_err']) if len(active) else float('nan'),'mean_rel_err_active':float(active['rel_err'].mean()) if len(active) else float('nan'),'max_rel_err_active':float(active['rel_err'].max()) if len(active) else float('nan')})
    overview=pd.DataFrame(rows)
    overview.to_csv(OUT/'comparison.csv', index=False)
    lines=['# Early-active interval report-force benchmark','',
           'Baseline uses the usual step-end contact force.',
           'Active-interval uses the onset-aware active-interval averaged reported force on the onset step.','']
    for case in ['sphere_centered','flat_punch_centered']:
        lines.append(f'## {case}')
        sub=overview[overview.case==case]
        for _,r in sub.iterrows():
            lines.append(f"### {r['mode']}")
            lines.append(f"- first_active_time = {r['first_active_time']:.6f}")
            lines.append(f"- first_active_rel_err = {r['first_active_rel_err']:.6f}")
            lines.append(f"- mean_rel_err_active = {r['mean_rel_err_active']:.6f}")
            lines.append(f"- max_rel_err_active = {r['max_rel_err_active']:.6f}")
            lines.append('')
    (OUT/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT/'comparison.csv')
    print('saved', OUT/'summary.md')

if __name__=='__main__':
    main()
