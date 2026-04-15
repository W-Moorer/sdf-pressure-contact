from __future__ import annotations
from pathlib import Path
import math, sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sdf_contact import (
    SpatialInertia, RigidBody6D, BodyState6D, Pose6D, SDFGeometryDomainSource,
    SphereGeometry, BoxGeometry, PlaneGeometry,
    PolygonPatchConfig, SheetExtractConfig, FormalPressureFieldConfig,
    FormalEndpointBandSheetEvaluator, ContactManager, GlobalImplicitSystemSolver6D,
    IntegratorConfig, make_world, cap_volume,
)
OUT = ROOT / 'results' / 'onset_event_benchmarks'
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

def make_solver(onset_focus_enabled: bool):
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, IntegratorConfig(dt=DT, collect_diagnostics=True, onset_focus_enabled=onset_focus_enabled))

def _plane_source(name='ground'):
    return SDFGeometryDomainSource(geometry=PlaneGeometry((0.0,1.0,0.0),0.0), pose=Pose6D(np.zeros(3), np.array([1.0,0.0,0.0,0.0])), name=name, hint_radius=1.0, reference_center=np.zeros(3))

def _sphere_inertia(m,R):
    I=(2.0/5.0)*m*R*R; return np.diag([I,I,I])

def _box_inertia(m,ext):
    x,y,z=[float(v) for v in ext]; return (m/12.0)*np.diag([y*y+z*z,x*x+z*z,x*x+y*y])

def _sphere_world(R,h0):
    return make_world(bodies=[RigidBody6D('body', SpatialInertia(MASS,_sphere_inertia(MASS,R)), SphereGeometry(R), BodyState6D(Pose6D(np.array([0.0,R+h0,0.0]), np.array([1.0,0.0,0.0,0.0])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)], domain_sources=[_plane_source()], gravity=(0.0,-GRAVITY,0.0))

def _box_world(ext,h0):
    return make_world(bodies=[RigidBody6D('body', SpatialInertia(MASS,_box_inertia(MASS,ext)), BoxGeometry(ext), BodyState6D(Pose6D(np.array([0.0,0.5*ext[1]+h0,0.0]), np.array([1.0,0.0,0.0,0.0])), np.zeros(3), np.zeros(3)), linear_damping=0.0, angular_damping=0.0)], domain_sources=[_plane_source()], gravity=(0.0,-GRAVITY,0.0))

def _run_window(case_name, world, solver, delta_fn, static_force_fn, active_steps_target=2, max_steps=25):
    rows=[]; first_active=None; sim_touch=float('nan'); refine_reason=''
    active_seen=0
    for step in range(max_steps):
        infos=solver.step_world(world)
        diag=solver.last_step_diagnostics or {}
        info_map={int(x['body_index']):x for x in infos}
        body=world.bodies[0]
        Fy=float(np.asarray(info_map.get(0,{}).get('contact_force',np.zeros(3)),dtype=float)[1])
        active=Fy>1e-8
        delta=max(0.0,float(delta_fn(float(body.state.pose.position[1]))))
        t=(step+1)*DT
        onset_offset=float(diag.get('onset_time_offset', float('nan')))
        if first_active is None and active:
            first_active=len(rows)
            sim_touch=step*DT + (onset_offset if np.isfinite(onset_offset) else DT)
            refine_reason=str(diag.get('refine_reason',''))
        if active:
            active_seen += 1
        rows.append({
            'time': t,
            'y': float(body.state.pose.position[1]),
            'vy': float(body.state.linear_velocity[1]),
            'delta': delta,
            'Fy': Fy,
            'static_Fy_at_delta': float(static_force_fn(delta)),
            'contact_active': int(active),
            'solver_substeps': int(diag.get('substeps',1)) if isinstance(diag.get('substeps',1),(int,float)) else 1,
            'used_local_refinement': int(bool(diag.get('used_local_refinement', False))),
            'refine_reason': str(diag.get('refine_reason','')),
            'onset_time_offset': onset_offset,
        })
        if first_active is not None and active_seen >= active_steps_target:
            break
    df=pd.DataFrame(rows)
    return df, sim_touch, refine_reason

def _summarize(df, sim_touch, analytic_touch, case, mode, refine_reason):
    if (df['contact_active']>0).any():
        act=df[df['contact_active']>0].copy()
        rel=((act['Fy']-act['static_Fy_at_delta']).abs()/act['static_Fy_at_delta'].abs().clip(lower=1e-15))
        first=act.iloc[0]
        return {
            'case':case,'mode':mode,
            'analytic_touch_time':analytic_touch,
            'sim_touch_time':sim_touch,
            'abs_touch_time_error':abs(sim_touch-analytic_touch),
            'first_active_delta':float(first['delta']),
            'first_active_Fy':float(first['Fy']),
            'first_active_static_Fy':float(first['static_Fy_at_delta']),
            'mean_rel_err_onset_window':float(rel.mean()),
            'max_rel_err_onset_window':float(rel.max()),
            'max_solver_substeps':int(df['solver_substeps'].max()),
            'n_local_refine_steps':int(df['used_local_refinement'].sum()),
            'onset_refine_reason':refine_reason,
        }
    return {'case':case,'mode':mode,'analytic_touch_time':analytic_touch,'sim_touch_time':float('nan'),'abs_touch_time_error':float('nan'),'first_active_delta':float('nan'),'first_active_Fy':float('nan'),'first_active_static_Fy':float('nan'),'mean_rel_err_onset_window':float('nan'),'max_rel_err_onset_window':float('nan'),'max_solver_substeps':0,'n_local_refine_steps':0,'onset_refine_reason':refine_reason}

def _plot(off,on,title,out_png):
    plt.figure(figsize=(7.4,4.6))
    plt.plot(off['time'], off['Fy'], label='Fy onset_focus_off')
    plt.plot(on['time'], on['Fy'], label='Fy onset_focus_on')
    plt.plot(on['time'], on['static_Fy_at_delta'], label='static Fy')
    if len(on):
        ymax=max(float(off['Fy'].max()), float(on['Fy'].max()), float(on['static_Fy_at_delta'].max()), 1.0)
        plt.vlines(on['time'][on['used_local_refinement']>0], 0.0, ymax, colors='k', linestyles=':', alpha=0.25)
    plt.xlabel('time [s]'); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def main():
    rows=[]
    R=0.12; h0s=0.02; analytic_touch=math.sqrt(2.0*h0s/GRAVITY)
    off_df, off_touch, off_reason = _run_window('sphere_centered_onset', _sphere_world(R,h0s), make_solver(False), lambda y:R-y, lambda d:K_EQ*cap_volume(R,d))
    on_df, on_touch, on_reason = _run_window('sphere_centered_onset', _sphere_world(R,h0s), make_solver(True), lambda y:R-y, lambda d:K_EQ*cap_volume(R,d))
    off_df.to_csv(OUT/'sphere_centered_onset_off.csv', index=False)
    on_df.to_csv(OUT/'sphere_centered_onset_on.csv', index=False)
    _plot(off_df,on_df,'Sphere onset window', OUT/'sphere_centered_onset_comparison.png')
    rows.append(_summarize(off_df,off_touch,analytic_touch,'sphere_centered_onset','onset_focus_off',off_reason))
    rows.append(_summarize(on_df,on_touch,analytic_touch,'sphere_centered_onset','onset_focus_on',on_reason))

    ext=np.array([0.2,0.1,0.15],dtype=float); area=float(ext[0]*ext[2]); h0b=0.01; analytic_touch=math.sqrt(2.0*h0b/GRAVITY)
    off_df, off_touch, off_reason = _run_window('flat_punch_centered_onset', _box_world(ext,h0b), make_solver(False), lambda y:0.5*ext[1]-y, lambda d: area*K_EQ*d)
    on_df, on_touch, on_reason = _run_window('flat_punch_centered_onset', _box_world(ext,h0b), make_solver(True), lambda y:0.5*ext[1]-y, lambda d: area*K_EQ*d)
    off_df.to_csv(OUT/'flat_punch_centered_onset_off.csv', index=False)
    on_df.to_csv(OUT/'flat_punch_centered_onset_on.csv', index=False)
    _plot(off_df,on_df,'Flat punch onset window', OUT/'flat_punch_centered_onset_comparison.png')
    rows.append(_summarize(off_df,off_touch,analytic_touch,'flat_punch_centered_onset','onset_focus_off',off_reason))
    rows.append(_summarize(on_df,on_touch,analytic_touch,'flat_punch_centered_onset','onset_focus_on',on_reason))

    overview=pd.DataFrame(rows)
    overview.to_csv(OUT/'comparison.csv', index=False)
    lines=['# Onset-event window benchmarks','']
    lines.append('This benchmark stops after the **first two active contact samples**. It isolates the onset stage rather than the whole release cycle.')
    lines.append('')
    for case in ['sphere_centered_onset','flat_punch_centered_onset']:
        lines.append(f'## {case}')
        sub=overview[overview['case']==case]
        for _,row in sub.iterrows():
            lines.append(f"### {row['mode']}")
            lines.append(f"- sim_touch_time = {row['sim_touch_time']:.6f}")
            lines.append(f"- analytic_touch_time = {row['analytic_touch_time']:.6f}")
            lines.append(f"- abs_touch_time_error = {row['abs_touch_time_error']:.6f}")
            lines.append(f"- first_active_delta = {row['first_active_delta']:.6f}")
            lines.append(f"- first_active_Fy = {row['first_active_Fy']:.6f}")
            lines.append(f"- first_active_static_Fy = {row['first_active_static_Fy']:.6f}")
            lines.append(f"- mean_rel_err_onset_window = {row['mean_rel_err_onset_window']:.6f}")
            lines.append(f"- max_rel_err_onset_window = {row['max_rel_err_onset_window']:.6f}")
            lines.append(f"- max_solver_substeps = {int(row['max_solver_substeps'])}")
            lines.append(f"- n_local_refine_steps = {int(row['n_local_refine_steps'])}")
            lines.append(f"- onset_refine_reason = {row['onset_refine_reason']}")
            lines.append('')
    (OUT/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT/'comparison.csv')
    print('saved', OUT/'summary.md')

if __name__=='__main__':
    main()
