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

OUT = ROOT / 'results' / 'onset_aligned_force_benchmarks'
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

def _run_until_first_contact(world, solver, delta_fn, static_force_fn, max_steps=30):
    rows=[]
    result=None
    for step in range(max_steps):
        infos = solver.step_world(world)
        diag = solver.last_step_diagnostics or {}
        info_map = {int(x['body_index']): x for x in infos}
        body = world.bodies[0]
        Fy_end = float(np.asarray(info_map.get(0, {}).get('contact_force', np.zeros(3)), dtype=float)[1])
        t_end = (step + 1) * DT
        delta_end = max(0.0, float(delta_fn(float(body.state.pose.position[1]))))
        static_end = float(static_force_fn(delta_end))
        rows.append({'step': step + 1, 'sample': 'step_end', 'time': t_end, 'delta': delta_end, 'Fy': Fy_end, 'static_Fy': static_end, 'rel_err': abs(Fy_end - static_end)/max(abs(static_end),1e-15)})
        if Fy_end > 1e-8 and result is None:
            result = {
                'step_end_time': t_end,
                'step_end_delta': delta_end,
                'step_end_Fy': Fy_end,
                'step_end_static_Fy': static_end,
                'step_end_rel_err': abs(Fy_end - static_end)/max(abs(static_end),1e-15),
                'onset_time_offset': float(diag.get('onset_time_offset', float('nan'))),
                'refine_reason': str(diag.get('refine_reason', '')),
                'solver_substeps': int(diag.get('substeps', 1)) if isinstance(diag.get('substeps',1),(int,float)) else 1,
            }
            oa = diag.get('onset_aligned')
            if isinstance(oa, dict):
                b0 = (oa.get('bodies') or {}).get(0, {})
                pos = np.asarray(b0.get('position', np.zeros(3)), dtype=float)
                force = np.asarray(b0.get('contact_force', np.zeros(3)), dtype=float)
                y0 = float(pos[1]) if pos.size >= 2 else float(body.state.pose.position[1])
                t0 = step * DT + float(oa.get('time_offset', DT))
                d0 = max(0.0, float(delta_fn(y0)))
                Fy0 = float(force[1]) if force.size >= 2 else Fy_end
                static0 = float(static_force_fn(d0))
                result.update({
                    'onset_aligned_time': t0,
                    'onset_aligned_delta': d0,
                    'onset_aligned_Fy': Fy0,
                    'onset_aligned_static_Fy': static0,
                    'onset_aligned_rel_err': abs(Fy0 - static0)/max(abs(static0),1e-15),
                })
                rows.append({'step': step + 1, 'sample': 'onset_aligned', 'time': t0, 'delta': d0, 'Fy': Fy0, 'static_Fy': static0, 'rel_err': abs(Fy0 - static0)/max(abs(static0),1e-15)})
            break
    return pd.DataFrame(rows), result

def _plot(df_off, df_on, title, out_png):
    plt.figure(figsize=(7.4,4.6))
    plt.plot(df_off['time'], df_off['Fy'], 'o-', label='coarse step-end Fy')
    plt.plot(df_off['time'], df_off['static_Fy'], '--', label='coarse static Fy')
    if len(df_on):
        plt.plot(df_on[df_on['sample']=='step_end']['time'], df_on[df_on['sample']=='step_end']['Fy'], 's-', label='onset-focus step-end Fy')
        sel = df_on[df_on['sample']=='onset_aligned']
        if len(sel):
            plt.scatter(sel['time'], sel['Fy'], marker='x', s=80, label='onset-aligned Fy')
            plt.scatter(sel['time'], sel['static_Fy'], marker='+', s=90, label='onset-aligned static Fy')
    plt.xlabel('time [s]'); plt.ylabel('Fy'); plt.title(title); plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def main():
    rows=[]
    # sphere
    R=0.12; h0=0.02; analytic_touch=math.sqrt(2.0*h0/GRAVITY)
    off_df, off_res = _run_until_first_contact(_sphere_world(R,h0), make_solver(False), lambda y:R-y, lambda d:K_EQ*cap_volume(R,d))
    on_df, on_res = _run_until_first_contact(_sphere_world(R,h0), make_solver(True), lambda y:R-y, lambda d:K_EQ*cap_volume(R,d))
    off_df.to_csv(OUT/'sphere_first_contact_off.csv', index=False)
    on_df.to_csv(OUT/'sphere_first_contact_on.csv', index=False)
    _plot(off_df, on_df, 'Sphere first-contact force', OUT/'sphere_first_contact.png')
    rows.append({'case':'sphere_centered','mode':'coarse_step_end','analytic_touch_time':analytic_touch, **(off_res or {})})
    rows.append({'case':'sphere_centered','mode':'onset_localized','analytic_touch_time':analytic_touch, **(on_res or {})})
    # flat
    ext=np.array([0.2,0.1,0.15],dtype=float); area=float(ext[0]*ext[2]); h0=0.01; analytic_touch=math.sqrt(2.0*h0/GRAVITY)
    off_df, off_res = _run_until_first_contact(_box_world(ext,h0), make_solver(False), lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d)
    on_df, on_res = _run_until_first_contact(_box_world(ext,h0), make_solver(True), lambda y:0.5*ext[1]-y, lambda d:area*K_EQ*d)
    off_df.to_csv(OUT/'flat_first_contact_off.csv', index=False)
    on_df.to_csv(OUT/'flat_first_contact_on.csv', index=False)
    _plot(off_df, on_df, 'Flat-punch first-contact force', OUT/'flat_first_contact.png')
    rows.append({'case':'flat_punch_centered','mode':'coarse_step_end','analytic_touch_time':analytic_touch, **(off_res or {})})
    rows.append({'case':'flat_punch_centered','mode':'onset_localized','analytic_touch_time':analytic_touch, **(on_res or {})})
    overview=pd.DataFrame(rows)
    overview.to_csv(OUT/'comparison.csv', index=False)
    lines=['# Onset-aligned first-contact force benchmarks','']
    lines.append('This benchmark compares the first contact force measured at the coarse end-of-step state versus the new onset-localized state reconstructed inside the first active coarse step.')
    lines.append('')
    for case in ['sphere_centered','flat_punch_centered']:
        lines.append(f'## {case}')
        sub=overview[overview['case']==case]
        for _,row in sub.iterrows():
            lines.append(f"### {row['mode']}")
            lines.append(f"- analytic_touch_time = {row['analytic_touch_time']:.6f}")
            lines.append(f"- step_end_time = {row['step_end_time']:.6f}")
            lines.append(f"- step_end_delta = {row['step_end_delta']:.6f}")
            lines.append(f"- step_end_Fy = {row['step_end_Fy']:.6f}")
            lines.append(f"- step_end_static_Fy = {row['step_end_static_Fy']:.6f}")
            lines.append(f"- step_end_rel_err = {row['step_end_rel_err']:.6f}")
            if pd.notna(row.get('onset_aligned_time', np.nan)):
                lines.append(f"- onset_aligned_time = {row['onset_aligned_time']:.6f}")
                lines.append(f"- onset_aligned_delta = {row['onset_aligned_delta']:.6f}")
                lines.append(f"- onset_aligned_Fy = {row['onset_aligned_Fy']:.6f}")
                lines.append(f"- onset_aligned_static_Fy = {row['onset_aligned_static_Fy']:.6f}")
                lines.append(f"- onset_aligned_rel_err = {row['onset_aligned_rel_err']:.6f}")
            lines.append(f"- refine_reason = {row.get('refine_reason','')}")
            lines.append(f"- solver_substeps = {int(row.get('solver_substeps',1))}")
            lines.append('')
    (OUT/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT/'comparison.csv')
    print('saved', OUT/'summary.md')

if __name__ == '__main__':
    main()
