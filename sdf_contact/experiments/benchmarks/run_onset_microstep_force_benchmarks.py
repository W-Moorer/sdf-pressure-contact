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

OUT = ROOT / 'results' / 'onset_microstep_force_benchmarks'
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
    cfg = IntegratorConfig(dt=DT, collect_diagnostics=True, onset_focus_enabled=onset_focus_enabled)
    cfg.onset_microstep_fraction = 0.8
    cfg.onset_microstep_min_fraction = 0.10
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, cfg)


def _plane_source(name='ground'):
    return SDFGeometryDomainSource(
        geometry=PlaneGeometry((0.0, 1.0, 0.0), 0.0),
        pose=Pose6D(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        name=name,
        hint_radius=1.0,
        reference_center=np.zeros(3),
    )


def _sphere_inertia(m, R):
    I = (2.0 / 5.0) * m * R * R
    return np.diag([I, I, I])


def _box_inertia(m, ext):
    x, y, z = [float(v) for v in ext]
    return (m / 12.0) * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def _sphere_world(R, h0):
    return make_world(
        bodies=[RigidBody6D(
            'body',
            SpatialInertia(MASS, _sphere_inertia(MASS, R)),
            SphereGeometry(R),
            BodyState6D(Pose6D(np.array([0.0, R + h0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])), np.zeros(3), np.zeros(3)),
            linear_damping=0.0,
            angular_damping=0.0,
        )],
        domain_sources=[_plane_source()],
        gravity=(0.0, -GRAVITY, 0.0),
    )


def _box_world(ext, h0):
    return make_world(
        bodies=[RigidBody6D(
            'body',
            SpatialInertia(MASS, _box_inertia(MASS, ext)),
            BoxGeometry(ext),
            BodyState6D(Pose6D(np.array([0.0, 0.5 * ext[1] + h0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0])), np.zeros(3), np.zeros(3)),
            linear_damping=0.0,
            angular_damping=0.0,
        )],
        domain_sources=[_plane_source()],
        gravity=(0.0, -GRAVITY, 0.0),
    )


def _extract_payload_sample(step_idx: int, payload: dict, delta_fn, static_force_fn):
    b0 = (payload.get('bodies') or {}).get(0, {})
    pos = np.asarray(b0.get('position', np.zeros(3)), dtype=float)
    force = np.asarray(b0.get('contact_force', np.zeros(3)), dtype=float)
    y = float(pos[1]) if pos.size >= 2 else 0.0
    t = step_idx * DT + float(payload.get('time_offset', DT))
    d = max(0.0, float(delta_fn(y)))
    Fy = float(force[1]) if force.size >= 2 else 0.0
    static_Fy = float(static_force_fn(d))
    rel = abs(Fy - static_Fy) / max(abs(static_Fy), 1.0e-15)
    return {'time': t, 'delta': d, 'Fy': Fy, 'static_Fy': static_Fy, 'rel_err': rel}


def _run_until_first_contact(world, solver, delta_fn, static_force_fn, max_steps=20):
    rows = []
    result = None
    for step in range(max_steps):
        infos = solver.step_world(world)
        diag = solver.last_step_diagnostics or {}
        info_map = {int(x['body_index']): x for x in infos}
        body = world.bodies[0]
        Fy_end = float(np.asarray(info_map.get(0, {}).get('contact_force', np.zeros(3)), dtype=float)[1])
        t_end = (step + 1) * DT
        delta_end = max(0.0, float(delta_fn(float(body.state.pose.position[1]))))
        static_end = float(static_force_fn(delta_end))
        rows.append({'step': step + 1, 'sample': 'step_end', 'time': t_end, 'delta': delta_end, 'Fy': Fy_end, 'static_Fy': static_end, 'rel_err': abs(Fy_end - static_end) / max(abs(static_end), 1.0e-15)})
        if Fy_end > 1.0e-8 and result is None:
            result = {
                'step_end_time': t_end,
                'step_end_delta': delta_end,
                'step_end_Fy': Fy_end,
                'step_end_static_Fy': static_end,
                'step_end_rel_err': abs(Fy_end - static_end) / max(abs(static_end), 1.0e-15),
                'solver_substeps': int(diag.get('substeps', 1)) if isinstance(diag.get('substeps', 1), (int, float)) else 1,
            }
            for key in ('onset_state', 'onset_aligned', 'onset_microstep'):
                payload = diag.get(key)
                if isinstance(payload, dict):
                    sample = _extract_payload_sample(step, payload, delta_fn, static_force_fn)
                    rows.append({'step': step + 1, 'sample': key, **sample})
                    result.update({f'{key}_{k}': v for k, v in sample.items()})
            break
    return pd.DataFrame(rows), result


def _plot(df, title, out_png):
    plt.figure(figsize=(7.4, 4.6))
    base = df[df['sample'] == 'step_end']
    plt.plot(base['time'], base['Fy'], 'o-', label='step-end Fy')
    plt.plot(base['time'], base['static_Fy'], '--', label='step-end static Fy')
    styles = {
        'onset_state': ('^', 'onset-state Fy'),
        'onset_aligned': ('x', 'onset-aligned Fy'),
        'onset_microstep': ('d', 'onset-microstep Fy'),
    }
    for sample_name, (marker, label) in styles.items():
        sel = df[df['sample'] == sample_name]
        if len(sel):
            plt.scatter(sel['time'], sel['Fy'], marker=marker, s=80, label=label)
            plt.scatter(sel['time'], sel['static_Fy'], marker='+', s=90)
    plt.xlabel('time [s]')
    plt.ylabel('Fy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    rows = []
    # sphere
    R = 0.12
    h0 = 0.02
    sphere_df, sphere_res = _run_until_first_contact(_sphere_world(R, h0), make_solver(True), lambda y: R - y, lambda d: K_EQ * cap_volume(R, d), max_steps=20)
    sphere_df.to_csv(OUT / 'sphere_first_contact.csv', index=False)
    _plot(sphere_df, 'Sphere first-contact force with onset micro-step', OUT / 'sphere_first_contact.png')
    rows.append({'case': 'sphere_centered', **(sphere_res or {})})

    # flat punch
    ext = np.array([0.2, 0.1, 0.15], dtype=float)
    area = float(ext[0] * ext[2])
    h0 = 0.01
    flat_df, flat_res = _run_until_first_contact(_box_world(ext, h0), make_solver(True), lambda y: 0.5 * ext[1] - y, lambda d: area * K_EQ * d, max_steps=14)
    flat_df.to_csv(OUT / 'flat_first_contact.csv', index=False)
    _plot(flat_df, 'Flat-punch first-contact force with onset micro-step', OUT / 'flat_first_contact.png')
    rows.append({'case': 'flat_punch_centered', **(flat_res or {})})

    overview = pd.DataFrame(rows)
    overview.to_csv(OUT / 'comparison.csv', index=False)
    lines = ['# Onset micro-step first-contact force benchmarks', '']
    lines.append('This benchmark compares first-contact force samples reconstructed at the coarse step end, the onset state itself, an onset-aligned post-state, and a post-onset micro-step state.')
    lines.append('')
    for _, row in overview.iterrows():
        lines.append(f"## {row['case']}")
        lines.append(f"- step_end_rel_err = {row['step_end_rel_err']:.6f}")
        for prefix in ('onset_state', 'onset_aligned', 'onset_microstep'):
            if pd.notna(row.get(f'{prefix}_rel_err', np.nan)):
                lines.append(f"- {prefix}_rel_err = {row[f'{prefix}_rel_err']:.6f}")
                lines.append(f"- {prefix}_time = {row[f'{prefix}_time']:.6f}")
                lines.append(f"- {prefix}_Fy = {row[f'{prefix}_Fy']:.6f}")
                lines.append(f"- {prefix}_static_Fy = {row[f'{prefix}_static_Fy']:.6f}")
        lines.append('')
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('saved', OUT / 'comparison.csv')
    print('saved', OUT / 'summary.md')


if __name__ == '__main__':
    main()
