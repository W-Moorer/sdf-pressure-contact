from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sdf_contact import (
    SpatialInertia,
    RigidBody6D,
    BodyState6D,
    Pose6D,
    SDFGeometryDomainSource,
    PlaneGeometry,
    BoxGeometry,
    PolygonPatchConfig,
    SheetExtractConfig,
    FormalPressureFieldConfig,
    FormalEndpointBandSheetEvaluator,
    ContactManager,
    GlobalImplicitSystemSolver6D,
    IntegratorConfig,
    make_world,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'flat_punch_support_area_prior'
OUT.mkdir(parents=True, exist_ok=True)

K_EQUAL = 40000.0
K_EQ = K_EQUAL * K_EQUAL / (K_EQUAL + K_EQUAL)
MASS = 0.05
DT = 0.005
GRAVITY = 9.81
EXT = np.array([0.2, 0.1, 0.15], dtype=float)
AREA = float(EXT[0] * EXT[2])


def plane_source(name: str = 'ground'):
    return SDFGeometryDomainSource(
        geometry=PlaneGeometry((0.0, 1.0, 0.0), 0.0),
        pose=Pose6D(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        name=name,
        hint_radius=1.0,
        reference_center=np.zeros(3, dtype=float),
    )


def box_inertia(m: float, ext: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in ext]
    return (m / 12.0) * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def box_plane_world(*, y: float, vy: float = 0.0):
    body = RigidBody6D(
        name='box',
        inertia=SpatialInertia(MASS, box_inertia(MASS, EXT)),
        geometry=BoxGeometry(EXT),
        state=BodyState6D(
            pose=Pose6D(np.array([0.0, y, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.array([0.0, vy, 0.0], dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
        linear_damping=0.0,
        angular_damping=0.0,
    )
    return make_world(bodies=[body], domain_sources=[plane_source()], gravity=(0.0, -GRAVITY, 0.0))


def make_evaluator(area_prior_control: bool):
    patch_cfg = PolygonPatchConfig(
        raster_cells=4,
        max_patch_radius=0.50,
        support_radius_floor_scale=0.90,
        area_prior_control=area_prior_control,
    )
    sheet_cfg = SheetExtractConfig(bisection_steps=8)
    pressure_cfg = FormalPressureFieldConfig(stiffness_default=K_EQUAL, damping_gamma=0.0)
    return FormalEndpointBandSheetEvaluator(patch_cfg=patch_cfg, sheet_cfg=sheet_cfg, pressure_cfg=pressure_cfg)


def make_solver(area_prior_control: bool):
    evaluator = make_evaluator(area_prior_control)
    manager = ContactManager(evaluator)
    cfg = IntegratorConfig(dt=DT, collect_diagnostics=True)
    return manager, GlobalImplicitSystemSolver6D(manager, cfg)


def solve_delta_max_box(h0: float, v0: float = 0.0) -> float:
    energy = MASS * GRAVITY * h0 + 0.5 * MASS * v0 * v0
    a = 0.5 * AREA * K_EQ
    b = -MASS * GRAVITY
    c = -energy
    disc = max(b * b - 4.0 * a * c, 0.0)
    return (-b + math.sqrt(disc)) / (2.0 * a)


def static_scan(area_prior_control: bool) -> tuple[pd.DataFrame, dict]:
    evaluator = make_evaluator(area_prior_control)
    manager = ContactManager(evaluator)
    rows = []
    for delta in np.linspace(0.002, 0.020, 10):
        y = 0.5 * EXT[1] - float(delta)
        world = box_plane_world(y=float(y))
        contacts = manager.compute_all_contacts(world)
        body = contacts['box']
        rec = body.pair_records[0]
        ideal = AREA * K_EQ * float(delta)
        rows.append({
            'delta': float(delta),
            'Fy': float(body.total_force[1]),
            'ideal_Fy': float(ideal),
            'rel_err_Fy': abs(float(body.total_force[1]) - ideal) / max(abs(ideal), 1.0e-15),
            'raw_support_area': float(rec.meta.get('raw_support_area', 0.0)),
            'prior_support_area': float(rec.meta.get('prior_support_area', 0.0)),
            'total_sheet_area': float(rec.meta.get('total_sheet_area', 0.0)),
            'patch_radius': float(rec.meta.get('patch_radius', 0.0)),
            'area_prior_control': int(area_prior_control),
        })
    df = pd.DataFrame(rows)
    summary = {
        'mean_rel_err_Fy': float(df['rel_err_Fy'].mean()),
        'max_rel_err_Fy': float(df['rel_err_Fy'].max()),
        'mean_total_sheet_area': float(df['total_sheet_area'].mean()),
        'mean_prior_support_area': float(df['prior_support_area'].mean()),
    }
    return df, summary


def free_contact_release(area_prior_control: bool) -> tuple[pd.DataFrame, dict]:
    manager, solver = make_solver(area_prior_control)
    world = box_plane_world(y=0.065, vy=-0.15)
    rows = []
    first_contact = None
    first_release = None
    release_buffer = None
    y0 = float(world.bodies[0].state.pose.position[1])
    E0 = 0.5 * MASS * float(np.dot(world.bodies[0].state.linear_velocity, world.bodies[0].state.linear_velocity)) + MASS * GRAVITY * y0
    delta_max_analytic = solve_delta_max_box(max(0.0, 0.5 * EXT[1] - y0), v0=0.15)

    for step in range(50):
        solver.step_world(world)
        step_diag = solver.last_step_diagnostics or {}
        contacts = manager.compute_all_contacts(world)
        body = world.bodies[0]
        agg_body = contacts['box']
        t = (step + 1) * DT
        y = float(body.state.pose.position[1])
        vy = float(body.state.linear_velocity[1])
        delta = max(0.0, 0.5 * EXT[1] - y)
        Fy = float(agg_body.total_force[1])
        active = Fy > 1.0e-8
        static_Fy = AREA * K_EQ * delta
        total_energy = 0.5 * MASS * float(np.dot(body.state.linear_velocity, body.state.linear_velocity)) + MASS * GRAVITY * y
        rows.append({
            'time': t,
            'y': y,
            'vy': vy,
            'delta': delta,
            'Fy': Fy,
            'static_Fy_at_delta': static_Fy,
            'rel_err_static_force_law': abs(Fy - static_Fy) / max(abs(static_Fy), 1.0e-15) if static_Fy > 0 else 0.0,
            'contact_active': int(active),
            'total_energy': total_energy,
            'energy_minus_initial': total_energy - E0,
            'solver_substeps': int(step_diag.get('substeps', 1)) if isinstance(step_diag.get('substeps', 1), (int, float)) else 1,
            'area_prior_control': int(area_prior_control),
        })
        if first_contact is None and active:
            first_contact = len(rows) - 1
        elif first_contact is not None and first_release is None and not active:
            first_release = len(rows) - 1
            release_buffer = 2
        elif first_release is not None:
            release_buffer -= 1
            if release_buffer <= 0:
                break

    df = pd.DataFrame(rows)
    if first_contact is None:
        raise RuntimeError('No contact detected in free-contact benchmark')
    if first_release is None:
        first_release = len(df) - 1
    first_cycle = df.iloc[first_contact:first_release].copy()
    release_row = df.iloc[first_release]
    touch_speed = math.sqrt(2.0 * GRAVITY * max(0.0, y0 - 0.5 * EXT[1]) + 0.15 * 0.15)
    summary = {
        'first_contact_time': float(df.iloc[first_contact]['time']),
        'first_release_time': float(df.iloc[first_release]['time']),
        'analytic_delta_max': float(delta_max_analytic),
        'sim_delta_max_first_cycle': float(first_cycle['delta'].max()),
        'rel_err_delta_max': abs(float(first_cycle['delta'].max()) - delta_max_analytic) / max(delta_max_analytic, 1.0e-15),
        'touch_speed_analytic': float(touch_speed),
        'release_speed_sim': abs(float(release_row['vy'])),
        'rel_err_release_speed': abs(abs(float(release_row['vy'])) - touch_speed) / max(touch_speed, 1.0e-15),
        'release_energy_drift_rel_drop_scale': float(release_row['energy_minus_initial']) / max(MASS * GRAVITY * max(0.0, y0 - 0.5 * EXT[1]), 1.0e-15),
        'max_rel_err_static_force_law_first_cycle': float(first_cycle['rel_err_static_force_law'].max()),
        'mean_rel_err_static_force_law_first_cycle': float(first_cycle['rel_err_static_force_law'].mean()),
        'mean_solver_substeps_active': float(first_cycle.loc[first_cycle['contact_active'] == 1, 'solver_substeps'].mean()),
    }
    return df, summary


def main() -> None:
    static_off, s_off = static_scan(False)
    static_on, s_on = static_scan(True)
    free_off, f_off = free_contact_release(False)
    free_on, f_on = free_contact_release(True)

    static_off.to_csv(OUT / 'static_flat_punch_area_prior_off.csv', index=False)
    static_on.to_csv(OUT / 'static_flat_punch_area_prior_on.csv', index=False)
    free_off.to_csv(OUT / 'free_flat_punch_area_prior_off.csv', index=False)
    free_on.to_csv(OUT / 'free_flat_punch_area_prior_on.csv', index=False)

    comparison = pd.DataFrame([
        {'benchmark': 'static_flat_punch', 'variant': 'area_prior_off', **s_off},
        {'benchmark': 'static_flat_punch', 'variant': 'area_prior_on', **s_on},
        {'benchmark': 'free_flat_punch', 'variant': 'area_prior_off', **f_off},
        {'benchmark': 'free_flat_punch', 'variant': 'area_prior_on', **f_on},
    ])
    comparison.to_csv(OUT / 'comparison.csv', index=False)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(static_off['delta'], static_off['Fy'], 'o-', label='static off')
    plt.plot(static_on['delta'], static_on['Fy'], 's-', label='static on')
    plt.plot(static_on['delta'], static_on['ideal_Fy'], '-', label='analytic')
    plt.xlabel('delta')
    plt.ylabel('Fy')
    plt.title('Flat punch static: area-prior-controlled support')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'static_flat_punch_comparison.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    act_off = free_off[free_off['contact_active'] == 1]
    act_on = free_on[free_on['contact_active'] == 1]
    plt.plot(act_off['delta'], act_off['Fy'], 'o-', label='free off')
    plt.plot(act_on['delta'], act_on['Fy'], 's-', label='free on')
    plt.plot(act_on['delta'], act_on['static_Fy_at_delta'], '-', label='analytic A keq delta')
    plt.xlabel('delta during first contact cycle')
    plt.ylabel('Fy')
    plt.title('Flat punch free-body: area-prior-controlled support')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'free_flat_punch_comparison.png', dpi=180)
    plt.close()

    summary = []
    summary.append('# Flat punch support area prior benchmarks')
    summary.append('')
    summary.append('This benchmark isolates the new area-prior-controlled support recovery for large flat support contact.')
    summary.append('The same endpoint contact architecture is used in both variants; only `area_prior_control` is toggled on/off.')
    summary.append('')
    summary.append('## Static centered flat punch (same low-resolution configuration as free-body benchmark)')
    summary.append(f"- area_prior_off mean relative Fy error: {100.0 * s_off['mean_rel_err_Fy']:.4f}%")
    summary.append(f"- area_prior_off max relative Fy error: {100.0 * s_off['max_rel_err_Fy']:.4f}%")
    summary.append(f"- area_prior_on mean relative Fy error: {100.0 * s_on['mean_rel_err_Fy']:.4f}%")
    summary.append(f"- area_prior_on max relative Fy error: {100.0 * s_on['max_rel_err_Fy']:.4f}%")
    summary.append(f"- mean recovered sheet area, area_prior_off: {s_off['mean_total_sheet_area']:.6f}")
    summary.append(f"- mean recovered sheet area, area_prior_on: {s_on['mean_total_sheet_area']:.6f}")
    summary.append(f"- mean prior support area: {s_on['mean_prior_support_area']:.6f}")
    summary.append('')
    summary.append('## Free-body centered flat punch')
    summary.append(f"- area_prior_off max relative static force-law error during first cycle: {100.0 * f_off['max_rel_err_static_force_law_first_cycle']:.4f}%")
    summary.append(f"- area_prior_on max relative static force-law error during first cycle: {100.0 * f_on['max_rel_err_static_force_law_first_cycle']:.4f}%")
    summary.append(f"- area_prior_off release speed error: {100.0 * f_off['rel_err_release_speed']:.4f}%")
    summary.append(f"- area_prior_on release speed error: {100.0 * f_on['rel_err_release_speed']:.4f}%")
    summary.append(f"- area_prior_off release energy drift / drop scale: {f_off['release_energy_drift_rel_drop_scale']:.6f}")
    summary.append(f"- area_prior_on release energy drift / drop scale: {f_on['release_energy_drift_rel_drop_scale']:.6f}")
    summary.append('')
    summary.append('## Reading')
    summary.append('- The support-area prior is intended to fix the support footprint / measure layer first.')
    summary.append('- If the static flat-punch curve moves onto `A * k_eq * delta` while the free-body first-cycle force mismatch stays high, the remaining bottleneck is no longer support geometry; it is time-domain dynamics / event resolution.')
    (OUT / 'summary.md').write_text('\n'.join(summary), encoding='utf-8')


if __name__ == '__main__':
    main()
