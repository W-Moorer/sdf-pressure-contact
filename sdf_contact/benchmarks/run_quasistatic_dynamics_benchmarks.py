from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sdf_contact import (
    SpatialInertia,
    RigidBody6D,
    BodyState6D,
    Pose6D,
    SDFGeometryDomainSource,
    SphereGeometry,
    BoxGeometry,
    PlaneGeometry,
    PolygonPatchConfig,
    SheetExtractConfig,
    FormalPressureFieldConfig,
    FormalEndpointBandSheetEvaluator,
    ContactManager,
    make_world,
    cap_volume,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'quasistatic_dynamics_benchmarks'
OUT.mkdir(parents=True, exist_ok=True)

# Shared untuned evaluator configuration used for all time-domain cases.
PATCH_CFG = PolygonPatchConfig(raster_cells=18, max_patch_radius=0.50, support_radius_floor_scale=0.90)
SHEET_CFG = SheetExtractConfig(bisection_steps=18)
K_EQUAL = 12000.0
K_EQ = K_EQUAL * K_EQUAL / (K_EQUAL + K_EQUAL)
EVALUATOR = FormalEndpointBandSheetEvaluator(
    patch_cfg=PATCH_CFG,
    sheet_cfg=SHEET_CFG,
    pressure_cfg=FormalPressureFieldConfig(stiffness_default=K_EQUAL, damping_gamma=0.0),
)
CONTACT_MANAGER = ContactManager(EVALUATOR)


def _plane_source(name: str = 'ground'):
    return SDFGeometryDomainSource(
        geometry=PlaneGeometry((0.0, 1.0, 0.0), 0.0),
        pose=Pose6D(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        name=name,
        hint_radius=1.0,
        reference_center=np.zeros(3, dtype=float),
    )


def _sphere_plane_world(*, x: float, y: float, vy: float, R: float):
    body = RigidBody6D(
        name='ball',
        inertia=SpatialInertia(1.0, np.eye(3)),
        geometry=SphereGeometry(R),
        state=BodyState6D(
            pose=Pose6D(np.array([x, y, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.array([0.0, vy, 0.0], dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
    )
    return make_world(bodies=[body], domain_sources=[_plane_source()])


def _box_plane_world(*, x: float, y: float, vy: float, extents: np.ndarray):
    body = RigidBody6D(
        name='box',
        inertia=SpatialInertia(1.0, np.eye(3)),
        geometry=BoxGeometry(extents),
        state=BodyState6D(
            pose=Pose6D(np.array([x, y, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.array([0.0, vy, 0.0], dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
    )
    return make_world(bodies=[body], domain_sources=[_plane_source()])


def _raised_cosine_press_release(t: np.ndarray, *, z_hi: float, z_lo: float, T: float):
    # Starts at z_hi, reaches z_lo at T/2, returns to z_hi at T.
    omega = 2.0 * math.pi / T
    amp = 0.5 * (z_hi - z_lo)
    mean = 0.5 * (z_hi + z_lo)
    z = mean + amp * np.cos(omega * t)
    vz = -amp * omega * np.sin(omega * t)
    return z, vz


def _active_rel_err(num: pd.Series, den: pd.Series, tol: float = 1.0e-8) -> pd.Series:
    out = []
    for a, b in zip(num, den):
        if abs(float(b)) <= tol:
            out.append(np.nan)
        else:
            out.append(abs(float(a) - float(b)) / abs(float(b)))
    return pd.Series(out, index=num.index, dtype=float)


def _summarize_case(df: pd.DataFrame, force_col: str = 'Fy', ideal_force_col: str = 'ideal_Fy', moment_col: str | None = None, ideal_moment_col: str | None = None):
    active = df[df['ideal_Fy'] > 1.0e-8].copy()
    out = {
        'n_steps': int(len(df)),
        'n_active_steps': int(len(active)),
        'max_delta': float(df['delta'].max()),
        'max_abs_vy': float(np.abs(df['vy']).max()),
    }
    if len(active) > 0:
        out['mean_rel_err_Fy_active'] = float(active['rel_err_Fy'].mean())
        out['max_rel_err_Fy_active'] = float(active['rel_err_Fy'].max())
    else:
        out['mean_rel_err_Fy_active'] = float('nan')
        out['max_rel_err_Fy_active'] = float('nan')
    if moment_col is not None and ideal_moment_col is not None:
        active_m = active[active[ideal_moment_col].abs() > 1.0e-8].copy()
        if len(active_m) > 0:
            out['mean_rel_err_M_active'] = float(active_m['rel_err_M'].mean())
            out['max_rel_err_M_active'] = float(active_m['rel_err_M'].max())
        else:
            out['mean_rel_err_M_active'] = float('nan')
            out['max_rel_err_M_active'] = float('nan')
    return out


def _run_centered_sphere_press_release() -> tuple[pd.DataFrame, dict]:
    R = 0.12
    T = 1.0
    nsteps = 5
    ts = np.linspace(0.0, T, nsteps)
    y, vy = _raised_cosine_press_release(ts, z_hi=0.125, z_lo=0.085, T=T)

    rows = []
    for t, yi, vyi in zip(ts, y, vy):
        world = _sphere_plane_world(x=0.0, y=float(yi), vy=float(vyi), R=R)
        contacts = CONTACT_MANAGER.compute_all_contacts(world)
        body = contacts['ball']
        delta = max(0.0, R - float(yi))
        ideal = K_EQ * cap_volume(R, delta)
        rows.append({
            'time': float(t),
            'y': float(yi),
            'vy': float(vyi),
            'delta': float(delta),
            'ideal_Fy': float(ideal),
            'Fy': float(body.total_force[1]),
            'rel_err_Fy': abs(float(body.total_force[1]) - float(ideal)) / max(abs(float(ideal)), 1.0e-15),
            'body_moment_norm': float(np.linalg.norm(body.total_moment)),
            'num_pair_patch_points': int(body.num_pair_patch_points),
            'num_pair_sheet_points': int(body.num_pair_sheet_points),
            'num_pair_tractions': int(body.num_pair_tractions),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'sphere_centered_press_release.csv', index=False)

    plt.figure(figsize=(7.4, 4.6))
    plt.plot(df['time'], df['Fy'], label='endpoint evaluator Fy')
    plt.plot(df['time'], df['ideal_Fy'], label='static analytic baseline')
    plt.xlabel('time [s]')
    plt.ylabel('Fy')
    plt.title('Quasi-static dynamics: centered sphere press/release')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'sphere_centered_press_release.png', dpi=180)
    plt.close()

    return df, _summarize_case(df)


def _run_centered_flat_punch_press_release() -> tuple[pd.DataFrame, dict]:
    ext = np.array([0.2, 0.1, 0.15], dtype=float)
    area = float(ext[0] * ext[2])
    T = 1.0
    nsteps = 5
    ts = np.linspace(0.0, T, nsteps)
    y, vy = _raised_cosine_press_release(ts, z_hi=0.5 * ext[1] + 0.004, z_lo=0.5 * ext[1] - 0.012, T=T)

    rows = []
    for t, yi, vyi in zip(ts, y, vy):
        world = _box_plane_world(x=0.0, y=float(yi), vy=float(vyi), extents=ext)
        contacts = CONTACT_MANAGER.compute_all_contacts(world)
        body = contacts['box']
        delta = max(0.0, 0.5 * ext[1] - float(yi))
        ideal = area * K_EQ * float(delta)
        rows.append({
            'time': float(t),
            'y': float(yi),
            'vy': float(vyi),
            'delta': float(delta),
            'ideal_Fy': float(ideal),
            'Fy': float(body.total_force[1]),
            'rel_err_Fy': abs(float(body.total_force[1]) - float(ideal)) / max(abs(float(ideal)), 1.0e-15),
            'body_force_lateral_norm': float(np.linalg.norm(body.total_force[[0, 2]])),
            'body_moment_norm': float(np.linalg.norm(body.total_moment)),
            'num_pair_patch_points': int(body.num_pair_patch_points),
            'num_pair_sheet_points': int(body.num_pair_sheet_points),
            'num_pair_tractions': int(body.num_pair_tractions),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'flat_punch_centered_press_release.csv', index=False)

    plt.figure(figsize=(7.4, 4.6))
    plt.plot(df['time'], df['Fy'], label='endpoint evaluator Fy')
    plt.plot(df['time'], df['ideal_Fy'], label='static analytic baseline')
    plt.xlabel('time [s]')
    plt.ylabel('Fy')
    plt.title('Quasi-static dynamics: centered flat punch press/release')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'flat_punch_centered_press_release.png', dpi=180)
    plt.close()

    return df, _summarize_case(df)


def _run_offaxis_sphere_press_release() -> tuple[pd.DataFrame, dict]:
    R = 0.12
    x_off = 0.04
    T = 1.0
    nsteps = 5
    ts = np.linspace(0.0, T, nsteps)
    y, vy = _raised_cosine_press_release(ts, z_hi=0.125, z_lo=0.085, T=T)

    rows = []
    for t, yi, vyi in zip(ts, y, vy):
        world = _sphere_plane_world(x=x_off, y=float(yi), vy=float(vyi), R=R)
        contacts = CONTACT_MANAGER.compute_all_contacts(world)
        body = contacts['ball']
        ground = contacts['ground']
        delta = max(0.0, R - float(yi))
        ideal_F = K_EQ * cap_volume(R, delta)
        ideal_Mz = -x_off * ideal_F
        rows.append({
            'time': float(t),
            'x': float(x_off),
            'y': float(yi),
            'vy': float(vyi),
            'delta': float(delta),
            'ideal_Fy': float(ideal_F),
            'Fy': float(body.total_force[1]),
            'rel_err_Fy': abs(float(body.total_force[1]) - float(ideal_F)) / max(abs(float(ideal_F)), 1.0e-15),
            'ideal_ground_Mz': float(ideal_Mz),
            'ground_Mz': float(ground.total_moment[2]),
            'rel_err_M': abs(float(ground.total_moment[2]) - float(ideal_Mz)) / max(abs(float(ideal_Mz)), 1.0e-15),
            'body_moment_norm': float(np.linalg.norm(body.total_moment)),
            'num_pair_patch_points': int(body.num_pair_patch_points),
            'num_pair_sheet_points': int(body.num_pair_sheet_points),
            'num_pair_tractions': int(body.num_pair_tractions),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'sphere_offaxis_press_release.csv', index=False)

    plt.figure(figsize=(7.4, 4.6))
    plt.plot(df['time'], df['ground_Mz'], label='endpoint evaluator ground Mz')
    plt.plot(df['time'], df['ideal_ground_Mz'], label='static analytic baseline')
    plt.xlabel('time [s]')
    plt.ylabel('ground Mz')
    plt.title('Quasi-static dynamics: off-axis sphere press/release')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'sphere_offaxis_press_release_moment.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7.4, 4.6))
    plt.plot(df['time'], df['Fy'], label='endpoint evaluator Fy')
    plt.plot(df['time'], df['ideal_Fy'], label='static analytic baseline')
    plt.xlabel('time [s]')
    plt.ylabel('Fy')
    plt.title('Quasi-static dynamics: off-axis sphere press/release')
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / 'sphere_offaxis_press_release_force.png', dpi=180)
    plt.close()

    return df, _summarize_case(df, moment_col='ground_Mz', ideal_moment_col='ideal_ground_Mz')


def run() -> None:
    df_sphere_center, sum_sphere_center = _run_centered_sphere_press_release()
    df_flat_center, sum_flat_center = _run_centered_flat_punch_press_release()
    df_sphere_off, sum_sphere_off = _run_offaxis_sphere_press_release()

    overview = pd.DataFrame([
        {'case': 'sphere_centered_press_release', **sum_sphere_center},
        {'case': 'flat_punch_centered_press_release', **sum_flat_center},
        {'case': 'sphere_offaxis_press_release', **sum_sphere_off},
    ])
    overview.to_csv(OUT / 'overview.csv', index=False)

    lines = []
    lines.append('# Quasi-static dynamics benchmarks')
    lines.append('')
    lines.append('These are **time-domain driven quasi-static benchmarks** for the endpoint evaluator.')
    lines.append('The body pose follows a smooth press/release trajectory, and the same endpoint evaluator configuration is used in all cases.')
    lines.append('This benchmark layer is intentionally used before free rigid-body impact tests so that we can verify that the endpoint contact law still matches the static analytic baseline inside a time-domain workflow.')
    lines.append('')
    lines.append('## Shared evaluator configuration')
    lines.append(f'- patch raster_cells = {PATCH_CFG.raster_cells}')
    lines.append(f'- patch support_radius_floor_scale = {PATCH_CFG.support_radius_floor_scale}')
    lines.append(f'- sheet bisection_steps = {SHEET_CFG.bisection_steps}')
    lines.append(f'- equal body/ground stiffness = {K_EQUAL:.1f}')
    lines.append('- damping_gamma = 0.0')
    lines.append('- no case-specific evaluator branching was used')
    lines.append('')
    lines.append('## Centered sphere press/release')
    lines.append(f"- active-step mean relative Fy error vs static analytic baseline: {sum_sphere_center['mean_rel_err_Fy_active']:.4%}")
    lines.append(f"- active-step max relative Fy error vs static analytic baseline: {sum_sphere_center['max_rel_err_Fy_active']:.4%}")
    lines.append(f"- max penetration depth: {sum_sphere_center['max_delta']:.6f}")
    lines.append(f"- max |vy|: {sum_sphere_center['max_abs_vy']:.6f}")
    lines.append('')
    lines.append('## Centered flat punch press/release')
    lines.append(f"- active-step mean relative Fy error vs static analytic baseline: {sum_flat_center['mean_rel_err_Fy_active']:.4%}")
    lines.append(f"- active-step max relative Fy error vs static analytic baseline: {sum_flat_center['max_rel_err_Fy_active']:.4%}")
    lines.append(f"- max penetration depth: {sum_flat_center['max_delta']:.6f}")
    lines.append(f"- max |vy|: {sum_flat_center['max_abs_vy']:.6f}")
    lines.append('')
    lines.append('## Off-axis sphere press/release')
    lines.append(f"- active-step mean relative Fy error vs static analytic baseline: {sum_sphere_off['mean_rel_err_Fy_active']:.4%}")
    lines.append(f"- active-step max relative Fy error vs static analytic baseline: {sum_sphere_off['max_rel_err_Fy_active']:.4%}")
    lines.append(f"- active-step mean relative ground Mz error vs static analytic baseline: {sum_sphere_off['mean_rel_err_M_active']:.4%}")
    lines.append(f"- active-step max relative ground Mz error vs static analytic baseline: {sum_sphere_off['max_rel_err_M_active']:.4%}")
    lines.append(f"- max penetration depth: {sum_sphere_off['max_delta']:.6f}")
    lines.append(f"- max |vy|: {sum_sphere_off['max_abs_vy']:.6f}")
    lines.append('')
    lines.append('## Reading of the result')
    lines.append('- If these time-domain driven cases stay close to the static analytic baseline, it shows that the endpoint evaluator keeps its static correctness when moved into a benchmark workflow with trajectories and velocities.')
    lines.append('- This is the right first dynamics gate before debugging free-body impacts, rebounds, and friction.')
    lines.append('- These are not yet free rigid-body impact benchmarks; they are deliberate quasi-static dynamic regressions used to isolate contact-law correctness from global time-integration instability.')
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
