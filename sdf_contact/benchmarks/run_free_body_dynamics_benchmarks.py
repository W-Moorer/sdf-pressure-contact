from __future__ import annotations

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    GlobalImplicitSystemSolver6D,
    IntegratorConfig,
    make_world,
    cap_volume,
)

OUT = ROOT / 'results' / 'free_body_dynamics_benchmarks'
OUT.mkdir(parents=True, exist_ok=True)

# Shared default endpoint configuration for all true-dynamics cases.
PATCH_CFG = PolygonPatchConfig(raster_cells=4, max_patch_radius=0.50, support_radius_floor_scale=0.90)
SHEET_CFG = SheetExtractConfig(bisection_steps=8)
K_EQUAL = 40000.0
MASS = 0.05
DT = 0.005
GRAVITY = 9.81

EVALUATOR = FormalEndpointBandSheetEvaluator(
    patch_cfg=PATCH_CFG,
    sheet_cfg=SHEET_CFG,
    pressure_cfg=FormalPressureFieldConfig(stiffness_default=K_EQUAL, damping_gamma=0.0),
)
CONTACT_MANAGER = ContactManager(EVALUATOR)
SOLVER_CFG = IntegratorConfig(dt=DT, collect_diagnostics=True)
K_EQ = K_EQUAL * K_EQUAL / (K_EQUAL + K_EQUAL)


def _solver():
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, SOLVER_CFG)


def _plane_source(name: str = 'ground'):
    return SDFGeometryDomainSource(
        geometry=PlaneGeometry((0.0, 1.0, 0.0), 0.0),
        pose=Pose6D(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        name=name,
        hint_radius=1.0,
        reference_center=np.zeros(3, dtype=float),
    )


def _sphere_inertia(m: float, R: float) -> np.ndarray:
    I = (2.0 / 5.0) * m * R * R
    return np.diag([I, I, I])


def _box_inertia(m: float, ext: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in ext]
    return (m / 12.0) * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def _sphere_plane_world(*, x: float, y: float, vy: float, R: float):
    body = RigidBody6D(
        name='ball',
        inertia=SpatialInertia(MASS, _sphere_inertia(MASS, R)),
        geometry=SphereGeometry(R),
        state=BodyState6D(
            pose=Pose6D(np.array([x, y, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.array([0.0, vy, 0.0], dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
        linear_damping=0.0,
        angular_damping=0.0,
    )
    return make_world(bodies=[body], domain_sources=[_plane_source()], gravity=(0.0, -GRAVITY, 0.0))


def _box_plane_world(*, x: float, y: float, vy: float, extents: np.ndarray):
    body = RigidBody6D(
        name='box',
        inertia=SpatialInertia(MASS, _box_inertia(MASS, extents)),
        geometry=BoxGeometry(extents),
        state=BodyState6D(
            pose=Pose6D(np.array([x, y, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.array([0.0, vy, 0.0], dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
        linear_damping=0.0,
        angular_damping=0.0,
    )
    return make_world(bodies=[body], domain_sources=[_plane_source()], gravity=(0.0, -GRAVITY, 0.0))


def _sphere_contact_potential(R: float, delta: float) -> float:
    d = max(0.0, min(float(delta), 2.0 * R))
    return K_EQ * math.pi * (R * d ** 3 / 3.0 - d ** 4 / 12.0)


def _box_contact_potential(area: float, delta: float) -> float:
    d = max(0.0, float(delta))
    return 0.5 * area * K_EQ * d * d


def _solve_delta_max_sphere(R: float, h0: float, v0: float = 0.0) -> float:
    energy = MASS * GRAVITY * h0 + 0.5 * MASS * v0 * v0
    def f(d: float) -> float:
        return _sphere_contact_potential(R, d) - MASS * GRAVITY * d - energy
    lo, hi = 0.0, min(2.0 * R, max(0.02, 2.0 * h0 + 0.05))
    while f(hi) < 0.0 and hi < 2.0 * R:
        hi = min(2.0 * R, hi * 1.5 + 0.01)
        if hi >= 2.0 * R:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _solve_delta_max_box(area: float, h0: float, v0: float = 0.0) -> float:
    energy = MASS * GRAVITY * h0 + 0.5 * MASS * v0 * v0
    a = 0.5 * area * K_EQ
    b = -MASS * GRAVITY
    c = -energy
    disc = max(b * b - 4.0 * a * c, 0.0)
    return (-b + math.sqrt(disc)) / (2.0 * a)


def _run_until_first_release(world, body_name: str, delta_fn, potential_fn, x_ref: float = 0.0, max_steps: int = 50, post_release_steps: int = 2):
    solver = _solver()
    rows = []
    first_contact_idx = None
    first_release_idx = None
    release_buffer = None

    body0 = next(b for b in world.bodies if b.name == body_name)
    y0 = float(body0.state.pose.position[1])
    v0 = float(body0.state.linear_velocity[1])
    E0 = 0.5 * MASS * float(np.dot(body0.state.linear_velocity, body0.state.linear_velocity)) + MASS * GRAVITY * y0

    for step in range(max_steps):
        infos = solver.step_world(world)
        step_diag = solver.last_step_diagnostics or {}
        contacts = CONTACT_MANAGER.compute_all_contacts(world)
        body = next(b for b in world.bodies if b.name == body_name)
        agg_body = contacts[body_name]
        agg_ground = contacts['ground']
        t = (step + 1) * DT
        x = float(body.state.pose.position[0])
        y = float(body.state.pose.position[1])
        vx = float(body.state.linear_velocity[0])
        vy = float(body.state.linear_velocity[1])
        omega = body.state.angular_velocity.copy()
        delta = max(0.0, float(delta_fn(y)))
        U = float(potential_fn(delta))
        kinetic = 0.5 * MASS * float(np.dot(body.state.linear_velocity, body.state.linear_velocity))
        total_energy = kinetic + MASS * GRAVITY * y + U
        Fy = float(agg_body.total_force[1])
        active = Fy > 1.0e-8
        rows.append({
            'time': t,
            'x': x,
            'y': y,
            'vx': vx,
            'vy': vy,
            'omega_x': float(omega[0]),
            'omega_y': float(omega[1]),
            'omega_z': float(omega[2]),
            'delta': delta,
            'contact_active': int(active),
            'Fy': Fy,
            'F_lateral_norm': float(np.linalg.norm(agg_body.total_force[[0, 2]])),
            'body_moment_norm': float(np.linalg.norm(agg_body.total_moment)),
            'ground_Mz': float(agg_ground.total_moment[2]),
            'static_Fy_at_delta': 0.0,  # filled by case runner
            'static_ground_Mz_at_delta': 0.0,
            'total_energy': total_energy,
            'energy_minus_initial': total_energy - E0,
            'x_ref': x_ref,
            'num_pair_patch_points': int(agg_body.num_pair_patch_points),
            'num_pair_sheet_points': int(agg_body.num_pair_sheet_points),
            'num_pair_tractions': int(agg_body.num_pair_tractions),
            'solver_final_residual': float(step_diag.get('final_residual_norm', float('nan'))),
            'solver_substeps': int(step_diag.get('substeps', 1)) if isinstance(step_diag.get('substeps', 1), (int, float)) else 1,
        })

        if first_contact_idx is None and active:
            first_contact_idx = len(rows) - 1
        elif first_contact_idx is not None and first_release_idx is None and not active:
            first_release_idx = len(rows) - 1
            release_buffer = post_release_steps
        elif first_release_idx is not None:
            release_buffer -= 1
            if release_buffer <= 0:
                break

    df = pd.DataFrame(rows)
    return df, first_contact_idx, first_release_idx, E0


def _summarize_common(df: pd.DataFrame, first_contact_idx: int | None, first_release_idx: int | None, *, h0: float, analytic_delta_max: float, touch_speed: float, x_ref: float = 0.0):
    out = {
        'n_steps': int(len(df)),
        'first_contact_time': float(df.iloc[first_contact_idx]['time']) if first_contact_idx is not None else float('nan'),
        'first_release_time': float(df.iloc[first_release_idx]['time']) if first_release_idx is not None else float('nan'),
        'touch_speed_analytic': float(touch_speed),
        'analytic_delta_max': float(analytic_delta_max),
        'contact_transitions_total': int(sum(int(df.iloc[i]['contact_active']) != int(df.iloc[i - 1]['contact_active']) for i in range(1, len(df)))) if len(df) > 1 else 0,
    }
    if first_contact_idx is None:
        out.update({
            'sim_delta_max_first_cycle': float('nan'),
            'rel_err_delta_max': float('nan'),
            'release_speed_sim': float('nan'),
            'rel_err_release_speed': float('nan'),
            'release_energy_drift_rel_drop_scale': float('nan'),
            'max_rel_err_static_force_law_first_cycle': float('nan'),
            'mean_rel_err_static_force_law_first_cycle': float('nan'),
            'max_abs_x_drift': float(np.max(np.abs(df['x'] - x_ref))) if len(df) else float('nan'),
            'max_abs_vx': float(np.max(np.abs(df['vx']))) if len(df) else float('nan'),
            'max_body_moment_norm_first_cycle': float('nan'),
            'max_rel_ground_Mz_relation_first_cycle': float('nan'),
        })
        return out

    end_idx = first_release_idx if first_release_idx is not None else len(df) - 1
    cyc = df.iloc[first_contact_idx:end_idx + 1].copy()
    out['sim_delta_max_first_cycle'] = float(cyc['delta'].max())
    out['rel_err_delta_max'] = abs(out['sim_delta_max_first_cycle'] - analytic_delta_max) / max(abs(analytic_delta_max), 1.0e-15)
    active = cyc[(cyc['contact_active'] > 0) & (cyc['static_Fy_at_delta'] > 1.0e-8)].copy()
    if len(active) > 0:
        rel_force = (active['Fy'] - active['static_Fy_at_delta']).abs() / active['static_Fy_at_delta'].abs().clip(lower=1.0e-15)
        out['max_rel_err_static_force_law_first_cycle'] = float(rel_force.max())
        out['mean_rel_err_static_force_law_first_cycle'] = float(rel_force.mean())
    else:
        out['max_rel_err_static_force_law_first_cycle'] = float('nan')
        out['mean_rel_err_static_force_law_first_cycle'] = float('nan')

    if first_release_idx is not None:
        rel = df.iloc[first_release_idx]
        out['release_speed_sim'] = max(0.0, float(rel['vy']))
        out['rel_err_release_speed'] = abs(out['release_speed_sim'] - touch_speed) / max(abs(touch_speed), 1.0e-15)
        drop_scale = max(MASS * GRAVITY * h0, 1.0e-15)
        out['release_energy_drift_rel_drop_scale'] = float(rel['energy_minus_initial']) / drop_scale
    else:
        out['release_speed_sim'] = float('nan')
        out['rel_err_release_speed'] = float('nan')
        out['release_energy_drift_rel_drop_scale'] = float('nan')

    out['max_abs_x_drift'] = float(np.max(np.abs(df['x'] - x_ref)))
    out['max_abs_vx'] = float(np.max(np.abs(df['vx'])))
    out['max_body_moment_norm_first_cycle'] = float(cyc['body_moment_norm'].max())
    rel_m = []
    for _, row in cyc.iterrows():
        if int(row['contact_active']) <= 0:
            continue
        ideal = row['static_ground_Mz_at_delta']
        if abs(float(ideal)) <= 1.0e-8:
            continue
        rel_m.append(abs(float(row['ground_Mz']) - float(ideal)) / abs(float(ideal)))
    out['max_rel_ground_Mz_relation_first_cycle'] = float(max(rel_m)) if rel_m else float('nan')
    return out


def _plot_force_energy(df: pd.DataFrame, title: str, out_png: Path, include_moment: bool = False):
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(df['time'], df['Fy'], label='contact Fy')
    plt.plot(df['time'], df['static_Fy_at_delta'], label='static Fy at same delta')
    if include_moment:
        plt.plot(df['time'], df['ground_Mz'], label='ground Mz')
        plt.plot(df['time'], df['static_ground_Mz_at_delta'], label='static ground Mz at same delta')
    plt.plot(df['time'], df['energy_minus_initial'], label='energy - initial')
    plt.xlabel('time [s]')
    plt.title(title)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def run_centered_sphere_free_contact_release() -> tuple[pd.DataFrame, dict]:
    R = 0.12
    h0 = 0.02
    world = _sphere_plane_world(x=0.0, y=R + h0, vy=0.0, R=R)
    df, ic, ir, _ = _run_until_first_release(world, 'ball', delta_fn=lambda y: R - y, potential_fn=lambda d: _sphere_contact_potential(R, d))
    df['static_Fy_at_delta'] = [K_EQ * cap_volume(R, float(d)) for d in df['delta']]
    df['static_ground_Mz_at_delta'] = 0.0
    df.to_csv(OUT / 'sphere_centered_free_contact_release.csv', index=False)
    _plot_force_energy(df, 'Free-body slow contact/release: centered sphere', OUT / 'sphere_centered_free_contact_release.png', include_moment=False)
    summary = _summarize_common(df, ic, ir, h0=h0, analytic_delta_max=_solve_delta_max_sphere(R, h0), touch_speed=math.sqrt(2.0 * GRAVITY * h0), x_ref=0.0)
    summary['case'] = 'sphere_centered_free_contact_release'
    return df, summary


def run_centered_flat_punch_free_contact_release() -> tuple[pd.DataFrame, dict]:
    ext = np.array([0.2, 0.1, 0.15], dtype=float)
    area = float(ext[0] * ext[2])
    h0 = 0.01
    world = _box_plane_world(x=0.0, y=0.5 * ext[1] + h0, vy=0.0, extents=ext)
    df, ic, ir, _ = _run_until_first_release(world, 'box', delta_fn=lambda y: 0.5 * ext[1] - y, potential_fn=lambda d: _box_contact_potential(area, d))
    df['static_Fy_at_delta'] = area * K_EQ * df['delta']
    df['static_ground_Mz_at_delta'] = 0.0
    df.to_csv(OUT / 'flat_punch_centered_free_contact_release.csv', index=False)
    _plot_force_energy(df, 'Free-body slow contact/release: centered flat punch', OUT / 'flat_punch_centered_free_contact_release.png', include_moment=False)
    summary = _summarize_common(df, ic, ir, h0=h0, analytic_delta_max=_solve_delta_max_box(area, h0), touch_speed=math.sqrt(2.0 * GRAVITY * h0), x_ref=0.0)
    summary['case'] = 'flat_punch_centered_free_contact_release'
    return df, summary


def run_offaxis_sphere_free_contact_release() -> tuple[pd.DataFrame, dict]:
    R = 0.12
    h0 = 0.02
    x_off = 0.04
    world = _sphere_plane_world(x=x_off, y=R + h0, vy=0.0, R=R)
    df, ic, ir, _ = _run_until_first_release(world, 'ball', delta_fn=lambda y: R - y, potential_fn=lambda d: _sphere_contact_potential(R, d), x_ref=x_off)
    df['static_Fy_at_delta'] = [K_EQ * cap_volume(R, float(d)) for d in df['delta']]
    df['static_ground_Mz_at_delta'] = -x_off * df['static_Fy_at_delta']
    df.to_csv(OUT / 'sphere_offaxis_free_contact_release.csv', index=False)
    _plot_force_energy(df, 'Free-body slow contact/release: off-axis sphere', OUT / 'sphere_offaxis_free_contact_release.png', include_moment=True)
    summary = _summarize_common(df, ic, ir, h0=h0, analytic_delta_max=_solve_delta_max_sphere(R, h0), touch_speed=math.sqrt(2.0 * GRAVITY * h0), x_ref=x_off)
    summary['case'] = 'sphere_offaxis_free_contact_release'
    return df, summary


def main() -> None:
    rows = []
    _, s0 = run_centered_sphere_free_contact_release(); rows.append(s0)
    _, s1 = run_centered_flat_punch_free_contact_release(); rows.append(s1)
    _, s2 = run_offaxis_sphere_free_contact_release(); rows.append(s2)
    overview = pd.DataFrame(rows)
    overview.to_csv(OUT / 'overview.csv', index=False)

    lines = []
    lines.append('# Free-body slow contact and release benchmarks')
    lines.append('')
    lines.append('Shared configuration (no case-specific tuning):')
    lines.append(f'- patch raster_cells = {PATCH_CFG.raster_cells}')
    lines.append(f'- sheet bisection_steps = {SHEET_CFG.bisection_steps}')
    lines.append(f'- stiffness per side = {K_EQUAL}')
    lines.append(f'- damping_gamma = 0.0')
    lines.append(f'- dt = {DT}')
    lines.append(f'- gravity = {GRAVITY}')
    lines.append('')
    lines.append('The main diagnostics are:')
    lines.append('- first-cycle max penetration vs conservative analytic turning-point prediction')
    lines.append('- first-release upward speed vs conservative analytic touch speed')
    lines.append('- total-energy drift at first release, normalized by the initial drop-energy scale')
    lines.append('- static force-law consistency during the first contact cycle')
    lines.append('- off-axis symmetry diagnostics: x drift, vx drift, body moment pollution, ground Mz consistency')
    lines.append('')
    for _, row in overview.iterrows():
        lines.append(f"## {row['case']}")
        lines.append(f"- first_contact_time = {row['first_contact_time']:.6f}")
        lines.append(f"- first_release_time = {row['first_release_time']:.6f}")
        lines.append(f"- analytic_delta_max = {row['analytic_delta_max']:.6f}")
        lines.append(f"- sim_delta_max_first_cycle = {row['sim_delta_max_first_cycle']:.6f}")
        lines.append(f"- rel_err_delta_max = {row['rel_err_delta_max']:.6f}")
        lines.append(f"- touch_speed_analytic = {row['touch_speed_analytic']:.6f}")
        lines.append(f"- release_speed_sim = {row['release_speed_sim']:.6f}")
        lines.append(f"- rel_err_release_speed = {row['rel_err_release_speed']:.6f}")
        lines.append(f"- release_energy_drift_rel_drop_scale = {row['release_energy_drift_rel_drop_scale']:.6f}")
        lines.append(f"- max_rel_err_static_force_law_first_cycle = {row['max_rel_err_static_force_law_first_cycle']:.6f}")
        if not math.isnan(float(row['max_rel_ground_Mz_relation_first_cycle'])):
            lines.append(f"- max_rel_ground_Mz_relation_first_cycle = {row['max_rel_ground_Mz_relation_first_cycle']:.6f}")
        lines.append(f"- max_abs_x_drift = {row['max_abs_x_drift']:.6e}")
        lines.append(f"- max_abs_vx = {row['max_abs_vx']:.6e}")
        lines.append(f"- max_body_moment_norm_first_cycle = {row['max_body_moment_norm_first_cycle']:.6e}")
        lines.append('')

    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved overview -> {OUT / "overview.csv"}')
    print(f'Saved summary -> {OUT / "summary.md"}')


if __name__ == '__main__':
    main()
