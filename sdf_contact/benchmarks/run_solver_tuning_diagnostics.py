#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(ROOT))

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

OUT = ROOT / 'results' / 'solver_tuning_diagnostics'
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

LEGACY_CFG = IntegratorConfig(
    dt=DT,
    newton_max_iter=2,
    newton_tol=1.0e-8,
    fd_eps=1.0e-4,
    line_search_factors=(1.0, 0.5),
    scheme='backward_euler',
    jacobian_mode='forward',
    predictor_mode='current_velocity',
    linear_regularization=1.0e-10,
    collect_diagnostics=True,
)

TUNED_CFG = IntegratorConfig(
    dt=DT,
    newton_max_iter=4,
    newton_tol=1.0e-9,
    fd_eps=5.0e-5,
    line_search_factors=(1.0, 0.75, 0.5, 0.25),
    scheme='implicit_midpoint',
    jacobian_mode='central',
    predictor_mode='explicit_force',
    linear_regularization=1.0e-8,
    collect_diagnostics=True,
)


def _solver(cfg: IntegratorConfig):
    return GlobalImplicitSystemSolver6D(CONTACT_MANAGER, cfg)


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


def _run_until_first_release(world, body_name: str, delta_fn, potential_fn, solver: GlobalImplicitSystemSolver6D, x_ref: float = 0.0, max_steps: int = 80, post_release_steps: int = 2):
    rows = []
    first_contact_idx = None
    first_release_idx = None
    release_buffer = None

    body0 = next(b for b in world.bodies if b.name == body_name)
    y0 = float(body0.state.pose.position[1])
    E0 = MASS * GRAVITY * y0

    for step in range(max_steps):
        infos = solver.step_world(world)
        contacts = CONTACT_MANAGER.compute_all_contacts(world)
        body = next(b for b in world.bodies if b.name == body_name)
        agg_body = contacts[body_name]
        agg_ground = contacts['ground']
        t = (step + 1) * DT
        x = float(body.state.pose.position[0])
        y = float(body.state.pose.position[1])
        vx = float(body.state.linear_velocity[0])
        vy = float(body.state.linear_velocity[1])
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
            'delta': delta,
            'contact_active': int(active),
            'Fy': Fy,
            'F_lateral_norm': float(np.linalg.norm(agg_body.total_force[[0, 2]])),
            'body_moment_norm': float(np.linalg.norm(agg_body.total_moment)),
            'ground_Mz': float(agg_ground.total_moment[2]),
            'static_Fy_at_delta': 0.0,
            'static_ground_Mz_at_delta': 0.0,
            'total_energy': total_energy,
            'energy_minus_initial': total_energy - E0,
            'x_ref': x_ref,
            'solver_final_residual_norm': float(solver.last_step_diagnostics['final_residual_norm']),
            'solver_newton_iters': int(len(solver.last_step_diagnostics['iterations'])),
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
    return pd.DataFrame(rows), first_contact_idx, first_release_idx


def _summarize_common(df: pd.DataFrame, first_contact_idx: int | None, first_release_idx: int | None, *, h0: float, analytic_delta_max: float, touch_speed: float, x_ref: float = 0.0):
    out = {
        'n_steps': int(len(df)),
        'first_contact_time': float(df.iloc[first_contact_idx]['time']) if first_contact_idx is not None else float('nan'),
        'first_release_time': float(df.iloc[first_release_idx]['time']) if first_release_idx is not None else float('nan'),
        'touch_speed_analytic': float(touch_speed),
        'analytic_delta_max': float(analytic_delta_max),
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
            'max_solver_final_residual_norm_first_cycle': float('nan'),
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
    out['max_solver_final_residual_norm_first_cycle'] = float(cyc['solver_final_residual_norm'].max())
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


def _run_case(case_name: str, solver_name: str, cfg: IntegratorConfig):
    solver = _solver(cfg)
    if case_name == 'sphere_centered':
        R = 0.12; h0 = 0.02
        world = _sphere_plane_world(x=0.0, y=R + h0, vy=0.0, R=R)
        df, ic, ir = _run_until_first_release(world, 'ball', delta_fn=lambda y: R - y, potential_fn=lambda d: _sphere_contact_potential(R, d), solver=solver)
        df['static_Fy_at_delta'] = [K_EQ * cap_volume(R, float(d)) for d in df['delta']]
        df['static_ground_Mz_at_delta'] = 0.0
        summary = _summarize_common(df, ic, ir, h0=h0, analytic_delta_max=_solve_delta_max_sphere(R, h0), touch_speed=math.sqrt(2.0 * GRAVITY * h0), x_ref=0.0)
        include_moment = False
    elif case_name == 'flat_punch_centered':
        ext = np.array([0.2, 0.1, 0.15], dtype=float); area = float(ext[0] * ext[2]); h0 = 0.01
        world = _box_plane_world(x=0.0, y=0.5 * ext[1] + h0, vy=0.0, extents=ext)
        df, ic, ir = _run_until_first_release(world, 'box', delta_fn=lambda y: 0.5 * ext[1] - y, potential_fn=lambda d: _box_contact_potential(area, d), solver=solver)
        df['static_Fy_at_delta'] = area * K_EQ * df['delta']
        df['static_ground_Mz_at_delta'] = 0.0
        summary = _summarize_common(df, ic, ir, h0=h0, analytic_delta_max=_solve_delta_max_box(area, h0), touch_speed=math.sqrt(2.0 * GRAVITY * h0), x_ref=0.0)
        include_moment = False
    elif case_name == 'sphere_offaxis':
        R = 0.12; h0 = 0.02; x_off = 0.04
        world = _sphere_plane_world(x=x_off, y=R + h0, vy=0.0, R=R)
        df, ic, ir = _run_until_first_release(world, 'ball', delta_fn=lambda y: R - y, potential_fn=lambda d: _sphere_contact_potential(R, d), solver=solver, x_ref=x_off)
        df['static_Fy_at_delta'] = [K_EQ * cap_volume(R, float(d)) for d in df['delta']]
        df['static_ground_Mz_at_delta'] = -x_off * df['static_Fy_at_delta']
        summary = _summarize_common(df, ic, ir, h0=h0, analytic_delta_max=_solve_delta_max_sphere(R, h0), touch_speed=math.sqrt(2.0 * GRAVITY * h0), x_ref=x_off)
        include_moment = True
    else:
        raise ValueError(case_name)
    summary['case'] = case_name
    summary['solver'] = solver_name
    df.to_csv(OUT / f'{case_name}_{solver_name}.csv', index=False)
    _plot_force_energy(df, f'{case_name} [{solver_name}]', OUT / f'{case_name}_{solver_name}.png', include_moment=include_moment)
    return df, summary


def _find_precontact_world(cfg: IntegratorConfig):
    R = 0.12
    world = _sphere_plane_world(x=0.0, y=R + 0.02, vy=0.0, R=R)
    solver = _solver(cfg)
    prev = copy.deepcopy(world)
    for _ in range(40):
        prev = copy.deepcopy(world)
        infos = solver.step_world(world)
        if infos[0]['contact_force'][1] > 1.0e-8:
            return prev, solver
    return prev, solver


def _jacobian_for_mode(solver: GlobalImplicitSystemSolver6D, world, mode: str):
    dyn = solver._dynamic_indices(world)
    U = solver._predict_initial_unknowns(world, dyn)
    R, _, _ = solver._eval_global_residual(world, dyn, U)
    old = solver.cfg.jacobian_mode
    solver.cfg.jacobian_mode = mode
    J = solver._build_fd_jacobian(world, dyn, U, R)
    solver.cfg.jacobian_mode = old
    return dyn, U, R, J


def run_linearization_diagnostics() -> None:
    rows = []
    for name, cfg in [('legacy', copy.deepcopy(LEGACY_CFG)), ('tuned', copy.deepcopy(TUNED_CFG))]:
        pre, solver = _find_precontact_world(cfg)
        info = solver.linearize_current_step(pre)
        dyn, U, R, Jf = _jacobian_for_mode(solver, pre, 'forward')
        _, _, _, Jc = _jacobian_for_mode(solver, pre, 'central')
        rel_fd_gap = float(np.linalg.norm(Jf - Jc) / max(np.linalg.norm(Jc), 1.0e-15))
        try:
            dUf = np.linalg.lstsq(Jf, -R, rcond=None)[0]
            pred_f = float(np.linalg.norm(R + Jf @ dUf)) / max(float(np.linalg.norm(R)), 1.0e-15)
        except np.linalg.LinAlgError:
            pred_f = float('nan')
        try:
            dUc = np.linalg.lstsq(Jc, -R, rcond=None)[0]
            pred_c = float(np.linalg.norm(R + Jc @ dUc)) / max(float(np.linalg.norm(R)), 1.0e-15)
        except np.linalg.LinAlgError:
            pred_c = float('nan')
        rows.append({
            'solver': name,
            'scheme': cfg.scheme,
            'predictor_mode': cfg.predictor_mode,
            'residual_norm_at_predictor': float(np.linalg.norm(R)),
            'jacobian_cond_active_mode': float(info['cond']),
            'sigma_min_active_mode': float(info['sigma_min']),
            'sigma_max_active_mode': float(info['sigma_max']),
            'rel_gap_forward_vs_central': rel_fd_gap,
            'predicted_linear_residual_ratio_forward': pred_f,
            'predicted_linear_residual_ratio_central': pred_c,
        })

        # Response curve with respect to candidate vertical end-velocity.
        vy0 = float(U[1])
        samples = []
        for dvy in np.linspace(-0.35, 0.35, 41):
            Us = U.copy(); Us[1] = vy0 + dvy
            Rv, ew, forces = solver._eval_global_residual(pre, dyn, Us)
            body = ew.bodies[0]
            Fy = float(forces[0].contact_force[1])
            samples.append({
                'solver': name,
                'candidate_vy': float(Us[1]),
                'trial_y_eval': float(body.state.pose.position[1]),
                'trial_vy_eval': float(body.state.linear_velocity[1]),
                'trial_Fy': Fy,
                'residual_norm': float(np.linalg.norm(Rv)),
                'residual_y': float(Rv[1]),
            })
        sdf = pd.DataFrame(samples)
        sdf.to_csv(OUT / f'contact_response_scan_{name}.csv', index=False)
        plt.figure(figsize=(7.4, 4.6))
        plt.plot(sdf['trial_y_eval'], sdf['trial_Fy'], label='trial Fy')
        plt.plot(sdf['trial_y_eval'], sdf['residual_y'], label='residual y')
        plt.gca().invert_xaxis()
        plt.xlabel('trial evaluation y')
        plt.title(f'Contact response scan [{name}]')
        plt.legend(); plt.tight_layout()
        plt.savefig(OUT / f'contact_response_scan_{name}.png', dpi=180)
        plt.close()

    pd.DataFrame(rows).to_csv(OUT / 'linearization_diagnostics.csv', index=False)


def _write_single_case_json(case: str, solver_name: str) -> None:
    cfg = copy.deepcopy(LEGACY_CFG if solver_name == 'legacy' else TUNED_CFG)
    _, summary = _run_case(case, solver_name, cfg)
    out = OUT / 'json'
    out.mkdir(parents=True, exist_ok=True)
    (out / f'{case}_{solver_name}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


def _build_summary_from_json() -> None:
    json_dir = OUT / 'json'
    rows = [json.loads(fp.read_text()) for fp in sorted(json_dir.glob('*.json'))]
    overview = pd.DataFrame(rows)
    overview.to_csv(OUT / 'overview.csv', index=False)

    compare_rows = []
    for case in ('sphere_centered', 'flat_punch_centered', 'sphere_offaxis'):
        leg = overview[(overview['case'] == case) & (overview['solver'] == 'legacy')].iloc[0]
        tun = overview[(overview['case'] == case) & (overview['solver'] == 'tuned')].iloc[0]
        compare_rows.append({
            'case': case,
            'release_speed_error_legacy': float(leg['rel_err_release_speed']),
            'release_speed_error_tuned': float(tun['rel_err_release_speed']),
            'energy_drift_legacy': float(leg['release_energy_drift_rel_drop_scale']),
            'energy_drift_tuned': float(tun['release_energy_drift_rel_drop_scale']),
            'delta_error_legacy': float(leg['rel_err_delta_max']),
            'delta_error_tuned': float(tun['rel_err_delta_max']),
            'force_law_max_err_legacy': float(leg['max_rel_err_static_force_law_first_cycle']),
            'force_law_max_err_tuned': float(tun['max_rel_err_static_force_law_first_cycle']),
            'ground_mz_err_legacy': float(leg['max_rel_ground_Mz_relation_first_cycle']) if not pd.isna(leg['max_rel_ground_Mz_relation_first_cycle']) else float('nan'),
            'ground_mz_err_tuned': float(tun['max_rel_ground_Mz_relation_first_cycle']) if not pd.isna(tun['max_rel_ground_Mz_relation_first_cycle']) else float('nan'),
        })
    pd.DataFrame(compare_rows).to_csv(OUT / 'comparison.csv', index=False)

    lin = pd.read_csv(OUT / 'linearization_diagnostics.csv')
    lines = []
    lines.append('# Global implicit solver tuning diagnostics')
    lines.append('')
    lines.append('This report compares the previous free-body solver configuration against a tuned configuration that changes only the global solver layer:')
    lines.append('')
    lines.append('- legacy: backward Euler + forward-difference Jacobian + current-velocity predictor')
    lines.append('- tuned: implicit midpoint kinematics/force evaluation + central-difference Jacobian + explicit-force predictor')
    lines.append('')
    lines.append('Shared contact evaluator/configuration is unchanged across all cases.')
    lines.append('')
    for case in ('sphere_centered', 'flat_punch_centered', 'sphere_offaxis'):
        leg = overview[(overview['case'] == case) & (overview['solver'] == 'legacy')].iloc[0]
        tun = overview[(overview['case'] == case) & (overview['solver'] == 'tuned')].iloc[0]
        lines.append(f'## {case}')
        lines.append(f"- release speed error: legacy {leg['rel_err_release_speed']:.4f} -> tuned {tun['rel_err_release_speed']:.4f}")
        lines.append(f"- release energy drift / drop scale: legacy {leg['release_energy_drift_rel_drop_scale']:.4f} -> tuned {tun['release_energy_drift_rel_drop_scale']:.4f}")
        lines.append(f"- first-cycle delta error: legacy {leg['rel_err_delta_max']:.4f} -> tuned {tun['rel_err_delta_max']:.4f}")
        lines.append(f"- first-cycle max static-force-law error: legacy {leg['max_rel_err_static_force_law_first_cycle']:.4f} -> tuned {tun['max_rel_err_static_force_law_first_cycle']:.4f}")
        if not pd.isna(tun['max_rel_ground_Mz_relation_first_cycle']):
            lines.append(f"- off-axis ground Mz relation max error: legacy {leg['max_rel_ground_Mz_relation_first_cycle']:.4f} -> tuned {tun['max_rel_ground_Mz_relation_first_cycle']:.4f}")
        lines.append('')
    lines.append('## Linearization / Jacobian diagnostics near first contact')
    for _, row in lin.iterrows():
        lines.append(f"### {row['solver']}")
        lines.append(f"- scheme = {row['scheme']}")
        lines.append(f"- predictor_mode = {row['predictor_mode']}")
        lines.append(f"- predictor residual norm = {row['residual_norm_at_predictor']:.6e}")
        lines.append(f"- Jacobian condition number = {row['jacobian_cond_active_mode']:.6e}")
        lines.append(f"- sigma_min / sigma_max = {row['sigma_min_active_mode']:.6e} / {row['sigma_max_active_mode']:.6e}")
        lines.append(f"- relative gap between forward and central FD Jacobians = {row['rel_gap_forward_vs_central']:.6e}")
        lines.append(f"- linearized residual ratio with forward FD = {row['predicted_linear_residual_ratio_forward']:.6e}")
        lines.append(f"- linearized residual ratio with central FD = {row['predicted_linear_residual_ratio_central']:.6e}")
        lines.append('')
    PLACEHOLDER


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-case', default='')
    ap.add_argument('--solver', default='')
    args = ap.parse_args()
    if args.single_case:
        if args.solver not in {'legacy', 'tuned'}:
            raise ValueError('--solver must be legacy or tuned when using --single-case')
        _write_single_case_json(args.single_case, args.solver)
        print(f'Saved single case -> {args.single_case} [{args.solver}]')
        return

    for solver_name in ('legacy', 'tuned'):
        for case_name in ('sphere_centered', 'flat_punch_centered', 'sphere_offaxis'):
            print(f'Running {case_name} [{solver_name}]', flush=True)
            subprocess.run([sys.executable, __file__, '--single-case', case_name, '--solver', solver_name], check=True)
    print('Running linearization diagnostics', flush=True)
    run_linearization_diagnostics()
    _build_summary_from_json()
    print(f'Saved -> {OUT / "overview.csv"}')
    print(f'Saved -> {OUT / "comparison.csv"}')
    print(f'Saved -> {OUT / "linearization_diagnostics.csv"}')
    print(f'Saved -> {OUT / "summary.md"}')


if __name__ == '__main__':
    main()
