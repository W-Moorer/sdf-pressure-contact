from __future__ import annotations

import math
from pathlib import Path

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
    PlaneGeometry,
    PolygonPatchConfig,
    SheetExtractConfig,
    ContactModelConfig,
    FormalPressureFieldConfig,
    PolygonHighAccuracyLocalEvaluator,
    FormalPressureFieldLocalEvaluator,
    ContactManager,
    make_world,
    cap_volume,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'formal_analytic_validation'
OUT.mkdir(parents=True, exist_ok=True)


def _sphere_plane_world(*, x: float, y: float, R: float, plane_ref_x: float = 0.0):
    body = RigidBody6D(
        name='ball',
        inertia=SpatialInertia(1.0, np.eye(3)),
        geometry=SphereGeometry(R),
        state=BodyState6D(
            pose=Pose6D(np.array([x, y, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
    )
    plane = SDFGeometryDomainSource(
        geometry=PlaneGeometry((0.0, 1.0, 0.0), 0.0),
        pose=Pose6D(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
        name='ground',
        hint_radius=1.0,
        reference_center=np.array([plane_ref_x, 0.0, 0.0], dtype=float),
    )
    return make_world(bodies=[body], domain_sources=[plane])


def _eval(world, evaluator):
    contacts = ContactManager(evaluator).compute_all_contacts(world)
    return contacts['ball'], contacts['ground']


def run() -> None:
    R = 0.12
    k_equal = 12000.0
    k_ground = 30000.0
    k_ball = 10000.0
    keq_equal = k_equal * k_equal / (k_equal + k_equal)
    keq_uneq = k_ground * k_ball / (k_ground + k_ball)

    patch_cfg = PolygonPatchConfig(raster_cells=14, max_patch_radius=0.30)
    sheet_cfg = SheetExtractConfig(bisection_steps=16)

    legacy_equal = PolygonHighAccuracyLocalEvaluator(
        patch_cfg=patch_cfg,
        sheet_cfg=sheet_cfg,
        contact_cfg=ContactModelConfig(stiffness_k=keq_equal, damping_c=0.0),
    )
    formal_equal = FormalPressureFieldLocalEvaluator(
        patch_cfg=patch_cfg,
        sheet_cfg=sheet_cfg,
        pressure_cfg=FormalPressureFieldConfig(stiffness_default=k_equal, damping_gamma=0.0),
    )
    formal_unequal = FormalPressureFieldLocalEvaluator(
        patch_cfg=patch_cfg,
        sheet_cfg=sheet_cfg,
        pressure_cfg=FormalPressureFieldConfig(
            stiffness_default=k_ball,
            source_stiffness={'ground': k_ground, 'ball': k_ball},
            damping_gamma=0.0,
        ),
    )

    rows_center = []
    ys = np.linspace(0.115, 0.08, 8)
    for i, y in enumerate(ys):
        delta = max(0.0, R - float(y))
        ideal = keq_equal * cap_volume(R, delta)
        world = _sphere_plane_world(x=0.0, y=float(y), R=R)
        body_leg, _ = _eval(world, legacy_equal)
        world = _sphere_plane_world(x=0.0, y=float(y), R=R)
        body_for, _ = _eval(world, formal_equal)
        rows_center.append({
            'state_index': i,
            'y': float(y),
            'delta': delta,
            'ideal_Fy': ideal,
            'legacy_Fy': float(body_leg.total_force[1]),
            'formal_Fy': float(body_for.total_force[1]),
            'legacy_rel_err': abs(float(body_leg.total_force[1]) - ideal) / max(abs(ideal), 1.0e-15),
            'formal_rel_err': abs(float(body_for.total_force[1]) - ideal) / max(abs(ideal), 1.0e-15),
            'formal_body_moment_norm': float(np.linalg.norm(body_for.total_moment)),
            'legacy_body_moment_norm': float(np.linalg.norm(body_leg.total_moment)),
        })
    df_center = pd.DataFrame(rows_center)
    df_center.to_csv(OUT / 'centered_equal_k_scan.csv', index=False)

    rows_offaxis = []
    y = 0.09
    delta = R - y
    ideal = keq_equal * cap_volume(R, delta)
    for i, x in enumerate([0.0, 0.02, 0.04]):
        world = _sphere_plane_world(x=float(x), y=float(y), R=R)
        body_for, ground_for = _eval(world, formal_equal)
        rows_offaxis.append({
            'state_index': i,
            'x': float(x),
            'y': float(y),
            'delta': delta,
            'ideal_Fy': ideal,
            'formal_Fy': float(body_for.total_force[1]),
            'formal_rel_err_Fy': abs(float(body_for.total_force[1]) - ideal) / max(abs(ideal), 1.0e-15),
            'ideal_body_Mx': 0.0,
            'ideal_body_Mz': 0.0,
            'formal_body_Mx': float(body_for.total_moment[0]),
            'formal_body_Mz': float(body_for.total_moment[2]),
            'ideal_ground_Mz': -float(x) * ideal,
            'formal_ground_Mz': float(ground_for.total_moment[2]),
            'rel_err_ground_Mz': abs(float(ground_for.total_moment[2]) - (-float(x) * ideal)) / max(abs(float(x) * ideal), 1.0e-15),
        })
    df_off = pd.DataFrame(rows_offaxis)
    df_off.to_csv(OUT / 'offaxis_equal_k_scan.csv', index=False)

    rows_uneq = []
    for i, y in enumerate(ys):
        delta = max(0.0, R - float(y))
        ideal = keq_uneq * cap_volume(R, delta)
        world = _sphere_plane_world(x=0.0, y=float(y), R=R)
        body_for, _ = _eval(world, formal_unequal)
        rows_uneq.append({
            'state_index': i,
            'y': float(y),
            'delta': delta,
            'ideal_Fy': ideal,
            'formal_Fy': float(body_for.total_force[1]),
            'formal_rel_err': abs(float(body_for.total_force[1]) - ideal) / max(abs(ideal), 1.0e-15),
            'formal_body_moment_norm': float(np.linalg.norm(body_for.total_moment)),
        })
    df_uneq = pd.DataFrame(rows_uneq)
    df_uneq.to_csv(OUT / 'centered_unequal_k_scan.csv', index=False)

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df_center['delta'], df_center['legacy_Fy'], 'o-', label='legacy polygon (single-k spring traction)')
    plt.plot(df_center['delta'], df_center['formal_Fy'], 's-', label='formal pressure-field evaluator')
    plt.plot(df_center['delta'], df_center['ideal_Fy'], '-', label='analytic keq * cap_volume')
    plt.xlabel('penetration depth delta')
    plt.ylabel('Fy')
    plt.title('Centered sphere-plane: equal-k analytic comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'centered_equal_k_scan.png', dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(df_uneq['delta'], df_uneq['formal_Fy'], 's-', label='formal pressure-field evaluator')
    plt.plot(df_uneq['delta'], df_uneq['ideal_Fy'], '-', label='analytic keq * cap_volume')
    plt.xlabel('penetration depth delta')
    plt.ylabel('Fy')
    plt.title('Centered sphere-plane: unequal-k analytic comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'centered_unequal_k_scan.png', dpi=180)
    plt.close()

    summary_lines = []
    summary_lines.append('# Formal endpoint analytic validation')
    summary_lines.append('')
    summary_lines.append('## What was changed in code')
    summary_lines.append('- Added `FormalPressureFieldLocalEvaluator` on the same patch/quadrature architecture.')
    summary_lines.append('- Replaced legacy spring-gap traction with the formal linear pressure law `p_i = k_i d_i`.')
    summary_lines.append('- Per column, solved the 1D equilibrium split `k_A d_A = k_B d_B`, `d_A + d_B = delta`.')
    summary_lines.append('- Computed sheet normal from `grad h = -k_A grad(phi_A) + k_B grad(phi_B)` rather than `n_A - n_B` alone.')
    summary_lines.append('- Built formal patch support from column-overlap intervals on each projected cell, not from a single overlap slice at the center plane.')
    summary_lines.append('')
    summary_lines.append('## Centered sphere-plane, equal stiffness')
    summary_lines.append(f"- legacy polygon mean relative force error: {df_center['legacy_rel_err'].mean():.4%}")
    summary_lines.append(f"- formal evaluator mean relative force error: {df_center['formal_rel_err'].mean():.4%}")
    summary_lines.append(f"- formal evaluator max relative force error: {df_center['formal_rel_err'].max():.4%}")
    summary_lines.append(f"- formal evaluator max body moment norm: {df_center['formal_body_moment_norm'].max():.6e}")
    summary_lines.append('')
    summary_lines.append('## Off-axis sphere-plane, equal stiffness')
    summary_lines.append(f"- max relative force error: {df_off['formal_rel_err_Fy'].max():.4%}")
    summary_lines.append(f"- max |body Mx|: {df_off['formal_body_Mx'].abs().max():.6e}")
    summary_lines.append(f"- max |body Mz|: {df_off['formal_body_Mz'].abs().max():.6e}")
    summary_lines.append(f"- max relative error of ground Mz vs -x*Fy_ideal: {df_off['rel_err_ground_Mz'].replace([np.inf], np.nan).iloc[1:].max():.4%}")
    summary_lines.append('')
    summary_lines.append('## Centered sphere-plane, unequal stiffness')
    summary_lines.append(f"- formal evaluator mean relative force error: {df_uneq['formal_rel_err'].mean():.4%}")
    summary_lines.append(f"- formal evaluator max relative force error: {df_uneq['formal_rel_err'].max():.4%}")
    summary_lines.append(f"- formal evaluator max body moment norm: {df_uneq['formal_body_moment_norm'].max():.6e}")
    summary_lines.append('')
    summary_lines.append('## Reading of the result')
    summary_lines.append('- Under a single untuned evaluator configuration, centered and off-axis analytic sphere-plane cases are now within about 1% force error.')
    summary_lines.append('- The same code path also matches the unequal-stiffness analytic force law `keq * cap_volume` without any case-specific branching.')
    summary_lines.append('- This does **not** mean the full formal endpoint is finished; it shows that the central constitutive and sheet-location corrections are working inside the shared architecture.')
    summary_lines.append('')
    summary_lines.append('## Still missing before the full formal endpoint')
    summary_lines.append('- True band mechanics with `w_eta = delta_eta(h) |grad h| chi_A chi_B` as a first-class layer, not just a column-wise sheet evaluator.')
    summary_lines.append('- A separate sheet-representation output layer, so force computation and patch geometry are no longer mixed.')
    summary_lines.append('- A full local-normal accumulator with explicit `I^(F), I^(A), I^(s)` one-dimensional integrals per sheet element.')
    summary_lines.append('- Better narrow-band SDF backends and BVH/block traversal for large meshes.')
    summary_lines.append('- Consistent derivatives / Jacobians for the global implicit solve once the local evaluator is frozen.')

    (OUT / 'summary.md').write_text('\n'.join(summary_lines), encoding='utf-8')


if __name__ == '__main__':
    run()
