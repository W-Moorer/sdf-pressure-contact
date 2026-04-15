from __future__ import annotations

import math
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh

from sdf_contact import (
    SpatialInertia,
    RigidBody6D,
    BodyState6D,
    Pose6D,
    SDFGeometryDomainSource,
    SphereGeometry,
    PlaneGeometry,
    MeshGeometryFactoryConfig,
    make_mesh_rigidbody,
    make_world,
    BaselinePatchConfig,
    PolygonPatchConfig,
    SheetExtractConfig,
    ContactModelConfig,
    BaselineGridLocalEvaluator,
    PolygonHighAccuracyLocalEvaluator,
    ContactManager,
    cap_volume,
)
from sdf_contact.core import quat_from_rotvec

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'local_benchmarks'
OUT.mkdir(parents=True, exist_ok=True)


def _primitive_sphere_world(*, x: float, y: float, R: float, plane_ref_x: float = 0.0):
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


def _primitive_default_evaluators(k_contact: float):
    base = BaselineGridLocalEvaluator(
        BaselinePatchConfig(Nuv=10, quad_order=2, max_patch_radius=0.3),
        SheetExtractConfig(),
        ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
    )
    poly = PolygonHighAccuracyLocalEvaluator(
        PolygonPatchConfig(raster_cells=14, max_patch_radius=0.3),
        SheetExtractConfig(),
        ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
    )
    poly_ref = PolygonHighAccuracyLocalEvaluator(
        PolygonPatchConfig(raster_cells=28, max_patch_radius=0.3),
        SheetExtractConfig(),
        ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
    )
    return {'baseline_default': base, 'polygon_default': poly, 'polygon_reference': poly_ref}


def _eval_world(world, evaluator, body_name='ball', domain_name='ground'):
    t0 = time.perf_counter()
    contacts = ContactManager(evaluator).compute_all_contacts(world)
    elapsed = time.perf_counter() - t0
    body = contacts[body_name]
    domain = contacts[domain_name]
    return {
        'elapsed_s': elapsed,
        'body_force_x': float(body.total_force[0]),
        'body_force_y': float(body.total_force[1]),
        'body_force_z': float(body.total_force[2]),
        'body_moment_x': float(body.total_moment[0]),
        'body_moment_y': float(body.total_moment[1]),
        'body_moment_z': float(body.total_moment[2]),
        'ground_force_x': float(domain.total_force[0]),
        'ground_force_y': float(domain.total_force[1]),
        'ground_force_z': float(domain.total_force[2]),
        'ground_moment_x': float(domain.total_moment[0]),
        'ground_moment_y': float(domain.total_moment[1]),
        'ground_moment_z': float(domain.total_moment[2]),
        'num_pair_patch_points': int(body.num_pair_patch_points),
        'num_pair_sheet_points': int(body.num_pair_sheet_points),
        'num_pair_tractions': int(body.num_pair_tractions),
    }


def run_centered_sphere_scan():
    R = 0.12
    k_contact = 12000.0
    evaluators = _primitive_default_evaluators(k_contact)
    ys = np.linspace(0.115, 0.08, 8)
    rows: list[dict] = []

    for state_idx, y in enumerate(ys):
        world = _primitive_sphere_world(x=0.0, y=float(y), R=R, plane_ref_x=0.0)
        delta = max(0.0, R - float(y))
        ideal_fy = k_contact * cap_volume(R, delta)
        for eval_name, evaluator in evaluators.items():
            rec = _eval_world(world, evaluator)
            fy = rec['body_force_y']
            rows.append({
                'case': 'centered_sphere_plane_scan',
                'state_index': state_idx,
                'x': 0.0,
                'y': float(y),
                'delta': delta,
                'evaluator': eval_name,
                'Fy': fy,
                'Mz_body': rec['body_moment_z'],
                'elapsed_s': rec['elapsed_s'],
                'ideal_Fy': ideal_fy,
                'abs_err_Fy': abs(fy - ideal_fy),
                'rel_err_Fy': abs(fy - ideal_fy) / max(abs(ideal_fy), 1.0e-12),
                'num_pair_patch_points': rec['num_pair_patch_points'],
                'num_pair_sheet_points': rec['num_pair_sheet_points'],
                'num_pair_tractions': rec['num_pair_tractions'],
            })

    df = pd.DataFrame(rows)
    out_csv = OUT / 'centered_sphere_plane_scan.csv'
    df.to_csv(out_csv, index=False)

    # calibration: one scalar alpha per evaluator to fit ideal_Fy = alpha * Fy
    cal_rows = []
    for eval_name, sub in df.groupby('evaluator'):
        f = sub['Fy'].to_numpy(dtype=float)
        ideal = sub['ideal_Fy'].to_numpy(dtype=float)
        alpha = float(np.dot(f, ideal) / max(np.dot(f, f), 1.0e-15))
        cal = alpha * f
        cal_rows.append({
            'evaluator': eval_name,
            'alpha_best_fit': alpha,
            'rmse_calibrated': float(np.sqrt(np.mean((cal - ideal) ** 2))),
            'mean_rel_err_calibrated': float(np.mean(np.abs(cal - ideal) / np.maximum(np.abs(ideal), 1.0e-12))),
            'mean_rel_err_raw': float(sub['rel_err_Fy'].mean()),
            'mean_elapsed_s': float(sub['elapsed_s'].mean()),
        })
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(OUT / 'centered_sphere_plane_calibration_summary.csv', index=False)

    plt.figure(figsize=(7.0, 4.5))
    for eval_name, sub in df.groupby('evaluator'):
        plt.plot(sub['delta'], sub['Fy'], marker='o', label=eval_name)
    ref = df[df['evaluator'] == 'baseline_default'][['delta', 'ideal_Fy']].sort_values('delta')
    plt.plot(ref['delta'], ref['ideal_Fy'], label='ideal cap-volume')
    plt.xlabel('penetration depth delta')
    plt.ylabel('normal force Fy')
    plt.title('Centered sphere-plane scan')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'centered_sphere_plane_scan.png', dpi=180)
    plt.close()

    return df, cal_df


def run_centered_sphere_convergence():
    R = 0.12
    y = 0.09
    delta = R - y
    k_contact = 12000.0
    ideal_fy = k_contact * cap_volume(R, delta)
    world = _primitive_sphere_world(x=0.0, y=y, R=R, plane_ref_x=0.0)

    rows: list[dict] = []
    for Nuv in [4, 6, 8, 10, 12, 16, 20, 24, 28]:
        evaluator = BaselineGridLocalEvaluator(
            BaselinePatchConfig(Nuv=Nuv, quad_order=2, max_patch_radius=0.3),
            SheetExtractConfig(),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        )
        rec = _eval_world(world, evaluator)
        rows.append({
            'family': 'baseline',
            'resolution': Nuv,
            'Fy': rec['body_force_y'],
            'ideal_Fy': ideal_fy,
            'abs_err_Fy': abs(rec['body_force_y'] - ideal_fy),
            'elapsed_s': rec['elapsed_s'],
            'num_pair_patch_points': rec['num_pair_patch_points'],
        })
    for raster in [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28]:
        evaluator = PolygonHighAccuracyLocalEvaluator(
            PolygonPatchConfig(raster_cells=raster, max_patch_radius=0.3),
            SheetExtractConfig(),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        )
        rec = _eval_world(world, evaluator)
        rows.append({
            'family': 'polygon',
            'resolution': raster,
            'Fy': rec['body_force_y'],
            'ideal_Fy': ideal_fy,
            'abs_err_Fy': abs(rec['body_force_y'] - ideal_fy),
            'elapsed_s': rec['elapsed_s'],
            'num_pair_patch_points': rec['num_pair_patch_points'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'centered_sphere_plane_convergence.csv', index=False)

    plt.figure(figsize=(7.0, 4.5))
    for fam, sub in df.groupby('family'):
        plt.plot(sub['resolution'], sub['Fy'], marker='o', label=fam)
    plt.axhline(ideal_fy, linewidth=1.0, linestyle='--', label='ideal cap-volume')
    plt.xlabel('resolution parameter')
    plt.ylabel('normal force Fy')
    plt.title('Centered sphere-plane convergence at fixed penetration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'centered_sphere_plane_convergence.png', dpi=180)
    plt.close()

    return df


def run_mesh_tilted_box_scan():
    k_contact = 12000.0
    body_cfg = MeshGeometryFactoryConfig(recenter_mode='center_mass')
    box_mesh = trimesh.creation.box(extents=[0.12, 0.06, 0.08])
    states = [
        {'state_index': 0, 'tilt_deg': 8.0, 'x': 0.01, 'y': 0.026},
        {'state_index': 1, 'tilt_deg': 12.0, 'x': 0.01, 'y': 0.024},
        {'state_index': 2, 'tilt_deg': 16.0, 'x': 0.01, 'y': 0.023},
    ]
    evaluators = {
        'baseline_default': BaselineGridLocalEvaluator(
            BaselinePatchConfig(Nuv=10, quad_order=2, max_patch_radius=0.2),
            SheetExtractConfig(bisection_steps=12),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        ),
        'polygon_default': PolygonHighAccuracyLocalEvaluator(
            PolygonPatchConfig(raster_cells=14, max_patch_radius=0.2),
            SheetExtractConfig(bisection_steps=12),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        ),
        'polygon_reference': PolygonHighAccuracyLocalEvaluator(
            PolygonPatchConfig(raster_cells=24, max_patch_radius=0.2),
            SheetExtractConfig(bisection_steps=16),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        ),
    }

    rows: list[dict] = []
    for st in states:
        q = quat_from_rotvec(np.array([0.0, 0.0, math.radians(st['tilt_deg'])], dtype=float))
        body = make_mesh_rigidbody(
            name='box',
            mesh_or_path=box_mesh,
            mass=1.0,
            position=[st['x'], st['y'], 0.0],
            orientation=q,
            cfg=body_cfg,
        )
        plane = SDFGeometryDomainSource(
            geometry=PlaneGeometry((0.0, 1.0, 0.0), 0.0),
            pose=Pose6D(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            name='ground',
            hint_radius=1.0,
            reference_center=np.array([st['x'], 0.0, 0.0], dtype=float),
        )
        world = make_world(bodies=[body], domain_sources=[plane])
        ref_fy = None
        ref_mz = None
        rec_cache: dict[str, dict] = {}
        for eval_name, evaluator in evaluators.items():
            rec_cache[eval_name] = _eval_world(world, evaluator, body_name='box', domain_name='ground')
        ref_fy = rec_cache['polygon_reference']['body_force_y']
        ref_mz = rec_cache['polygon_reference']['body_moment_z']
        for eval_name, rec in rec_cache.items():
            rows.append({
                'case': 'mesh_tilted_box_plane_scan',
                'state_index': st['state_index'],
                'tilt_deg': st['tilt_deg'],
                'x': st['x'],
                'y': st['y'],
                'evaluator': eval_name,
                'Fy': rec['body_force_y'],
                'Mz_body': rec['body_moment_z'],
                'elapsed_s': rec['elapsed_s'],
                'Fy_ref_polygon': ref_fy,
                'Mz_ref_polygon': ref_mz,
                'abs_err_vs_ref_Fy': abs(rec['body_force_y'] - ref_fy),
                'abs_err_vs_ref_Mz': abs(rec['body_moment_z'] - ref_mz),
                'num_pair_patch_points': rec['num_pair_patch_points'],
                'num_pair_sheet_points': rec['num_pair_sheet_points'],
                'num_pair_tractions': rec['num_pair_tractions'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'mesh_tilted_box_plane_scan.csv', index=False)

    plt.figure(figsize=(7.0, 4.5))
    for eval_name, sub in df.groupby('evaluator'):
        plt.plot(sub['tilt_deg'], sub['Fy'], marker='o', label=eval_name)
    plt.xlabel('tilt [deg]')
    plt.ylabel('normal force Fy')
    plt.title('Tilted mesh box vs plane')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / 'mesh_tilted_box_plane_scan.png', dpi=180)
    plt.close()

    return df


def run_offaxis_reference_center_sensitivity():
    R = 0.12
    y = 0.09
    x = 0.04
    k_contact = 12000.0
    evaluators = {
        'baseline_default': BaselineGridLocalEvaluator(
            BaselinePatchConfig(Nuv=10, quad_order=2, max_patch_radius=0.3),
            SheetExtractConfig(),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        ),
        'polygon_default': PolygonHighAccuracyLocalEvaluator(
            PolygonPatchConfig(raster_cells=14, max_patch_radius=0.3),
            SheetExtractConfig(),
            ContactModelConfig(stiffness_k=k_contact, damping_c=0.0),
        ),
    }

    rows: list[dict] = []
    for ref_mode, ref_x in [('origin_reference', 0.0), ('aligned_reference', x)]:
        world = _primitive_sphere_world(x=x, y=y, R=R, plane_ref_x=ref_x)
        for eval_name, evaluator in evaluators.items():
            rec = _eval_world(world, evaluator)
            fy = rec['body_force_y']
            fx = rec['body_force_x']
            rows.append({
                'case': 'offaxis_reference_center_sensitivity',
                'reference_mode': ref_mode,
                'evaluator': eval_name,
                'x': x,
                'y': y,
                'body_force_x': fx,
                'body_force_y': fy,
                'horizontal_to_vertical_ratio': abs(fx) / max(abs(fy), 1.0e-12),
                'elapsed_s': rec['elapsed_s'],
                'num_pair_patch_points': rec['num_pair_patch_points'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'offaxis_reference_center_sensitivity.csv', index=False)
    return df


def write_summary(centered_df: pd.DataFrame, calibration_df: pd.DataFrame, convergence_df: pd.DataFrame, mesh_df: pd.DataFrame, offaxis_df: pd.DataFrame):
    lines: list[str] = []
    lines.append('# Local evaluator benchmark summary')
    lines.append('')
    lines.append('## 1) Centered primitive sphere-plane scan')
    raw = centered_df.groupby('evaluator').agg(mean_rel_err_Fy=('rel_err_Fy', 'mean'), mean_elapsed_s=('elapsed_s', 'mean'))
    lines.append(raw.to_markdown())
    lines.append('')
    lines.append('Best-fit single-scale calibration against the ideal cap-volume force:')
    lines.append(calibration_df.to_markdown(index=False))
    lines.append('')
    lines.append('- Observation: baseline and polygon-default both under-predict the ideal force by a similar raw margin; after one scalar calibration, polygon has lower RMSE than baseline, but the gain is modest.')
    lines.append('')

    lines.append('## 2) Fixed-penetration convergence')
    conv_tail = convergence_df.sort_values(['family', 'resolution']).groupby('family').tail(1)[['family', 'resolution', 'Fy', 'ideal_Fy', 'abs_err_Fy', 'elapsed_s']]
    lines.append(conv_tail.to_markdown(index=False))
    lines.append('')
    lines.append('- Observation: both evaluator families converge to roughly the same force plateau, still well below the ideal cap-volume prediction. This indicates that the dominant remaining error is not just coarse patch quadrature.')
    lines.append('')

    lines.append('## 3) Tilted mesh box vs plane')
    mesh_summary = mesh_df[mesh_df['evaluator'] != 'polygon_reference'].groupby('evaluator').agg(
        mean_abs_err_vs_ref_Fy=('abs_err_vs_ref_Fy', 'mean'),
        mean_abs_err_vs_ref_Mz=('abs_err_vs_ref_Mz', 'mean'),
        mean_elapsed_s=('elapsed_s', 'mean'),
    )
    lines.append(mesh_summary.to_markdown())
    lines.append('')
    lines.append('- Observation: polygon-default is usually closer to the higher-resolution polygon reference on moment and on some force states, but the improvement is not uniform across all tilted mesh cases.')
    lines.append('')

    lines.append('## 4) Off-axis sensitivity to plane reference center')
    lines.append(offaxis_df.to_markdown(index=False))
    lines.append('')
    lines.append('- Observation: after the geometry-driven pair-frame update, moving only the plane reference center no longer changes the computed force in any meaningful way for this off-axis sphere-plane case. The spurious horizontal force ratio collapses from order 1e-1 to numerical-noise level.')
    lines.append('')

    lines.append('## Overall conclusion')
    lines.append('- The local high-accuracy evaluator is worth keeping, but the benchmark does **not** yet justify a global-solver rewrite.')
    lines.append('- The initial pair-frame / reference-center sensitivity has been largely removed in this benchmark suite. The next high-value fix is now the force-law calibration and broader geometry-driven validation, rather than a first-pass refactor of the global implicit layer.')
    lines.append('- After that, rerun the same benchmark suite and only then decide whether the remaining error is small enough that global-layer work is worth the cost.')

    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    centered_df, calibration_df = run_centered_sphere_scan()
    convergence_df = run_centered_sphere_convergence()
    mesh_df = run_mesh_tilted_box_scan()
    offaxis_df = run_offaxis_reference_center_sensitivity()
    write_summary(centered_df, calibration_df, convergence_df, mesh_df, offaxis_df)
    print(f'Saved benchmark outputs to: {OUT}')


if __name__ == '__main__':
    main()
