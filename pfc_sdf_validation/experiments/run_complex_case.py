
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pfcsdf.dynamics.benchmarks_complex import (
    ComplexBodyDropSetup,
    benchmark_complex_body_error,
    run_complex_body_drop_benchmark,
    run_high_resolution_complex_reference,
)
from pfcsdf.dynamics.events import EventAwareControllerConfig
from pfcsdf.geometry.complex_bodies import build_capsule_flat_edge_body_profile


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / 'results'
    fig_dir = results_dir / 'figures'
    tab_dir = results_dir / 'tables'
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    profile = build_capsule_flat_edge_body_profile(nx=81, width=0.6)
    setup = ComplexBodyDropSetup(
        initial_height=0.18,
        initial_velocity=-0.7,
        initial_angle=np.deg2rad(8.0),
        initial_omega=0.0,
        mass=2.0,
        inertia=0.045,
        contact_stiffness=2800.0,
        gravity=9.81,
        t_final=0.75,
    )
    reference = run_high_resolution_complex_reference(setup, profile)
    baseline = run_complex_body_drop_benchmark(
        setup,
        profile,
        dt=0.01,
        controller=EventAwareControllerConfig(max_depth=0, min_dt=1e-5, force_relative_jump_tol=10.0, active_measure_relative_jump_tol=10.0, predictor_corrector_mismatch_fraction_tol=0.99),
        continuity_enabled=False,
    )
    full = run_complex_body_drop_benchmark(
        setup,
        profile,
        dt=0.01,
        controller=EventAwareControllerConfig(max_depth=7, min_dt=1e-4, force_relative_jump_tol=0.2, active_measure_relative_jump_tol=0.15, predictor_corrector_mismatch_fraction_tol=0.12),
        continuity_enabled=True,
    )

    rows = []
    for name, hist in [('baseline', baseline), ('full_method', full)]:
        err = benchmark_complex_body_error(hist, reference)
        rows.append({
            'case': name,
            'z_rms': err.z_rms,
            'theta_rms': err.theta_rms,
            'force_rms': err.force_rms,
            'torque_rms': err.torque_rms,
            'peak_force_error': err.peak_force_error,
            'peak_torque_error': err.peak_torque_error,
            'release_timing_error': err.release_timing_error,
            'mean_component_count_error': err.mean_component_count_error,
            'mean_candidate_count': float(np.mean(hist.candidate_count)),
            'mean_recompute_count': float(np.mean(hist.recompute_count)),
            'mean_substeps': float(np.mean(hist.used_substeps)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(tab_dir / 'complex_case_summary.csv', index=False)
    df.to_markdown(tab_dir / 'complex_case_summary.md', index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(reference.times, reference.force_z, label='reference force')
    plt.plot(reference.times, reference.torque_y, label='reference torque')
    plt.plot(full.times, full.force_z, '--', label='full force')
    plt.plot(full.times, full.torque_y, '--', label='full torque')
    plt.xlabel('time')
    plt.ylabel('force / torque')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'complex_force_torque_time.pdf')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(reference.times, np.rad2deg(reference.theta), label='reference angle')
    plt.plot(full.times, np.rad2deg(full.theta), '--', label='full angle')
    plt.plot(reference.times, reference.z, label='reference z')
    plt.plot(full.times, full.z, '--', label='full z')
    plt.xlabel('time')
    plt.ylabel('pose')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'complex_pose_time.pdf')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(full.times, full.active_measure, label='active measure')
    plt.plot(full.times, full.active_components, label='components')
    plt.plot(full.times, full.mismatch_fraction, label='mismatch fraction')
    plt.xlabel('time')
    plt.ylabel('controller stats')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'complex_controller_stats.pdf')
    plt.close()


if __name__ == '__main__':
    main()
