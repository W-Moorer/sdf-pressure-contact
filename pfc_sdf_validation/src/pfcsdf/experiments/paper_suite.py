from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import (
    AnalyticLinearSphereContactModel,
    DynamicHistory,
    FlatImpactSetup,
    NativeBandFlatContactModel,
    SphereImpactSetup,
    benchmark_flat_impact_error,
    benchmark_sphere_impact_error,
    exact_flat_impact_state,
    run_flat_impact_benchmark,
    run_sphere_impact_benchmark,
    sphere_reference_history,
)
from pfcsdf.dynamics.events import EventAwareControllerConfig
from pfcsdf.experiments.ablation import AblationCaseConfig, build_default_ablation_case_configs, generate_ablation_table
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw
from pfcsdf.reporting.export import write_csv, write_latex_table, write_markdown_table
from pfcsdf.reporting.plots import (
    plot_active_measure_histories,
    plot_controller_statistics,
    plot_energy_histories,
    plot_force_histories,
)
from pfcsdf.reporting.tables import (
    build_efficiency_continuity_table,
    build_efficiency_traction_table,
    build_main_ablation_table,
)


ROOT = Path(__file__).resolve().parents[3]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


SCHEME_ORDER = [
    'event_aware_midpoint',
    'event_aware_midpoint_impulse_corrected',
    'event_aware_midpoint_work_consistent',
    'event_aware_midpoint_work_consistent_consistent_traction',
]


def _native_model_from_cfg(model_cfg: dict[str, Any], *, consistent_traction: bool | None = None, continuity_enabled: bool | None = None, boundary_only: bool | None = None) -> NativeBandFlatContactModel:
    grid_cfg = model_cfg['grid']
    footprint_cfg = model_cfg['footprint']
    continuity_cfg = model_cfg.get('continuity', {})
    traction_cfg = model_cfg.get('traction', {})
    return NativeBandFlatContactModel(
        mass=float(model_cfg['mass']),
        grid=UniformGrid3D(
            origin=np.asarray(grid_cfg['origin'], dtype=float),
            spacing=np.asarray(grid_cfg['spacing'], dtype=float),
            shape=tuple(int(v) for v in grid_cfg['shape']),
        ),
        footprint=BoxFootprint(float(footprint_cfg['width']), float(footprint_cfg['height'])),
        law_a=LinearPressureLaw(float(model_cfg['law_a_stiffness'])),
        law_b=LinearPressureLaw(float(model_cfg['law_b_stiffness'])),
        config=NativeBandAccumulatorConfig(
            eta=float(model_cfg['band']['eta']),
            band_half_width=float(model_cfg['band']['band_half_width']),
        ),
        max_depth_a=float(model_cfg['max_depth_a']),
        max_depth_b=float(model_cfg['max_depth_b']),
        use_continuity_warm_start=continuity_cfg.get('use_warm_start', True) if continuity_enabled is None else bool(continuity_enabled),
        boundary_only_update=continuity_cfg.get('boundary_only_update', True) if boundary_only is None else bool(boundary_only),
        continuity_dilation_radius=int(continuity_cfg.get('dilation_radius', 1)),
        consistent_traction_reconstruction=traction_cfg.get('consistent_reconstruction', True) if consistent_traction is None else bool(consistent_traction),
    )


def _controller_from_cfg(cfg: dict[str, Any] | None) -> EventAwareControllerConfig | None:
    if cfg is None:
        return None
    return EventAwareControllerConfig(
        max_depth=int(cfg.get('max_depth', 4)),
        min_dt=float(cfg.get('min_dt', 1e-4)),
        force_relative_jump_tol=float(cfg.get('force_relative_jump_tol', cfg.get('force_jump_relative_tol', 0.20))),
        acceleration_relative_jump_tol=float(cfg.get('acceleration_relative_jump_tol', 0.20)),
        active_measure_relative_jump_tol=float(cfg.get('active_measure_relative_jump_tol', cfg.get('active_measure_jump_relative_tol', 0.15))),
        predictor_force_relative_jump_tol=float(cfg.get('predictor_force_relative_jump_tol', 0.20)),
        predictor_active_measure_relative_jump_tol=float(cfg.get('predictor_active_measure_relative_jump_tol', 0.15)),
        predictor_corrector_jaccard_tol=float(cfg.get('predictor_corrector_jaccard_tol', 0.85)),
        predictor_corrector_mismatch_fraction_tol=float(cfg.get('predictor_corrector_mismatch_fraction_tol', cfg.get('predictor_corrector_mismatch_tol', 0.10))),
        work_mismatch_relative_tol=float(cfg.get('work_mismatch_relative_tol', 0.02)),
        mismatch_repair_dilation_radius=int(cfg.get('mismatch_repair_dilation_radius', 1)),
        use_predictor_corrector_continuity=bool(cfg.get('use_predictor_corrector_continuity', True)),
    )


def build_case_configs_from_config(config: dict[str, Any]) -> list[AblationCaseConfig]:
    schemes = list(config['schemes'])
    benchmark_cfg = config['benchmarks']
    cases: list[AblationCaseConfig] = []
    for benchmark, cfg in benchmark_cfg.items():
        if benchmark == 'analytic_flat':
            setup = FlatImpactSetup(
                initial_gap=float(cfg['initial_gap']),
                initial_velocity=float(cfg['initial_velocity']),
                mass=float(cfg['mass']),
                contact_stiffness=float(cfg['contact_stiffness']),
                t_final=float(cfg['t_final']),
            )
            controller = _controller_from_cfg(cfg.get('controller'))
            dt = float(cfg['dt'])
            for scheme in schemes:
                notes = '解析接触模型不存在 traction reconstruction，自然与上一行相同。' if scheme.endswith('consistent_traction') else ''
                cases.append(AblationCaseConfig(benchmark=benchmark, scheme=scheme, dt=dt, setup=setup, controller=controller, notes=notes))
        elif benchmark == 'analytic_sphere':
            setup = SphereImpactSetup(
                initial_gap=float(cfg['initial_gap']),
                initial_velocity=float(cfg['initial_velocity']),
                mass=float(cfg['mass']),
                sphere_radius=float(cfg['sphere_radius']),
                sphere_stiffness=float(cfg['sphere_stiffness']),
                plane_stiffness=float(cfg['plane_stiffness']),
                t_final=float(cfg['t_final']),
            )
            controller = _controller_from_cfg(cfg.get('controller'))
            dt = float(cfg['dt'])
            for scheme in schemes:
                notes = '解析球-平面基准不存在 traction reconstruction，自然与上一行相同。' if scheme.endswith('consistent_traction') else ''
                cases.append(AblationCaseConfig(benchmark=benchmark, scheme=scheme, dt=dt, setup=setup, controller=controller, notes=notes))
        elif benchmark == 'native_band_flat':
            setup = FlatImpactSetup(
                initial_gap=float(cfg['initial_gap']),
                initial_velocity=float(cfg['initial_velocity']),
                mass=float(cfg['mass']),
                contact_stiffness=float(cfg['contact_stiffness']),
                t_final=float(cfg['t_final']),
            )
            controller = _controller_from_cfg(cfg.get('controller'))
            dt = float(cfg['dt'])
            for scheme in schemes:
                cases.append(
                    AblationCaseConfig(
                        benchmark=benchmark,
                        scheme=scheme,
                        dt=dt,
                        setup=setup,
                        controller=controller,
                        model_factory=lambda s=scheme, c=cfg['model']: _native_model_from_cfg(c, consistent_traction=s.endswith('consistent_traction')),
                    )
                )
        else:
            raise ValueError(f'Unknown benchmark {benchmark}')
    return cases


def generate_main_outputs(config: dict[str, Any], *, output_root: str | Path | None = None) -> dict[str, Path]:
    out_root = ROOT if output_root is None else Path(output_root)
    results_dir = out_root / config.get('output_root', 'results')
    tables_dir = results_dir / 'tables'
    cases = build_case_configs_from_config(config)
    raw_df = generate_ablation_table(cases)
    raw_csv = write_csv(raw_df, tables_dir / 'long_horizon_ablation.csv')
    raw_md = write_markdown_table(raw_df, tables_dir / 'long_horizon_ablation_raw.md', title='Raw long-horizon ablation')
    main_df = build_main_ablation_table(raw_df)
    main_csv = write_csv(main_df, tables_dir / 'main_ablation.csv')
    main_md = write_markdown_table(main_df, tables_dir / 'main_ablation.md', title='Main long-horizon ablation')
    main_tex = write_latex_table(main_df, tables_dir / 'main_ablation.tex', caption='长时程动力学主表', label='tab:main_ablation')
    return {
        'raw_csv': raw_csv,
        'raw_md': raw_md,
        'main_csv': main_csv,
        'main_md': main_md,
        'main_tex': main_tex,
    }


def _run_native_flat_case(*, setup: FlatImpactSetup, dt: float, scheme: str, model: NativeBandFlatContactModel, controller: EventAwareControllerConfig | None) -> tuple[DynamicHistory, dict[str, float]]:
    start = perf_counter()
    hist = run_flat_impact_benchmark(setup, dt=dt, scheme=scheme, model=model, controller=controller)
    runtime = perf_counter() - start
    err = benchmark_flat_impact_error(setup, hist)
    row = {
        'runtime_seconds': runtime,
        'force_error': err.force_error,
        'impulse_error': err.impulse_error,
        'state_error': err.state_error,
        'energy_drift': hist.energy_drift,
        'mean_candidate_count': float(np.nanmean(hist.candidate_count)),
        'mean_recompute_count': float(np.nanmean(hist.recompute_count)),
        'mean_predictor_corrector_jaccard': float(np.nanmean(hist.predictor_corrector_jaccard)),
        'mean_substeps': float(np.mean(hist.used_substeps)),
        'max_used_substeps': int(np.max(hist.used_substeps)),
        'final_force': float(hist.forces[-1]),
    }
    return hist, row


def generate_efficiency_outputs(config: dict[str, Any], *, output_root: str | Path | None = None) -> dict[str, Path]:
    out_root = ROOT if output_root is None else Path(output_root)
    results_dir = out_root / config.get('output_root', 'results')
    tables_dir = results_dir / 'tables'
    eff_cfg = config['native_band_flat']
    setup = FlatImpactSetup(
        initial_gap=float(eff_cfg['initial_gap']),
        initial_velocity=float(eff_cfg['initial_velocity']),
        mass=float(eff_cfg['mass']),
        contact_stiffness=float(eff_cfg['contact_stiffness']),
        t_final=float(eff_cfg['t_final']),
    )
    controller = _controller_from_cfg(eff_cfg.get('controller'))
    dt = float(eff_cfg['dt'])

    continuity_rows = []
    for variant, continuity_enabled, boundary_only in [
        ('Dense baseline', False, False),
        ('Continuity-aware', True, True),
    ]:
        model = _native_model_from_cfg(eff_cfg['model'], continuity_enabled=continuity_enabled, boundary_only=boundary_only, consistent_traction=False)
        hist, row = _run_native_flat_case(
            setup=setup,
            dt=dt,
            scheme=eff_cfg['continuity_scheme'],
            model=model,
            controller=controller,
        )
        continuity_rows.append({
            'variant': variant,
            'scheme_display': eff_cfg['continuity_scheme_display'],
            **row,
        })
    continuity_df = pd.DataFrame(continuity_rows)
    continuity_table = build_efficiency_continuity_table(continuity_df)

    traction_rows = []
    for consistent in [False, True]:
        model = _native_model_from_cfg(eff_cfg['model'], continuity_enabled=True, boundary_only=True, consistent_traction=consistent)
        hist, row = _run_native_flat_case(
            setup=setup,
            dt=dt,
            scheme=eff_cfg['traction_scheme'],
            model=model,
            controller=controller,
        )
        traction_rows.append({
            'consistent_traction_reconstruction': consistent,
            **row,
        })
    traction_df = pd.DataFrame(traction_rows)
    traction_table = build_efficiency_traction_table(traction_df)

    cont_csv = write_csv(continuity_df, tables_dir / 'efficiency_continuity_raw.csv')
    cont_tex = write_latex_table(continuity_table, tables_dir / 'efficiency_continuity.tex', caption='continuity-aware sparse traversal 的效率收益', label='tab:efficiency_continuity')
    cont_md = write_markdown_table(continuity_table, tables_dir / 'efficiency_continuity.md', title='Continuity efficiency')
    tr_csv = write_csv(traction_df, tables_dir / 'efficiency_traction_raw.csv')
    tr_tex = write_latex_table(traction_table, tables_dir / 'efficiency_traction.tex', caption='consistent traction reconstruction 的补充收益', label='tab:efficiency_traction')
    tr_md = write_markdown_table(traction_table, tables_dir / 'efficiency_traction.md', title='Consistent traction efficiency')
    return {
        'continuity_csv': cont_csv,
        'continuity_tex': cont_tex,
        'continuity_md': cont_md,
        'traction_csv': tr_csv,
        'traction_tex': tr_tex,
        'traction_md': tr_md,
    }


def _history_map_for_analytic_flat(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, DynamicHistory]]:
    cfg = config['benchmarks']['analytic_flat']
    setup = FlatImpactSetup(
        initial_gap=float(cfg['initial_gap']),
        initial_velocity=float(cfg['initial_velocity']),
        mass=float(cfg['mass']),
        contact_stiffness=float(cfg['contact_stiffness']),
        t_final=float(cfg['t_final']),
    )
    controller = _controller_from_cfg(cfg.get('controller'))
    dt = float(cfg['dt'])
    histories = {}
    for scheme in config['schemes'][:3]:
        histories[scheme] = run_flat_impact_benchmark(setup, dt=dt, scheme=scheme, controller=controller)
    tref = np.linspace(0.0, setup.t_final, 600)
    fref = np.array([exact_flat_impact_state(setup, t)[2] for t in tref], dtype=float)
    return tref, fref, histories


def _history_map_for_analytic_sphere(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, DynamicHistory], DynamicHistory]:
    cfg = config['benchmarks']['analytic_sphere']
    setup = SphereImpactSetup(
        initial_gap=float(cfg['initial_gap']),
        initial_velocity=float(cfg['initial_velocity']),
        mass=float(cfg['mass']),
        sphere_radius=float(cfg['sphere_radius']),
        sphere_stiffness=float(cfg['sphere_stiffness']),
        plane_stiffness=float(cfg['plane_stiffness']),
        t_final=float(cfg['t_final']),
    )
    controller = _controller_from_cfg(cfg.get('controller'))
    dt = float(cfg['dt'])
    histories = {}
    for scheme in config['schemes'][:3]:
        histories[scheme] = run_sphere_impact_benchmark(setup, dt=dt, scheme=scheme, controller=controller)
    ref = sphere_reference_history(setup)
    return ref.times, ref.forces, histories, ref


def _history_map_for_native_band(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, DynamicHistory]]:
    cfg = config['benchmarks']['native_band_flat']
    setup = FlatImpactSetup(
        initial_gap=float(cfg['initial_gap']),
        initial_velocity=float(cfg['initial_velocity']),
        mass=float(cfg['mass']),
        contact_stiffness=float(cfg['contact_stiffness']),
        t_final=float(cfg['t_final']),
    )
    controller = _controller_from_cfg(cfg.get('controller'))
    dt = float(cfg['dt'])
    histories = {}
    model_cfg = cfg['model']
    baseline_model = _native_model_from_cfg(model_cfg, consistent_traction=False)
    best_model = _native_model_from_cfg(model_cfg, consistent_traction=True)
    histories['event_aware_midpoint'] = run_flat_impact_benchmark(setup, dt=dt, scheme='event_aware_midpoint', model=baseline_model, controller=controller)
    histories['event_aware_midpoint_work_consistent'] = run_flat_impact_benchmark(setup, dt=dt, scheme='event_aware_midpoint_work_consistent', model=baseline_model, controller=controller)
    histories['event_aware_midpoint_work_consistent_consistent_traction'] = run_flat_impact_benchmark(setup, dt=dt, scheme='event_aware_midpoint_work_consistent', model=best_model, controller=controller)
    tref = np.linspace(0.0, setup.t_final, 600)
    fref = np.array([exact_flat_impact_state(setup, t)[2] for t in tref], dtype=float)
    return tref, fref, histories


def generate_plot_outputs(config: dict[str, Any], *, output_root: str | Path | None = None) -> dict[str, Path]:
    out_root = ROOT if output_root is None else Path(output_root)
    results_dir = out_root / config.get('output_root', 'results')
    figs_dir = results_dir / 'figures'
    paths: dict[str, Path] = {}

    t_flat, f_flat, h_flat = _history_map_for_analytic_flat(config)
    paths['flat_force'] = plot_force_histories(t_flat, f_flat, h_flat, figs_dir / 'flat_force_time.pdf', title='Analytic flat impact: force-time')
    paths['flat_energy'] = plot_energy_histories(h_flat, figs_dir / 'flat_energy_time.pdf', title='Analytic flat impact: energy-time')

    t_sphere, f_sphere, h_sphere, _ = _history_map_for_analytic_sphere(config)
    paths['sphere_force'] = plot_force_histories(t_sphere, f_sphere, h_sphere, figs_dir / 'sphere_force_time.pdf', title='Analytic sphere impact: force-time')

    t_native, f_native, h_native = _history_map_for_native_band(config)
    paths['native_force'] = plot_force_histories(t_native, f_native, h_native, figs_dir / 'native_band_force_time.pdf', title='Native-band flat impact: force-time')
    native_named = {
        'baseline': h_native['event_aware_midpoint'],
        'work-consistent': h_native['event_aware_midpoint_work_consistent'],
        'consistent traction': h_native['event_aware_midpoint_work_consistent_consistent_traction'],
    }
    paths['active_measure'] = plot_active_measure_histories(native_named, figs_dir / 'native_band_active_measure_time.pdf', title='Native-band active measure evolution')
    paths['controller_stats'] = plot_controller_statistics(h_native['event_aware_midpoint_work_consistent_consistent_traction'], figs_dir / 'native_band_controller_statistics.pdf', title='Native-band controller statistics')
    return paths
