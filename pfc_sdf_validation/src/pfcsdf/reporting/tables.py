from __future__ import annotations

import pandas as pd

BENCHMARK_DISPLAY = {
    'analytic_flat': 'Analytic flat',
    'analytic_sphere': 'Analytic sphere',
    'native_band_flat': 'Native-band flat',
}


def build_main_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        'benchmark', 'scheme_display', 'peak_force_error', 'impulse_error', 'energy_drift',
        'release_timing_error', 'max_penetration_error', 'rebound_velocity_error'
    ]
    out = df.loc[:, cols].copy()
    out['benchmark'] = out['benchmark'].map(BENCHMARK_DISPLAY).fillna(out['benchmark'])
    out = out.rename(columns={
        'benchmark': 'Benchmark',
        'scheme_display': 'Scheme',
        'peak_force_error': 'Peak force error',
        'impulse_error': 'Impulse error',
        'energy_drift': 'Energy drift',
        'release_timing_error': 'Release timing error',
        'max_penetration_error': 'Max penetration error',
        'rebound_velocity_error': 'Rebound velocity error',
    })
    return out


def build_efficiency_continuity_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.rename(columns={
        'variant': 'Variant',
        'scheme_display': 'Scheme',
        'runtime_seconds': 'Runtime (s)',
        'force_error': 'Force error',
        'impulse_error': 'Impulse error',
        'energy_drift': 'Energy drift',
        'mean_candidate_count': 'Mean candidate cells',
        'mean_recompute_count': 'Mean recompute cells',
        'mean_predictor_corrector_jaccard': 'Mean pred./corr. Jaccard',
        'mean_substeps': 'Mean substeps',
    })
    keep = [
        'Variant', 'Scheme', 'Runtime (s)', 'Force error', 'Impulse error', 'Energy drift',
        'Mean candidate cells', 'Mean recompute cells', 'Mean pred./corr. Jaccard', 'Mean substeps'
    ]
    return out.loc[:, keep]


def build_efficiency_traction_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['consistent_traction_reconstruction'] = out['consistent_traction_reconstruction'].map({False: 'False', True: 'True'})
    out = out.rename(columns={
        'consistent_traction_reconstruction': 'Consistent traction',
        'runtime_seconds': 'Runtime (s)',
        'force_error': 'Force error',
        'impulse_error': 'Impulse error',
        'state_error': 'State error',
        'energy_drift': 'Energy drift',
        'mean_candidate_count': 'Mean candidate cells',
        'mean_recompute_count': 'Mean recompute cells',
    })
    keep = [
        'Consistent traction', 'Runtime (s)', 'Force error', 'Impulse error', 'State error',
        'Energy drift', 'Mean candidate cells', 'Mean recompute cells'
    ]
    return out.loc[:, keep]
