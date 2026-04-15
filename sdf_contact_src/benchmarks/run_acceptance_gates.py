from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results' / 'local_benchmarks'
OUT.mkdir(parents=True, exist_ok=True)

RESEARCH_THRESHOLDS = {
    'max_centered_moment_ratio': 1.0e-3,
    'tail_spread_rel': 2.0e-2,
    'last_step_rel_change': 2.0e-2,
    'max_horizontal_to_vertical_ratio': 1.0e-4,
    'fy_reference_mode_rel_diff': 1.0e-4,
    'mean_rel_err_calibrated': 2.0e-2,
    'mean_rel_err_vs_ref_Fy': 3.0e-2,
    'max_rel_err_vs_ref_Fy': 8.0e-2,
    'mean_rel_err_vs_ref_Mz': 3.0e-2,
    'max_rel_err_vs_ref_Mz': 8.0e-2,
}

PRODUCTION_THRESHOLDS = {
    'max_centered_moment_ratio': 1.0e-4,
    'tail_spread_rel': 5.0e-3,
    'last_step_rel_change': 5.0e-3,
    'max_horizontal_to_vertical_ratio': 1.0e-5,
    'fy_reference_mode_rel_diff': 1.0e-5,
    'mean_rel_err_raw': 1.0e-1,
    'mean_rel_err_calibrated': 1.0e-2,
    'mean_rel_err_vs_ref_Fy': 1.0e-2,
    'max_rel_err_vs_ref_Fy': 3.0e-2,
    'mean_rel_err_vs_ref_Mz': 1.0e-2,
    'max_rel_err_vs_ref_Mz': 3.0e-2,
}


def _load_required(name: str) -> pd.DataFrame:
    path = OUT / name
    if not path.exists():
        raise FileNotFoundError(
            f'Missing benchmark file: {path}. Run benchmarks/run_local_evaluator_benchmarks.py first.'
        )
    return pd.read_csv(path)


def _centered_metrics(centered: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    centered = centered.copy()
    centered['centered_moment_ratio'] = centered['Mz_body'].abs() / centered['Fy'].abs().clip(lower=1.0e-12)
    moment = centered.groupby('evaluator')['centered_moment_ratio'].max().rename('max_centered_moment_ratio')
    raw = centered.groupby('evaluator')['rel_err_Fy'].mean().rename('mean_rel_err_raw')
    cal = calibration.set_index('evaluator')['mean_rel_err_calibrated']
    return pd.concat([moment, raw, cal], axis=1).reset_index()


def _convergence_metrics(convergence: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, sub in convergence.groupby('family'):
        sub = sub.sort_values('resolution')
        fy = sub['Fy'].to_numpy(dtype=float)
        tail = fy[-3:]
        tail_spread_rel = float((tail.max() - tail.min()) / max(abs(tail.mean()), 1.0e-12))
        last_step_rel_change = float(abs(fy[-1] - fy[-2]) / max(abs(fy[-1]), 1.0e-12))
        rows.append({
            'family': family,
            'tail_spread_rel': tail_spread_rel,
            'last_step_rel_change': last_step_rel_change,
        })
    out = pd.DataFrame(rows)
    family_to_eval = {'baseline': 'baseline_default', 'polygon': 'polygon_default'}
    out['evaluator'] = out['family'].map(family_to_eval)
    return out[['evaluator', 'tail_spread_rel', 'last_step_rel_change']]


def _mesh_metrics(mesh: pd.DataFrame) -> pd.DataFrame:
    sub = mesh[mesh['evaluator'] != 'polygon_reference'].copy()
    sub['rel_err_vs_ref_Fy'] = sub['abs_err_vs_ref_Fy'] / sub['Fy_ref_polygon'].abs().clip(lower=1.0e-12)
    sub['rel_err_vs_ref_Mz'] = sub['abs_err_vs_ref_Mz'] / sub['Mz_ref_polygon'].abs().clip(lower=1.0e-12)
    out = sub.groupby('evaluator').agg(
        mean_rel_err_vs_ref_Fy=('rel_err_vs_ref_Fy', 'mean'),
        max_rel_err_vs_ref_Fy=('rel_err_vs_ref_Fy', 'max'),
        mean_rel_err_vs_ref_Mz=('rel_err_vs_ref_Mz', 'mean'),
        max_rel_err_vs_ref_Mz=('rel_err_vs_ref_Mz', 'max'),
    ).reset_index()
    return out


def _offaxis_metrics(offaxis: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for evaluator, sub in offaxis.groupby('evaluator'):
        max_ratio = float(sub['horizontal_to_vertical_ratio'].max())
        fy_by_mode = {row['reference_mode']: float(row['body_force_y']) for _, row in sub.iterrows()}
        fy_values = list(fy_by_mode.values())
        fy_rel_diff = float((max(fy_values) - min(fy_values)) / max(abs(np.mean(fy_values)), 1.0e-12))
        rows.append({
            'evaluator': evaluator,
            'max_horizontal_to_vertical_ratio': max_ratio,
            'fy_reference_mode_rel_diff': fy_rel_diff,
        })
    return pd.DataFrame(rows)


def _merge_metrics(centered, convergence, mesh, offaxis) -> pd.DataFrame:
    metrics = centered.merge(convergence, on='evaluator', how='left')
    metrics = metrics.merge(mesh, on='evaluator', how='left')
    metrics = metrics.merge(offaxis, on='evaluator', how='left')
    return metrics


def _evaluate_profile(row: pd.Series, thresholds: dict[str, float]) -> tuple[bool, dict[str, bool]]:
    checks = {}
    for metric, threshold in thresholds.items():
        value = row.get(metric)
        if pd.isna(value):
            checks[metric] = False
        else:
            checks[metric] = bool(float(value) <= float(threshold))
    passed = all(checks.values())
    return passed, checks


def _row_to_status(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    overview_rows = []
    status_json: dict[str, dict] = {}
    for _, row in metrics.iterrows():
        evaluator = row['evaluator']
        if evaluator == 'polygon_reference':
            research_pass = False
            production_pass = False
            tier = 'reference_only'
            research_checks = {}
            production_checks = {}
        else:
            research_pass, research_checks = _evaluate_profile(row, RESEARCH_THRESHOLDS)
            production_pass, production_checks = _evaluate_profile(row, PRODUCTION_THRESHOLDS)
            if production_pass:
                tier = 'production'
            elif research_pass:
                tier = 'research'
            else:
                tier = 'below_research'
        overview_rows.append({
            'evaluator': evaluator,
            'tier': tier,
            'research_pass': research_pass,
            'production_pass': production_pass,
        })
        status_json[evaluator] = {
            'tier': tier,
            'research_pass': research_pass,
            'production_pass': production_pass,
            'research_checks': research_checks,
            'production_checks': production_checks,
            'metrics': {k: (None if pd.isna(v) else float(v)) for k, v in row.items() if k != 'evaluator'},
        }
    return pd.DataFrame(overview_rows), status_json


def _write_markdown(metrics: pd.DataFrame, overview: pd.DataFrame, status_json: dict):
    lines: list[str] = []
    lines.append('# Benchmark acceptance report')
    lines.append('')
    lines.append('This report evaluates the current benchmark outputs against the admission profiles defined in `docs/BENCHMARK_ACCEPTANCE.md`.')
    lines.append('')
    lines.append('## Overview')
    lines.append(overview.to_markdown(index=False))
    lines.append('')
    lines.append('## Raw metrics')
    lines.append(metrics.to_markdown(index=False))
    lines.append('')
    lines.append('## Profile thresholds')
    lines.append('### Research')
    lines.append(pd.DataFrame([RESEARCH_THRESHOLDS]).to_markdown(index=False))
    lines.append('')
    lines.append('### Production')
    lines.append(pd.DataFrame([PRODUCTION_THRESHOLDS]).to_markdown(index=False))
    lines.append('')

    for evaluator in overview['evaluator']:
        item = status_json[evaluator]
        lines.append(f'## Evaluator: `{evaluator}`')
        lines.append(f"- Tier: **{item['tier']}**")
        lines.append(f"- Research pass: **{item['research_pass']}**")
        lines.append(f"- Production pass: **{item['production_pass']}**")
        lines.append('')

        failed_research = [k for k, ok in item['research_checks'].items() if not ok]
        failed_prod = [k for k, ok in item['production_checks'].items() if not ok]
        lines.append(f"- Failed research checks: {', '.join(failed_research) if failed_research else 'none'}")
        lines.append(f"- Failed production checks: {', '.join(failed_prod) if failed_prod else 'none'}")
        lines.append('')

        if item['tier'] == 'production':
            lines.append('- Interpretation: this evaluator is stable enough to be treated as production-accurate under the current benchmark scope.')
        elif item['tier'] == 'research':
            lines.append('- Interpretation: this evaluator is accurate enough for research iteration, but still not calibrated or consistent enough to justify production assumptions.')
        elif item['tier'] == 'reference_only':
            lines.append('- Interpretation: this entry is a numerical reference configuration, not a candidate production evaluator, so profile gating is not applied to it.')
        else:
            lines.append('- Interpretation: this evaluator still fails at least one mandatory research-profile gate. Fix those gates before considering wider refactors.')
        lines.append('')

    lines.append('## Current strategic interpretation')
    lines.append('- If an evaluator is below the research profile, do not refactor the global solver yet.')
    lines.append('- If an evaluator passes research but fails production, keep improving force-law calibration and general-mesh validation before major global-layer work.')
    lines.append('- Only when at least one evaluator passes production with margin should the remaining error budget be attributed mainly to the global layer.')
    lines.append('')

    (OUT / 'acceptance_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    centered = _load_required('centered_sphere_plane_scan.csv')
    calibration = _load_required('centered_sphere_plane_calibration_summary.csv')
    convergence = _load_required('centered_sphere_plane_convergence.csv')
    mesh = _load_required('mesh_tilted_box_plane_scan.csv')
    offaxis = _load_required('offaxis_reference_center_sensitivity.csv')

    centered_metrics = _centered_metrics(centered, calibration)
    convergence_metrics = _convergence_metrics(convergence)
    mesh_metrics = _mesh_metrics(mesh)
    offaxis_metrics = _offaxis_metrics(offaxis)

    metrics = _merge_metrics(centered_metrics, convergence_metrics, mesh_metrics, offaxis_metrics)
    overview, status_json = _row_to_status(metrics)

    metrics.to_csv(OUT / 'acceptance_metrics.csv', index=False)
    overview.to_csv(OUT / 'acceptance_overview.csv', index=False)
    (OUT / 'acceptance_status.json').write_text(json.dumps(status_json, indent=2), encoding='utf-8')
    _write_markdown(metrics, overview, status_json)
    print(f'Wrote acceptance outputs to: {OUT}')


if __name__ == '__main__':
    main()
