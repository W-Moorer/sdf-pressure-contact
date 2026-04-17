from __future__ import annotations

from pathlib import Path

import pandas as pd

from pfcsdf.experiments.paper_suite import generate_efficiency_outputs, load_yaml_config


def test_generate_efficiency_outputs_from_yaml_produces_both_tables(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / 'configs' / 'efficiency.yaml')
    config['native_band_flat']['controller']['max_depth'] = 0
    config['native_band_flat']['dt'] = 0.06
    outputs = generate_efficiency_outputs(config, output_root=tmp_path)
    for key in ['continuity_csv', 'continuity_tex', 'continuity_md', 'traction_csv', 'traction_tex', 'traction_md']:
        assert outputs[key].exists(), key

    continuity = pd.read_csv(outputs['continuity_csv'])
    assert set(continuity['variant']) == {'Dense baseline', 'Continuity-aware'}
    dense = continuity.loc[continuity['variant'] == 'Dense baseline', 'mean_candidate_count'].iloc[0]
    sparse = continuity.loc[continuity['variant'] == 'Continuity-aware', 'mean_candidate_count'].iloc[0]
    assert sparse < dense

    traction = pd.read_csv(outputs['traction_csv'])
    assert set(traction['consistent_traction_reconstruction']) == {False, True}
