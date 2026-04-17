from __future__ import annotations

from pathlib import Path

import pandas as pd

from pfcsdf.experiments.paper_suite import generate_main_outputs, load_yaml_config


def test_generate_main_outputs_from_yaml_produces_csv_markdown_and_tex(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / 'configs' / 'main_tables.yaml')
    config['benchmarks']['native_band_flat']['controller']['max_depth'] = 0
    config['benchmarks']['native_band_flat']['dt'] = 0.06
    outputs = generate_main_outputs(config, output_root=tmp_path)

    for key in ['raw_csv', 'raw_md', 'main_csv', 'main_md', 'main_tex']:
        assert outputs[key].exists(), key

    df = pd.read_csv(outputs['main_csv'])
    assert set(df['Benchmark']) == {'Analytic flat', 'Analytic sphere', 'Native-band flat'}
    assert '+ work consistency' in set(df['Scheme'])
    tex = outputs['main_tex'].read_text(encoding='utf-8')
    assert '\\begin{table}' in tex or '\\begin{tabular}' in tex
