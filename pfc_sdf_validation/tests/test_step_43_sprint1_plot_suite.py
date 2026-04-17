from __future__ import annotations

from pathlib import Path

from pfcsdf.experiments.paper_suite import generate_plot_outputs, load_yaml_config


def test_generate_plot_outputs_produces_paper_ready_figures(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(project_root / 'configs' / 'main_tables.yaml')
    config['benchmarks']['native_band_flat']['controller']['max_depth'] = 0
    config['benchmarks']['native_band_flat']['dt'] = 0.06
    outputs = generate_plot_outputs(config, output_root=tmp_path)
    required = {
        'flat_force',
        'flat_energy',
        'sphere_force',
        'native_force',
        'active_measure',
        'controller_stats',
    }
    assert set(outputs) == required
    for path in outputs.values():
        assert path.exists()
        assert path.stat().st_size > 0
