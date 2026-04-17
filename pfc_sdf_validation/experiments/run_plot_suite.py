from pathlib import Path

from pfcsdf.experiments.paper_suite import generate_plot_outputs, load_yaml_config


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / 'configs' / 'main_tables.yaml')
    outputs = generate_plot_outputs(config, output_root=root)
    for key, value in outputs.items():
        print(f'{key}: {value}')
