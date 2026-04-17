from pathlib import Path

from pfcsdf.experiments.paper_suite import load_yaml_config
from pfcsdf.reporting.export import write_markdown_table
import pandas as pd


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / 'configs' / 'appendix.yaml')
    df = pd.DataFrame([
        {'planned_appendix_artifact': 'controller_statistics', 'status': 'planned'},
        {'planned_appendix_artifact': 'sensitivity_tables', 'status': 'planned'},
    ])
    out = root / config.get('output_root', 'results') / 'tables' / 'appendix_plan.md'
    write_markdown_table(df, out, title='Appendix plan placeholders')
    print(out)
