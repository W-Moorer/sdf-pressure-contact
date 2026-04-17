from __future__ import annotations

from pathlib import Path
import pandas as pd

from pfcsdf.dynamics.benchmarks_complex_6dof import (
    benchmark_complex_rigid_6dof_error,
    default_complex_6dof_setup,
    export_complex_rigid_6dof_outputs,
    run_complex_rigid_6dof_benchmark,
    run_high_resolution_complex_6dof_reference,
)
from pfcsdf.dynamics.rigid_controller import RigidEventAwareControllerConfig


def main(outdir: str | Path = 'results') -> None:
    outdir = Path(outdir)
    setup, support_cloud = default_complex_6dof_setup()
    controller = RigidEventAwareControllerConfig()
    history = run_complex_rigid_6dof_benchmark(setup, support_cloud, dt=0.01, controller=controller, continuity_enabled=True)
    reference = run_high_resolution_complex_6dof_reference(setup, support_cloud)
    summary = benchmark_complex_rigid_6dof_error(history, reference)
    tables = outdir / 'tables'
    figs = outdir / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([summary.__dict__])
    df.to_csv(tables / 'complex6dof_summary.csv', index=False)
    (tables / 'complex6dof_summary.md').write_text(df.to_markdown(index=False))
    export_complex_rigid_6dof_outputs(history, reference, figs)


if __name__ == '__main__':
    main()
