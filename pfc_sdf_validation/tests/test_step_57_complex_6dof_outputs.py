from __future__ import annotations

from pathlib import Path
import importlib.util


def test_complex_6dof_outputs(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / 'experiments' / 'run_complex_case_6dof.py'
    spec = importlib.util.spec_from_file_location('run_complex_case_6dof', script)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    out = tmp_path / 'out'
    mod.main(out)
    assert (out / 'tables' / 'complex6dof_summary.csv').exists()
    assert (out / 'figures' / 'complex6dof_force_torque_time.pdf').exists()
    assert (out / 'figures' / 'complex6dof_pose_time.pdf').exists()
    assert (out / 'figures' / 'complex6dof_controller_stats.pdf').exists()
