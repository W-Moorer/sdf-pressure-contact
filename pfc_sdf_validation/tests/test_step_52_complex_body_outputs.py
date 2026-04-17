
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_complex_case_outputs(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    env = dict(**__import__('os').environ)
    env['PYTHONPATH'] = str(repo / 'src')
    subprocess.run([sys.executable, str(repo / 'experiments' / 'run_complex_case.py')], cwd=repo, env=env, check=True)
    assert (repo / 'results' / 'tables' / 'complex_case_summary.csv').exists()
    assert (repo / 'results' / 'figures' / 'complex_force_torque_time.pdf').exists()
    assert (repo / 'results' / 'figures' / 'complex_pose_time.pdf').exists()
    assert (repo / 'results' / 'figures' / 'complex_controller_stats.pdf').exists()
