from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np

from pfcsdf.dynamics.benchmarks import DynamicHistory


def _finalize_plot(path: str | Path, *, xlabel: str, ylabel: str, title: str) -> Path:
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_force_histories(reference_times: np.ndarray, reference_forces: np.ndarray, histories: Mapping[str, DynamicHistory], path: str | Path, *, title: str) -> Path:
    plt.figure(figsize=(7.4, 4.6))
    plt.plot(reference_times, reference_forces, label='Reference')
    for label, history in histories.items():
        plt.plot(history.times, history.forces, marker='o', markersize=3, label=label)
    return _finalize_plot(path, xlabel='Time', ylabel='Normal force', title=title)


def plot_energy_histories(histories: Mapping[str, DynamicHistory], path: str | Path, *, title: str) -> Path:
    plt.figure(figsize=(7.4, 4.6))
    for label, history in histories.items():
        plt.plot(history.times, history.total_energy, marker='o', markersize=3, label=label)
    return _finalize_plot(path, xlabel='Time', ylabel='Total energy', title=title)


def plot_active_measure_histories(histories: Mapping[str, DynamicHistory], path: str | Path, *, title: str) -> Path:
    plt.figure(figsize=(7.4, 4.6))
    for label, history in histories.items():
        plt.plot(history.times, history.active_measure, marker='o', markersize=3, label=label)
    return _finalize_plot(path, xlabel='Time', ylabel='Active measure', title=title)


def plot_controller_statistics(history: DynamicHistory, path: str | Path, *, title: str) -> Path:
    plt.figure(figsize=(7.6, 4.8))
    plt.plot(history.times, history.predictor_corrector_mismatch_fraction, marker='o', markersize=3, label='Mismatch fraction')
    plt.plot(history.times, history.work_mismatch, marker='o', markersize=3, label='Work mismatch')
    plt.plot(history.times, history.predictor_force_jump, marker='o', markersize=3, label='Force jump')
    return _finalize_plot(path, xlabel='Time', ylabel='Controller statistics', title=title)
