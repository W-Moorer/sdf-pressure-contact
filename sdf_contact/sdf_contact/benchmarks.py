from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable
import numpy as np

from .geometry import make_world
from .pipeline import ContactManager


@dataclass
class EvaluatorComparisonRow:
    state_index: int
    body_position: np.ndarray
    body_linear_velocity: np.ndarray
    baseline_force: np.ndarray
    high_accuracy_force: np.ndarray
    ideal_force: np.ndarray | None
    baseline_moment: np.ndarray
    high_accuracy_moment: np.ndarray
    ideal_moment: np.ndarray | None
    baseline_meta: dict
    high_accuracy_meta: dict


def cap_volume(R: float, delta: float) -> float:
    if delta <= 0.0:
        return 0.0
    if delta >= 2.0 * R:
        delta = 2.0 * R
    return math.pi * delta * delta * (R - delta / 3.0)


def compare_local_evaluators(*, body_factory: Callable[[np.ndarray, np.ndarray], object], domain_source_factory: Callable[[], object], states: Iterable[dict], baseline_evaluator, high_accuracy_evaluator, ideal_force_fn: Callable[[dict], np.ndarray] | None = None, ideal_moment_fn: Callable[[dict], np.ndarray] | None = None):
    baseline_manager = ContactManager(baseline_evaluator)
    high_manager = ContactManager(high_accuracy_evaluator)
    rows: list[EvaluatorComparisonRow] = []

    for idx, st in enumerate(states):
        position = np.asarray(st['position'], dtype=float)
        linear_velocity = np.asarray(st.get('linear_velocity', [0.0, 0.0, 0.0]), dtype=float)
        body_a = body_factory(position, linear_velocity)
        domain_a = domain_source_factory()
        world_a = make_world(bodies=[body_a], domain_sources=[domain_a])
        base_contacts = baseline_manager.compute_all_contacts(world_a)
        base_agg = base_contacts[body_a.name]

        body_b = body_factory(position, linear_velocity)
        domain_b = domain_source_factory()
        world_b = make_world(bodies=[body_b], domain_sources=[domain_b])
        high_contacts = high_manager.compute_all_contacts(world_b)
        high_agg = high_contacts[body_b.name]

        rows.append(
            EvaluatorComparisonRow(
                state_index=idx,
                body_position=position.copy(),
                body_linear_velocity=linear_velocity.copy(),
                baseline_force=base_agg.total_force.copy(),
                high_accuracy_force=high_agg.total_force.copy(),
                ideal_force=None if ideal_force_fn is None else np.asarray(ideal_force_fn(st), dtype=float),
                baseline_moment=base_agg.total_moment.copy(),
                high_accuracy_moment=high_agg.total_moment.copy(),
                ideal_moment=None if ideal_moment_fn is None else np.asarray(ideal_moment_fn(st), dtype=float),
                baseline_meta={
                    'num_pair_patch_points': base_agg.num_pair_patch_points,
                    'num_pair_sheet_points': base_agg.num_pair_sheet_points,
                    'num_pair_tractions': base_agg.num_pair_tractions,
                },
                high_accuracy_meta={
                    'num_pair_patch_points': high_agg.num_pair_patch_points,
                    'num_pair_sheet_points': high_agg.num_pair_sheet_points,
                    'num_pair_tractions': high_agg.num_pair_tractions,
                },
            )
        )
    return rows
