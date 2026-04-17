from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalDynamicsState:
    """Minimal 1D normal dynamics state for contact benchmarks.

    gap > 0 means separated, gap < 0 means penetrated.
    velocity is the time derivative of gap.
    """

    time: float
    gap: float
    velocity: float
    mass: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")

    @property
    def is_active(self) -> bool:
        return self.gap <= 0.0

    def with_state(self, *, time: float | None = None, gap: float | None = None, velocity: float | None = None) -> "NormalDynamicsState":
        return NormalDynamicsState(
            time=self.time if time is None else float(time),
            gap=self.gap if gap is None else float(gap),
            velocity=self.velocity if velocity is None else float(velocity),
            mass=self.mass,
        )


@dataclass(frozen=True)
class ModelEvaluation:
    force: float
    acceleration: float
    active: bool
    active_measure: float | None = None
    active_snapshot: Any | None = None


@dataclass(frozen=True)
class StepDiagnostics:
    force: float
    acceleration: float
    active: bool
    used_substeps: int = 1
    onset_detected: bool = False
    release_detected: bool = False
    onset_time_estimate: float | None = None
    release_time_estimate: float | None = None
    active_measure: float | None = None
    relative_force_jump: float | None = None
    relative_acceleration_jump: float | None = None
    active_measure_jump: float | None = None
    continuity_jaccard: float | None = None
    predictor_force_jump: float | None = None
    predictor_active_measure_jump: float | None = None
    predictor_corrector_jaccard: float | None = None
    predictor_corrector_mismatch_fraction: float | None = None
    work_mismatch: float | None = None


@dataclass(frozen=True)
class IntegrationStepResult:
    state: NormalDynamicsState
    diagnostics: StepDiagnostics
