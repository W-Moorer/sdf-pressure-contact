from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

ArrayLike = np.ndarray


@dataclass(frozen=True)
class ActiveSetSnapshot:
    mask: ArrayLike
    measure: float

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "measure", float(self.measure))

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def total_count(self) -> int:
        return int(self.mask.size)

    @property
    def active_indices(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(v) for v in idx) for idx in np.argwhere(self.mask))

    @classmethod
    def empty(cls, shape: tuple[int, ...]) -> "ActiveSetSnapshot":
        return cls(mask=np.zeros(shape, dtype=bool), measure=0.0)


@dataclass(frozen=True)
class ActiveSetContinuityState:
    previous: ActiveSetSnapshot | None = None
    step_index: int = 0


@dataclass(frozen=True)
class ActiveSetContinuityReport:
    previous_count: int
    current_count: int
    retained_count: int
    entered_count: int
    exited_count: int
    measure_previous: float
    measure_current: float
    relative_measure_jump: float
    jaccard_index: float
    warm_start_mask: ArrayLike


@dataclass(frozen=True)
class ActiveSetMismatchReport:
    predictor_count: int
    corrector_count: int
    retained_count: int
    mismatch_count: int
    mismatch_fraction: float
    measure_predictor: float
    measure_corrector: float
    relative_measure_jump: float
    jaccard_index: float
    mismatch_mask: ArrayLike


@dataclass(frozen=True)
class ActiveSetUpdatePlan:
    previous_mask: ArrayLike
    candidate_mask: ArrayLike
    recompute_mask: ArrayLike
    retained_interior_mask: ArrayLike
    boundary_mask: ArrayLike

    @property
    def candidate_count(self) -> int:
        return int(np.count_nonzero(self.candidate_mask))

    @property
    def recompute_count(self) -> int:
        return int(np.count_nonzero(self.recompute_mask))

    @property
    def retained_count(self) -> int:
        return int(np.count_nonzero(self.retained_interior_mask))


def _normalize_mask(mask: ArrayLike) -> ArrayLike:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("active-set masks must be 3D")
    return mask


def _shift_mask(mask: ArrayLike, offset: tuple[int, int, int]) -> ArrayLike:
    shifted = np.zeros_like(mask, dtype=bool)
    src = []
    dst = []
    for axis, delta in enumerate(offset):
        if delta >= 0:
            src.append(slice(0, mask.shape[axis] - delta))
            dst.append(slice(delta, mask.shape[axis]))
        else:
            src.append(slice(-delta, mask.shape[axis]))
            dst.append(slice(0, mask.shape[axis] + delta))
    shifted[tuple(dst)] = mask[tuple(src)]
    return shifted


def dilate_mask(mask: ArrayLike, radius: int = 1) -> ArrayLike:
    mask = _normalize_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask.copy()
    out = np.zeros_like(mask, dtype=bool)
    for offset in product(range(-radius, radius + 1), repeat=mask.ndim):
        out |= _shift_mask(mask, tuple(int(v) for v in offset))
    return out


def erode_mask(mask: ArrayLike, radius: int = 1) -> ArrayLike:
    mask = _normalize_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask.copy()
    return ~dilate_mask(~mask, radius=radius)


def boundary_mask(mask: ArrayLike, radius: int = 1) -> ArrayLike:
    mask = _normalize_mask(mask)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    return mask & ~erode_mask(mask, radius=radius)


def continuity_update_plan(
    previous_mask: ArrayLike,
    *,
    extra_mask: ArrayLike | None = None,
    dilation_radius: int = 1,
) -> ActiveSetUpdatePlan:
    prev = _normalize_mask(previous_mask)
    support_mask = np.ones_like(prev, dtype=bool) if extra_mask is None else np.asarray(extra_mask, dtype=bool)
    if support_mask.shape != prev.shape:
        raise ValueError("extra_mask shape mismatch")

    prev = prev & support_mask
    if not np.any(prev):
        empty = np.zeros_like(prev, dtype=bool)
        return ActiveSetUpdatePlan(
            previous_mask=prev,
            candidate_mask=empty,
            recompute_mask=empty,
            retained_interior_mask=empty,
            boundary_mask=empty,
        )

    boundary = boundary_mask(prev, radius=1)
    retained = erode_mask(prev, radius=1) & support_mask
    recompute = dilate_mask(boundary, radius=dilation_radius) & support_mask
    candidate = (retained | recompute) & support_mask
    return ActiveSetUpdatePlan(
        previous_mask=prev,
        candidate_mask=candidate,
        recompute_mask=recompute,
        retained_interior_mask=retained,
        boundary_mask=boundary,
    )



def repair_update_plan(
    previous_mask: ArrayLike,
    repair_mask: ArrayLike,
    *,
    extra_mask: ArrayLike | None = None,
) -> ActiveSetUpdatePlan:
    prev = _normalize_mask(previous_mask)
    repair = np.asarray(repair_mask, dtype=bool)
    if repair.shape != prev.shape:
        raise ValueError("repair_mask shape mismatch")
    support_mask = np.ones_like(prev, dtype=bool) if extra_mask is None else np.asarray(extra_mask, dtype=bool)
    if support_mask.shape != prev.shape:
        raise ValueError("extra_mask shape mismatch")
    prev = prev & support_mask
    repair = repair & support_mask
    retained = prev & (~repair)
    candidate = retained | repair
    return ActiveSetUpdatePlan(
        previous_mask=prev,
        candidate_mask=candidate,
        recompute_mask=repair,
        retained_interior_mask=retained,
        boundary_mask=boundary_mask(prev, radius=1),
    )



def active_set_mismatch_report(
    predictor: ActiveSetSnapshot,
    corrector: ActiveSetSnapshot,
) -> ActiveSetMismatchReport:
    pred_mask = _normalize_mask(predictor.mask)
    corr_mask = _normalize_mask(corrector.mask)
    if pred_mask.shape != corr_mask.shape:
        raise ValueError("predictor/corrector shape mismatch")
    retained = pred_mask & corr_mask
    mismatch = pred_mask ^ corr_mask
    union = pred_mask | corr_mask
    retained_count = int(np.count_nonzero(retained))
    mismatch_count = int(np.count_nonzero(mismatch))
    union_count = int(np.count_nonzero(union))
    jaccard = 1.0 if union_count == 0 else float(retained_count / union_count)
    mismatch_fraction = 0.0 if union_count == 0 else float(mismatch_count / union_count)
    denom = max(abs(predictor.measure), 1e-14)
    rel_measure_jump = float(abs(corrector.measure - predictor.measure) / denom) if predictor.measure != 0.0 else (1.0 if corrector.measure != 0.0 else 0.0)
    return ActiveSetMismatchReport(
        predictor_count=predictor.active_count,
        corrector_count=corrector.active_count,
        retained_count=retained_count,
        mismatch_count=mismatch_count,
        mismatch_fraction=mismatch_fraction,
        measure_predictor=predictor.measure,
        measure_corrector=corrector.measure,
        relative_measure_jump=rel_measure_jump,
        jaccard_index=jaccard,
        mismatch_mask=mismatch,
    )



def continuity_report(
    state: ActiveSetContinuityState,
    current: ActiveSetSnapshot,
) -> tuple[ActiveSetContinuityState, ActiveSetContinuityReport]:
    prev = state.previous
    if prev is None:
        warm = np.zeros_like(current.mask, dtype=bool)
        report = ActiveSetContinuityReport(
            previous_count=0,
            current_count=current.active_count,
            retained_count=0,
            entered_count=current.active_count,
            exited_count=0,
            measure_previous=0.0,
            measure_current=current.measure,
            relative_measure_jump=1.0 if current.measure > 0.0 else 0.0,
            jaccard_index=1.0 if current.active_count == 0 else 0.0,
            warm_start_mask=warm,
        )
        return ActiveSetContinuityState(previous=current, step_index=state.step_index + 1), report

    prev_mask = np.asarray(prev.mask, dtype=bool)
    curr_mask = np.asarray(current.mask, dtype=bool)
    retained = prev_mask & curr_mask
    entered = (~prev_mask) & curr_mask
    exited = prev_mask & (~curr_mask)
    union = prev_mask | curr_mask
    union_count = int(np.count_nonzero(union))
    retained_count = int(np.count_nonzero(retained))
    jaccard = 1.0 if union_count == 0 else float(retained_count / union_count)
    denom = max(abs(prev.measure), 1e-14)
    rel_measure_jump = float(abs(current.measure - prev.measure) / denom) if prev.measure != 0.0 else (1.0 if current.measure != 0.0 else 0.0)
    report = ActiveSetContinuityReport(
        previous_count=prev.active_count,
        current_count=current.active_count,
        retained_count=retained_count,
        entered_count=int(np.count_nonzero(entered)),
        exited_count=int(np.count_nonzero(exited)),
        measure_previous=prev.measure,
        measure_current=current.measure,
        relative_measure_jump=rel_measure_jump,
        jaccard_index=jaccard,
        warm_start_mask=prev_mask.copy(),
    )
    return ActiveSetContinuityState(previous=current, step_index=state.step_index + 1), report



def transport_snapshot_by_rigid_motion(
    snapshot: ActiveSetSnapshot | None,
    *,
    previous_rotation: ArrayLike | None = None,
    previous_position: ArrayLike | None = None,
    new_rotation: ArrayLike | None = None,
    new_position: ArrayLike | None = None,
) -> ActiveSetSnapshot | None:
    """Transport an active-set snapshot under rigid motion.

    For body-fixed support discretizations, rigid transport preserves index-space
    identity exactly, so the transported warm-start mask is the same boolean mask.
    The optional pose arguments are accepted to make the interface explicit and
    future-proof for world-space transport schemes.
    """
    if snapshot is None:
        return None
    return ActiveSetSnapshot(mask=np.asarray(snapshot.mask, dtype=bool).copy(), measure=float(snapshot.measure))
