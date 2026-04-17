from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pfcsdf.contact.active_set import ActiveSetSnapshot, ActiveSetUpdatePlan, continuity_update_plan, repair_update_plan
from pfcsdf.contact.band import hat_delta
from pfcsdf.contact.local_normal import solve_column_equilibrium
from pfcsdf.contact.wrench import PairWrench
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.depth import depth_from_phi
from pfcsdf.physics.pressure import LinearPressureLaw

ArrayLike = np.ndarray


def _sdf_gradient(sdf: Any, point: ArrayLike, eps: float = 1e-6) -> ArrayLike:
    if hasattr(sdf, "gradient"):
        return np.asarray(sdf.gradient(point), dtype=float)

    point = np.asarray(point, dtype=float)
    grad = np.zeros(3, dtype=float)
    for axis in range(3):
        step = np.zeros(3, dtype=float)
        step[axis] = eps
        plus = float(sdf.signed_distance(point + step))
        minus = float(sdf.signed_distance(point - step))
        grad[axis] = (plus - minus) / (2.0 * eps)
    return grad


def _depth_gradient(phi: float, phi_grad: ArrayLike, max_depth: float) -> ArrayLike:
    depth = float(depth_from_phi(phi, max_depth))
    if depth <= 0.0 or depth >= max_depth:
        return np.zeros(3, dtype=float)
    return -np.asarray(phi_grad, dtype=float)


def _safe_unit(v: ArrayLike, fallback: ArrayLike | None = None) -> ArrayLike:
    v = np.asarray(v, dtype=float)
    norm = float(np.linalg.norm(v))
    if norm > 1e-14:
        return v / norm
    if fallback is None:
        fallback = np.array([0.0, 0.0, 1.0], dtype=float)
    fallback = np.asarray(fallback, dtype=float)
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm <= 1e-14:
        raise ValueError("fallback normal must be non-zero")
    return fallback / fallback_norm


def _project_to_balance_surface(point: ArrayLike, balance: float, grad_h: ArrayLike) -> ArrayLike:
    point = np.asarray(point, dtype=float)
    grad_h = np.asarray(grad_h, dtype=float)
    grad_sq = float(np.dot(grad_h, grad_h))
    if grad_sq <= 1e-14:
        return point.copy()
    return point - (balance / grad_sq) * grad_h


def _clip_depth(depth: float, max_depth: float) -> float:
    return float(np.clip(depth, 0.0, max_depth))


@dataclass(frozen=True)
class NativeBandAccumulatorConfig:
    eta: float
    band_half_width: float

    def __post_init__(self) -> None:
        if self.eta <= 0.0:
            raise ValueError("eta must be positive")
        if self.band_half_width <= 0.0:
            raise ValueError("band_half_width must be positive")
        if self.band_half_width < self.eta:
            raise ValueError("band_half_width must be at least eta")


@dataclass(frozen=True)
class SampledPFCBalanceFields:
    grid: UniformGrid3D
    depth_a: ArrayLike
    depth_b: ArrayLike
    pressure_a: ArrayLike
    pressure_b: ArrayLike
    pressure_common: ArrayLike
    balance: ArrayLike
    grad_h: ArrayLike
    grad_depth_a: ArrayLike
    grad_depth_b: ArrayLike
    grad_pressure_a: ArrayLike
    grad_pressure_b: ArrayLike
    stiffness_a: float
    stiffness_b: float
    max_depth_a: float
    max_depth_b: float

    def __post_init__(self) -> None:
        shape = self.grid.shape
        expected_vec_shape = shape + (3,)
        if np.asarray(self.depth_a).shape != shape:
            raise ValueError("depth_a shape mismatch")
        if np.asarray(self.depth_b).shape != shape:
            raise ValueError("depth_b shape mismatch")
        if np.asarray(self.pressure_a).shape != shape:
            raise ValueError("pressure_a shape mismatch")
        if np.asarray(self.pressure_b).shape != shape:
            raise ValueError("pressure_b shape mismatch")
        if np.asarray(self.pressure_common).shape != shape:
            raise ValueError("pressure_common shape mismatch")
        if np.asarray(self.balance).shape != shape:
            raise ValueError("balance shape mismatch")
        if np.asarray(self.grad_h).shape != expected_vec_shape:
            raise ValueError("grad_h shape mismatch")
        if np.asarray(self.grad_depth_a).shape != expected_vec_shape:
            raise ValueError("grad_depth_a shape mismatch")
        if np.asarray(self.grad_depth_b).shape != expected_vec_shape:
            raise ValueError("grad_depth_b shape mismatch")
        if np.asarray(self.grad_pressure_a).shape != expected_vec_shape:
            raise ValueError("grad_pressure_a shape mismatch")
        if np.asarray(self.grad_pressure_b).shape != expected_vec_shape:
            raise ValueError("grad_pressure_b shape mismatch")
        if self.stiffness_a <= 0.0 or self.stiffness_b <= 0.0:
            raise ValueError("stiffnesses must be positive")
        if self.max_depth_a <= 0.0 or self.max_depth_b <= 0.0:
            raise ValueError("max depths must be positive")


@dataclass(frozen=True)
class NativeBandWrenchResult:
    wrench: PairWrench
    active_mask: ArrayLike
    weighted_measure: float

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(self.active_mask))

    @property
    def total_count(self) -> int:
        return int(self.active_mask.size)


@dataclass(frozen=True)
class SparseActiveCell:
    index: tuple[int, int, int]
    point: ArrayLike
    projected_point: ArrayLike
    normal: ArrayLike
    depth_a: float
    depth_b: float
    overlap: float
    balance: float
    grad_norm: float
    weight: float
    cell_volume: float
    pressure_cell_center: float
    pressure_local_normal: float


@dataclass(frozen=True)
class SparseBandTraversal:
    active_cells: tuple[SparseActiveCell, ...]
    active_mask: ArrayLike
    candidate_mask: ArrayLike | None = None
    update_plan: ActiveSetUpdatePlan | None = None

    @property
    def active_count(self) -> int:
        return len(self.active_cells)

    @property
    def total_count(self) -> int:
        return int(self.active_mask.size)

    @property
    def candidate_count(self) -> int:
        if self.candidate_mask is None:
            return self.total_count
        return int(np.count_nonzero(self.candidate_mask))

    @property
    def recompute_count(self) -> int:
        if self.update_plan is None:
            return self.candidate_count
        return self.update_plan.recompute_count

    @property
    def retained_count(self) -> int:
        if self.update_plan is None:
            return 0
        return self.update_plan.retained_count


@dataclass(frozen=True)
class SparseNativeBandWrenchResult(NativeBandWrenchResult):
    traversal: SparseBandTraversal


@dataclass(frozen=True)
class PerCellCubatureRule:
    offsets: ArrayLike
    weights: ArrayLike

    def __post_init__(self) -> None:
        offsets = np.asarray(self.offsets, dtype=float)
        weights = np.asarray(self.weights, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 3:
            raise ValueError("offsets must have shape (n, 3)")
        if weights.ndim != 1 or weights.shape[0] != offsets.shape[0]:
            raise ValueError("weights must be a 1D array matching offsets")
        if np.any(weights <= 0.0):
            raise ValueError("cubature weights must be positive")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-12):
            raise ValueError("cubature weights must sum to 1")
        object.__setattr__(self, "offsets", offsets)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def gauss_legendre_tensor_2(cls) -> "PerCellCubatureRule":
        a = 0.5 / np.sqrt(3.0)
        axes = (-a, a)
        offsets = np.array([[sx, sy, sz] for sx in axes for sy in axes for sz in axes], dtype=float)
        weights = np.full(offsets.shape[0], 1.0 / offsets.shape[0], dtype=float)
        return cls(offsets=offsets, weights=weights)


@dataclass(frozen=True)
class HigherOrderCubatureSample:
    point: ArrayLike
    projected_point: ArrayLike
    normal: ArrayLike
    depth_a: float
    depth_b: float
    overlap: float
    depth_a_projected: float
    depth_b_projected: float
    projected_overlap: float
    balance: float
    grad_norm: float
    weight: float
    pressure_linearized: float
    pressure_projected_linearized: float
    pressure_local_normal: float


@dataclass(frozen=True)
class HigherOrderSparseActiveCell:
    base_cell: SparseActiveCell
    cubature_samples: tuple[HigherOrderCubatureSample, ...]

    @property
    def integrated_weight(self) -> float:
        return float(sum(sample.weight for sample in self.cubature_samples))


@dataclass(frozen=True)
class HigherOrderSparseBandTraversal:
    active_cells: tuple[HigherOrderSparseActiveCell, ...]
    active_mask: ArrayLike
    cubature_rule: PerCellCubatureRule
    candidate_mask: ArrayLike | None = None
    update_plan: ActiveSetUpdatePlan | None = None

    @property
    def active_count(self) -> int:
        return len(self.active_cells)

    @property
    def total_count(self) -> int:
        return int(self.active_mask.size)

    @property
    def candidate_count(self) -> int:
        if self.candidate_mask is None:
            return self.total_count
        return int(np.count_nonzero(self.candidate_mask))

    @property
    def recompute_count(self) -> int:
        if self.update_plan is None:
            return self.candidate_count
        return self.update_plan.recompute_count

    @property
    def retained_count(self) -> int:
        if self.update_plan is None:
            return 0
        return self.update_plan.retained_count


@dataclass(frozen=True)
class HigherOrderSparseNativeBandWrenchResult(NativeBandWrenchResult):
    traversal: HigherOrderSparseBandTraversal



def sample_linear_pfc_balance_fields(
    grid: UniformGrid3D,
    sdf_a: Any,
    sdf_b: Any,
    law_a: LinearPressureLaw,
    law_b: LinearPressureLaw,
    *,
    max_depth_a: float,
    max_depth_b: float,
) -> SampledPFCBalanceFields:
    shape = grid.shape
    depth_a = np.zeros(shape, dtype=float)
    depth_b = np.zeros(shape, dtype=float)
    pressure_a = np.zeros(shape, dtype=float)
    pressure_b = np.zeros(shape, dtype=float)
    pressure_common = np.zeros(shape, dtype=float)
    balance = np.zeros(shape, dtype=float)
    grad_h = np.zeros(shape + (3,), dtype=float)
    grad_depth_a = np.zeros(shape + (3,), dtype=float)
    grad_depth_b = np.zeros(shape + (3,), dtype=float)
    grad_pressure_a = np.zeros(shape + (3,), dtype=float)
    grad_pressure_b = np.zeros(shape + (3,), dtype=float)

    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                x = grid.cell_center_point(i, j, k)
                phi_a = float(sdf_a.signed_distance(x))
                phi_b = float(sdf_b.signed_distance(x))
                d_a = float(depth_from_phi(phi_a, max_depth_a))
                d_b = float(depth_from_phi(phi_b, max_depth_b))
                p_a = float(law_a.pressure(d_a))
                p_b = float(law_b.pressure(d_b))
                grad_phi_a = _sdf_gradient(sdf_a, x)
                grad_phi_b = _sdf_gradient(sdf_b, x)
                g_d_a = _depth_gradient(phi_a, grad_phi_a, max_depth_a)
                g_d_b = _depth_gradient(phi_b, grad_phi_b, max_depth_b)
                g_p_a = law_a.stiffness * g_d_a
                g_p_b = law_b.stiffness * g_d_b
                g_h = g_p_a - g_p_b

                depth_a[i, j, k] = d_a
                depth_b[i, j, k] = d_b
                pressure_a[i, j, k] = p_a
                pressure_b[i, j, k] = p_b
                pressure_common[i, j, k] = 0.5 * (p_a + p_b)
                balance[i, j, k] = p_a - p_b
                grad_h[i, j, k] = g_h
                grad_depth_a[i, j, k] = g_d_a
                grad_depth_b[i, j, k] = g_d_b
                grad_pressure_a[i, j, k] = g_p_a
                grad_pressure_b[i, j, k] = g_p_b

    return SampledPFCBalanceFields(
        grid=grid,
        depth_a=depth_a,
        depth_b=depth_b,
        pressure_a=pressure_a,
        pressure_b=pressure_b,
        pressure_common=pressure_common,
        balance=balance,
        grad_h=grad_h,
        grad_depth_a=grad_depth_a,
        grad_depth_b=grad_depth_b,
        grad_pressure_a=grad_pressure_a,
        grad_pressure_b=grad_pressure_b,
        stiffness_a=law_a.stiffness,
        stiffness_b=law_b.stiffness,
        max_depth_a=max_depth_a,
        max_depth_b=max_depth_b,
    )



def active_contact_mask(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    *,
    extra_mask: ArrayLike | None = None,
) -> ArrayLike:
    mask = (fields.depth_a > 0.0) & (fields.depth_b > 0.0) & (np.abs(fields.balance) <= config.band_half_width)
    if extra_mask is not None:
        mask = mask & np.asarray(extra_mask, dtype=bool)
    return mask



def _active_contact_mask_on_region(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    region_mask: ArrayLike,
    *,
    extra_mask: ArrayLike | None = None,
) -> ArrayLike:
    region_mask = np.asarray(region_mask, dtype=bool)
    if region_mask.shape != fields.grid.shape:
        raise ValueError("region_mask shape mismatch")
    mask = region_mask & (fields.depth_a > 0.0) & (fields.depth_b > 0.0) & (np.abs(fields.balance) <= config.band_half_width)
    if extra_mask is not None:
        mask = mask & np.asarray(extra_mask, dtype=bool)
    return mask



def build_sparse_active_traversal(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    *,
    extra_mask: ArrayLike | None = None,
    local_normal_correction: bool = True,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
    continuity_dilation_radius: int = 1,
    boundary_only_update: bool = False,
    repair_mask: ArrayLike | None = None,
) -> SparseBandTraversal:
    update_plan: ActiveSetUpdatePlan | None = None
    candidate_mask: ArrayLike | None = None
    if warm_start_snapshot is not None and warm_start_snapshot.active_count > 0:
        if repair_mask is not None:
            update_plan = repair_update_plan(
                warm_start_snapshot.mask,
                repair_mask,
                extra_mask=extra_mask,
            )
        else:
            update_plan = continuity_update_plan(
                warm_start_snapshot.mask,
                extra_mask=extra_mask,
                dilation_radius=continuity_dilation_radius,
            )
        candidate_mask = update_plan.candidate_mask
        if boundary_only_update or repair_mask is not None:
            active = update_plan.retained_interior_mask.copy()
            if update_plan.recompute_count > 0:
                active |= _active_contact_mask_on_region(
                    fields,
                    config,
                    update_plan.recompute_mask,
                    extra_mask=extra_mask,
                )
        else:
            active = _active_contact_mask_on_region(
                fields,
                config,
                candidate_mask,
                extra_mask=extra_mask,
            )
    else:
        active = active_contact_mask(fields, config, extra_mask=extra_mask)
        candidate_mask = np.asarray(extra_mask, dtype=bool) if extra_mask is not None else np.ones(fields.grid.shape, dtype=bool)

    kernel = hat_delta(fields.balance, config.eta)
    grad_norm = np.linalg.norm(fields.grad_h, axis=-1)
    weights = np.where(active, kernel * grad_norm * fields.grid.cell_volume, 0.0)
    points = fields.grid.stacked_cell_centers()
    law_a = LinearPressureLaw(fields.stiffness_a)
    law_b = LinearPressureLaw(fields.stiffness_b)

    active_cells: list[SparseActiveCell] = []
    for i, j, k in np.argwhere(active):
        point = points[i, j, k]
        grad = fields.grad_h[i, j, k]
        normal = _safe_unit(grad)
        projected_point = _project_to_balance_surface(point, float(fields.balance[i, j, k]), grad)
        depth_a = float(fields.depth_a[i, j, k])
        depth_b = float(fields.depth_b[i, j, k])
        overlap = depth_a + depth_b
        pressure_cell_center = float(fields.pressure_common[i, j, k])
        if local_normal_correction:
            eq = solve_column_equilibrium(overlap, law_a, law_b)
            pressure_local_normal = float(eq.pressure)
        else:
            pressure_local_normal = pressure_cell_center
        active_cells.append(
            SparseActiveCell(
                index=(int(i), int(j), int(k)),
                point=np.asarray(point, dtype=float),
                projected_point=np.asarray(projected_point, dtype=float),
                normal=normal,
                depth_a=depth_a,
                depth_b=depth_b,
                overlap=overlap,
                balance=float(fields.balance[i, j, k]),
                grad_norm=float(grad_norm[i, j, k]),
                weight=float(weights[i, j, k]),
                cell_volume=float(fields.grid.cell_volume),
                pressure_cell_center=pressure_cell_center,
                pressure_local_normal=pressure_local_normal,
            )
        )
    return SparseBandTraversal(
        active_cells=tuple(active_cells),
        active_mask=active,
        candidate_mask=np.asarray(candidate_mask, dtype=bool) if candidate_mask is not None else None,
        update_plan=update_plan,
    )



def _linearized_sample_at_offset(
    fields: SampledPFCBalanceFields,
    idx: tuple[int, int, int],
    offset_world: ArrayLike,
    *,
    cubature_weight: float,
    config: NativeBandAccumulatorConfig,
    local_normal_correction: bool,
) -> HigherOrderCubatureSample:
    i, j, k = idx
    offset_world = np.asarray(offset_world, dtype=float)

    point = fields.grid.cell_center_point(i, j, k) + offset_world
    grad_h = np.asarray(fields.grad_h[i, j, k], dtype=float)
    grad_norm = float(np.linalg.norm(grad_h))
    normal = _safe_unit(grad_h)

    base_depth_a = float(fields.depth_a[i, j, k])
    base_depth_b = float(fields.depth_b[i, j, k])
    base_pressure_a = float(fields.pressure_a[i, j, k])
    base_pressure_b = float(fields.pressure_b[i, j, k])

    total_offset = np.asarray(offset_world, dtype=float)
    depth_a = _clip_depth(base_depth_a + float(np.dot(fields.grad_depth_a[i, j, k], total_offset)), fields.max_depth_a)
    depth_b = _clip_depth(base_depth_b + float(np.dot(fields.grad_depth_b[i, j, k], total_offset)), fields.max_depth_b)
    pressure_a = max(0.0, base_pressure_a + float(np.dot(fields.grad_pressure_a[i, j, k], total_offset)))
    pressure_b = max(0.0, base_pressure_b + float(np.dot(fields.grad_pressure_b[i, j, k], total_offset)))
    balance = float(fields.balance[i, j, k] + np.dot(grad_h, total_offset))

    overlap = depth_a + depth_b
    projected_point = _project_to_balance_surface(point, balance, grad_h)
    projected_offset = projected_point - fields.grid.cell_center_point(i, j, k)
    depth_a_projected = _clip_depth(base_depth_a + float(np.dot(fields.grad_depth_a[i, j, k], projected_offset)), fields.max_depth_a)
    depth_b_projected = _clip_depth(base_depth_b + float(np.dot(fields.grad_depth_b[i, j, k], projected_offset)), fields.max_depth_b)
    projected_overlap = depth_a_projected + depth_b_projected
    pressure_a_projected = max(0.0, base_pressure_a + float(np.dot(fields.grad_pressure_a[i, j, k], projected_offset)))
    pressure_b_projected = max(0.0, base_pressure_b + float(np.dot(fields.grad_pressure_b[i, j, k], projected_offset)))

    if depth_a <= 0.0 or depth_b <= 0.0 or overlap <= 0.0:
        kernel = 0.0
        pressure_linearized = 0.0
        pressure_projected_linearized = 0.0
        pressure_local_normal = 0.0
    else:
        kernel = float(hat_delta(balance, config.eta))
        pressure_linearized = 0.5 * (pressure_a + pressure_b)
        pressure_projected_linearized = 0.5 * (pressure_a_projected + pressure_b_projected)
        if local_normal_correction:
            law_a = LinearPressureLaw(fields.stiffness_a)
            law_b = LinearPressureLaw(fields.stiffness_b)
            pressure_local_normal = float(solve_column_equilibrium(projected_overlap, law_a, law_b).pressure)
        else:
            pressure_local_normal = pressure_projected_linearized

    weight = cubature_weight * kernel * grad_norm * fields.grid.cell_volume
    return HigherOrderCubatureSample(
        point=np.asarray(point, dtype=float),
        projected_point=np.asarray(projected_point, dtype=float),
        normal=normal,
        depth_a=depth_a,
        depth_b=depth_b,
        overlap=overlap,
        depth_a_projected=depth_a_projected,
        depth_b_projected=depth_b_projected,
        projected_overlap=projected_overlap,
        balance=balance,
        grad_norm=grad_norm,
        weight=weight,
        pressure_linearized=pressure_linearized,
        pressure_projected_linearized=pressure_projected_linearized,
        pressure_local_normal=pressure_local_normal,
    )



def build_higher_order_sparse_active_traversal(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    *,
    cubature_rule: PerCellCubatureRule | None = None,
    extra_mask: ArrayLike | None = None,
    local_normal_correction: bool = True,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
    continuity_dilation_radius: int = 1,
    boundary_only_update: bool = False,
    repair_mask: ArrayLike | None = None,
) -> HigherOrderSparseBandTraversal:
    if cubature_rule is None:
        cubature_rule = PerCellCubatureRule.gauss_legendre_tensor_2()

    base_traversal = build_sparse_active_traversal(
        fields,
        config,
        extra_mask=extra_mask,
        local_normal_correction=local_normal_correction,
        warm_start_snapshot=warm_start_snapshot,
        continuity_dilation_radius=continuity_dilation_radius,
        boundary_only_update=boundary_only_update,
        repair_mask=repair_mask,
    )

    higher_cells: list[HigherOrderSparseActiveCell] = []
    for base_cell in base_traversal.active_cells:
        samples: list[HigherOrderCubatureSample] = []
        for offset_ref, weight_ref in zip(cubature_rule.offsets, cubature_rule.weights, strict=True):
            offset_world = fields.grid.spacing * np.asarray(offset_ref, dtype=float)
            sample = _linearized_sample_at_offset(
                fields,
                base_cell.index,
                offset_world,
                cubature_weight=float(weight_ref),
                config=config,
                local_normal_correction=local_normal_correction,
            )
            samples.append(sample)
        higher_cells.append(HigherOrderSparseActiveCell(base_cell=base_cell, cubature_samples=tuple(samples)))

    return HigherOrderSparseBandTraversal(
        active_cells=tuple(higher_cells),
        active_mask=base_traversal.active_mask,
        cubature_rule=cubature_rule,
        candidate_mask=base_traversal.candidate_mask,
        update_plan=base_traversal.update_plan,
    )



def accumulate_sdf_native_band_wrench(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    *,
    reference: ArrayLike | None = None,
    extra_mask: ArrayLike | None = None,
) -> NativeBandWrenchResult:
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    active = active_contact_mask(fields, config, extra_mask=extra_mask)
    kernel = hat_delta(fields.balance, config.eta)
    grad_norm = np.linalg.norm(fields.grad_h, axis=-1)
    safe_grad_norm = np.where(grad_norm > 1e-14, grad_norm, 1.0)
    normal = fields.grad_h / safe_grad_norm[..., None]
    weights = kernel * grad_norm * fields.grid.cell_volume
    weights = np.where(active, weights, 0.0)

    force_density = fields.pressure_common[..., None] * weights[..., None] * normal
    points = fields.grid.stacked_cell_centers()
    lever = points - reference
    torque_density = np.cross(lever, force_density)

    force = np.sum(force_density, axis=(0, 1, 2))
    torque = np.sum(torque_density, axis=(0, 1, 2))
    weighted_measure = float(np.sum(weights))
    return NativeBandWrenchResult(
        wrench=PairWrench(force=force, torque=torque),
        active_mask=active,
        weighted_measure=weighted_measure,
    )



def accumulate_sparse_sdf_native_band_wrench(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    *,
    reference: ArrayLike | None = None,
    extra_mask: ArrayLike | None = None,
    local_normal_correction: bool = True,
    use_projected_points: bool = True,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
    continuity_dilation_radius: int = 1,
    boundary_only_update: bool = False,
    repair_mask: ArrayLike | None = None,
) -> SparseNativeBandWrenchResult:
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    traversal = build_sparse_active_traversal(
        fields,
        config,
        extra_mask=extra_mask,
        local_normal_correction=local_normal_correction,
        warm_start_snapshot=warm_start_snapshot,
        continuity_dilation_radius=continuity_dilation_radius,
        boundary_only_update=boundary_only_update,
        repair_mask=repair_mask,
    )

    force = np.zeros(3, dtype=float)
    torque = np.zeros(3, dtype=float)
    weighted_measure = 0.0
    for cell in traversal.active_cells:
        application_point = cell.projected_point if use_projected_points else cell.point
        pressure = cell.pressure_local_normal if local_normal_correction else cell.pressure_cell_center
        cell_force = pressure * cell.weight * cell.normal
        cell_torque = np.cross(application_point - reference, cell_force)
        force += cell_force
        torque += cell_torque
        weighted_measure += cell.weight

    return SparseNativeBandWrenchResult(
        wrench=PairWrench(force=force, torque=torque),
        active_mask=traversal.active_mask,
        weighted_measure=float(weighted_measure),
        traversal=traversal,
    )



def accumulate_higher_order_sparse_sdf_native_band_wrench(
    fields: SampledPFCBalanceFields,
    config: NativeBandAccumulatorConfig,
    *,
    reference: ArrayLike | None = None,
    cubature_rule: PerCellCubatureRule | None = None,
    extra_mask: ArrayLike | None = None,
    local_normal_correction: bool = True,
    use_projected_points: bool = True,
    consistent_traction_reconstruction: bool = True,
    warm_start_snapshot: ActiveSetSnapshot | None = None,
    continuity_dilation_radius: int = 1,
    boundary_only_update: bool = False,
    repair_mask: ArrayLike | None = None,
) -> HigherOrderSparseNativeBandWrenchResult:
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    traversal = build_higher_order_sparse_active_traversal(
        fields,
        config,
        cubature_rule=cubature_rule,
        extra_mask=extra_mask,
        local_normal_correction=local_normal_correction,
        warm_start_snapshot=warm_start_snapshot,
        continuity_dilation_radius=continuity_dilation_radius,
        boundary_only_update=boundary_only_update,
        repair_mask=repair_mask,
    )

    force = np.zeros(3, dtype=float)
    torque = np.zeros(3, dtype=float)
    weighted_measure = 0.0
    for cell in traversal.active_cells:
        if consistent_traction_reconstruction:
            cell_force = np.zeros(3, dtype=float)
            scalar_weights = []
            application_points = []
            for sample in cell.cubature_samples:
                application_point = sample.projected_point if use_projected_points else sample.point
                pressure = sample.pressure_local_normal if local_normal_correction else sample.pressure_projected_linearized
                sample_force = pressure * sample.weight * sample.normal
                cell_force += sample_force
                scalar_weights.append(max(pressure * sample.weight, 0.0))
                application_points.append(application_point)
                weighted_measure += sample.weight
            scalar_weights_arr = np.asarray(scalar_weights, dtype=float)
            if float(np.sum(scalar_weights_arr)) > 1e-14:
                application_points_arr = np.asarray(application_points, dtype=float)
                cell_point = np.average(application_points_arr, axis=0, weights=scalar_weights_arr)
            else:
                cell_point = cell.base_cell.projected_point if use_projected_points else cell.base_cell.point
            cell_torque = np.cross(cell_point - reference, cell_force)
            force += cell_force
            torque += cell_torque
        else:
            for sample in cell.cubature_samples:
                application_point = sample.projected_point if use_projected_points else sample.point
                pressure = sample.pressure_local_normal if local_normal_correction else sample.pressure_projected_linearized
                sample_force = pressure * sample.weight * sample.normal
                sample_torque = np.cross(application_point - reference, sample_force)
                force += sample_force
                torque += sample_torque
                weighted_measure += sample.weight

    return HigherOrderSparseNativeBandWrenchResult(
        wrench=PairWrench(force=force, torque=torque),
        active_mask=traversal.active_mask,
        weighted_measure=float(weighted_measure),
        traversal=traversal,
    )
