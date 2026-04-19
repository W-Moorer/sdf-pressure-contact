from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from pfcsdf.contact.active_set import ActiveSetContinuityState, ActiveSetSnapshot, continuity_report
from pfcsdf.contact.native_band import (
    NativeBandAccumulatorConfig,
    accumulate_higher_order_sparse_sdf_native_band_wrench,
    sample_linear_pfc_balance_fields,
)
from pfcsdf.geometry.base import SignedDistanceGeometry
from pfcsdf.geometry.mesh_factory import MeshAssetGeometryBuildResult, build_mesh_asset_sdf_geometry
from pfcsdf.geometry.mesh_io import build_uv_sphere_triangle_mesh
from pfcsdf.geometry.mesh_sdf import build_mesh_grid_sdf
from pfcsdf.geometry.primitives import BoxFootprint, PlaneSDF, SphereSDF
from pfcsdf.geometry.transforms import TransformedGeometry
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw
from pfcsdf.solvers.static import compute_sphere_plane_contact_linear_exact

from .events import EventAwareControllerConfig
from .integrators import event_aware_midpoint_impulse_corrected_step, event_aware_midpoint_step, event_aware_midpoint_work_consistent_step, midpoint_contact_step, midpoint_contact_substep, semi_implicit_euler_step
from .metrics import (
    DynamicErrorSummary,
    energy_drift,
    integrate_impulse,
    kinetic_energy,
    linear_contact_potential,
    relative_error,
    timing_error,
)
from .state import IntegrationStepResult, ModelEvaluation, NormalDynamicsState


@dataclass(frozen=True)
class FlatImpactSetup:
    initial_gap: float
    initial_velocity: float
    mass: float
    contact_stiffness: float
    t_final: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.contact_stiffness <= 0.0:
            raise ValueError("contact_stiffness must be positive")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive")
        if self.initial_velocity >= 0.0:
            raise ValueError("initial_velocity should be negative for impact benchmarks")
        if self.initial_gap < 0.0:
            raise ValueError("initial_gap must be non-negative")

    @property
    def natural_frequency(self) -> float:
        return float(np.sqrt(self.contact_stiffness / self.mass))

    @property
    def onset_time_exact(self) -> float:
        return float(self.initial_gap / (-self.initial_velocity))

    @property
    def release_time_exact(self) -> float:
        return float(self.onset_time_exact + np.pi / self.natural_frequency)

    def initial_state(self) -> NormalDynamicsState:
        return NormalDynamicsState(time=0.0, gap=self.initial_gap, velocity=self.initial_velocity, mass=self.mass)




@dataclass(frozen=True)
class SphereImpactSetup:
    initial_gap: float
    initial_velocity: float
    mass: float
    sphere_radius: float
    sphere_stiffness: float
    plane_stiffness: float
    t_final: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if self.sphere_radius <= 0.0:
            raise ValueError("sphere_radius must be positive")
        if self.sphere_stiffness <= 0.0 or self.plane_stiffness <= 0.0:
            raise ValueError("sphere_stiffness and plane_stiffness must be positive")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive")
        if self.initial_velocity >= 0.0:
            raise ValueError("initial_velocity should be negative for impact benchmarks")
        if self.initial_gap < 0.0:
            raise ValueError("initial_gap must be non-negative")

    @property
    def onset_time_exact(self) -> float:
        return float(self.initial_gap / (-self.initial_velocity))

    @property
    def equivalent_stiffness(self) -> float:
        return float(self.sphere_stiffness * self.plane_stiffness / (self.sphere_stiffness + self.plane_stiffness))

    def force(self, gap: float) -> float:
        overlap = max(-float(gap), 0.0)
        if overlap <= 0.0:
            return 0.0
        exact = compute_sphere_plane_contact_linear_exact(
            overlap,
            self.sphere_radius,
            LinearPressureLaw(self.sphere_stiffness),
            LinearPressureLaw(self.plane_stiffness),
            normal=np.array([0.0, 0.0, 1.0]),
        )
        return float(exact.wrench.force[2])

    def potential(self, gap: float) -> float:
        overlap = max(-float(gap), 0.0)
        if overlap <= 0.0:
            return 0.0
        k_eq = self.equivalent_stiffness
        R = self.sphere_radius
        return float(np.pi * k_eq * (R * overlap**3 / 3.0 - overlap**4 / 12.0))

    def initial_state(self) -> NormalDynamicsState:
        return NormalDynamicsState(time=0.0, gap=self.initial_gap, velocity=self.initial_velocity, mass=self.mass)


class AnalyticLinearSphereContactModel:
    def __init__(self, setup: SphereImpactSetup) -> None:
        self.setup = setup
        self.mass = float(setup.mass)

    def force(self, gap: float) -> float:
        return self.setup.force(gap)

    def __call__(self, state: NormalDynamicsState) -> ModelEvaluation:
        force = self.force(state.gap)
        return ModelEvaluation(force=force, acceleration=force / self.mass, active=force > 0.0)


class NativeBandSphereContactModel:
    def __init__(
        self,
        setup: SphereImpactSetup,
        grid: UniformGrid3D,
        config: NativeBandAccumulatorConfig,
        *,
        max_depth_a: float,
        max_depth_b: float,
        use_continuity_warm_start: bool = True,
        boundary_only_update: bool = True,
        continuity_dilation_radius: int = 1,
        consistent_traction_reconstruction: bool = True,
    ) -> None:
        self.setup = setup
        self.mass = float(setup.mass)
        self.grid = grid
        self.config = config
        self.max_depth_a = float(max_depth_a)
        self.max_depth_b = float(max_depth_b)
        self.use_continuity_warm_start = bool(use_continuity_warm_start)
        self.boundary_only_update = bool(boundary_only_update)
        self.continuity_dilation_radius = int(continuity_dilation_radius)
        self.consistent_traction_reconstruction = bool(consistent_traction_reconstruction)
        self.law_a = LinearPressureLaw(setup.sphere_stiffness)
        self.law_b = LinearPressureLaw(setup.plane_stiffness)
        points = self.grid.stacked_cell_centers()
        self.extra_mask = np.zeros(self.grid.shape, dtype=bool)
        radial_limit = setup.sphere_radius + 0.1
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                x, y, _ = points[i, j, 0]
                self.extra_mask[i, j, :] = (x * x + y * y) <= radial_limit * radial_limit
        self.reset_continuity()

    def reset_continuity(self) -> None:
        self._continuity_state = ActiveSetContinuityState()
        self.last_continuity_report = None
        self.last_candidate_count = self.grid.shape[0] * self.grid.shape[1] * self.grid.shape[2]
        self.last_recompute_count = self.last_candidate_count
        self.last_retained_count = 0

    def commit_active_snapshot(self, snapshot: ActiveSetSnapshot | None) -> None:
        if snapshot is None:
            return
        self._continuity_state, self.last_continuity_report = continuity_report(self._continuity_state, snapshot)

    @property
    def warm_start_snapshot(self) -> ActiveSetSnapshot | None:
        if not self.use_continuity_warm_start:
            return None
        return self._continuity_state.previous

    def build_contact_geometries(self, gap: float) -> tuple[SignedDistanceGeometry, SignedDistanceGeometry]:
        sphere = SphereSDF(center=np.array([0.0, 0.0, self.setup.sphere_radius + gap]), radius=self.setup.sphere_radius)
        plane = PlaneSDF(point=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
        return sphere, plane

    def evaluate(
        self,
        gap: float,
        *,
        warm_start_snapshot: ActiveSetSnapshot | None = None,
        boundary_only_update: bool | None = None,
        repair_mask: np.ndarray | None = None,
    ) -> ModelEvaluation:
        overlap = max(-float(gap), 0.0)
        if overlap <= 0.0:
            self.last_candidate_count = 0
            self.last_recompute_count = 0
            self.last_retained_count = 0
            empty = ActiveSetSnapshot.empty(self.grid.shape)
            return ModelEvaluation(force=0.0, acceleration=0.0, active=False, active_measure=0.0, active_snapshot=empty)
        if boundary_only_update is None:
            boundary_only_update = self.boundary_only_update
        if warm_start_snapshot is None:
            warm_start_snapshot = self.warm_start_snapshot
        sphere, plane = self.build_contact_geometries(gap)
        fields = sample_linear_pfc_balance_fields(
            self.grid, sphere, plane, self.law_a, self.law_b, max_depth_a=self.max_depth_a, max_depth_b=self.max_depth_b
        )
        result = accumulate_higher_order_sparse_sdf_native_band_wrench(
            fields,
            self.config,
            extra_mask=self.extra_mask,
            local_normal_correction=True,
            use_projected_points=True,
            warm_start_snapshot=warm_start_snapshot,
            continuity_dilation_radius=self.continuity_dilation_radius,
            boundary_only_update=bool(boundary_only_update),
            repair_mask=repair_mask,
            consistent_traction_reconstruction=self.consistent_traction_reconstruction,
        )
        self.last_candidate_count = result.traversal.candidate_count
        self.last_recompute_count = result.traversal.recompute_count
        self.last_retained_count = result.traversal.retained_count
        force = float(result.wrench.force[2])
        snapshot = ActiveSetSnapshot(mask=result.active_mask, measure=result.weighted_measure)
        return ModelEvaluation(
            force=force,
            acceleration=force / self.mass,
            active=force > 0.0,
            active_measure=result.weighted_measure,
            active_snapshot=snapshot,
        )

    def __call__(self, state: NormalDynamicsState) -> ModelEvaluation:
        return self.evaluate(state.gap)


class MeshNativeBandSphereContactModel(NativeBandSphereContactModel):
    """Sphere-plane native-band validation model using a mesh-derived local SDF."""

    def __init__(
        self,
        setup: SphereImpactSetup,
        grid: UniformGrid3D,
        config: NativeBandAccumulatorConfig,
        *,
        sphere_geometry_local: SignedDistanceGeometry,
        max_depth_a: float,
        max_depth_b: float,
        use_continuity_warm_start: bool = True,
        boundary_only_update: bool = True,
        continuity_dilation_radius: int = 1,
        consistent_traction_reconstruction: bool = True,
    ) -> None:
        super().__init__(
            setup,
            grid,
            config,
            max_depth_a=max_depth_a,
            max_depth_b=max_depth_b,
            use_continuity_warm_start=use_continuity_warm_start,
            boundary_only_update=boundary_only_update,
            continuity_dilation_radius=continuity_dilation_radius,
            consistent_traction_reconstruction=consistent_traction_reconstruction,
        )
        self.sphere_geometry_local = sphere_geometry_local

    def build_contact_geometries(self, gap: float) -> tuple[SignedDistanceGeometry, SignedDistanceGeometry]:
        sphere = TransformedGeometry.from_translation(
            self.sphere_geometry_local,
            np.array([0.0, 0.0, self.setup.sphere_radius + gap], dtype=float),
        )
        plane = PlaneSDF(point=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 1.0]))
        return sphere, plane


@dataclass(frozen=True)
class MeshSphereDynamicValidationResult:
    """Short-horizon analytic-vs-mesh sphere impact comparison on a shared benchmark setup."""

    analytic_history: DynamicHistory
    mesh_history: DynamicHistory
    analytic_error: DynamicErrorSummary
    mesh_error: DynamicErrorSummary
    peak_force_relative_difference: float
    impulse_relative_difference: float
    max_penetration_relative_difference: float
    onset_time_difference: float


@dataclass(frozen=True)
class MeshSphereDynamicFactoryValidationResult:
    """Dynamic mesh-backed sphere validation built through the mesh asset factory."""

    mesh_geometry: MeshAssetGeometryBuildResult
    comparison: MeshSphereDynamicValidationResult
    dt: float
    scheme: str


@dataclass(frozen=True)
class MeshSphereSensitivityCase:
    """One factor-setting in a controlled mesh-backed sphere sensitivity study."""

    axis: str
    label: str
    mesh_segments: int
    mesh_stacks: int
    mesh_sdf_spacing: float
    native_band_spacing: tuple[float, float, float]
    native_band_shape: tuple[int, int, int]
    analytic_static_force: float
    mesh_static_force: float
    static_force_relative_difference: float
    analytic_peak_force: float | None = None
    mesh_peak_force: float | None = None
    peak_force_relative_difference: float | None = None
    analytic_impulse: float | None = None
    mesh_impulse: float | None = None
    impulse_relative_difference: float | None = None
    max_penetration_relative_difference: float | None = None
    onset_time_difference: float | None = None


@dataclass(frozen=True)
class MeshSphereSensitivityStudy:
    """Small one-factor-at-a-time diagnostics for the mesh-backed sphere path."""

    static_overlap: float
    axes_scanned: tuple[str, ...]
    cases: tuple[MeshSphereSensitivityCase, ...]

    def axis_cases(self, axis: str) -> tuple[MeshSphereSensitivityCase, ...]:
        return tuple(case for case in self.cases if case.axis == axis)


@dataclass(frozen=True)
class DynamicHistory:
    times: np.ndarray
    gaps: np.ndarray
    velocities: np.ndarray
    forces: np.ndarray
    active: np.ndarray
    used_substeps: np.ndarray
    total_energy: np.ndarray
    onset_estimates: np.ndarray
    release_estimates: np.ndarray
    active_measure: np.ndarray
    continuity_jaccard: np.ndarray
    predictor_force_jump: np.ndarray | None = None
    predictor_active_measure_jump: np.ndarray | None = None
    predictor_corrector_jaccard: np.ndarray | None = None
    predictor_corrector_mismatch_fraction: np.ndarray | None = None
    work_mismatch: np.ndarray | None = None
    candidate_count: np.ndarray | None = None
    recompute_count: np.ndarray | None = None

    @property
    def onset_time(self) -> float | None:
        finite = self.onset_estimates[np.isfinite(self.onset_estimates)]
        if finite.size:
            return float(np.min(finite))
        indices = np.flatnonzero(self.active)
        if indices.size == 0:
            return None
        return float(self.times[int(indices[0])])

    @property
    def release_time(self) -> float | None:
        finite = self.release_estimates[np.isfinite(self.release_estimates)]
        if finite.size:
            return float(np.min(finite))
        active_int = self.active.astype(int)
        diffs = np.diff(active_int)
        rel = np.flatnonzero(diffs < 0)
        if rel.size == 0:
            return None
        return float(self.times[int(rel[0] + 1)])

    @property
    def impulse(self) -> float:
        return integrate_impulse(self.times, self.forces)

    @property
    def energy_drift(self) -> float:
        return energy_drift(self.total_energy)


class AnalyticLinearFlatContactModel:
    def __init__(self, mass: float, contact_stiffness: float) -> None:
        if mass <= 0.0 or contact_stiffness <= 0.0:
            raise ValueError("mass and contact_stiffness must be positive")
        self.mass = float(mass)
        self.contact_stiffness = float(contact_stiffness)

    def force(self, gap: float) -> float:
        return float(max(-self.contact_stiffness * gap, 0.0))

    def __call__(self, state: NormalDynamicsState) -> ModelEvaluation:
        force = self.force(state.gap)
        return ModelEvaluation(force=force, acceleration=force / self.mass, active=force > 0.0)


class _UpperOverlapPlane:
    def __init__(self, overlap: float) -> None:
        self.overlap = float(overlap)

    def signed_distance(self, x: np.ndarray) -> float:
        return float(-(x[2] + 0.5 * self.overlap))

    def gradient(self, x: np.ndarray | None = None) -> np.ndarray:
        return np.array([0.0, 0.0, -1.0], dtype=float)


class _LowerOverlapPlane:
    def __init__(self, overlap: float) -> None:
        self.overlap = float(overlap)

    def signed_distance(self, x: np.ndarray) -> float:
        return float(x[2] - 0.5 * self.overlap)

    def gradient(self, x: np.ndarray | None = None) -> np.ndarray:
        return np.array([0.0, 0.0, 1.0], dtype=float)


class NativeBandFlatContactModel:
    def __init__(
        self,
        mass: float,
        grid: UniformGrid3D,
        footprint: BoxFootprint,
        law_a: LinearPressureLaw,
        law_b: LinearPressureLaw,
        config: NativeBandAccumulatorConfig,
        *,
        max_depth_a: float,
        max_depth_b: float,
        use_continuity_warm_start: bool = True,
        boundary_only_update: bool = True,
        continuity_dilation_radius: int = 1,
        consistent_traction_reconstruction: bool = True,
    ) -> None:
        self.mass = float(mass)
        self.grid = grid
        self.footprint = footprint
        self.law_a = law_a
        self.law_b = law_b
        self.config = config
        self.max_depth_a = float(max_depth_a)
        self.max_depth_b = float(max_depth_b)
        self.use_continuity_warm_start = bool(use_continuity_warm_start)
        self.boundary_only_update = bool(boundary_only_update)
        self.continuity_dilation_radius = int(continuity_dilation_radius)
        self.consistent_traction_reconstruction = bool(consistent_traction_reconstruction)
        points = self.grid.stacked_cell_centers()
        self.extra_mask = np.zeros(self.grid.shape, dtype=bool)
        for i in range(self.grid.shape[0]):
            for j in range(self.grid.shape[1]):
                xy = points[i, j, 0]
                self.extra_mask[i, j, :] = self.footprint.contains_xy(xy)
        self.effective_area = float(np.count_nonzero(self.extra_mask[:, :, 0]) * self.grid.spacing[0] * self.grid.spacing[1])
        self.reset_continuity()

    def reset_continuity(self) -> None:
        self._continuity_state = ActiveSetContinuityState()
        self.last_continuity_report = None
        self.last_candidate_count = self.grid.shape[0] * self.grid.shape[1] * self.grid.shape[2]
        self.last_recompute_count = self.last_candidate_count
        self.last_retained_count = 0

    def commit_active_snapshot(self, snapshot: ActiveSetSnapshot | None) -> None:
        if snapshot is None:
            return
        self._continuity_state, self.last_continuity_report = continuity_report(self._continuity_state, snapshot)

    @property
    def warm_start_snapshot(self) -> ActiveSetSnapshot | None:
        if not self.use_continuity_warm_start:
            return None
        return self._continuity_state.previous

    def evaluate(
        self,
        gap: float,
        *,
        warm_start_snapshot: ActiveSetSnapshot | None = None,
        boundary_only_update: bool | None = None,
        repair_mask: np.ndarray | None = None,
    ) -> ModelEvaluation:
        overlap = max(-float(gap), 0.0)
        if overlap <= 0.0:
            self.last_candidate_count = 0
            self.last_recompute_count = 0
            self.last_retained_count = 0
            empty = ActiveSetSnapshot.empty(self.grid.shape)
            return ModelEvaluation(force=0.0, acceleration=0.0, active=False, active_measure=0.0, active_snapshot=empty)
        if boundary_only_update is None:
            boundary_only_update = self.boundary_only_update
        if warm_start_snapshot is None:
            warm_start_snapshot = self.warm_start_snapshot
        upper = _UpperOverlapPlane(overlap)
        lower = _LowerOverlapPlane(overlap)
        fields = sample_linear_pfc_balance_fields(
            self.grid,
            upper,
            lower,
            self.law_a,
            self.law_b,
            max_depth_a=self.max_depth_a,
            max_depth_b=self.max_depth_b,
        )
        result = accumulate_higher_order_sparse_sdf_native_band_wrench(
            fields,
            self.config,
            extra_mask=self.extra_mask,
            local_normal_correction=True,
            use_projected_points=True,
            warm_start_snapshot=warm_start_snapshot,
            continuity_dilation_radius=self.continuity_dilation_radius,
            boundary_only_update=bool(boundary_only_update),
            repair_mask=repair_mask,
            consistent_traction_reconstruction=self.consistent_traction_reconstruction,
        )
        self.last_candidate_count = result.traversal.candidate_count
        self.last_recompute_count = result.traversal.recompute_count
        self.last_retained_count = result.traversal.retained_count
        force = float(result.wrench.force[2])
        snapshot = ActiveSetSnapshot(mask=result.active_mask, measure=result.weighted_measure)
        return ModelEvaluation(
            force=force,
            acceleration=force / self.mass,
            active=force > 0.0,
            active_measure=result.weighted_measure,
            active_snapshot=snapshot,
        )

    def evaluate_state(
        self,
        state: NormalDynamicsState,
        *,
        warm_start_snapshot: ActiveSetSnapshot | None = None,
        boundary_only_update: bool | None = None,
        repair_mask: np.ndarray | None = None,
    ) -> ModelEvaluation:
        return self.evaluate(
            state.gap,
            warm_start_snapshot=warm_start_snapshot,
            boundary_only_update=boundary_only_update,
            repair_mask=repair_mask,
        )

    def force(self, gap: float) -> float:
        return self.evaluate(gap).force

    def __call__(self, state: NormalDynamicsState) -> ModelEvaluation:
        return self.evaluate(state.gap)



def exact_flat_impact_state(setup: FlatImpactSetup, time: float) -> tuple[float, float, float]:
    t = float(time)
    if t <= setup.onset_time_exact:
        gap = setup.initial_gap + setup.initial_velocity * t
        velocity = setup.initial_velocity
        force = 0.0
        return gap, velocity, force

    tau = t - setup.onset_time_exact
    w = setup.natural_frequency
    if tau <= np.pi / w:
        gap = (setup.initial_velocity / w) * np.sin(w * tau)
        velocity = setup.initial_velocity * np.cos(w * tau)
        force = max(-setup.contact_stiffness * gap, 0.0)
        return gap, velocity, force

    gap = 0.0
    velocity = -setup.initial_velocity
    force = 0.0
    return gap, velocity, force



def _simulate_reference_history_rk4(
    initial_state: NormalDynamicsState,
    t_final: float,
    *,
    force_fn: Callable[[float], float],
    potential_fn: Callable[[float], float],
    dt_ref: float,
) -> DynamicHistory:
    def deriv(gap: float, vel: float, mass: float) -> tuple[float, float]:
        force = force_fn(gap)
        return vel, force / mass

    state = initial_state
    times = [state.time]
    gaps = [state.gap]
    velocities = [state.velocity]
    force0 = force_fn(state.gap)
    forces = [force0]
    active = [force0 > 0.0]
    used_substeps = [1]
    total_energy = [kinetic_energy(state.mass, state.velocity) + potential_fn(state.gap)]
    onset_estimates = [np.nan]
    release_estimates = [np.nan]
    active_measure = [np.nan]
    continuity_jaccard = [np.nan]
    while state.time < t_final - 1e-15:
        h = min(dt_ref, t_final - state.time)
        g0, v0, m = state.gap, state.velocity, state.mass
        k1g, k1v = deriv(g0, v0, m)
        k2g, k2v = deriv(g0 + 0.5 * h * k1g, v0 + 0.5 * h * k1v, m)
        k3g, k3v = deriv(g0 + 0.5 * h * k2g, v0 + 0.5 * h * k2v, m)
        k4g, k4v = deriv(g0 + h * k3g, v0 + h * k3v, m)
        g1 = g0 + (h / 6.0) * (k1g + 2.0 * k2g + 2.0 * k3g + k4g)
        v1 = v0 + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        state = NormalDynamicsState(time=state.time + h, gap=float(g1), velocity=float(v1), mass=state.mass)
        f1 = force_fn(state.gap)
        times.append(state.time)
        gaps.append(state.gap)
        velocities.append(state.velocity)
        forces.append(f1)
        active.append(f1 > 0.0)
        used_substeps.append(1)
        total_energy.append(kinetic_energy(state.mass, state.velocity) + potential_fn(state.gap))
        onset_estimates.append(np.nan)
        release_estimates.append(np.nan)
        active_measure.append(np.nan)
        continuity_jaccard.append(np.nan)

    n = len(times)
    return DynamicHistory(
        times=np.asarray(times, dtype=float),
        gaps=np.asarray(gaps, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        forces=np.asarray(forces, dtype=float),
        active=np.asarray(active, dtype=bool),
        used_substeps=np.asarray(used_substeps, dtype=int),
        total_energy=np.asarray(total_energy, dtype=float),
        onset_estimates=np.asarray(onset_estimates, dtype=float),
        release_estimates=np.asarray(release_estimates, dtype=float),
        active_measure=np.asarray(active_measure, dtype=float),
        continuity_jaccard=np.asarray(continuity_jaccard, dtype=float),
        predictor_force_jump=np.full(n, np.nan, dtype=float),
        predictor_active_measure_jump=np.full(n, np.nan, dtype=float),
        predictor_corrector_jaccard=np.full(n, np.nan, dtype=float),
        predictor_corrector_mismatch_fraction=np.full(n, np.nan, dtype=float),
        work_mismatch=np.full(n, np.nan, dtype=float),
        candidate_count=np.full(n, np.nan, dtype=float),
        recompute_count=np.full(n, np.nan, dtype=float),
    )


def sphere_reference_history(setup: SphereImpactSetup, *, dt_ref: float = 5e-4) -> DynamicHistory:
    model = AnalyticLinearSphereContactModel(setup)
    return _simulate_reference_history_rk4(
        setup.initial_state(),
        setup.t_final,
        force_fn=model.force,
        potential_fn=setup.potential,
        dt_ref=dt_ref,
    )


def benchmark_history_error_against_reference(reference: DynamicHistory, history: DynamicHistory, *, onset_time_reference: float) -> DynamicErrorSummary:
    ref_gaps = np.interp(history.times, reference.times, reference.gaps)
    ref_velocities = np.interp(history.times, reference.times, reference.velocities)
    ref_forces = np.interp(history.times, reference.times, reference.forces)
    state_error = float(np.hypot(history.gaps[-1] - ref_gaps[-1], history.velocities[-1] - ref_velocities[-1]))
    force_error = relative_error(history.forces[-1], ref_forces[-1])
    impulse_error = relative_error(history.impulse, reference.impulse)
    onset_numeric = history.onset_time if history.onset_time is not None else history.times[-1]
    onset_error = timing_error(onset_numeric, onset_time_reference)
    release_ref = reference.release_time if reference.release_time is not None else reference.times[-1]
    release_numeric = history.release_time if history.release_time is not None else history.times[-1]
    release_error = timing_error(release_numeric, release_ref)
    peak_force_error = relative_error(float(np.max(history.forces)), float(np.max(reference.forces)))
    max_penetration_error = relative_error(float(-np.min(history.gaps)), float(-np.min(reference.gaps)))
    rebound_velocity_error = relative_error(float(history.velocities[-1]), float(reference.velocities[-1]))
    return DynamicErrorSummary(
        state_error=state_error,
        force_error=force_error,
        impulse_error=impulse_error,
        onset_timing_error=onset_error,
        release_timing_error=release_error,
        peak_force_error=peak_force_error,
        max_penetration_error=max_penetration_error,
        rebound_velocity_error=rebound_velocity_error,
    )


def benchmark_sphere_impact_error(setup: SphereImpactSetup, history: DynamicHistory, *, dt_ref: float = 5e-4) -> DynamicErrorSummary:
    reference = sphere_reference_history(setup, dt_ref=dt_ref)
    return benchmark_history_error_against_reference(reference, history, onset_time_reference=setup.onset_time_exact)


def _as_vector3(value: float | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(3, float(arr), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a scalar or have shape (3,)")
    return arr


def _build_matching_native_band_grid(
    *,
    cell_center_minimum: np.ndarray,
    cell_center_maximum: np.ndarray,
    spacing: float | np.ndarray,
) -> UniformGrid3D:
    spacing_vec = _as_vector3(spacing, name="spacing")
    if np.any(spacing_vec <= 0.0):
        raise ValueError("spacing must be strictly positive")
    center_span = np.asarray(cell_center_maximum, dtype=float) - np.asarray(cell_center_minimum, dtype=float)
    shape = tuple(int(max(2, np.ceil(center_span[axis] / spacing_vec[axis] - 1e-12) + 1)) for axis in range(3))
    origin = np.asarray(cell_center_minimum, dtype=float) - 0.5 * spacing_vec
    return UniformGrid3D(origin=origin, spacing=spacing_vec, shape=shape)


def _build_mesh_backed_sphere_local_geometry(
    *,
    radius: float,
    mesh_segments: int,
    mesh_stacks: int,
    mesh_sdf_spacing: float,
    mesh_padding: float | np.ndarray,
) -> SignedDistanceGeometry:
    mesh = build_uv_sphere_triangle_mesh(
        radius=radius,
        segments=mesh_segments,
        stacks=mesh_stacks,
    )
    return build_mesh_grid_sdf(
        mesh,
        spacing=mesh_sdf_spacing,
        padding=mesh_padding,
    ).geometry


def _build_mesh_sphere_sensitivity_case(
    setup: SphereImpactSetup,
    *,
    dt: float,
    scheme: str,
    config: NativeBandAccumulatorConfig,
    grid: UniformGrid3D,
    sphere_geometry_local: SignedDistanceGeometry,
    mesh_segments: int,
    mesh_stacks: int,
    mesh_sdf_spacing: float,
    static_overlap: float,
    axis: str,
    label: str,
    max_depth_a: float,
    max_depth_b: float,
    dt_ref: float,
    include_dynamic_metrics: bool,
) -> MeshSphereSensitivityCase:
    analytic_model = NativeBandSphereContactModel(
        setup,
        grid,
        config,
        max_depth_a=max_depth_a,
        max_depth_b=max_depth_b,
    )
    mesh_model = MeshNativeBandSphereContactModel(
        setup,
        grid,
        config,
        sphere_geometry_local=sphere_geometry_local,
        max_depth_a=max_depth_a,
        max_depth_b=max_depth_b,
    )
    analytic_static_force = analytic_model.evaluate(-static_overlap).force
    mesh_static_force = mesh_model.evaluate(-static_overlap).force
    static_force_relative_difference = relative_error(mesh_static_force, analytic_static_force)

    if not include_dynamic_metrics:
        return MeshSphereSensitivityCase(
            axis=axis,
            label=label,
            mesh_segments=mesh_segments,
            mesh_stacks=mesh_stacks,
            mesh_sdf_spacing=float(mesh_sdf_spacing),
            native_band_spacing=tuple(float(v) for v in grid.spacing),
            native_band_shape=grid.shape,
            analytic_static_force=analytic_static_force,
            mesh_static_force=mesh_static_force,
            static_force_relative_difference=static_force_relative_difference,
        )

    comparison = validate_mesh_native_band_sphere_dynamic_against_analytic(
        setup,
        dt=dt,
        scheme=scheme,
        analytic_model=analytic_model,
        mesh_model=mesh_model,
        dt_ref=dt_ref,
    )
    return MeshSphereSensitivityCase(
        axis=axis,
        label=label,
        mesh_segments=mesh_segments,
        mesh_stacks=mesh_stacks,
        mesh_sdf_spacing=float(mesh_sdf_spacing),
        native_band_spacing=tuple(float(v) for v in grid.spacing),
        native_band_shape=grid.shape,
        analytic_static_force=analytic_static_force,
        mesh_static_force=mesh_static_force,
        static_force_relative_difference=static_force_relative_difference,
        analytic_peak_force=float(np.max(comparison.analytic_history.forces)),
        mesh_peak_force=float(np.max(comparison.mesh_history.forces)),
        peak_force_relative_difference=comparison.peak_force_relative_difference,
        analytic_impulse=comparison.analytic_history.impulse,
        mesh_impulse=comparison.mesh_history.impulse,
        impulse_relative_difference=comparison.impulse_relative_difference,
        max_penetration_relative_difference=comparison.max_penetration_relative_difference,
        onset_time_difference=comparison.onset_time_difference,
    )


def validate_mesh_native_band_sphere_dynamic_against_analytic(
    setup: SphereImpactSetup,
    *,
    dt: float,
    scheme: str,
    analytic_model: NativeBandSphereContactModel,
    mesh_model: MeshNativeBandSphereContactModel,
    controller: EventAwareControllerConfig | None = None,
    dt_ref: float = 5e-4,
) -> MeshSphereDynamicValidationResult:
    """Run a short controlled sphere-impact comparison with only the sphere geometry source changed."""

    analytic_history = run_sphere_impact_benchmark(
        setup,
        dt=dt,
        scheme=scheme,
        model=analytic_model,
        controller=controller,
    )
    mesh_history = run_sphere_impact_benchmark(
        setup,
        dt=dt,
        scheme=scheme,
        model=mesh_model,
        controller=controller,
    )
    analytic_error = benchmark_sphere_impact_error(setup, analytic_history, dt_ref=dt_ref)
    mesh_error = benchmark_sphere_impact_error(setup, mesh_history, dt_ref=dt_ref)

    analytic_onset = analytic_history.onset_time if analytic_history.onset_time is not None else setup.t_final
    mesh_onset = mesh_history.onset_time if mesh_history.onset_time is not None else setup.t_final
    return MeshSphereDynamicValidationResult(
        analytic_history=analytic_history,
        mesh_history=mesh_history,
        analytic_error=analytic_error,
        mesh_error=mesh_error,
        peak_force_relative_difference=relative_error(
            float(np.max(mesh_history.forces)),
            float(np.max(analytic_history.forces)),
        ),
        impulse_relative_difference=relative_error(
            mesh_history.impulse,
            analytic_history.impulse,
        ),
        max_penetration_relative_difference=relative_error(
            float(-np.min(mesh_history.gaps)),
            float(-np.min(analytic_history.gaps)),
        ),
        onset_time_difference=timing_error(mesh_onset, analytic_onset),
    )


def validate_mesh_native_band_sphere_dynamic_from_asset(
    asset_path: str | Path,
    setup: SphereImpactSetup,
    *,
    grid: UniformGrid3D,
    config: NativeBandAccumulatorConfig,
    dt: float,
    scheme: str,
    padding: float | np.ndarray = 0.12,
    mesh_sdf_spacing: float | np.ndarray | None = None,
    recommended_spacing_scale: float = 0.9,
    max_depth_a: float = 2.0,
    max_depth_b: float = 2.0,
    controller: EventAwareControllerConfig | None = None,
    dt_ref: float = 5e-4,
) -> MeshSphereDynamicFactoryValidationResult:
    """Build a mesh-backed sphere through the asset factory and run a dynamic comparison.

    When ``mesh_sdf_spacing`` is omitted, the mesh factory uses the current
    recommended sphere-validation policy tied to ``grid.spacing``.
    """

    mesh_geometry = build_mesh_asset_sdf_geometry(
        asset_path,
        spacing=mesh_sdf_spacing,
        native_band_spacing=grid.spacing,
        recommended_spacing_scale=recommended_spacing_scale,
        padding=padding,
    )
    analytic_model = NativeBandSphereContactModel(
        setup,
        grid,
        config,
        max_depth_a=max_depth_a,
        max_depth_b=max_depth_b,
    )
    mesh_model = MeshNativeBandSphereContactModel(
        setup,
        grid,
        config,
        sphere_geometry_local=mesh_geometry.local_geometry,
        max_depth_a=max_depth_a,
        max_depth_b=max_depth_b,
    )
    comparison = validate_mesh_native_band_sphere_dynamic_against_analytic(
        setup,
        dt=dt,
        scheme=scheme,
        analytic_model=analytic_model,
        mesh_model=mesh_model,
        controller=controller,
        dt_ref=dt_ref,
    )
    return MeshSphereDynamicFactoryValidationResult(
        mesh_geometry=mesh_geometry,
        comparison=comparison,
        dt=float(dt),
        scheme=scheme,
    )


def run_mesh_backed_sphere_sensitivity_study(
    setup: SphereImpactSetup,
    *,
    dt: float,
    scheme: str,
    config: NativeBandAccumulatorConfig,
    static_overlap: float = 0.14,
    mesh_sdf_spacing_levels: tuple[float, ...] = (0.16, 0.12, 0.08),
    mesh_resolution_levels: tuple[tuple[int, int], ...] = ((12, 6), (16, 8), (20, 10)),
    native_band_spacing_levels: tuple[tuple[float, float, float], ...] = (
        (0.1, 0.1, 0.05),
        (0.09, 0.09, 0.04),
        (0.08, 0.08, 0.04),
    ),
    baseline_mesh_segments: int = 12,
    baseline_mesh_stacks: int = 6,
    baseline_mesh_sdf_spacing: float = 0.12,
    baseline_native_band_origin: np.ndarray = np.array([-1.1, -1.1, -0.14]),
    baseline_native_band_spacing: np.ndarray = np.array([0.1, 0.1, 0.05]),
    baseline_native_band_shape: tuple[int, int, int] = (23, 23, 24),
    mesh_padding: float | np.ndarray = 0.12,
    max_depth_a: float = 2.0,
    max_depth_b: float = 2.0,
    dt_ref: float = 2e-3,
    dynamic_axes: tuple[str, ...] = ("mesh_sdf_spacing",),
) -> MeshSphereSensitivityStudy:
    """Run a compact one-factor-at-a-time sensitivity study for the mesh-backed sphere path.

    The current implementation keeps the short dynamic comparison on the
    grid-SDF-spacing axis by default, then uses static force comparisons to
    isolate mesh-resolution and native-band-grid effects at much lower cost.
    """

    baseline_origin = np.asarray(baseline_native_band_origin, dtype=float)
    baseline_spacing = _as_vector3(baseline_native_band_spacing, name="baseline_native_band_spacing")
    baseline_grid = UniformGrid3D(
        origin=baseline_origin,
        spacing=baseline_spacing,
        shape=baseline_native_band_shape,
    )
    baseline_cell_center_minimum = baseline_grid.cell_center_origin
    baseline_cell_center_maximum = baseline_grid.cell_center_origin + baseline_grid.spacing * (np.asarray(baseline_grid.shape) - 1.0)
    mesh_padding_vec = _as_vector3(mesh_padding, name="mesh_padding")
    cases: list[MeshSphereSensitivityCase] = []

    geometry_cache: dict[tuple[int, int, float, tuple[float, float, float]], SignedDistanceGeometry] = {}

    def get_geometry(mesh_segments: int, mesh_stacks: int, mesh_sdf_spacing: float) -> SignedDistanceGeometry:
        key = (
            int(mesh_segments),
            int(mesh_stacks),
            float(mesh_sdf_spacing),
            tuple(float(v) for v in mesh_padding_vec),
        )
        geometry = geometry_cache.get(key)
        if geometry is None:
            geometry = _build_mesh_backed_sphere_local_geometry(
                radius=setup.sphere_radius,
                mesh_segments=mesh_segments,
                mesh_stacks=mesh_stacks,
                mesh_sdf_spacing=mesh_sdf_spacing,
                mesh_padding=mesh_padding_vec,
            )
            geometry_cache[key] = geometry
        return geometry

    for mesh_sdf_spacing in mesh_sdf_spacing_levels:
        cases.append(
            _build_mesh_sphere_sensitivity_case(
                setup,
                dt=dt,
                scheme=scheme,
                config=config,
                grid=baseline_grid,
                sphere_geometry_local=get_geometry(baseline_mesh_segments, baseline_mesh_stacks, float(mesh_sdf_spacing)),
                mesh_segments=baseline_mesh_segments,
                mesh_stacks=baseline_mesh_stacks,
                mesh_sdf_spacing=float(mesh_sdf_spacing),
                static_overlap=static_overlap,
                axis="mesh_sdf_spacing",
                label=f"mesh_sdf_spacing={mesh_sdf_spacing:.3f}",
                max_depth_a=max_depth_a,
                max_depth_b=max_depth_b,
                dt_ref=dt_ref,
                include_dynamic_metrics="mesh_sdf_spacing" in dynamic_axes,
            )
        )

    for mesh_segments, mesh_stacks in mesh_resolution_levels:
        cases.append(
            _build_mesh_sphere_sensitivity_case(
                setup,
                dt=dt,
                scheme=scheme,
                config=config,
                grid=baseline_grid,
                sphere_geometry_local=get_geometry(int(mesh_segments), int(mesh_stacks), baseline_mesh_sdf_spacing),
                mesh_segments=int(mesh_segments),
                mesh_stacks=int(mesh_stacks),
                mesh_sdf_spacing=baseline_mesh_sdf_spacing,
                static_overlap=static_overlap,
                axis="mesh_resolution",
                label=f"mesh_resolution={mesh_segments}x{mesh_stacks}",
                max_depth_a=max_depth_a,
                max_depth_b=max_depth_b,
                dt_ref=dt_ref,
                include_dynamic_metrics="mesh_resolution" in dynamic_axes,
            )
        )

    for native_band_spacing in native_band_spacing_levels:
        grid = _build_matching_native_band_grid(
            cell_center_minimum=baseline_cell_center_minimum,
            cell_center_maximum=baseline_cell_center_maximum,
            spacing=np.asarray(native_band_spacing, dtype=float),
        )
        cases.append(
            _build_mesh_sphere_sensitivity_case(
                setup,
                dt=dt,
                scheme=scheme,
                config=config,
                grid=grid,
                sphere_geometry_local=get_geometry(baseline_mesh_segments, baseline_mesh_stacks, baseline_mesh_sdf_spacing),
                mesh_segments=baseline_mesh_segments,
                mesh_stacks=baseline_mesh_stacks,
                mesh_sdf_spacing=baseline_mesh_sdf_spacing,
                static_overlap=static_overlap,
                axis="native_band_spacing",
                label=(
                    "native_band_spacing="
                    f"({native_band_spacing[0]:.3f},{native_band_spacing[1]:.3f},{native_band_spacing[2]:.3f})"
                ),
                max_depth_a=max_depth_a,
                max_depth_b=max_depth_b,
                dt_ref=dt_ref,
                include_dynamic_metrics="native_band_spacing" in dynamic_axes,
            )
        )

    return MeshSphereSensitivityStudy(
        static_overlap=float(static_overlap),
        axes_scanned=("mesh_sdf_spacing", "mesh_resolution", "native_band_spacing"),
        cases=tuple(cases),
    )


def run_sphere_impact_benchmark(
    setup: SphereImpactSetup,
    *,
    dt: float,
    scheme: str,
    model: Callable[[NormalDynamicsState], ModelEvaluation | tuple[float, float, bool]] | None = None,
    controller: EventAwareControllerConfig | None = None,
) -> DynamicHistory:
    if model is None:
        model = AnalyticLinearSphereContactModel(setup)

    initial_state = setup.initial_state()
    if hasattr(model, "reset_continuity"):
        model.reset_continuity()
    if scheme == "semi_implicit":
        stepper = semi_implicit_euler_step
        kwargs: dict = {}
    elif scheme == "midpoint":
        stepper = midpoint_contact_step
        kwargs = {}
    elif scheme == "midpoint_substep":
        stepper = midpoint_contact_substep
        kwargs = {"max_depth": 10}
    elif scheme == "event_aware_midpoint":
        stepper = event_aware_midpoint_step
        kwargs = {"controller": EventAwareControllerConfig(max_depth=10) if controller is None else controller}
    elif scheme == "event_aware_midpoint_impulse_corrected":
        stepper = event_aware_midpoint_impulse_corrected_step
        kwargs = {"controller": EventAwareControllerConfig(max_depth=10) if controller is None else controller}
    elif scheme == "event_aware_midpoint_work_consistent":
        stepper = event_aware_midpoint_work_consistent_step
        kwargs = {"controller": EventAwareControllerConfig(max_depth=10) if controller is None else controller}
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return simulate_history(
        initial_state,
        dt,
        setup.t_final,
        model,
        stepper,
        stepper_kwargs=kwargs,
        contact_potential_fn=setup.potential,
    )




def simulate_history(
    initial_state: NormalDynamicsState,
    dt: float,
    t_final: float,
    model: Callable[[NormalDynamicsState], ModelEvaluation | tuple[float, float, bool]],
    stepper: Callable[..., IntegrationStepResult],
    *,
    stepper_kwargs: dict | None = None,
    contact_stiffness_for_energy: float | None = None,
    contact_potential_fn: Callable[[float], float] | None = None,
) -> DynamicHistory:
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    stepper_kwargs = {} if stepper_kwargs is None else dict(stepper_kwargs)

    def eval_model(state: NormalDynamicsState) -> ModelEvaluation:
        raw = model(state)
        if isinstance(raw, ModelEvaluation):
            return raw
        force, acceleration, active = raw
        return ModelEvaluation(force=float(force), acceleration=float(acceleration), active=bool(active))

    times = [initial_state.time]
    gaps = [initial_state.gap]
    velocities = [initial_state.velocity]
    eval0 = eval_model(initial_state)
    forces = [eval0.force]
    active = [eval0.active]
    used_substeps = [1]
    active_measure = [np.nan if eval0.active_measure is None else eval0.active_measure]
    continuity_jaccard = [np.nan]
    predictor_force_jump = [np.nan]
    predictor_active_measure_jump = [np.nan]
    predictor_corrector_jaccard = [np.nan]
    predictor_corrector_mismatch_fraction = [np.nan]
    work_mismatch = [np.nan]
    candidate_count = [float(getattr(model, "last_candidate_count", np.nan))]
    recompute_count = [float(getattr(model, "last_recompute_count", np.nan))]
    continuity_state = ActiveSetContinuityState()
    if eval0.active_snapshot is not None:
        continuity_state, initial_report = continuity_report(continuity_state, eval0.active_snapshot)
        continuity_jaccard[0] = initial_report.jaccard_index
        if hasattr(model, "commit_active_snapshot"):
            model.commit_active_snapshot(eval0.active_snapshot)
    if contact_potential_fn is None:
        if contact_stiffness_for_energy is None:
            contact_stiffness_for_energy = 0.0
        contact_potential_fn = lambda gap: linear_contact_potential(contact_stiffness_for_energy, gap)
    total_energy = [kinetic_energy(initial_state.mass, initial_state.velocity) + contact_potential_fn(initial_state.gap)]
    onset_estimates = [np.nan]
    release_estimates = [np.nan]

    state = initial_state
    n_steps = int(np.ceil((t_final - initial_state.time) / dt))
    for _ in range(n_steps):
        if state.time >= t_final - 1e-15:
            break
        this_dt = min(dt, t_final - state.time)
        result = stepper(state, this_dt, model, **stepper_kwargs)
        state = result.state
        times.append(state.time)
        gaps.append(state.gap)
        velocities.append(state.velocity)
        forces.append(result.diagnostics.force)
        active.append(result.diagnostics.active)
        used_substeps.append(result.diagnostics.used_substeps)
        total_energy.append(kinetic_energy(state.mass, state.velocity) + contact_potential_fn(state.gap))
        onset_estimates.append(np.nan if result.diagnostics.onset_time_estimate is None else result.diagnostics.onset_time_estimate)
        release_estimates.append(np.nan if result.diagnostics.release_time_estimate is None else result.diagnostics.release_time_estimate)
        active_measure.append(np.nan if result.diagnostics.active_measure is None else result.diagnostics.active_measure)
        predictor_force_jump.append(np.nan if result.diagnostics.predictor_force_jump is None else result.diagnostics.predictor_force_jump)
        predictor_active_measure_jump.append(
            np.nan if result.diagnostics.predictor_active_measure_jump is None else result.diagnostics.predictor_active_measure_jump
        )
        predictor_corrector_jaccard.append(
            np.nan if result.diagnostics.predictor_corrector_jaccard is None else result.diagnostics.predictor_corrector_jaccard
        )
        predictor_corrector_mismatch_fraction.append(
            np.nan
            if result.diagnostics.predictor_corrector_mismatch_fraction is None
            else result.diagnostics.predictor_corrector_mismatch_fraction
        )
        work_mismatch.append(np.nan if result.diagnostics.work_mismatch is None else result.diagnostics.work_mismatch)

        eval_end = eval_model(state)
        candidate_count.append(float(getattr(model, "last_candidate_count", np.nan)))
        recompute_count.append(float(getattr(model, "last_recompute_count", np.nan)))
        if eval_end.active_snapshot is None:
            continuity_jaccard.append(np.nan)
        else:
            continuity_state, report = continuity_report(continuity_state, eval_end.active_snapshot)
            continuity_jaccard.append(report.jaccard_index)
            if hasattr(model, "commit_active_snapshot"):
                model.commit_active_snapshot(eval_end.active_snapshot)

    return DynamicHistory(
        times=np.asarray(times, dtype=float),
        gaps=np.asarray(gaps, dtype=float),
        velocities=np.asarray(velocities, dtype=float),
        forces=np.asarray(forces, dtype=float),
        active=np.asarray(active, dtype=bool),
        used_substeps=np.asarray(used_substeps, dtype=int),
        total_energy=np.asarray(total_energy, dtype=float),
        onset_estimates=np.asarray(onset_estimates, dtype=float),
        release_estimates=np.asarray(release_estimates, dtype=float),
        active_measure=np.asarray(active_measure, dtype=float),
        continuity_jaccard=np.asarray(continuity_jaccard, dtype=float),
        predictor_force_jump=np.asarray(predictor_force_jump, dtype=float),
        predictor_active_measure_jump=np.asarray(predictor_active_measure_jump, dtype=float),
        predictor_corrector_jaccard=np.asarray(predictor_corrector_jaccard, dtype=float),
        predictor_corrector_mismatch_fraction=np.asarray(predictor_corrector_mismatch_fraction, dtype=float),
        work_mismatch=np.asarray(work_mismatch, dtype=float),
        candidate_count=np.asarray(candidate_count, dtype=float),
        recompute_count=np.asarray(recompute_count, dtype=float),
    )



def benchmark_flat_impact_error(
    setup: FlatImpactSetup,
    history: DynamicHistory,
) -> DynamicErrorSummary:
    gap_ref, vel_ref, force_ref = exact_flat_impact_state(setup, history.times[-1])
    state_error = float(np.hypot(history.gaps[-1] - gap_ref, history.velocities[-1] - vel_ref))
    force_error = relative_error(history.forces[-1], force_ref)

    exact_times = history.times
    exact_states = np.array([exact_flat_impact_state(setup, t) for t in exact_times], dtype=float)
    exact_forces = exact_states[:, 2]
    exact_gaps = exact_states[:, 0]
    exact_velocities = exact_states[:, 1]
    impulse_ref = integrate_impulse(exact_times, exact_forces)
    impulse_error = relative_error(history.impulse, impulse_ref)

    onset_numeric = history.onset_time if history.onset_time is not None else history.times[-1]
    onset_error = timing_error(onset_numeric, setup.onset_time_exact)

    release_numeric = history.release_time if history.release_time is not None else history.times[-1]
    release_error = timing_error(release_numeric, setup.release_time_exact)

    peak_force_error = relative_error(float(np.max(history.forces)), float(np.max(exact_forces)))
    max_penetration_error = relative_error(float(-np.min(history.gaps)), float(-np.min(exact_gaps)))
    rebound_velocity_error = relative_error(float(history.velocities[-1]), float(exact_velocities[-1]))

    return DynamicErrorSummary(
        state_error=state_error,
        force_error=force_error,
        impulse_error=impulse_error,
        onset_timing_error=onset_error,
        release_timing_error=release_error,
        peak_force_error=peak_force_error,
        max_penetration_error=max_penetration_error,
        rebound_velocity_error=rebound_velocity_error,
    )



def run_flat_impact_benchmark(
    setup: FlatImpactSetup,
    *,
    dt: float,
    scheme: str,
    model: Callable[[NormalDynamicsState], ModelEvaluation | tuple[float, float, bool]] | None = None,
    controller: EventAwareControllerConfig | None = None,
) -> DynamicHistory:
    if model is None:
        model = AnalyticLinearFlatContactModel(setup.mass, setup.contact_stiffness)

    initial_state = setup.initial_state()
    if hasattr(model, "reset_continuity"):
        model.reset_continuity()
    if scheme == "semi_implicit":
        stepper = semi_implicit_euler_step
        kwargs: dict = {}
    elif scheme == "midpoint":
        stepper = midpoint_contact_step
        kwargs = {}
    elif scheme == "midpoint_substep":
        stepper = midpoint_contact_substep
        kwargs = {"max_depth": 10}
    elif scheme == "event_aware_midpoint":
        stepper = event_aware_midpoint_step
        kwargs = {"controller": EventAwareControllerConfig(max_depth=10) if controller is None else controller}
    elif scheme == "event_aware_midpoint_impulse_corrected":
        stepper = event_aware_midpoint_impulse_corrected_step
        kwargs = {"controller": EventAwareControllerConfig(max_depth=10) if controller is None else controller}
    elif scheme == "event_aware_midpoint_work_consistent":
        stepper = event_aware_midpoint_work_consistent_step
        kwargs = {"controller": EventAwareControllerConfig(max_depth=10) if controller is None else controller}
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return simulate_history(
        initial_state,
        dt,
        setup.t_final,
        model,
        stepper,
        stepper_kwargs=kwargs,
        contact_stiffness_for_energy=setup.contact_stiffness,
    )
