"""PFC-SDF validation scaffold."""

from .physics.pressure import LinearPressureLaw
from .contact.local_normal import solve_column_equilibrium
from .solvers.static import (
    compute_sphere_plane_contact_linear_exact,
    compute_sphere_plane_contact_linear_quadrature,
    compute_uniform_flat_contact,
)
from .dynamics.benchmarks import (
    AnalyticLinearFlatContactModel,
    FlatImpactSetup,
    NativeBandFlatContactModel,
    benchmark_flat_impact_error,
    exact_flat_impact_state,
    run_flat_impact_benchmark,
)

__all__ = [
    "LinearPressureLaw",
    "solve_column_equilibrium",
    "compute_uniform_flat_contact",
    "compute_sphere_plane_contact_linear_exact",
    "compute_sphere_plane_contact_linear_quadrature",
    "FlatImpactSetup",
    "AnalyticLinearFlatContactModel",
    "NativeBandFlatContactModel",
    "exact_flat_impact_state",
    "run_flat_impact_benchmark",
    "benchmark_flat_impact_error",
]
