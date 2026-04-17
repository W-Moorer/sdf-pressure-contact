from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pfcsdf.contact.sheet import FlatSheetPatch
from pfcsdf.contact.wrench import PairWrench


@dataclass(frozen=True)
class BandDiscretization:
    eta: float
    band_half_width: float
    n_samples: int



def hat_delta(h: float | np.ndarray, eta: float) -> float | np.ndarray:
    """Compactly supported delta approximation with unit integral on [-eta, eta]."""
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    h_arr = np.asarray(h, dtype=float)
    out = np.maximum(0.0, 1.0 - np.abs(h_arr) / eta) / eta
    return out.item() if np.isscalar(h) else out



def sample_band_line(discretization: BandDiscretization) -> np.ndarray:
    if discretization.band_half_width <= 0.0:
        raise ValueError("band_half_width must be positive")
    if discretization.band_half_width < discretization.eta:
        raise ValueError("band_half_width must be at least eta")
    if discretization.n_samples < 3:
        raise ValueError("n_samples must be at least 3")
    return np.linspace(
        -discretization.band_half_width,
        discretization.band_half_width,
        discretization.n_samples,
    )



def coarea_weight_1d(discretization: BandDiscretization) -> float:
    """Approximate ∫ δη(h(z)) ||∇h|| dz for the flat case h(z)=z, ||∇h||=1."""
    z = sample_band_line(discretization)
    kernel = hat_delta(z, discretization.eta)
    return float(np.trapezoid(kernel, z))



def accumulate_flat_band_wrench(
    pressure: float,
    patch: FlatSheetPatch,
    discretization: BandDiscretization,
    *,
    reference: np.ndarray | None = None,
) -> PairWrench:
    """Band-mechanics counterpart of a uniform flat sheet traction.

    The 1D coarea factor supplies the mechanical measure. The sheet patch supplies
    the geometric semantics (area, center, normal).
    """
    if reference is None:
        reference = np.zeros(3)
    reference = np.asarray(reference, dtype=float)

    weight = coarea_weight_1d(discretization)
    force = pressure * patch.area * weight * patch.normal
    torque = np.cross(patch.center - reference, force)
    return PairWrench(force=force, torque=torque)
