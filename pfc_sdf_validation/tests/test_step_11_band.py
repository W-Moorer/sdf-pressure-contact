from __future__ import annotations

import numpy as np

from pfcsdf.contact.band import BandDiscretization, accumulate_flat_band_wrench, coarea_weight_1d
from pfcsdf.contact.sheet import build_flat_sheet_patch
from pfcsdf.contact.wrench import accumulate_uniform_pressure_wrench
from pfcsdf.geometry.primitives import BoxFootprint



def test_hat_delta_coarea_weight_is_unity() -> None:
    discretization = BandDiscretization(eta=0.05, band_half_width=0.2, n_samples=4001)
    weight = coarea_weight_1d(discretization)
    assert np.isclose(weight, 1.0, atol=5e-4)



def test_flat_band_wrench_matches_direct_sheet_wrench() -> None:
    footprint = BoxFootprint(lx=2.0, ly=4.0)
    center = np.array([1.5, -0.5, 0.25])
    reference = np.array([-0.25, 0.5, -1.0])
    patch = build_flat_sheet_patch(footprint, center=center, normal=np.array([0.0, 0.0, 1.0]))
    discretization = BandDiscretization(eta=0.05, band_half_width=0.2, n_samples=4001)
    pressure = 7.5

    band_wrench = accumulate_flat_band_wrench(
        pressure=pressure,
        patch=patch,
        discretization=discretization,
        reference=reference,
    )
    direct_wrench = accumulate_uniform_pressure_wrench(
        pressure=pressure,
        area=patch.area,
        normal=patch.normal,
        center=patch.center,
        reference=reference,
    )

    assert np.allclose(band_wrench.force, direct_wrench.force, atol=5e-3)
    assert np.allclose(band_wrench.torque, direct_wrench.torque, atol=5e-3)
