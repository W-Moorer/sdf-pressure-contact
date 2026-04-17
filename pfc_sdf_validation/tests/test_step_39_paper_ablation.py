from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pfcsdf.experiments.ablation import generate_ablation_table


REQUIRED_SCHEMES = {
    "event_aware_midpoint",
    "event_aware_midpoint_impulse_corrected",
    "event_aware_midpoint_work_consistent",
    "event_aware_midpoint_work_consistent_consistent_traction",
}


@pytest.fixture(scope="module")
def ablation_df() -> pd.DataFrame:
    return generate_ablation_table()


def test_ablation_table_contains_required_benchmarks_schemes_and_long_horizon_columns(ablation_df: pd.DataFrame) -> None:
    df = ablation_df

    assert set(df["benchmark"]) == {"analytic_flat", "analytic_sphere", "native_band_flat"}
    for benchmark in ["analytic_flat", "analytic_sphere", "native_band_flat"]:
        subset = df[df["benchmark"] == benchmark]
        assert set(subset["scheme"]) == REQUIRED_SCHEMES
        assert (subset["t_final"] > subset["release_time_exact"]).all()
        assert (subset["horizon_over_release"] > 1.0).all()

    required_columns = {
        "benchmark",
        "scheme",
        "scheme_display",
        "dt",
        "t_final",
        "release_time_exact",
        "horizon_over_release",
        "force_error",
        "peak_force_error",
        "impulse_error",
        "energy_drift",
        "state_error",
        "onset_timing_error",
        "release_timing_error",
        "max_penetration_error",
        "rebound_velocity_error",
        "mean_candidate_count",
        "mean_recompute_count",
        "max_used_substeps",
    }
    assert required_columns.issubset(df.columns)


def test_long_horizon_ablation_recovers_release_energy_and_cross_geometry_story(ablation_df: pd.DataFrame) -> None:
    df = ablation_df

    analytic = df[df["benchmark"] == "analytic_flat"].set_index("scheme")
    sphere = df[df["benchmark"] == "analytic_sphere"].set_index("scheme")
    native = df[df["benchmark"] == "native_band_flat"].set_index("scheme")

    # 长时程解析 flat：更重要的是 release/peak/energy，不再只盯住短时程 impulse。
    assert analytic.loc["event_aware_midpoint_impulse_corrected", "peak_force_error"] < analytic.loc["event_aware_midpoint", "peak_force_error"]
    assert abs(analytic.loc["event_aware_midpoint_work_consistent", "energy_drift"]) < abs(analytic.loc["event_aware_midpoint", "energy_drift"])
    assert analytic.loc["event_aware_midpoint_work_consistent", "release_timing_error"] < analytic.loc["event_aware_midpoint", "release_timing_error"]
    assert analytic.loc["event_aware_midpoint_work_consistent_consistent_traction", "force_error"] == analytic.loc[
        "event_aware_midpoint_work_consistent", "force_error"
    ]

    # 长时程 analytic_sphere：作为曲率主导对照组，work-consistent 应在 peak/release/energy 上更稳。
    assert sphere.loc["event_aware_midpoint_work_consistent", "peak_force_error"] < sphere.loc["event_aware_midpoint", "peak_force_error"]
    assert abs(sphere.loc["event_aware_midpoint_work_consistent", "energy_drift"]) < abs(sphere.loc["event_aware_midpoint", "energy_drift"])
    assert sphere.loc["event_aware_midpoint_work_consistent", "release_timing_error"] < sphere.loc["event_aware_midpoint", "release_timing_error"]
    assert sphere.loc["event_aware_midpoint_work_consistent_consistent_traction", "force_error"] == sphere.loc[
        "event_aware_midpoint_work_consistent", "force_error"
    ]

    # 长时程 native-band：release/energy 仍由 work-consistent 主导；consistent traction 不应引入额外 traversal 成本。
    assert native.loc["event_aware_midpoint_impulse_corrected", "peak_force_error"] < native.loc["event_aware_midpoint", "peak_force_error"]
    assert abs(native.loc["event_aware_midpoint_work_consistent", "energy_drift"]) <= abs(native.loc["event_aware_midpoint", "energy_drift"])
    assert native.loc["event_aware_midpoint_work_consistent", "release_timing_error"] < native.loc["event_aware_midpoint", "release_timing_error"]
    assert native.loc["event_aware_midpoint_work_consistent_consistent_traction", "mean_candidate_count"] == native.loc[
        "event_aware_midpoint_work_consistent", "mean_candidate_count"
    ]
    assert np.isfinite(native["mean_predictor_corrector_jaccard"].to_numpy()).all()
