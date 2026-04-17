from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from pfcsdf.contact.native_band import NativeBandAccumulatorConfig
from pfcsdf.dynamics.benchmarks import (
    FlatImpactSetup,
    NativeBandFlatContactModel,
    SphereImpactSetup,
    benchmark_flat_impact_error,
    benchmark_sphere_impact_error,
    run_flat_impact_benchmark,
    run_sphere_impact_benchmark,
    sphere_reference_history,
)
from pfcsdf.dynamics.events import EventAwareControllerConfig
from pfcsdf.geometry.primitives import BoxFootprint
from pfcsdf.geometry.volume import UniformGrid3D
from pfcsdf.physics.pressure import LinearPressureLaw


SCHEME_DISPLAY_NAMES = {
    "event_aware_midpoint": "event-aware midpoint",
    "event_aware_midpoint_impulse_corrected": "+ impulse correction",
    "event_aware_midpoint_work_consistent": "+ work consistency",
    "event_aware_midpoint_work_consistent_consistent_traction": "+ consistent traction reconstruction",
}


@dataclass(frozen=True)
class AblationCaseConfig:
    benchmark: str
    scheme: str
    dt: float
    setup: object
    model_factory: Callable[[], object] | None = None
    controller: EventAwareControllerConfig | None = None
    notes: str = ""


@dataclass(frozen=True)
class AblationRow:
    benchmark: str
    scheme: str
    scheme_display: str
    dt: float
    t_final: float
    release_time_exact: float
    horizon_over_release: float
    force_error: float
    impulse_error: float
    energy_drift: float
    state_error: float
    onset_timing_error: float
    release_timing_error: float
    peak_force_error: float
    max_penetration_error: float
    rebound_velocity_error: float
    mean_candidate_count: float
    mean_recompute_count: float
    max_used_substeps: int
    mean_predictor_corrector_jaccard: float
    mean_predictor_corrector_mismatch_fraction: float
    mean_work_mismatch: float
    notes: str = ""


def _safe_nanmean(values: np.ndarray | None) -> float:
    if values is None:
        return float("nan")
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))



def _build_native_model(*, consistent_traction: bool) -> NativeBandFlatContactModel:
    return NativeBandFlatContactModel(
        mass=1.0,
        grid=UniformGrid3D(origin=np.array([-0.4, -0.4, -0.1]), spacing=np.array([0.2, 0.2, 0.01]), shape=(4, 4, 20)),
        footprint=BoxFootprint(0.8, 0.8),
        law_a=LinearPressureLaw(200.0),
        law_b=LinearPressureLaw(200.0),
        config=NativeBandAccumulatorConfig(eta=8.0, band_half_width=12.0),
        max_depth_a=0.2,
        max_depth_b=0.2,
        use_continuity_warm_start=True,
        boundary_only_update=True,
        continuity_dilation_radius=1,
        consistent_traction_reconstruction=consistent_traction,
    )



def build_default_ablation_case_configs() -> list[AblationCaseConfig]:
    # 默认 ablation 不再只覆盖 onset 附近的短时段，而是至少跨越：
    # onset -> peak-compression -> release -> 一段 release 后自由段。
    analytic_setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=100.0,
        t_final=0.60,
    )
    analytic_dt = 0.04

    native_setup = FlatImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        contact_stiffness=64.0,
        t_final=0.55,
    )
    native_dt = 0.05
    native_controller = EventAwareControllerConfig(max_depth=2, work_mismatch_relative_tol=0.05)

    sphere_setup = SphereImpactSetup(
        initial_gap=0.05,
        initial_velocity=-1.0,
        mass=1.0,
        sphere_radius=1.0,
        sphere_stiffness=12.0,
        plane_stiffness=18.0,
        t_final=1.50,
    )
    sphere_dt = 0.05
    sphere_controller = EventAwareControllerConfig(max_depth=4, work_mismatch_relative_tol=0.02)

    cases: list[AblationCaseConfig] = []
    for scheme in [
        "event_aware_midpoint",
        "event_aware_midpoint_impulse_corrected",
        "event_aware_midpoint_work_consistent",
    ]:
        cases.append(
            AblationCaseConfig(
                benchmark="analytic_flat",
                scheme=scheme,
                dt=analytic_dt,
                setup=analytic_setup,
                model_factory=None,
            )
        )
    cases.append(
        AblationCaseConfig(
            benchmark="analytic_flat",
            scheme="event_aware_midpoint_work_consistent_consistent_traction",
            dt=analytic_dt,
            setup=analytic_setup,
            model_factory=None,
            notes="解析接触模型不存在 traction reconstruction，自然与上一行相同。",
        )
    )

    for scheme in [
        "event_aware_midpoint",
        "event_aware_midpoint_impulse_corrected",
        "event_aware_midpoint_work_consistent",
    ]:
        cases.append(
            AblationCaseConfig(
                benchmark="analytic_sphere",
                scheme=scheme,
                dt=sphere_dt,
                setup=sphere_setup,
                model_factory=None,
                controller=sphere_controller,
            )
        )
    cases.append(
        AblationCaseConfig(
            benchmark="analytic_sphere",
            scheme="event_aware_midpoint_work_consistent_consistent_traction",
            dt=sphere_dt,
            setup=sphere_setup,
            model_factory=None,
            controller=sphere_controller,
            notes="解析球-平面基准不存在 traction reconstruction，自然与上一行相同。",
        )
    )

    cases.append(
        AblationCaseConfig(
            benchmark="native_band_flat",
            scheme="event_aware_midpoint",
            dt=native_dt,
            setup=native_setup,
            model_factory=lambda: _build_native_model(consistent_traction=False),
            controller=native_controller,
        )
    )
    cases.append(
        AblationCaseConfig(
            benchmark="native_band_flat",
            scheme="event_aware_midpoint_impulse_corrected",
            dt=native_dt,
            setup=native_setup,
            model_factory=lambda: _build_native_model(consistent_traction=False),
            controller=native_controller,
        )
    )
    cases.append(
        AblationCaseConfig(
            benchmark="native_band_flat",
            scheme="event_aware_midpoint_work_consistent",
            dt=native_dt,
            setup=native_setup,
            model_factory=lambda: _build_native_model(consistent_traction=False),
            controller=native_controller,
        )
    )
    cases.append(
        AblationCaseConfig(
            benchmark="native_band_flat",
            scheme="event_aware_midpoint_work_consistent_consistent_traction",
            dt=native_dt,
            setup=native_setup,
            model_factory=lambda: _build_native_model(consistent_traction=True),
            controller=native_controller,
        )
    )
    return cases



def _normalize_scheme_for_runner(scheme: str) -> str:
    if scheme == "event_aware_midpoint_work_consistent_consistent_traction":
        return "event_aware_midpoint_work_consistent"
    return scheme



def generate_ablation_table(
    cases: Iterable[AblationCaseConfig] | None = None,
    *,
    csv_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> pd.DataFrame:
    if cases is None:
        cases = build_default_ablation_case_configs()

    rows: list[AblationRow] = []
    sphere_release_cache: dict[tuple[float, float, float, float, float, float], float] = {}
    previous_native_work_row: AblationRow | None = None
    for case in cases:
        if case.benchmark == "native_band_flat" and case.scheme == "event_aware_midpoint_work_consistent_consistent_traction" and previous_native_work_row is not None:
            rows.append(AblationRow(**{**asdict(previous_native_work_row), "scheme": case.scheme, "scheme_display": SCHEME_DISPLAY_NAMES[case.scheme], "notes": "长时程默认表复用 work-consistent 结果；consistent traction 的收益由 targeted benchmark 单独评估。"}))
            continue
        model = None if case.model_factory is None else case.model_factory()
        if case.benchmark in {"analytic_flat", "native_band_flat"}:
            history = run_flat_impact_benchmark(
                case.setup,
                dt=case.dt,
                scheme=_normalize_scheme_for_runner(case.scheme),
                model=model,
                controller=case.controller,
            )
            err = benchmark_flat_impact_error(case.setup, history)
            release_reference = case.setup.release_time_exact
        elif case.benchmark == "analytic_sphere":
            history = run_sphere_impact_benchmark(
                case.setup,
                dt=case.dt,
                scheme=_normalize_scheme_for_runner(case.scheme),
                model=model,
                controller=case.controller,
            )
            err = benchmark_sphere_impact_error(case.setup, history)
            key = (case.setup.initial_gap, case.setup.initial_velocity, case.setup.mass, case.setup.sphere_radius, case.setup.sphere_stiffness, case.setup.plane_stiffness)
            if key not in sphere_release_cache:
                ref = sphere_reference_history(case.setup)
                sphere_release_cache[key] = ref.release_time if ref.release_time is not None else case.setup.t_final
            release_reference = sphere_release_cache[key]
        else:
            raise ValueError(f"unknown benchmark: {case.benchmark}")
        candidate_mean = _safe_nanmean(history.candidate_count)
        recompute_mean = _safe_nanmean(history.recompute_count)
        row = AblationRow(
            benchmark=case.benchmark,
            scheme=case.scheme,
            scheme_display=SCHEME_DISPLAY_NAMES[case.scheme],
            dt=case.dt,
            t_final=case.setup.t_final,
            release_time_exact=release_reference,
            horizon_over_release=case.setup.t_final / release_reference,
            force_error=err.force_error,
            impulse_error=err.impulse_error,
            energy_drift=history.energy_drift,
            state_error=err.state_error,
            onset_timing_error=err.onset_timing_error,
            release_timing_error=err.release_timing_error,
            peak_force_error=err.peak_force_error,
            max_penetration_error=err.max_penetration_error,
            rebound_velocity_error=err.rebound_velocity_error,
            mean_candidate_count=candidate_mean,
            mean_recompute_count=recompute_mean,
            max_used_substeps=int(np.max(history.used_substeps)),
            mean_predictor_corrector_jaccard=_safe_nanmean(history.predictor_corrector_jaccard),
            mean_predictor_corrector_mismatch_fraction=_safe_nanmean(history.predictor_corrector_mismatch_fraction),
            mean_work_mismatch=_safe_nanmean(history.work_mismatch),
            notes=case.notes,
        )
        rows.append(row)
        if case.benchmark == "native_band_flat" and case.scheme == "event_aware_midpoint_work_consistent":
            previous_native_work_row = row

    df = pd.DataFrame([asdict(r) for r in rows])
    scheme_order = {scheme: i for i, scheme in enumerate(SCHEME_DISPLAY_NAMES)}
    bench_order = {"analytic_flat": 0, "analytic_sphere": 1, "native_band_flat": 2}
    df = df.sort_values(
        by=["benchmark", "scheme"],
        key=lambda s: s.map(bench_order if s.name == "benchmark" else scheme_order),
        ignore_index=True,
    )

    if csv_path is not None:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
    if markdown_path is not None:
        Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
        Path(markdown_path).write_text(render_ablation_markdown(df), encoding="utf-8")
    return df



def render_ablation_markdown(df: pd.DataFrame) -> str:
    lines: list[str] = []
    for benchmark, group in df.groupby("benchmark", sort=False):
        lines.append(f"## {benchmark}")
        headers = [
            "scheme_display",
            "force_error",
            "peak_force_error",
            "impulse_error",
            "energy_drift",
            "state_error",
            "release_timing_error",
            "max_penetration_error",
            "rebound_velocity_error",
            "mean_candidate_count",
            "mean_recompute_count",
            "max_used_substeps",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for _, row in group.iterrows():
            vals = []
            for h in headers:
                val = row[h]
                if isinstance(val, (float, np.floating)):
                    vals.append(f"{float(val):.6g}" if np.isfinite(val) else "nan")
                else:
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    csv_path = root / "paper_ablation_table.csv"
    md_path = root / "paper_ablation_table.md"
    df = generate_ablation_table(csv_path=csv_path, markdown_path=md_path)
    print(df.to_string(index=False))
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")
