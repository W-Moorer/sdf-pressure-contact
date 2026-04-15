#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo for the unified contact pipeline.

Runs the four-stage interface on the three requested shallow-indentation examples:
    - small cube into large cube
    - annular flat punch into large cube
    - horizontal cylinder into large cube

and compares:
    - direct_band_baseline()
    - run_contact_pipeline()
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from contact_pipeline_unified import (
    DomainSpec,
    PatchBuildConfig,
    SheetExtractConfig,
    HydrostaticPressureModel,
    SmallBoxPunch,
    AnnularFlatPunch,
    HorizontalCylinder,
    run_contact_pipeline,
    direct_band_baseline,
)

OUT_DIR = Path(__file__).resolve().parent

def main() -> None:
    domain = DomainSpec(cube_size=1.8, cube_height=0.25)
    pressure = HydrostaticPressureModel(k=10.0, top_y=0.0)

    shapes = [
        ("small_cube", SmallBoxPunch(lx=0.45, lz=0.45, rigid_height=0.40)),
        ("annular_punch", AnnularFlatPunch(ri=0.12, ro=0.30, rigid_height=0.40)),
        ("horizontal_cylinder", HorizontalCylinder(radius=0.28, length=0.80)),
    ]
    deltas = [0.01, 0.02, 0.03, 0.05, 0.08]

    rows = []

    for name, shape in shapes:
        quad_order = 2 if name == "small_cube" else 3
        patch_cfg = PatchBuildConfig(Nxz=48, quad_order=quad_order)
        sheet_cfg = SheetExtractConfig(top_y=domain.top_y)

        for delta in deltas:
            exact_force = shape.exact_force(pressure.k, delta)

            baseline = direct_band_baseline(
                shape=shape,
                delta=delta,
                domain=domain,
                pressure_model=pressure,
                N=64,
            )

            result = run_contact_pipeline(
                shape=shape,
                delta=delta,
                domain=domain,
                pressure_model=pressure,
                patch_cfg=patch_cfg,
                sheet_cfg=sheet_cfg,
            )

            wrench = result["wrench"]

            rows.append(
                {
                    "case": name,
                    "method": "direct_band_baseline",
                    "delta": delta,
                    "Fy_num": baseline.force[1],
                    "Fy_exact": exact_force,
                    "rel_err_Fy": abs(baseline.force[1] - exact_force) / abs(exact_force),
                    "num_projected_points": None,
                    "num_sheet_points": None,
                }
            )
            rows.append(
                {
                    "case": name,
                    "method": "pipeline",
                    "delta": delta,
                    "Fy_num": wrench.force[1],
                    "Fy_exact": exact_force,
                    "rel_err_Fy": abs(wrench.force[1] - exact_force) / abs(exact_force),
                    "num_projected_points": result["patches"].metadata["num_projected_points"],
                    "num_sheet_points": result["sheet"].metadata["num_sheet_points"],
                }
            )

        # quick geometry demo at delta = 0.03
        demo = run_contact_pipeline(
            shape=shape,
            delta=0.03,
            domain=domain,
            pressure_model=pressure,
            patch_cfg=patch_cfg,
            sheet_cfg=sheet_cfg,
        )

        patches = demo["patches"].samples
        plt.figure(figsize=(4.4, 4.0))
        if patches:
            xs = [p.x for p in patches]
            zs = [p.z for p in patches]
            plt.scatter(xs, zs, s=6)
        plt.gca().set_aspect("equal")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title(f"{name}: projected patches")
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"unified_pipeline_{name}_patches.png", dpi=180)
        plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "unified_contact_pipeline_demo_results.csv", index=False)

    for name, _ in shapes:
        sub = df[df["case"] == name].copy()
        plt.figure(figsize=(6.8, 4.4))
        for method, grp in sub.groupby("method"):
            grp = grp.sort_values("delta")
            plt.plot(grp["delta"], grp["Fy_num"], marker="o", label=method)
        exact = sub.drop_duplicates("delta").sort_values("delta")
        plt.plot(exact["delta"], exact["Fy_exact"], marker="x", label="exact")
        plt.xlabel("penetration depth δ")
        plt.ylabel("vertical force Fy")
        plt.title(f"{name}: unified pipeline demo")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"unified_pipeline_{name}_force.png", dpi=180)
        plt.close()

    print("Saved:")
    print(OUT_DIR / "unified_contact_pipeline_demo_results.csv")
    for name, _ in shapes:
        print(OUT_DIR / f"unified_pipeline_{name}_patches.png")
        print(OUT_DIR / f"unified_pipeline_{name}_force.png")

if __name__ == "__main__":
    main()
