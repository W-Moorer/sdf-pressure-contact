#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo scene:
    a small sphere falls from the air onto a fixed large cube

Requirements from the user:
    - run for a few seconds
    - at the end of the simulation the bodies must be in contact

This demo uses the unified implicit-contact framework.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from implicit_contact_framework import (
    Marker,
    Pose,
    RigidBody,
    SphereGeometry,
    DomainSpec,
    World,
    PatchBuildConfig,
    SheetExtractConfig,
    ContactModelConfig,
    IntegratorConfig,
    ImplicitEulerIntegrator,
    Simulator,
)

OUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    # -----------------------------
    # Scene setup
    # -----------------------------
    domain = DomainSpec(cube_size=1.6, cube_height=0.35, top_y=0.0)

    sphere = RigidBody(
        name="ball",
        mass=0.05,
        geometry=SphereGeometry(radius=0.18),
        pose=Pose(position=np.array([0.0, 0.75, 0.0], dtype=float)),
        linear_velocity=np.zeros(3, dtype=float),
        markers=[
            Marker("center", np.array([0.0, 0.0, 0.0], dtype=float)),
            Marker("bottom", np.array([0.0, -0.18, 0.0], dtype=float)),
            Marker("top", np.array([0.0, 0.18, 0.0], dtype=float)),
        ],
        linear_damping=0.05,
    )

    world = World(
        domain=domain,
        gravity=np.array([0.0, -9.81, 0.0], dtype=float),
        bodies=[sphere],
    )

    patch_cfg = PatchBuildConfig(Nxz=28, quad_order=3, bbox_padding_cells=1)
    sheet_cfg = SheetExtractConfig(bisection_steps=24, normal_step=1.0e-6)
    contact_cfg = ContactModelConfig(stiffness_k=10000.0, damping_c=120.0, top_y=domain.top_y)
    integ_cfg = IntegratorConfig(dt=0.01, newton_max_iter=8, newton_tol=1.0e-8, fd_eps=1.0e-5)

    integrator = ImplicitEulerIntegrator(
        patch_cfg=patch_cfg,
        sheet_cfg=sheet_cfg,
        contact_cfg=contact_cfg,
        cfg=integ_cfg,
    )

    sim = Simulator(world, integrator)

    # Run long enough that the sphere lands and remains in contact
    total_time = 2.0
    log = sim.run(total_time)

    # -----------------------------
    # Convert log to table
    # -----------------------------
    rows = []
    for entry in log:
        rows.append(
            {
                "time": entry.time,
                "x": entry.position[0],
                "y": entry.position[1],
                "z": entry.position[2],
                "vx": entry.linear_velocity[0],
                "vy": entry.linear_velocity[1],
                "vz": entry.linear_velocity[2],
                "Fx_contact": entry.contact_force[0],
                "Fy_contact": entry.contact_force[1],
                "Fz_contact": entry.contact_force[2],
                "Mx_contact": entry.contact_moment[0],
                "My_contact": entry.contact_moment[1],
                "Mz_contact": entry.contact_moment[2],
                "gap": entry.gap,
                "num_patches": entry.num_patches,
                "num_sheet_points": entry.num_sheet_points,
                "num_tractions": entry.num_tractions,
                "marker_center_y": entry.marker_positions["center"][1],
                "marker_bottom_y": entry.marker_positions["bottom"][1],
                "marker_top_y": entry.marker_positions["top"][1],
            }
        )

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "falling_sphere_implicit_results.csv"
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # Basic diagnostics
    # -----------------------------
    first_contact_time = None
    contact_mask = df["Fy_contact"] > 1.0e-6
    if contact_mask.any():
        first_contact_time = float(df.loc[contact_mask.idxmax(), "time"])

    final_row = df.iloc[-1]
    summary_lines = [
        f"Total simulated time: {total_time:.3f} s",
        f"Time step: {integ_cfg.dt:.4f} s",
        f"First detected contact time: {first_contact_time if first_contact_time is not None else 'none'}",
        f"Final center y: {final_row['y']:.6f}",
        f"Final gap (bottom - top face): {final_row['gap']:.6f}",
        f"Final vertical velocity: {final_row['vy']:.6f}",
        f"Final contact force Fy: {final_row['Fy_contact']:.6f}",
        f"Final num_sheet_points: {int(final_row['num_sheet_points'])}",
    ]
    summary_path = OUT_DIR / "falling_sphere_implicit_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # -----------------------------
    # Plots
    # -----------------------------
    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["time"], df["y"], label="sphere center y")
    plt.plot(df["time"], df["marker_bottom_y"], label="bottom marker y")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, label="cube top")
    plt.xlabel("time [s]")
    plt.ylabel("height")
    plt.title("Implicit simulation: falling sphere height")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "falling_sphere_implicit_height.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["time"], df["vy"])
    plt.xlabel("time [s]")
    plt.ylabel("vertical velocity vy")
    plt.title("Implicit simulation: falling sphere velocity")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "falling_sphere_implicit_velocity.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["time"], df["Fy_contact"])
    plt.xlabel("time [s]")
    plt.ylabel("contact force Fy")
    plt.title("Implicit simulation: contact force")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "falling_sphere_implicit_contact_force.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["time"], df["gap"])
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("time [s]")
    plt.ylabel("gap = bottom - cube_top")
    plt.title("Implicit simulation: signed gap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "falling_sphere_implicit_gap.png", dpi=180)
    plt.close()

    print("Saved:")
    print(csv_path)
    print(summary_path)
    print(OUT_DIR / "falling_sphere_implicit_height.png")
    print(OUT_DIR / "falling_sphere_implicit_velocity.png")
    print(OUT_DIR / "falling_sphere_implicit_contact_force.png")
    print(OUT_DIR / "falling_sphere_implicit_gap.png")
    print("")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
