#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_v8_global_unified_manager.py

Smoke test for:
    global multi-body implicit Newton solver
    + unified source-source contact manager

Scene:
    - two dynamic spheres
    - one fixed top-plane domain source
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from implicit_contact_framework_v8 import (
    Marker,
    Pose6D,
    BodyState6D,
    SpatialInertia,
    RigidBody6D,
    SphereGeometry,
    DomainSpec,
    World,
    PatchBuildConfig,
    BodyBodyPatchBuildConfig,
    SheetExtractConfig,
    ContactModelConfig,
    IntegratorConfig,
    GlobalImplicitSystemSolver6D,
    Simulator,
)

OUT_DIR = Path(__file__).resolve().parent


def make_sphere(name: str, radius: float, mass: float, pos: np.ndarray) -> RigidBody6D:
    I = (2.0 / 5.0) * mass * radius * radius
    return RigidBody6D(
        name=name,
        inertia=SpatialInertia(mass=mass, inertia_body=np.diag([I, I, I])),
        geometry=SphereGeometry(radius),
        state=BodyState6D(
            pose=Pose6D(position=pos.copy(), orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
        markers=[
            Marker("center", np.array([0.0, 0.0, 0.0], dtype=float)),
            Marker("bottom", np.array([0.0, -radius, 0.0], dtype=float)),
        ],
        linear_damping=0.03,
        angular_damping=0.01,
    )


def main() -> None:
    lower = make_sphere("lower", radius=0.16, mass=0.05, pos=np.array([0.0, 0.52, 0.0], dtype=float))
    upper = make_sphere("upper", radius=0.16, mass=0.05, pos=np.array([0.0, 0.88, 0.0], dtype=float))

    world = World(
        domain=DomainSpec(cube_size=1.6, cube_height=0.35, top_y=0.0),
        gravity=np.array([0.0, -9.81, 0.0], dtype=float),
        bodies=[lower, upper],
    )

    solver = GlobalImplicitSystemSolver6D(
        patch_cfg=PatchBuildConfig(Nxz=14, quad_order=2, bbox_padding_cells=1),
        pair_patch_cfg=BodyBodyPatchBuildConfig(
            Nuv=8,
            quad_order=2,
            radius_scale=1.2,
            min_patch_radius=0.01,
            max_patch_radius=0.18,
            ray_span_scale=1.1,
        ),
        sheet_cfg=SheetExtractConfig(bisection_steps=18, normal_step=1.0e-6),
        contact_cfg=ContactModelConfig(
            stiffness_k=8500.0,
            damping_c=100.0,
            top_y=0.0,
            pair_stiffness_k=14000.0,
            pair_damping_c=100.0,
        ),
        integ_cfg=IntegratorConfig(
            dt=0.02,
            newton_max_iter=6,
            newton_tol=1.0e-8,
            fd_eps=1.0e-5,
        ),
    )

    sim = Simulator(world, solver)
    total_time = 0.40
    log = sim.run(total_time)

    rows = []
    for e in log:
        rows.append({
            "time": e.time,
            "body": e.body_name,
            "x": e.position[0],
            "y": e.position[1],
            "z": e.position[2],
            "vx": e.linear_velocity[0],
            "vy": e.linear_velocity[1],
            "vz": e.linear_velocity[2],
            "Fx_contact": e.contact_force[0],
            "Fy_contact": e.contact_force[1],
            "Fz_contact": e.contact_force[2],
            "Mx_contact": e.contact_moment[0],
            "My_contact": e.contact_moment[1],
            "Mz_contact": e.contact_moment[2],
            "num_pairs": e.num_pairs,
            "num_pair_patch_pts": e.num_pair_patch_points,
            "num_pair_sheet_pts": e.num_pair_sheet_points,
            "num_pair_tractions": e.num_pair_tractions,
            "bottom_y": e.marker_positions["bottom"][1],
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v8_global_unified_manager_results.csv", index=False)

    lower_df = df[df["body"] == "lower"].copy()
    upper_df = df[df["body"] == "upper"].copy()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lower_df["time"], lower_df["y"], label="lower center y")
    plt.plot(upper_df["time"], upper_df["y"], label="upper center y")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, label="domain top")
    plt.xlabel("time [s]")
    plt.ylabel("height")
    plt.title("v8 global unified manager: heights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v8_global_unified_manager_height.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lower_df["time"], lower_df["Fy_contact"], label="lower Fy contact")
    plt.plot(upper_df["time"], upper_df["Fy_contact"], label="upper Fy contact")
    plt.xlabel("time [s]")
    plt.ylabel("contact force Fy")
    plt.title("v8 global unified manager: contact force")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v8_global_unified_manager_force.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lower_df["time"], lower_df["num_pair_patch_pts"], label="lower pair patch pts")
    plt.plot(upper_df["time"], upper_df["num_pair_patch_pts"], label="upper pair patch pts")
    plt.xlabel("time [s]")
    plt.ylabel("count")
    plt.title("v8 global unified manager: pair patch counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v8_global_unified_manager_counts.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
