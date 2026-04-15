#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_multibody_6dof_skeleton.py

Smoke-test style demo for the multi-body + 6-DOF framework skeleton.
This demo intentionally keeps the scene simple:
    one dynamic sphere
    one fixed compliant cube domain

The point is to show the new architecture, not to claim a full multi-contact engine.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from implicit_contact_framework_v2 import (
    Marker,
    Pose6D,
    BodyState6D,
    SpatialInertia,
    RigidBody6D,
    SphereGeometry,
    DomainSpec,
    World,
    PatchBuildConfig,
    SheetExtractConfig,
    ContactModelConfig,
    IntegratorConfig,
    ContactManager,
    ImplicitSystemSolver6D,
    Simulator,
)

OUT_DIR = Path(__file__).resolve().parent

def main() -> None:
    radius = 0.18
    mass = 0.05
    inertia_scalar = (2.0 / 5.0) * mass * radius * radius
    inertia = SpatialInertia(mass=mass, inertia_body=np.diag([inertia_scalar, inertia_scalar, inertia_scalar]))

    sphere = RigidBody6D(
        name="ball6d",
        inertia=inertia,
        geometry=SphereGeometry(radius=radius),
        state=BodyState6D(
            pose=Pose6D(
                position=np.array([0.0, 0.75, 0.0], dtype=float),
                orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            ),
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
        markers=[
            Marker("center", np.array([0.0, 0.0, 0.0], dtype=float)),
            Marker("bottom", np.array([0.0, -radius, 0.0], dtype=float)),
            Marker("top", np.array([0.0, radius, 0.0], dtype=float)),
        ],
        linear_damping=0.05,
        angular_damping=0.01,
    )

    world = World(
        domain=DomainSpec(cube_size=1.6, cube_height=0.35, top_y=0.0),
        gravity=np.array([0.0, -9.81, 0.0], dtype=float),
        bodies=[sphere],
    )

    patch_cfg = PatchBuildConfig(Nxz=24, quad_order=3, bbox_padding_cells=1)
    sheet_cfg = SheetExtractConfig(bisection_steps=24, normal_step=1.0e-6)
    contact_cfg = ContactModelConfig(stiffness_k=10000.0, damping_c=120.0, top_y=world.domain.top_y)
    integ_cfg = IntegratorConfig(dt=0.01, newton_max_iter=8, newton_tol=1.0e-8, fd_eps=1.0e-5)

    contact_manager = ContactManager(patch_cfg, sheet_cfg, contact_cfg)
    solver = ImplicitSystemSolver6D(contact_manager, integ_cfg)
    sim = Simulator(world, solver)

    total_time = 1.0
    log = sim.run(total_time)

    rows = []
    for e in log:
        rows.append(
            {
                "time": e.time,
                "body": e.body_name,
                "x": e.position[0],
                "y": e.position[1],
                "z": e.position[2],
                "qw": e.orientation[0],
                "qx": e.orientation[1],
                "qy": e.orientation[2],
                "qz": e.orientation[3],
                "vx": e.linear_velocity[0],
                "vy": e.linear_velocity[1],
                "vz": e.linear_velocity[2],
                "wx": e.angular_velocity[0],
                "wy": e.angular_velocity[1],
                "wz": e.angular_velocity[2],
                "Fx_contact": e.contact_force[0],
                "Fy_contact": e.contact_force[1],
                "Fz_contact": e.contact_force[2],
                "Mx_contact": e.contact_moment[0],
                "My_contact": e.contact_moment[1],
                "Mz_contact": e.contact_moment[2],
                "marker_bottom_y": e.marker_positions["bottom"][1],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "multibody_6dof_skeleton_demo_results.csv", index=False)

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["time"], df["y"], label="center y")
    plt.plot(df["time"], df["marker_bottom_y"], label="bottom marker y")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, label="cube top")
    plt.xlabel("time [s]")
    plt.ylabel("height")
    plt.title("6-DOF skeleton demo: falling sphere")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "multibody_6dof_skeleton_demo_height.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(df["time"], df["Fy_contact"])
    plt.xlabel("time [s]")
    plt.ylabel("contact force Fy")
    plt.title("6-DOF skeleton demo: contact force")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "multibody_6dof_skeleton_demo_force.png", dpi=180)
    plt.close()

    print("Saved:")
    print(OUT_DIR / "multibody_6dof_skeleton_demo_results.csv")
    print(OUT_DIR / "multibody_6dof_skeleton_demo_height.png")
    print(OUT_DIR / "multibody_6dof_skeleton_demo_force.png")


if __name__ == "__main__":
    main()
