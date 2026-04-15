#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_global_multibody_newton.py

Smoke test for the global multi-body implicit Newton solver.

Scene:
    - two dynamic spheres
    - fixed compliant cube domain
    - lower sphere lands on domain
    - upper sphere lands on lower sphere

This tests:
    1) body-domain contact
    2) body-body sphere-sphere contact
    3) monolithic global Newton solve over both dynamic bodies at once
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from implicit_contact_framework_v5 import (
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

    contact_manager = ContactManager(
        PatchBuildConfig(Nxz=18, quad_order=3, bbox_padding_cells=1),
        SheetExtractConfig(bisection_steps=22, normal_step=1.0e-6),
        ContactModelConfig(
            stiffness_k=9000.0,
            damping_c=110.0,
            top_y=0.0,
            sphere_sphere_stiffness=18000.0,
            sphere_sphere_damping=140.0,
            sphere_pair_margin=0.02,
        ),
    )
    solver = GlobalImplicitSystemSolver6D(
        contact_manager,
        IntegratorConfig(dt=0.01, newton_max_iter=7, newton_tol=1.0e-8, fd_eps=1.0e-5),
    )
    sim = Simulator(world, solver)

    total_time = 0.35
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
            "wx": e.angular_velocity[0],
            "wy": e.angular_velocity[1],
            "wz": e.angular_velocity[2],
            "Fx_contact": e.contact_force[0],
            "Fy_contact": e.contact_force[1],
            "Fz_contact": e.contact_force[2],
            "Mx_contact": e.contact_moment[0],
            "My_contact": e.contact_moment[1],
            "Mz_contact": e.contact_moment[2],
            "num_pairs": e.num_pairs,
            "bottom_y": e.marker_positions["bottom"][1],
        })

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "global_multibody_newton_demo_results.csv"
    df.to_csv(csv_path, index=False)

    lower_df = df[df["body"] == "lower"].copy()
    upper_df = df[df["body"] == "upper"].copy()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lower_df["time"], lower_df["y"], label="lower center y")
    plt.plot(upper_df["time"], upper_df["y"], label="upper center y")
    plt.axhline(0.0, linestyle="--", linewidth=1.0, label="cube top")
    plt.xlabel("time [s]")
    plt.ylabel("height")
    plt.title("Global multi-body Newton demo: heights")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_multibody_newton_demo_height.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lower_df["time"], lower_df["Fy_contact"], label="lower Fy contact")
    plt.plot(upper_df["time"], upper_df["Fy_contact"], label="upper Fy contact")
    plt.xlabel("time [s]")
    plt.ylabel("contact force Fy")
    plt.title("Global multi-body Newton demo: contact force")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_multibody_newton_demo_force.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7.2, 4.6))
    plt.plot(lower_df["time"], lower_df["num_pairs"], label="lower num_pairs")
    plt.plot(upper_df["time"], upper_df["num_pairs"], label="upper num_pairs")
    plt.xlabel("time [s]")
    plt.ylabel("active pair count")
    plt.title("Global multi-body Newton demo: pair counts")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "global_multibody_newton_demo_pairs.png", dpi=180)
    plt.close()

    summary = []
    for body_name, sub in [("lower", lower_df), ("upper", upper_df)]:
        last = sub.iloc[-1]
        summary.append(
            f"{body_name}: final y={last['y']:.6f}, final bottom_y={last['bottom_y']:.6f}, "
            f"final Fy_contact={last['Fy_contact']:.6f}, final num_pairs={int(last['num_pairs'])}"
        )
    summary_path = OUT_DIR / "global_multibody_newton_demo_summary.txt"
    summary_path.write_text("\\n".join(summary), encoding="utf-8")

    print("Saved:")
    print(csv_path)
    print(summary_path)
    print(OUT_DIR / "global_multibody_newton_demo_height.png")
    print(OUT_DIR / "global_multibody_newton_demo_force.png")
    print(OUT_DIR / "global_multibody_newton_demo_pairs.png")
    print("")
    print("\\n".join(summary))

if __name__ == "__main__":
    main()
