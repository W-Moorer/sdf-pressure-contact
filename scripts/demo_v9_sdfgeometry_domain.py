
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from implicit_contact_framework_v9 import (
    Marker, Pose6D, BodyState6D, SpatialInertia, RigidBody6D,
    SphereGeometry, PlaneGeometry, DomainSpec, World,
    SDFGeometryDomainSource, UnifiedPairPatchConfig, SheetExtractConfig,
    ContactModelConfig, IntegratorConfig, UnifiedContactManager,
    GlobalImplicitSystemSolver6D, Simulator,
)

OUT_DIR = Path(__file__).resolve().parent

def make_sphere(name, radius, mass, pos):
    I = (2.0/5.0) * mass * radius * radius
    return RigidBody6D(
        name=name,
        inertia=SpatialInertia(mass=mass, inertia_body=np.diag([I,I,I])),
        geometry=SphereGeometry(radius),
        state=BodyState6D(
            pose=Pose6D(position=pos.copy(), orientation=np.array([1.,0.,0.,0.])),
            linear_velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        ),
        markers=[Marker("bottom", np.array([0.,-radius,0.]))],
        linear_damping=0.03,
        angular_damping=0.01,
    )

def main():
    ball = make_sphere("ball", radius=0.16, mass=0.05, pos=np.array([0.0, 0.58, 0.0]))
    # fixed convex spherical domain geometry (not a plane)
    dome_source = SDFGeometryDomainSource(
        geometry=SphereGeometry(0.35),
        pose=Pose6D(position=np.array([0.0, -0.20, 0.0]), orientation=np.array([1.,0.,0.,0.])),
        name="fixed_dome",
        hint_radius=0.35,
        reference_center=np.array([0.0, -0.20, 0.0]),
    )
    world = World(
        domain=DomainSpec(cube_size=1.6, cube_height=0.35, top_y=0.0),
        gravity=np.array([0.0,-9.81,0.0]),
        bodies=[ball],
        domain_sources=[dome_source],
    )

    cm = UnifiedContactManager(
        UnifiedPairPatchConfig(Nuv=8, quad_order=2, radius_scale=1.2, min_patch_radius=0.01, max_patch_radius=0.16, ray_span_scale=1.15),
        SheetExtractConfig(bisection_steps=18, normal_step=1e-6),
        ContactModelConfig(stiffness_k=9000.0, damping_c=100.0),
    )
    solver = GlobalImplicitSystemSolver6D(cm, IntegratorConfig(dt=0.02, newton_max_iter=6, newton_tol=1e-8, fd_eps=1e-5))
    sim = Simulator(world, solver)
    log = sim.run(0.40)

    rows=[]
    for e in log:
        rows.append({
            "time": e.time,
            "body": e.body_name,
            "y": e.position[1],
            "vy": e.linear_velocity[1],
            "Fy_contact": e.contact_force[1],
            "num_pairs": e.num_pairs,
            "num_pair_patch_points": e.num_pair_patch_points,
            "num_pair_sheet_points": e.num_pair_sheet_points,
            "num_pair_tractions": e.num_pair_tractions,
            "bottom_y": e.marker_positions["bottom"][1],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v9_sdfgeometry_domain_results.csv", index=False)

    plt.figure(figsize=(7,4.4))
    plt.plot(df["time"], df["y"], label="ball center y")
    plt.plot(df["time"], df["bottom_y"], label="ball bottom y")
    plt.xlabel("time [s]"); plt.ylabel("height")
    plt.title("v9 SDFGeometryDomainSource demo: falling ball onto fixed dome")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "v9_sdfgeometry_domain_height.png", dpi=180); plt.close()

    plt.figure(figsize=(7,4.4))
    plt.plot(df["time"], df["Fy_contact"])
    plt.xlabel("time [s]"); plt.ylabel("contact force Fy")
    plt.title("v9 SDFGeometryDomainSource demo: contact force")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "v9_sdfgeometry_domain_force.png", dpi=180); plt.close()

    plt.figure(figsize=(7,4.4))
    plt.plot(df["time"], df["num_pair_patch_points"], label="pair patch pts")
    plt.plot(df["time"], df["num_pair_sheet_points"], label="pair sheet pts")
    plt.plot(df["time"], df["num_pair_tractions"], label="pair tractions")
    plt.xlabel("time [s]"); plt.ylabel("count")
    plt.title("v9 SDFGeometryDomainSource demo: pipeline counts")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "v9_sdfgeometry_domain_counts.png", dpi=180); plt.close()

    last=df.iloc[-1]
    (OUT_DIR/"v9_sdfgeometry_domain_summary.txt").write_text(
        f"final y={last['y']:.6f}\nfinal bottom_y={last['bottom_y']:.6f}\n"
        f"final Fy_contact={last['Fy_contact']:.6f}\nnum_pairs={int(last['num_pairs'])}\n"
        f"pair_patch_pts={int(last['num_pair_patch_points'])}\npair_sheet_pts={int(last['num_pair_sheet_points'])}\n"
        f"pair_tractions={int(last['num_pair_tractions'])}\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
