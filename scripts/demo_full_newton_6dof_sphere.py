#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from implicit_contact_framework_v3 import (
    Marker, Pose6D, BodyState6D, SpatialInertia, RigidBody6D, SphereGeometry,
    DomainSpec, World, PatchBuildConfig, SheetExtractConfig, ContactModelConfig,
    IntegratorConfig, ContactManager, ImplicitSystemSolver6D, Simulator
)

OUT_DIR = Path('/mnt/data')

def main():
    radius = 0.18
    mass = 0.05
    inertia_scalar = (2.0 / 5.0) * mass * radius * radius
    inertia = SpatialInertia(mass=mass, inertia_body=np.diag([inertia_scalar]*3))

    sphere = RigidBody6D(
        name='ball6d_full_newton',
        inertia=inertia,
        geometry=SphereGeometry(radius=radius),
        state=BodyState6D(
            pose=Pose6D(position=np.array([0.0, 0.75, 0.0]), orientation=np.array([1.0,0.0,0.0,0.0])),
            linear_velocity=np.zeros(3),
            angular_velocity=np.array([0.0, 0.2, 0.0]),
        ),
        markers=[Marker('center', np.array([0,0,0],dtype=float)), Marker('bottom', np.array([0,-radius,0],dtype=float))],
        linear_damping=0.05,
        angular_damping=0.02,
    )
    world = World(domain=DomainSpec(cube_size=1.6, cube_height=0.35, top_y=0.0), gravity=np.array([0.0,-9.81,0.0]), bodies=[sphere])
    patch_cfg = PatchBuildConfig(Nxz=22, quad_order=3, bbox_padding_cells=1)
    sheet_cfg = SheetExtractConfig(bisection_steps=20, normal_step=1e-6)
    contact_cfg = ContactModelConfig(stiffness_k=10000.0, damping_c=120.0, top_y=world.domain.top_y)
    integ_cfg = IntegratorConfig(dt=0.01, newton_max_iter=7, newton_tol=1e-8, fd_eps=1e-5)
    manager = ContactManager(patch_cfg, sheet_cfg, contact_cfg)
    solver = ImplicitSystemSolver6D(manager, integ_cfg)
    sim = Simulator(world, solver)
    log = sim.run(2.0)
    rows=[]
    for e in log:
        rows.append({
            'time':e.time,'y':e.position[1],'vy':e.linear_velocity[1],
            'wy':e.angular_velocity[1],'Fy_contact':e.contact_force[1],
            'bottom_y':e.marker_positions['bottom'][1]
        })
    df=pd.DataFrame(rows)
    df.to_csv(OUT_DIR/'full_newton_6dof_sphere_results.csv',index=False)
    final=df.iloc[-1]
    summary='\n'.join([
        f"final y = {final['y']:.6f}",
        f"final vy = {final['vy']:.6f}",
        f"final wy = {final['wy']:.6f}",
        f"final bottom_y = {final['bottom_y']:.6f}",
        f"final Fy_contact = {final['Fy_contact']:.6f}",
    ])
    (OUT_DIR/'full_newton_6dof_sphere_summary.txt').write_text(summary,encoding='utf-8')
    plt.figure(figsize=(7,4.4)); plt.plot(df['time'],df['y'],label='center y'); plt.plot(df['time'],df['bottom_y'],label='bottom y'); plt.axhline(0.0,ls='--',lw=1.0,label='cube top'); plt.legend(); plt.tight_layout(); plt.savefig(OUT_DIR/'full_newton_6dof_sphere_height.png',dpi=180); plt.close()
    plt.figure(figsize=(7,4.4)); plt.plot(df['time'],df['Fy_contact']); plt.tight_layout(); plt.savefig(OUT_DIR/'full_newton_6dof_sphere_force.png',dpi=180); plt.close()
    print(summary)

if __name__=='__main__':
    main()
