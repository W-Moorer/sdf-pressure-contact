from __future__ import annotations

import numpy as np
import trimesh

from sdf_contact import (
    Marker,
    MeshGeometryFactoryConfig,
    make_mesh_rigidbody,
    make_mesh_domain_source,
    BaselinePatchConfig,
    PolygonPatchConfig,
    SheetExtractConfig,
    ContactModelConfig,
    BaselineGridLocalEvaluator,
    PolygonHighAccuracyLocalEvaluator,
    compare_local_evaluators,
    cap_volume,
)


def main() -> None:
    R = 0.12
    k_contact = 12000.0
    sphere_mesh = trimesh.creation.icosphere(radius=R, subdivisions=1)
    box_mesh = trimesh.creation.box(extents=[1.0, 0.18, 1.0])

    body_cfg = MeshGeometryFactoryConfig(recenter_mode='center_mass')
    domain_cfg = MeshGeometryFactoryConfig(recenter_mode='bbox_center')

    def body_factory(position: np.ndarray, linear_velocity: np.ndarray):
        return make_mesh_rigidbody(
            name='ball',
            mesh_or_path=sphere_mesh,
            mass=0.2,
            position=position,
            linear_velocity=linear_velocity,
            markers=[Marker('center', np.array([0.0, 0.0, 0.0]))],
            cfg=body_cfg,
        )

    def domain_factory():
        return make_mesh_domain_source(
            name='ground',
            mesh_or_path=box_mesh,
            position=[0.0, -0.09, 0.0],
            cfg=domain_cfg,
        )

    baseline = BaselineGridLocalEvaluator(
        BaselinePatchConfig(Nuv=10, quad_order=2),
        SheetExtractConfig(),
        ContactModelConfig(stiffness_k=k_contact, damping_c=80.0),
    )
    high = PolygonHighAccuracyLocalEvaluator(
        PolygonPatchConfig(raster_cells=14),
        SheetExtractConfig(),
        ContactModelConfig(stiffness_k=k_contact, damping_c=80.0),
    )

    states = []
    for y in np.linspace(0.14, 0.08, 7):
        states.append({'position': np.array([0.0, y, 0.0]), 'linear_velocity': np.array([0.0, 0.0, 0.0])})

    def ideal_force_fn(st: dict) -> np.ndarray:
        bottom_y = float(st['position'][1] - R)
        delta = max(0.0, -bottom_y)
        return np.array([0.0, k_contact * cap_volume(R, delta), 0.0])

    rows = compare_local_evaluators(
        body_factory=body_factory,
        domain_source_factory=domain_factory,
        states=states,
        baseline_evaluator=baseline,
        high_accuracy_evaluator=high,
        ideal_force_fn=ideal_force_fn,
    )

    for row in rows:
        print(f'state={row.state_index:02d} y={row.body_position[1]:.4f}')
        print('  baseline Fy      =', row.baseline_force[1])
        print('  high_accuracy Fy =', row.high_accuracy_force[1])
        print('  ideal Fy         =', None if row.ideal_force is None else row.ideal_force[1])
        print('  baseline meta    =', row.baseline_meta)
        print('  high meta        =', row.high_accuracy_meta)


if __name__ == '__main__':
    main()
