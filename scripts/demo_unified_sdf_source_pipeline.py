from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from implicit_contact_framework_v7 import (
    Marker, Pose6D, BodyState6D, SpatialInertia, RigidBody6D,
    SphereGeometry, DomainSpec, World,
    UnifiedPairPatchConfig, SheetExtractConfig, ContactModelConfig,
    UnifiedContactManager,
)

OUT_DIR = Path('/mnt/data')


def make_sphere(name: str, radius: float, mass: float, pos: np.ndarray) -> RigidBody6D:
    I = (2.0/5.0) * mass * radius * radius
    return RigidBody6D(
        name=name,
        inertia=SpatialInertia(mass=mass, inertia_body=np.diag([I, I, I])),
        geometry=SphereGeometry(radius),
        state=BodyState6D(
            pose=Pose6D(position=pos.copy(), orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=float)),
            linear_velocity=np.zeros(3, dtype=float),
            angular_velocity=np.zeros(3, dtype=float),
        ),
        markers=[Marker('center', np.array([0.0, 0.0, 0.0], dtype=float)), Marker('bottom', np.array([0.0, -radius, 0.0], dtype=float))],
    )


def main() -> None:
    # fixed configuration chosen so that:
    #   lower sphere overlaps the top plane slightly
    #   upper sphere overlaps the lower sphere slightly
    lower = make_sphere('lower', radius=0.16, mass=0.05, pos=np.array([0.0, 0.150, 0.0], dtype=float))
    upper = make_sphere('upper', radius=0.16, mass=0.05, pos=np.array([0.0, 0.445, 0.0], dtype=float))

    world = World(
        domain=DomainSpec(cube_size=1.6, cube_height=0.35, top_y=0.0),
        gravity=np.array([0.0, -9.81, 0.0], dtype=float),
        bodies=[lower, upper],
    )

    manager = UnifiedContactManager(
        UnifiedPairPatchConfig(Nuv=8, quad_order=2, radius_scale=1.15, min_patch_radius=0.01, max_patch_radius=0.10, ray_span_scale=1.1),
        SheetExtractConfig(bisection_steps=18, normal_step=1.0e-6),
        ContactModelConfig(stiffness_k=12000.0, damping_c=0.0),
    )

    agg = manager.compute_all_contacts(world)

    rows = []
    for source_name, a in agg.items():
        rows.append({
            'source': source_name,
            'Fx': a.total_force[0],
            'Fy': a.total_force[1],
            'Fz': a.total_force[2],
            'Mx': a.total_moment[0],
            'My': a.total_moment[1],
            'Mz': a.total_moment[2],
            'num_pairs': len(a.pair_records),
            'num_pair_patch_points': a.num_pair_patch_points,
            'num_pair_sheet_points': a.num_pair_sheet_points,
            'num_pair_tractions': a.num_pair_tractions,
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / 'unified_sdf_source_pipeline_results.csv', index=False)

    pair_rows = []
    seen = set()
    for source_name, a in agg.items():
        for rec in a.pair_records:
            key = (rec.source_a_name, rec.source_b_name)
            if key in seen:
                continue
            seen.add(key)
            pair_rows.append({
                'pair_kind': rec.pair_kind,
                'source_a': rec.source_a_name,
                'source_b': rec.source_b_name,
                'patch_radius': rec.meta['patch_radius'],
                'num_pair_patch_points': rec.meta['num_pair_patch_points'],
                'num_pair_sheet_points': rec.meta['num_pair_sheet_points'],
                'num_pair_tractions': rec.meta['num_pair_tractions'],
            })
    pair_df = pd.DataFrame(pair_rows)
    pair_df.to_csv(OUT_DIR / 'unified_sdf_source_pipeline_pairs.csv', index=False)

    plt.figure(figsize=(7.0, 4.6))
    plt.bar(df['source'], df['Fy'])
    plt.ylabel('aggregated Fy')
    plt.title('Unified SDF source-source pipeline: aggregated vertical force')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'unified_sdf_source_pipeline_force.png', dpi=180)
    plt.close()

    if not pair_df.empty:
        x = np.arange(len(pair_df))
        w = 0.25
        plt.figure(figsize=(7.2, 4.8))
        plt.bar(x - w, pair_df['num_pair_patch_points'], width=w, label='patch')
        plt.bar(x, pair_df['num_pair_sheet_points'], width=w, label='sheet')
        plt.bar(x + w, pair_df['num_pair_tractions'], width=w, label='traction')
        plt.xticks(x, [f"{a}\nvs\n{b}" for a, b in zip(pair_df['source_a'], pair_df['source_b'])])
        plt.ylabel('count')
        plt.title('Unified source-source pipeline counts per pair')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / 'unified_sdf_source_pipeline_counts.png', dpi=180)
        plt.close()

    summary = []
    for _, row in df.iterrows():
        summary.append(
            f"{row['source']}: Fy={row['Fy']:.6f}, num_pairs={int(row['num_pairs'])}, "
            f"pair_patch_pts={int(row['num_pair_patch_points'])}, pair_sheet_pts={int(row['num_pair_sheet_points'])}, "
            f"pair_tractions={int(row['num_pair_tractions'])}"
        )
    (OUT_DIR / 'unified_sdf_source_pipeline_summary.txt').write_text('\n'.join(summary), encoding='utf-8')

    print('Saved:')
    print(OUT_DIR / 'unified_sdf_source_pipeline_results.csv')
    print(OUT_DIR / 'unified_sdf_source_pipeline_pairs.csv')
    print(OUT_DIR / 'unified_sdf_source_pipeline_summary.txt')
    print(OUT_DIR / 'unified_sdf_source_pipeline_force.png')
    print(OUT_DIR / 'unified_sdf_source_pipeline_counts.png')
    print('')
    print('\n'.join(summary))


if __name__ == '__main__':
    main()
