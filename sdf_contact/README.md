# Minimal SDF Contact Repository

A cleaned-up repository for **reasonably accurate normal contact on arbitrary watertight triangle meshes**.

This repo is intentionally organized around one main goal:

> Given two SDF-backed objects or meshes and their poses, compute a reasonably accurate normal contact force, moment, and local contact geometry.

## What was kept

- A **single main contact path** based on a high-accuracy local evaluator:
  contour extraction -> polygon footprint -> triangulation -> quadrature -> dual root solves -> traction accumulation.
- A **baseline evaluator** for comparison and regression tests.
- A **small optional implicit solver** so the evaluator can still be embedded into dynamics.
- A **benchmark module** so accuracy is defined by reproducible cases instead of intuition.

## What was removed

- Versioned framework sprawl (`v9`, `v10b`, `v10e`) as first-class entry points.
- Multiple overlapping contact managers.
- Geometry and solver logic mixed into the evaluator module.

## Minimal directory layout

```text
sdf_contact_minimal_repo/
├── README.md
├── pyproject.toml
├── requirements.txt
├── docs/
│   └── RESTRUCTURE_PLAN.md
├── examples/
│   └── local_accuracy_demo.py
└── sdf_contact/
    ├── __init__.py
    ├── core.py
    ├── geometry.py
    ├── evaluators.py
    ├── pipeline.py
    └── benchmarks.py
```

## Design principles

1. **Local evaluator first**: benchmark local contact accuracy before restructuring the global solver.
2. **One default evaluator**: the polygon high-accuracy evaluator is the main path.
3. **Benchmark-driven accuracy**: “accurate enough” is decided by force, moment, convergence, and robustness tests.
4. **Watertight meshes first**: the included mesh SDF backend is a self-contained brute-force reference implementation. It is not optimized. For production-scale models, swap in VDB/NanoVDB or another fast signed-distance backend.

## Suggested workflow

1. Use `PolygonHighAccuracyLocalEvaluator` for all new experiments.
2. Use `BaselineGridLocalEvaluator` only for comparison.
3. Validate on static / quasi-static benchmark families first.
4. Only then decide whether the global implicit solver is worth refactoring.
5. Run the admission gates in `docs/BENCHMARK_ACCEPTANCE.md` to decide whether an evaluator is merely research-usable or genuinely production-ready.

## Quick start

```python
import numpy as np
import trimesh
from sdf_contact import (
    Marker,
    MeshGeometryFactoryConfig,
    make_mesh_rigidbody,
    make_mesh_domain_source,
    make_world,
    PolygonPatchConfig,
    SheetExtractConfig,
    ContactModelConfig,
    PolygonHighAccuracyLocalEvaluator,
    ContactManager,
)

sphere = trimesh.creation.icosphere(radius=0.1, subdivisions=1)
plane_box = trimesh.creation.box(extents=[1.0, 0.2, 1.0])

body = make_mesh_rigidbody(
    name='ball',
    mesh_or_path=sphere,
    mass=1.0,
    position=[0.0, 0.09, 0.0],
    markers=[Marker('center', np.array([0.0, 0.0, 0.0]))],
    cfg=MeshGeometryFactoryConfig(recenter_mode='center_mass'),
)

domain = make_mesh_domain_source(
    name='ground',
    mesh_or_path=plane_box,
    position=[0.0, -0.1, 0.0],
    cfg=MeshGeometryFactoryConfig(recenter_mode='bbox_center'),
)

evaluator = PolygonHighAccuracyLocalEvaluator(
    PolygonPatchConfig(),
    SheetExtractConfig(),
    ContactModelConfig(stiffness_k=10000.0, damping_c=100.0),
)
manager = ContactManager(evaluator)
world = make_world(bodies=[body], domain_sources=[domain])
contacts = manager.compute_all_contacts(world)
print(contacts['ball'].total_force)
```

## Notes on accuracy

The included polygon evaluator is substantially closer to the formal “quad-sheet” route than the old uv-grid sampler, but it is still an approximation:

- overlap contours are extracted from a rasterized local overlap field;
- footprint clipping is polygon-accurate **after** that rasterization step;
- the sheet normal is still approximated from the two recovered surface normals.

This repository therefore gives you a **clean experimental baseline** for deciding what to improve next. The benchmark suite now also has an explicit admission standard in `docs/BENCHMARK_ACCEPTANCE.md`, together with an automated evaluator gate script in `benchmarks/run_acceptance_gates.py`.

This repository therefore gives you a **clean experimental baseline** for deciding what to improve next:

- replace the normal with `∇h / ||∇h||`;
- replace sheet-point traction with a true local-normal band accumulator;
- replace brute-force mesh distance with a fast sparse SDF backend.
