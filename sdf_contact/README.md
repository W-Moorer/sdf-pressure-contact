# SDF Contact Release Repository

A cleaned release-oriented repository for **reasonably accurate normal contact on arbitrary watertight triangle meshes**, with a single recommended mainline and archived experiments kept separately.

## Recommended mainline

Use the repository root as the main implementation path:

- **formal endpoint evaluator** for normal contact
- **flat support area prior** for large flat / large support contacts
- **tuned default solver**
  - implicit midpoint flavor
  - central finite-difference Jacobian
  - adaptive substepping enabled
- **`step_end`** as the current best-supported first-contact reporting quantity

## Repository layout

```text
.
├── sdf_contact/                # mainline library code
├── benchmarks/                 # recommended benchmark entry points
├── results/                    # current mainline benchmark outputs
├── docs/                       # mainline documentation
├── experiments/                # archived non-default research branches
├── examples/
├── README.md
├── pyproject.toml
└── requirements.txt
```

## What stays in the mainline

The root package keeps the branches that currently provide the best validated overall behavior:

- static/contact accuracy path
- formal analytic validation path
- quasi-static dynamics path
- free-body dynamics path
- flat-support-area-prior regression
- acceptance gates

## What moved into `experiments/`

Archived development branches include:

- onset-event refinement variants
- onset-aligned / micro-step force reporting variants
- early-active interval force reporting
- equivalent contact state and impulse-matched contact state experiments
- standalone solver tuning diagnostics

These remain useful as research artifacts but are not the current default recommendation.

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

## Recommended benchmark entry points

- `benchmarks/run_local_evaluator_benchmarks.py`
- `benchmarks/run_acceptance_gates.py`
- `benchmarks/run_formal_analytic_validation.py`
- `benchmarks/run_endpoint_final_validation.py`
- `benchmarks/run_quasistatic_dynamics_benchmarks.py`
- `benchmarks/run_free_body_dynamics_benchmarks.py`
- `benchmarks/run_flat_punch_support_area_prior_benchmarks.py`

## Mainline status

This repository is best understood as a **clean research release**:

- static evaluator path: strong
- free-body dynamics: usable and much improved from the original baseline
- first-contact experimental variants: preserved, but not promoted to default

See `docs/RELEASE_MAINLINE.md` for the rationale behind the current default choices.
