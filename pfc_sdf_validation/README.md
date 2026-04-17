# PFC-SDF Validation Project

This repository is a test-driven Python scaffold for a **SDF-native PFC dynamics** workflow.
It now includes a complete **Sprint 1 publication loop**:

- long-horizon default ablation tables
- efficiency tables
- paper-ready figures
- reproducible experiment scripts
- updated paper integration assets

## Main solver story

The default solver path is **not** an explicit sheet/mesh reconstruction loop. The main chain is:

```text
phi_A, phi_B
-> depth_A, depth_B
-> pressure_A, pressure_B
-> h = p_A - p_B
-> sparse active-cell traversal in |h| <= eta
-> local-normal / per-cell traction correction
-> direct narrow-band wrench accumulation
-> event-aware midpoint dynamics
```

Reference paths (sheet reconstruction / marching cubes) remain in the repository as validation and debugging tools, but not as the default production path.

## Sprint 1 deliverables now in the repo

### Experiment scripts

```text
experiments/run_main_tables.py
experiments/run_efficiency_tables.py
experiments/run_plot_suite.py
experiments/run_appendix_tables.py
```

### Configs

```text
configs/main_tables.yaml
configs/efficiency.yaml
configs/appendix.yaml
```

### Reporting helpers

```text
src/pfcsdf/reporting/export.py
src/pfcsdf/reporting/tables.py
src/pfcsdf/reporting/plots.py
```

### Output folders

```text
results/tables/
results/figures/
results/logs/
```

## Installation

```bash
python -m pip install -e .[dev]
```

## Reproduce the main paper artifacts

### 1. Main long-horizon tables

```bash
python experiments/run_main_tables.py
```

This generates:

- `results/tables/long_horizon_ablation.csv`
- `results/tables/long_horizon_ablation_raw.md`
- `results/tables/main_ablation.csv`
- `results/tables/main_ablation.md`
- `results/tables/main_ablation.tex`

The default long-horizon story covers:

- `analytic_flat`
- `analytic_sphere`
- `native_band_flat`

and all tables run beyond release, not only around first contact.

### 2. Efficiency tables

```bash
python experiments/run_efficiency_tables.py
```

This generates:

- `results/tables/efficiency_continuity_raw.csv`
- `results/tables/efficiency_continuity.md`
- `results/tables/efficiency_continuity.tex`
- `results/tables/efficiency_traction_raw.csv`
- `results/tables/efficiency_traction.md`
- `results/tables/efficiency_traction.tex`

These tables separate two roles:

- **continuity-aware traversal**: lowers candidate/recompute cost
- **consistent traction reconstruction**: improves per-cell traction quality with nearly unchanged traversal cost

### 3. Main figures

```bash
python experiments/run_plot_suite.py
```

This generates paper-ready figures under `results/figures/`:

- `flat_force_time.pdf`
- `flat_energy_time.pdf`
- `sphere_force_time.pdf`
- `native_band_force_time.pdf`
- `native_band_active_measure_time.pdf`
- `native_band_controller_statistics.pdf`

### 4. Appendix placeholders

```bash
python experiments/run_appendix_tables.py
```

This currently creates a small appendix plan placeholder, so later Sprint 2 / Sprint 3 content can plug into the same experiment structure without reshaping the repository.

## Tests

Run the full regression suite with:

```bash
pytest
```

Sprint 1 adds tests that lock in:

- long-horizon default ablation as the default publication story
- sphere long-horizon benchmark support
- generation of main tables, efficiency tables, and figure outputs

## Interpreting the current evidence

At the current stage, the strongest conclusions supported by the code are:

1. The **SDF-native dynamics chain is fully implemented**.
2. **Continuity-aware traversal** has clear engineering value: it lowers traversal cost without materially shifting the main benchmark result.
3. **Impulse correction** primarily reduces impulse-related error.
4. **Work-consistent accepted steps** primarily reduce energy drift and improve release-phase consistency.
5. **Consistent traction reconstruction** is a smaller but positive per-cell refinement, especially for impulse accumulation.

## Repository roadmap after Sprint 1

The next planned stages are:

- add eccentric flat and multipatch dynamic benchmarks
- formalize controller statistics for appendix tables
- add reference-comparison and sensitivity suites
- strengthen paper appendix implementation details



## Complex-case MVP benchmark

This repository now also contains a **complex-geometry eccentric long-horizon benchmark**.

Main additions:

- `src/pfcsdf/geometry/complex_bodies.py`
- `src/pfcsdf/dynamics/benchmarks_complex.py`
- `experiments/run_complex_case.py`
- `tests/test_step_50_complex_body_benchmark.py`
- `tests/test_step_51_complex_body_reference.py`
- `tests/test_step_52_complex_body_outputs.py`

The complex case uses a **composite body SDF/profile** (curved main body + local flat patch + asymmetric edge bump) and a reduced planar rigid-body dynamics model with:

- vertical translation
- one rotational DOF
- distributed contact force and torque
- continuity-aware active-set tracking
- predictor/corrector controller statistics
- high-resolution reference trajectory

Run the benchmark with:

```bash
PYTHONPATH=src python experiments/run_complex_case.py
```

It writes:

- `results/tables/complex_case_summary.csv`
- `results/figures/complex_force_torque_time.pdf`
- `results/figures/complex_pose_time.pdf`
- `results/figures/complex_controller_stats.pdf`

The current complex-case MVP is intended to provide a **more realistic dynamic geometry/torque/controller demonstration** than the flat/sphere canonical benchmarks, while still remaining lightweight enough for reproducible experiments.


## 6-DoF foundation modules

This worktree also includes the first 6-DoF foundation layer:

- `src/pfcsdf/dynamics/rotation.py`
- `src/pfcsdf/dynamics/rigid_state.py`
- `src/pfcsdf/dynamics/rigid_integrators.py`
- `tests/test_step_53_rigid_state_6dof.py`
- `tests/test_step_54_rigid_wrench_integration.py`

These modules provide:

- rigid body state on `SE(3)` using a rotation matrix,
- body/world inertia transforms,
- kinetic energy and angular momentum diagnostics,
- midpoint and semi-implicit rigid-body stepping from a `PairWrench`,
- wrench reference shifting utilities in `contact/wrench.py`.
