# Step 3: wrench-aware controller + rigid-motion-aware continuity + 6-DoF complex benchmark

This stage adds:

- rigid-motion-aware continuity transport for body-fixed support discretizations;
- a wrench-aware controller with force/torque/active-measure/orientation/work mismatch indicators;
- a 6-DoF complex rigid-body benchmark with distributed normal contact wrench;
- a high-resolution reference and exported summary plots.

Generated artifacts:

- `results/tables/complex6dof_summary.csv`
- `results/tables/complex6dof_summary.md`
- `results/figures/complex6dof_force_torque_time.pdf`
- `results/figures/complex6dof_pose_time.pdf`
- `results/figures/complex6dof_controller_stats.pdf`
