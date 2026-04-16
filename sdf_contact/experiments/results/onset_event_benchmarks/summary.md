# Onset-event window benchmarks

This benchmark isolates the **contact establishment stage**.

The onset-focused default solver does two new things:
- on the onset step, it replaces a single coarse step with a small uniform local refinement
- on the next first-contact-cycle steps, it checks one-step vs two-half-step consistency and only keeps the refined path when the discrepancy is significant

The flat-punch benchmark here is intentionally **cut at the first active contact sample** to keep the cost focused on onset timing rather than full release dynamics.

## sphere_centered_onset
### onset_focus_off
- sim_touch_time = 0.075000
- analytic_touch_time = 0.063855
- abs_touch_time_error = 0.011145
- first_active_delta = 0.007591
- first_active_Fy = 0.402195
- first_active_static_Fy = 0.425266
- mean_rel_err_onset_window = 0.036301
- max_rel_err_onset_window = 0.054252
- max_solver_substeps = 1
- n_local_refine_steps = 0
- onset_refine_reason = nan

### onset_focus_on
- sim_touch_time = 0.073750
- analytic_touch_time = 0.063855
- abs_touch_time_error = 0.009895
- first_active_delta = 0.007572
- first_active_Fy = 0.400933
- first_active_static_Fy = 0.423215
- mean_rel_err_onset_window = 0.035714
- max_rel_err_onset_window = 0.052648
- max_solver_substeps = 2
- n_local_refine_steps = 2
- onset_refine_reason = onset_uniform_refine

## flat_punch_centered_onset
### onset_focus_off
- sim_touch_time = 0.050000
- analytic_touch_time = 0.045152
- abs_touch_time_error = 0.004848
- first_active_delta = 0.002112
- first_active_Fy = 1.267506
- first_active_static_Fy = 1.267494
- mean_rel_err_onset_window = 0.000009
- max_rel_err_onset_window = 0.000009
- max_solver_substeps = 1
- n_local_refine_steps = 0
- onset_refine_reason = nan

### onset_focus_on
- sim_touch_time = 0.046250
- analytic_touch_time = 0.045152
- abs_touch_time_error = 0.001098
- first_active_delta = 0.002144
- first_active_Fy = 1.267658
- first_active_static_Fy = 1.286107
- mean_rel_err_onset_window = 0.014345
- max_rel_err_onset_window = 0.014345
- max_solver_substeps = 2
- n_local_refine_steps = 1
- onset_refine_reason = onset_uniform_refine
