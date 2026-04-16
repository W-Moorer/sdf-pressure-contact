# Onset-aligned first-contact force benchmarks

This benchmark compares the first contact force measured at the coarse end-of-step state versus the new onset-localized state reconstructed inside the first active coarse step.

## sphere_centered
### coarse_step_end
- analytic_touch_time = 0.063855
- step_end_time = 0.075000
- step_end_delta = 0.007591
- step_end_Fy = 0.402195
- step_end_static_Fy = 0.425266
- step_end_rel_err = 0.054252
- refine_reason = 
- solver_substeps = 1

### onset_localized
- analytic_touch_time = 0.063855
- step_end_time = 0.075000
- step_end_delta = 0.007572
- step_end_Fy = 0.400933
- step_end_static_Fy = 0.423215
- step_end_rel_err = 0.052648
- onset_aligned_time = 0.074082
- onset_aligned_delta = 0.006913
- onset_aligned_Fy = 0.326227
- onset_aligned_static_Fy = 0.353393
- onset_aligned_rel_err = 0.076872
- refine_reason = onset_uniform_refine
- solver_substeps = 2

## flat_punch_centered
### coarse_step_end
- analytic_touch_time = 0.045152
- step_end_time = 0.050000
- step_end_delta = 0.002112
- step_end_Fy = 1.267506
- step_end_static_Fy = 1.267494
- step_end_rel_err = 0.000009
- refine_reason = 
- solver_substeps = 1

### onset_localized
- analytic_touch_time = 0.045152
- step_end_time = 0.050000
- step_end_delta = 0.002144
- step_end_Fy = 1.267658
- step_end_static_Fy = 1.286107
- step_end_rel_err = 0.014345
- onset_aligned_time = 0.046328
- onset_aligned_delta = 0.000525
- onset_aligned_Fy = 0.309763
- onset_aligned_static_Fy = 0.315019
- onset_aligned_rel_err = 0.016682
- refine_reason = onset_uniform_refine
- solver_substeps = 2
