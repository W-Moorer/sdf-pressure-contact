# Free-body slow contact and release benchmarks

Shared configuration (no case-specific tuning):
- patch raster_cells = 4
- sheet bisection_steps = 8
- stiffness per side = 40000.0
- damping_gamma = 0.0
- dt = 0.005
- gravity = 9.81

The main diagnostics are:
- first-cycle max penetration vs conservative analytic turning-point prediction
- first-release upward speed vs conservative analytic touch speed
- total-energy drift at first release, normalized by the initial drop-energy scale
- static force-law consistency during the first contact cycle
- off-axis symmetry diagnostics: x drift, vx drift, body moment pollution, ground Mz consistency

## sphere_centered_free_contact_release
- first_contact_time = 0.075000
- first_release_time = 0.135000
- analytic_delta_max = 0.020148
- sim_delta_max_first_cycle = 0.020719
- rel_err_delta_max = 0.028374
- touch_speed_analytic = 0.626418
- release_speed_sim = 0.685508
- rel_err_release_speed = 0.094329
- release_energy_drift_rel_drop_scale = 0.031011
- max_rel_err_static_force_law_first_cycle = 0.067151
- max_abs_x_drift = 2.222470e-17
- max_abs_vx = 7.207806e-16
- max_body_moment_norm_first_cycle = 1.018227e-17

## flat_punch_centered_free_contact_release
- first_contact_time = 0.050000
- first_release_time = 0.085000
- analytic_delta_max = 0.004943
- sim_delta_max_first_cycle = 0.005944
- rel_err_delta_max = 0.202499
- touch_speed_analytic = 0.442945
- release_speed_sim = 0.431567
- rel_err_release_speed = 0.025686
- release_energy_drift_rel_drop_scale = -0.021466
- max_rel_err_static_force_law_first_cycle = 0.276892
- max_abs_x_drift = 1.641614e-16
- max_abs_vx = 6.304002e-15
- max_body_moment_norm_first_cycle = 6.463591e-16

## sphere_offaxis_free_contact_release
- first_contact_time = 0.075000
- first_release_time = 0.135000
- analytic_delta_max = 0.020148
- sim_delta_max_first_cycle = 0.020719
- rel_err_delta_max = 0.028374
- touch_speed_analytic = 0.626418
- release_speed_sim = 0.685508
- rel_err_release_speed = 0.094329
- release_energy_drift_rel_drop_scale = 0.031011
- max_rel_err_static_force_law_first_cycle = 0.067151
- max_rel_ground_Mz_relation_first_cycle = 0.067151
- max_abs_x_drift = 4.857226e-17
- max_abs_vx = 1.035774e-15
- max_body_moment_norm_first_cycle = 1.018227e-17
