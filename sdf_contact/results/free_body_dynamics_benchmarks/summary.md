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
- sim_delta_max_first_cycle = 0.017631
- rel_err_delta_max = 0.124900
- touch_speed_analytic = 0.626418
- release_speed_sim = 0.366025
- rel_err_release_speed = 0.415687
- release_energy_drift_rel_drop_scale = -0.901965
- max_rel_err_static_force_law_first_cycle = 0.062666
- max_abs_x_drift = 4.958727e-17
- max_abs_vx = 1.105604e-15
- max_body_moment_norm_first_cycle = 6.626285e-18

## flat_punch_centered_free_contact_release
- first_contact_time = 0.045000
- first_release_time = 0.095000
- analytic_delta_max = 0.004943
- sim_delta_max_first_cycle = 0.004263
- rel_err_delta_max = 0.137622
- touch_speed_analytic = 0.442945
- release_speed_sim = 0.088492
- rel_err_release_speed = 0.800219
- release_energy_drift_rel_drop_scale = -0.917829
- max_rel_err_static_force_law_first_cycle = 0.275754
- max_abs_x_drift = 1.973970e-17
- max_abs_vx = 5.099649e-16
- max_body_moment_norm_first_cycle = 6.031907e-17

## sphere_offaxis_free_contact_release
- first_contact_time = 0.075000
- first_release_time = 0.135000
- analytic_delta_max = 0.020148
- sim_delta_max_first_cycle = 0.017631
- rel_err_delta_max = 0.124900
- touch_speed_analytic = 0.626418
- release_speed_sim = 0.366025
- rel_err_release_speed = 0.415687
- release_energy_drift_rel_drop_scale = -0.901965
- max_rel_err_static_force_law_first_cycle = 0.062666
- max_rel_ground_Mz_relation_first_cycle = 0.062666
- max_abs_x_drift = 7.632783e-17
- max_abs_vx = 1.107877e-15
- max_body_moment_norm_first_cycle = 9.126155e-18
