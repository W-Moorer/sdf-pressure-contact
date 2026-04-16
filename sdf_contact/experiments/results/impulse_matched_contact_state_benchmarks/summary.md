# Impulse-matched equivalent contact state first-contact benchmarks

This diagnostic upgrades the equivalent-state idea from a fixed midpoint state to an impulse-matched equivalent contact state. The solver locates onset, reconstructs the active subinterval, computes the average contact force over that subinterval, then searches along the onset-to-end state path for a static-contact state whose static contact force best matches that average-force impulse target.

## Main result

The impulse-matched equivalent state is successfully reconstructed, but in these first-contact tests it does **not** outperform the existing coarse step-end sample. In both the sphere and flat-punch cases, the best match to the analytic static curve remains the coarse step-end force.

## sphere_centered
- step_end_rel_err = 0.052648
- onset_aligned_rel_err = 0.069688
- onset_equivalent_state_rel_err = 0.069688
- onset_impulse_matched_state_rel_err = 0.070110
- impulse_match_alpha = 0.489258

## flat_punch_centered
- step_end_rel_err = 0.014345
- onset_aligned_rel_err = 0.043619
- onset_equivalent_state_rel_err = 0.043619
- onset_impulse_matched_state_rel_err = 0.055442
- impulse_match_alpha = 0.560547
