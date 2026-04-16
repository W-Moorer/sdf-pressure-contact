# Early-active interval force reconstruction result

This benchmark compares two reporting schemes on the same dynamic trajectory update:
- **baseline**: ordinary step-end contact force
- **active_interval**: onset-aware active-interval averaged reported force on the onset step

The dynamic state update is unchanged; only the reported onset-step force is reconstructed differently.

## sphere_centered
### baseline
- first_active_time = 0.075000
- first_active_rel_err = 0.052648
- mean_rel_err_active = 0.028120
- max_rel_err_active = 0.052648
- n_active_steps = 3

### active_interval
- first_active_time = 0.075000
- first_active_rel_err = 0.202550
- mean_rel_err_active = 0.078087
- max_rel_err_active = 0.202550
- n_active_steps = 3

## flat_punch_centered
### baseline
- first_active_time = 0.050000
- first_active_rel_err = 0.014345
- mean_rel_err_active = 0.014345
- max_rel_err_active = 0.014345
- n_active_steps = 1

### active_interval
- first_active_time = 0.050000
- first_active_rel_err = 0.726778
- mean_rel_err_active = 0.726778
- max_rel_err_active = 0.726778
- n_active_steps = 1

## Conclusion
For the current implementation, onset-aware active-interval averaging does **not** pull the curve back to theory.
- Sphere: the first active-step error gets worse (about 5.26% -> 20.26%).
- Flat punch: the first active-step error gets much worse (about 1.43% -> 72.68%).
This means the current active-interval reconstruction is averaging over the onset remainder in a way that underestimates the equivalent instantaneous force needed for comparison with the static theory curve.