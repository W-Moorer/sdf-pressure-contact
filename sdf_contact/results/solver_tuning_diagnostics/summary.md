# Global implicit solver tuning diagnostics

This report compares the previous free-body solver configuration against a tuned configuration that changes only the global solver layer:

- legacy: backward Euler + forward-difference Jacobian + current-velocity predictor
- tuned: implicit midpoint kinematics/force evaluation + central-difference Jacobian + explicit-force predictor

Shared contact evaluator/configuration is unchanged across all cases.

## sphere_centered
- release speed error: legacy 0.4157 -> tuned 0.0943
- release energy drift / drop scale: legacy -0.9020 -> tuned 0.0310
- first-cycle delta error: legacy 0.1249 -> tuned 0.0284
- first-cycle max static-force-law error: legacy 0.0627 -> tuned 0.0672

## flat_punch_centered
- release speed error: legacy 0.8002 -> tuned 0.0257
- release energy drift / drop scale: legacy -0.9178 -> tuned -0.0215
- first-cycle delta error: legacy 0.1376 -> tuned 0.2025
- first-cycle max static-force-law error: legacy 0.2758 -> tuned 0.2769

## sphere_offaxis
- release speed error: legacy 0.4157 -> tuned 0.0943
- release energy drift / drop scale: legacy -0.9020 -> tuned 0.0310
- first-cycle delta error: legacy 0.1249 -> tuned 0.0284
- first-cycle max static-force-law error: legacy 0.0627 -> tuned 0.0672
- off-axis ground Mz relation max error: legacy 0.0627 -> tuned 0.0672

## Linearization / Jacobian diagnostics near first contact
### legacy
- scheme = backward_euler
- predictor_mode = current_velocity
- predictor residual norm = 5.007892e-04
- Jacobian condition number = 1.738273e+02
- sigma_min / sigma_max = 2.880000e-04 / 5.006226e-02
- relative gap between forward and central FD Jacobians = 2.243769e-08
- linearized residual ratio with forward FD = 1.039539e-14
- linearized residual ratio with central FD = 3.471887e-15

### tuned
- scheme = implicit_midpoint
- predictor_mode = explicit_force
- predictor residual norm = 1.734723e-18
- Jacobian condition number = 1.736111e+02
- sigma_min / sigma_max = 2.880000e-04 / 5.000000e-02
- relative gap between forward and central FD Jacobians = 5.007632e-14
- linearized residual ratio with forward FD = 0.000000e+00
- linearized residual ratio with central FD = 0.000000e+00
