# Free-body slow contact and release benchmarks

This benchmark layer is the first *true dynamics* check for the endpoint-style formal contact stack.
It is intentionally different from the earlier driven quasi-static scans:

- the body is a free rigid body,
- gravity is active,
- there is no kinematic driving after initialization,
- the evaluator and solver configuration are shared across all cases,
- no case-specific contact-law branch is introduced.

## What these tests measure

For the **first contact-release cycle** only, the benchmark reports:

1. **Turning-point penetration**
   - Compare the simulated maximum penetration to the conservative analytic prediction.
2. **First release speed**
   - Compare the simulated first release speed to the conservative analytic touch speed.
3. **Energy drift at first release**
   - Total energy error normalized by the initial drop-energy scale.
4. **Static force-law consistency during contact**
   - Check whether the instantaneous contact force still follows the same static analytic law at the same penetration depth.
5. **Symmetry / pollution diagnostics**
   - Off-axis cases track spurious x drift, vx drift, body moment pollution, and ground moment consistency.

## Shared configuration

These runs use one shared, untuned configuration:

- `patch raster_cells = 4`
- `sheet bisection_steps = 8`
- `stiffness per side = 40000`
- `damping_gamma = 0`
- `dt = 0.005`
- `gravity = 9.81`

## How to interpret the current results

The current endpoint evaluator + implicit solver stack shows a clear split:

- **Good:** during the first contact cycle, the instantaneous force law still tracks the static analytic law reasonably well, especially for the sphere cases.
- **Not yet good:** the first release speed and total-energy recovery are poor, meaning the current global time stepping is still strongly dissipative even with zero explicit damping.

So these benchmarks are primarily a **global dynamics debugging tool**:

- local contact geometry / force law are no longer the main bottleneck,
- the remaining issue is the free-body time integration / implicit solve layer.
