# Flat punch support-area prior benchmark rerun

This rerun isolates the new area-prior-controlled support polygon recovery for large flat support contact.

## Static flat punch with the same low-resolution evaluator settings used in free-body dynamics
- area_prior_off mean relative Fy error: 26.6548%
- area_prior_off max relative Fy error: 27.5802%
- area_prior_on mean relative Fy error: 0.3281%
- area_prior_on max relative Fy error: 1.3438%
- recovered mean sheet area, area_prior_off: 0.022022
- recovered mean sheet area, area_prior_on: 0.030000
- geometric support-area prior: 0.030000

## Free-body flat punch rerun after the patch
- rel_err_delta_max: 20.2499%
- max_rel_err_static_force_law_first_cycle: 27.6892%
- mean_rel_err_static_force_law_first_cycle: 26.6945%
- rel_err_release_speed: 2.5686%
- release_energy_drift_rel_drop_scale: -0.021466

## Reading
- The support-area prior fixes the support-measure layer very strongly in the static benchmark: the low-resolution flat-punch curve moves from roughly 26.7% mean force error to about 0.33% mean force error without changing the outer evaluator architecture.
- The free-body rerun does not improve by a comparable amount. The first-cycle force mismatch stays around 27.7%, even though the release-speed and energy metrics remain good.
- Therefore the next dominant error is no longer support polygon area recovery. The static curve now lines up with theory; the remaining mismatch in the free-body curve is primarily time-domain / event-resolution related, not footprint-area related.