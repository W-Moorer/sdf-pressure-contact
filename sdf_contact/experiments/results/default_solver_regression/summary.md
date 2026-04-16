# Default solver regression summary

The default solver is now the tuned midpoint/central/predictor configuration with adaptive substepping enabled.

Compared against the previous fixed-step tuned diagnostic baseline on the free-body release benchmarks:

## sphere_centered_free_contact_release
- old peak delta = 0.017631
- new peak delta = 0.020719
- old min energy drift = -0.009416
- new min energy drift = 0.000000
- new mean solver substeps = 1.000
- new max solver substeps = 1
- new max solver residual = 2.135e-10

## flat_punch_centered_free_contact_release
- old peak delta = 0.004263
- new peak delta = 0.005944
- old min energy drift = -0.004622
- new min energy drift = -0.000105
- new mean solver substeps = 1.000
- new max solver substeps = 1
- new max solver residual = 7.545e-09

## sphere_offaxis_free_contact_release
- old peak delta = 0.017631
- new peak delta = 0.020719
- old min energy drift = -0.009416
- new min energy drift = 0.000000
- new mean solver substeps = 1.000
- new max solver substeps = 1
- new max solver residual = 2.135e-10
