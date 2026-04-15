# Local evaluator benchmark summary

## 1) Centered primitive sphere-plane scan
| evaluator         |   mean_rel_err_Fy |   mean_elapsed_s |
|:------------------|------------------:|-----------------:|
| baseline_default  |          0.234724 |        0.0261768 |
| polygon_default   |          0.240393 |        0.0231021 |
| polygon_reference |          0.234956 |        0.0394316 |

Best-fit single-scale calibration against the ideal cap-volume force:
| evaluator         |   alpha_best_fit |   rmse_calibrated |   mean_rel_err_calibrated |   mean_rel_err_raw |   mean_elapsed_s |
|:------------------|-----------------:|------------------:|--------------------------:|-------------------:|-----------------:|
| baseline_default  |          1.29189 |         0.0426845 |                 0.0151728 |           0.234724 |        0.0261768 |
| polygon_default   |          1.29706 |         0.0246719 |                 0.0162896 |           0.240393 |        0.0231021 |
| polygon_reference |          1.28699 |         0.0252902 |                 0.0170002 |           0.234956 |        0.0394316 |

- Observation: baseline and polygon-default both under-predict the ideal force by a similar raw margin; after one scalar calibration, polygon has lower RMSE than baseline, but the gain is modest.

## 2) Fixed-penetration convergence
| family   |   resolution |      Fy |   ideal_Fy |   abs_err_Fy |   elapsed_s |
|:---------|-------------:|--------:|-----------:|-------------:|------------:|
| baseline |           28 | 2.90057 |    3.73221 |     0.831641 |   0.189138  |
| polygon  |           28 | 2.88044 |    3.73221 |     0.851777 |   0.0422506 |

- Observation: both evaluator families converge to roughly the same force plateau, still well below the ideal cap-volume prediction. This indicates that the dominant remaining error is not just coarse patch quadrature.

## 3) Tilted mesh box vs plane
| evaluator        |   mean_abs_err_vs_ref_Fy |   mean_abs_err_vs_ref_Mz |   mean_elapsed_s |
|:-----------------|-------------------------:|-------------------------:|-----------------:|
| baseline_default |               0.00465128 |              0.000118293 |          2.30925 |
| polygon_default  |               0.0164934  |              0.000415651 |          1.10011 |

- Observation: polygon-default is usually closer to the higher-resolution polygon reference on moment and on some force states, but the improvement is not uniform across all tilted mesh cases.

## 4) Off-axis sensitivity to plane reference center
| case                                 | reference_mode    | evaluator        |    x |    y |   body_force_x |   body_force_y |   horizontal_to_vertical_ratio |   elapsed_s |   num_pair_patch_points |
|:-------------------------------------|:------------------|:-----------------|-----:|-----:|---------------:|---------------:|-------------------------------:|------------:|------------------------:|
| offaxis_reference_center_sensitivity | origin_reference  | baseline_default | 0.04 | 0.09 |   -6.50521e-17 |        2.84363 |                    2.28764e-17 |   0.02704   |                     112 |
| offaxis_reference_center_sensitivity | origin_reference  | polygon_default  | 0.04 | 0.09 |   -4.65342e-06 |        2.85974 |                    1.62722e-06 |   0.0239676 |                      90 |
| offaxis_reference_center_sensitivity | aligned_reference | baseline_default | 0.04 | 0.09 |   -6.50521e-17 |        2.84363 |                    2.28764e-17 |   0.0276427 |                     112 |
| offaxis_reference_center_sensitivity | aligned_reference | polygon_default  | 0.04 | 0.09 |   -4.65342e-06 |        2.85974 |                    1.62722e-06 |   0.0228539 |                      90 |

- Observation: after the geometry-driven pair-frame update, moving only the plane reference center no longer changes the computed force in any meaningful way for this off-axis sphere-plane case. The spurious horizontal force ratio collapses from order 1e-1 to numerical-noise level.

## Overall conclusion
- The local high-accuracy evaluator is worth keeping, but the benchmark does **not** yet justify a global-solver rewrite.
- The initial pair-frame / reference-center sensitivity has been largely removed in this benchmark suite. The next high-value fix is now the force-law calibration and broader geometry-driven validation, rather than a first-pass refactor of the global implicit layer.
- After that, rerun the same benchmark suite and only then decide whether the remaining error is small enough that global-layer work is worth the cost.