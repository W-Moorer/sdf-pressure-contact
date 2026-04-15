# Benchmark acceptance report

This report evaluates the current benchmark outputs against the admission profiles defined in `docs/BENCHMARK_ACCEPTANCE.md`.

## Overview
| evaluator         | tier           | research_pass   | production_pass   |
|:------------------|:---------------|:----------------|:------------------|
| baseline_default  | research       | True            | False             |
| polygon_default   | research       | True            | False             |
| polygon_reference | reference_only | False           | False             |

## Raw metrics
| evaluator         |   max_centered_moment_ratio |   mean_rel_err_raw |   mean_rel_err_calibrated |   tail_spread_rel |   last_step_rel_change |   mean_rel_err_vs_ref_Fy |   max_rel_err_vs_ref_Fy |   mean_rel_err_vs_ref_Mz |   max_rel_err_vs_ref_Mz |   max_horizontal_to_vertical_ratio |   fy_reference_mode_rel_diff |
|:------------------|----------------------------:|-------------------:|--------------------------:|------------------:|-----------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|-----------------------------------:|-----------------------------:|
| baseline_default  |                 5.56172e-18 |           0.234724 |                 0.0151728 |        0.0135254  |            0.00842611  |               0.00630857 |               0.0180854 |               0.00648373 |               0.0183039 |                        2.28764e-17 |                            0 |
| polygon_default   |                 3.12767e-05 |           0.240393 |                 0.0162896 |        0.00164221 |            0.000610958 |               0.0219132  |               0.0650283 |               0.0218968  |               0.0649131 |                        1.62722e-06 |                            0 |
| polygon_reference |                 2.44014e-05 |           0.234956 |                 0.0170002 |      nan          |          nan           |             nan          |             nan         |             nan          |             nan         |                      nan           |                          nan |

## Profile thresholds
### Research
|   max_centered_moment_ratio |   tail_spread_rel |   last_step_rel_change |   max_horizontal_to_vertical_ratio |   fy_reference_mode_rel_diff |   mean_rel_err_calibrated |   mean_rel_err_vs_ref_Fy |   max_rel_err_vs_ref_Fy |   mean_rel_err_vs_ref_Mz |   max_rel_err_vs_ref_Mz |
|----------------------------:|------------------:|-----------------------:|-----------------------------------:|-----------------------------:|--------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|
|                       0.001 |              0.02 |                   0.02 |                             0.0001 |                       0.0001 |                      0.02 |                     0.03 |                    0.08 |                     0.03 |                    0.08 |

### Production
|   max_centered_moment_ratio |   tail_spread_rel |   last_step_rel_change |   max_horizontal_to_vertical_ratio |   fy_reference_mode_rel_diff |   mean_rel_err_raw |   mean_rel_err_calibrated |   mean_rel_err_vs_ref_Fy |   max_rel_err_vs_ref_Fy |   mean_rel_err_vs_ref_Mz |   max_rel_err_vs_ref_Mz |
|----------------------------:|------------------:|-----------------------:|-----------------------------------:|-----------------------------:|-------------------:|--------------------------:|-------------------------:|------------------------:|-------------------------:|------------------------:|
|                      0.0001 |             0.005 |                  0.005 |                              1e-05 |                        1e-05 |                0.1 |                      0.01 |                     0.01 |                    0.03 |                     0.01 |                    0.03 |

## Evaluator: `baseline_default`
- Tier: **research**
- Research pass: **True**
- Production pass: **False**

- Failed research checks: none
- Failed production checks: tail_spread_rel, last_step_rel_change, mean_rel_err_raw, mean_rel_err_calibrated

- Interpretation: this evaluator is accurate enough for research iteration, but still not calibrated or consistent enough to justify production assumptions.

## Evaluator: `polygon_default`
- Tier: **research**
- Research pass: **True**
- Production pass: **False**

- Failed research checks: none
- Failed production checks: mean_rel_err_raw, mean_rel_err_calibrated, mean_rel_err_vs_ref_Fy, max_rel_err_vs_ref_Fy, mean_rel_err_vs_ref_Mz, max_rel_err_vs_ref_Mz

- Interpretation: this evaluator is accurate enough for research iteration, but still not calibrated or consistent enough to justify production assumptions.

## Evaluator: `polygon_reference`
- Tier: **reference_only**
- Research pass: **False**
- Production pass: **False**

- Failed research checks: none
- Failed production checks: none

- Interpretation: this entry is a numerical reference configuration, not a candidate production evaluator, so profile gating is not applied to it.

## Current strategic interpretation
- If an evaluator is below the research profile, do not refactor the global solver yet.
- If an evaluator passes research but fails production, keep improving force-law calibration and general-mesh validation before major global-layer work.
- Only when at least one evaluator passes production with margin should the remaining error budget be attributed mainly to the global layer.
