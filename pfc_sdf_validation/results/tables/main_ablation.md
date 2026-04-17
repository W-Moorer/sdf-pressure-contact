# Main long-horizon ablation

| Benchmark        | Scheme                               |   Peak force error |   Impulse error |   Energy drift |   Release timing error |   Max penetration error |   Rebound velocity error |
|:-----------------|:-------------------------------------|-------------------:|----------------:|---------------:|-----------------------:|------------------------:|-------------------------:|
| Analytic flat    | event-aware midpoint                 |          0.017794  |        0.002377 |      0.013595  |              0.005424  |               0.007973  |                0.013504  |
| Analytic flat    | + impulse correction                 |          0.008525  |        0.020615 |     -0.003337  |              0.001849  |               0.008177  |                0.003343  |
| Analytic flat    | + work consistency                   |          0.0006591 |        0.008664 |      0.001132  |              0.0007424 |               0.0006591 |                0.001131  |
| Analytic flat    | + consistent traction reconstruction |          0.0006591 |        0.008664 |      0.001132  |              0.0007424 |               0.0006591 |                0.001131  |
| Analytic sphere  | event-aware midpoint                 |          0.003818  |        0.001034 |      0.001632  |              0.004601  |               0.0006597 |                0.001631  |
| Analytic sphere  | + impulse correction                 |          0.003252  |        0.002735 |     -0.0003596 |              0.0008314 |               0.001761  |                0.0003597 |
| Analytic sphere  | + work consistency                   |          0.001686  |        0.001966 |     -7.514e-05 |              0.0008206 |               0.0009179 |                7.514e-05 |
| Analytic sphere  | + consistent traction reconstruction |          0.001686  |        0.001966 |     -7.514e-05 |              0.0008206 |               0.0009179 |                7.514e-05 |
| Native-band flat | event-aware midpoint                 |          0.368463  |        0.026318 |      0.059124  |              0.093223  |               0.175832  |                0.057473  |
| Native-band flat | + impulse correction                 |          0.181206  |        0.063514 |     -0.007443  |              0.08287   |               0.242781  |                0.007471  |
| Native-band flat | + work consistency                   |          0.198292  |        0.030332 |      0.006026  |              0.075034  |               0.233093  |                0.006008  |
| Native-band flat | + consistent traction reconstruction |          0.198292  |        0.030332 |      0.006026  |              0.075034  |               0.233093  |                0.006008  |
