# Main long-horizon ablation

| Benchmark        | Scheme                               |   Peak force error |   Impulse error |   Energy drift |   Release timing error |   Max penetration error |   Rebound velocity error |
|:-----------------|:-------------------------------------|-------------------:|----------------:|---------------:|-----------------------:|------------------------:|-------------------------:|
| Analytic flat    | event-aware midpoint                 |          0.001667  |       0.0002492 |      0.000341  |              0.0004593 |               0.0001045 |                0.0003409 |
| Analytic flat    | + impulse correction                 |          0.0008291 |       0.001149  |     -5.472e-05 |              4.667e-05 |               0.0008277 |                5.472e-05 |
| Analytic flat    | + work consistency                   |          0.0003757 |       0.000389  |      1.451e-05 |              6.494e-05 |               0.0003757 |                1.451e-05 |
| Analytic flat    | + consistent traction reconstruction |          0.0003757 |       0.000389  |      1.451e-05 |              6.494e-05 |               0.0003757 |                1.451e-05 |
| Analytic sphere  | event-aware midpoint                 |          0.0002447 |       8.341e-06 |      1.331e-05 |              0.0004893 |               4.079e-05 |                1.331e-05 |
| Analytic sphere  | + impulse correction                 |          0.0002664 |       0.0001332 |     -1.655e-06 |              0.0002977 |               0.000145  |                1.655e-06 |
| Analytic sphere  | + work consistency                   |          0.000204  |       2.271e-05 |      9.952e-07 |              0.00025   |               0.000111  |                9.952e-07 |
| Analytic sphere  | + consistent traction reconstruction |          0.000204  |       2.271e-05 |      9.952e-07 |              0.00025   |               0.000111  |                9.952e-07 |
| Native-band flat | event-aware midpoint                 |          0.014842  |       0.0009253 |      0.001415  |              8.431e-06 |               0.013669  |                0.001414  |
| Native-band flat | + impulse correction                 |          0.013235  |       0.0004417 |      0.001624  |              0.0004533 |               0.013236  |                0.001623  |
| Native-band flat | + work consistency                   |          0.013539  |       0.000916  |      0.001551  |              0.000552  |               0.013539  |                0.00155   |
| Native-band flat | + consistent traction reconstruction |          0.013539  |       0.000916  |      0.001551  |              0.000552  |               0.013539  |                0.00155   |
