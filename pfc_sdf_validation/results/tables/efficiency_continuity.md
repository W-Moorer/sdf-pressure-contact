# Continuity efficiency

| Variant          | Scheme             |   Runtime (s) |   Force error |   Impulse error |   Energy drift |   Mean candidate cells |   Mean recompute cells |   Mean pred./corr. Jaccard |   Mean substeps |
|:-----------------|:-------------------|--------------:|--------------:|----------------:|---------------:|-----------------------:|-----------------------:|---------------------------:|----------------:|
| Dense baseline   | + work consistency |       5.86523 |             0 |        0.030332 |       0.006026 |                286.364 |                286.364 |                   0.747619 |               1 |
| Continuity-aware | + work consistency |       5.40942 |             0 |        0.030332 |       0.006026 |                122.727 |                106.818 |                   0.86     |               1 |
