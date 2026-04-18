# Continuity efficiency

| Variant          | Scheme             |   Runtime (s) |   Force error |   Impulse error |   Energy drift |   Mean candidate cells |   Mean recompute cells |   Mean pred./corr. Jaccard |   Mean substeps |
|:-----------------|:-------------------|--------------:|--------------:|----------------:|---------------:|-----------------------:|-----------------------:|---------------------------:|----------------:|
| Dense baseline   | + work consistency |       16.7939 |             0 |        0.000916 |       0.001551 |               228.571  |               228.571  |                          1 |               1 |
| Continuity-aware | + work consistency |       16.8721 |             0 |        0.000916 |       0.001551 |                94.8571 |                78.2857 |                          1 |               1 |
