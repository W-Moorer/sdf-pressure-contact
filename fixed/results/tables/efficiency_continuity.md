# Continuity efficiency

| Variant          | Scheme             |   Runtime (s) |   Force error |   Impulse error |   Energy drift |   Mean candidate cells |   Mean recompute cells |   Mean pred./corr. Jaccard |   Mean substeps |
|:-----------------|:-------------------|--------------:|--------------:|----------------:|---------------:|-----------------------:|-----------------------:|---------------------------:|----------------:|
| Dense baseline   | + work consistency |       3.92819 |             0 |        0.007236 |        -0.0048 |                203.636 |                203.636 |                   0.9      |               1 |
| Continuity-aware | + work consistency |       4.11475 |             0 |        0.007236 |        -0.0048 |                116.364 |                101.818 |                   0.933333 |               1 |
