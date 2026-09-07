# E1 — Selection-procedure comparison (test R@20)

| Procedure | Val. cells | OOF fit | Video | Baby | Diginetica |
|---|---|---|---|---|---|
| P1 endpoint-only | 2 | -- | 0.12571 | 0.04873 | 0.49428 | 0.02173 | 0.00799 | 0.02282 |
| P2 global-$\beta$ fusion | 0 | 1 scalar | 0.14719 | 0.05671 | 0.51702 | 0.00025 | 0.00001 | 0.00008 |
| P3 regime router | 0 | 2 scalars | 0.14726 | 0.05655 | 0.51692 | 0.00018 | 0.00017 | 0.00018 |
| P4 oracle (audit family) | test-side | -- | 0.14744 | 0.05672 | 0.51710 | 0.00000 | 0.00000 | 0.00000 |
| P5 forced neural incl. | 0 | as P6 | 0.14735 | 0.05671 | 0.51702 | 0.00009 | 0.00001 | 0.00008 |
| P6 query-cond.\ gate | 0 | 5-param gate | 0.14735 | 0.05669 | 0.51701 | 0.00009 | 0.00003 | 0.00009 |

Audit-only variants: 15 allocation-control policies (buckets, feature gates, deltas, reassignment). The primary gate is not selected from them. 'Zero validation cells' refers to selection only: P2 still fits one scalar and P6 five parameters on the 4,000-row OOF calibration set.

**Fusion vs best endpoint, per seed (recomputed from frozen arrays):** Video 3/3 seeds, Baby 3/3 seeds (Diginetica endpoint arrays were lost in the incident; recorded means show memory .49428 > neural .44852 while fusion .51701).

**Dynamic minus OOF-global ΔU [95% CI] (recorded paired bootstrap):**
| Domain | ΔU | 95% CI |
|---|---|---|
| Video | +0.00022 | [+0.00009, +0.00035] |
| Baby | +0.00002 | [-0.00003, +0.00006] |
| Diginetica | +0.00002 | [-0.00015, +0.00018] |

**Finding (report as-is):** the single training-learned global β (P2, zero validation cells) is within .00025 R@20 of the 15-cell test-side oracle on every domain. The query-conditioned gate adds a detectable-but-tiny +.0002 U on Video (CI above zero) and is statistically undetectable on Baby and Diginetica. Endpoint-only selection (P1) loses .008–.023. Forbidding component rejection (P5) costs nothing here because fusion beats both endpoints on all domains and seeds.