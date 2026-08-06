# Temporary Ranking Reversals in State-Space Session Recommendation

## Abstract

Recent state-space recommenders motivate the hypothesis that selective
state-space encoders become more competitive as the amount of interaction data
increases. We test this hypothesis under a controlled session-recommendation
protocol on the public Diginetica dataset. GRU4Rec and a local pure-PyTorch
selective-SSM proxy share the same item embeddings, prediction head, loss,
validation procedure, and training budget. We evaluate four training sizes
(10K, 12K, 15K, and 30K sessions), three independently sampled training draws,
and three model-initialization seeds. Query-level paired bootstrap intervals
are computed separately within each fixed draw. The SSM proxy has higher mean
Recall@20 on all three draws at 15K, but the differences are small and their
confidence intervals include zero. At 30K, it wins one draw and loses two; its
overall mean Recall@20 is 0.0987 compared with 0.1093 for GRU4Rec. Three of nine
30K SSM runs also converge to substantially lower-performing solutions. These
results show a temporary, draw-sensitive ranking reversal rather than a stable
scaling crossover. The study demonstrates why data-sampling and optimization
uncertainty must be separated in sequential-recommender comparisons. The local
SSM proxy is not presented as a reproduction of Mamba4Rec or SIGMA, and no
state-of-the-art claim is made.

## 1. Introduction

Sequential recommenders are often compared on one fixed preprocessing split
and a small number of initialization seeds. This protocol can underestimate a
second source of uncertainty: which training sessions are sampled. The issue
is especially important for sparse, large-catalogue data, where changing the
training subset changes item coverage, transition counts, and the number of
examples available to the prediction head.

Selective state-space models (SSMs) provide a linear-time alternative to
attention and recurrence. Mamba4Rec and SIGMA show that SSM components can be
useful in sequential recommendation, but those complete architectures include
specific blocks and training choices that are unavailable in the present MPS
environment. We therefore evaluate a deliberately scoped local selective-SSM
proxy rather than describing it as an official reproduction.

Our research question is:

> Does the relative ranking of GRU4Rec and a local selective-SSM proxy persist
> across training sizes, independently sampled data draws, and model seeds?

The contributions are:

1. A factorial evaluation that separates data-draw randomness from
   initialization randomness.
2. Query-level paired inference conditional on each draw, together with
   cross-draw robustness and variance decomposition.
3. Evidence that the apparent SSM advantage at 15K is a temporary,
   non-monotonic sign reversal rather than a stable scaling crossover.
4. Audited manifests and retained failed seeds, preventing selection of only
   favorable SSM runs.

## 2. Related work

GRU4Rec introduced recurrent neural networks for session recommendation.
SASRec and BERT4Rec later established attention-based sequential models, while
SR-GNN represents session transitions as graphs. More recent work applies
selective state-space models to recommendation. Mamba4Rec uses Mamba blocks for
sequential recommendation, and SIGMA combines selective gating, PF-Mamba, and
feature-extracting GRU components.

Our work does not propose another complete recommender architecture. It is a
controlled reliability study inspired by these models. This distinction is
important: using a selective recurrence does not make the local proxy a
reproduction of Mamba4Rec or SIGMA.

Recommender-system research has repeatedly shown that weak baselines,
inconsistent protocols, and incomplete tuning can reverse conclusions. We
extend this reliability concern from model seeds to independently sampled
training subsets.

## 3. Models and protocol

### 3.1 Shared recommendation protocol

Both models use the same item vocabulary, sequence construction, item
embeddings, full-softmax prediction head, objective, batch construction,
validation queries, early-stopping logic, and maximum training budget. The
only intended difference is the sequence encoder.

GRU4Rec uses a gated recurrent unit. The SSM proxy uses an input-dependent
keep/write recurrence followed by a gated projection. A diagnostic
`contractive_ssm` variant bounds the write and output gates; it is analyzed as
an optimization control and is not proposed as a new method.

### 3.2 Dataset and training subsets

We use the public Diginetica session dataset. Training subsets contain 10,000,
12,000, 15,000, or 30,000 sessions. For every size, three independent draws
are generated with seeds 42, 123, and 456. Within each fixed draw, both models
are trained with initialization seeds 42, 123, and 456. The primary design is
therefore:

\[
4\ \text{sizes}\times3\ \text{draws}\times3\ \text{model seeds}
\times2\ \text{encoders}=72\ \text{runs}.
\]

Historical diagnostic artifacts are audited separately and are not treated as
additional observations in the primary factorial inference.

### 3.3 Metrics and uncertainty

We report Recall@20 and NDCG@20. Predictions are first averaged across model
seeds within one data draw. Paired bootstrap resampling is then performed over
the fixed draw's test queries. These intervals quantify query uncertainty
conditional on that data draw; they do not represent uncertainty across data
draws. Cross-draw robustness is assessed separately from the three draw-level
effects. We also decompose observed variation into between-draw and within-draw
model-seed components.

## 4. Results

### 4.1 Draw-level scaling results

Each cell below is the mean Recall@20 over model seeds 42, 123, and 456 for one
fixed training-data draw.

| Sessions | Draw 42: GRU | Draw 42: SSM | Draw 123: GRU | Draw 123: SSM | Draw 456: GRU | Draw 456: SSM |
|---:|---:|---:|---:|---:|---:|---:|
| 10K | .0702 | .0548 | .0747 | .0722 | .0669 | .0536 |
| 12K | .0758 | .0622 | .0800 | .0764 | .0773 | .0764 |
| 15K | .0858 | .0878 | .0904 | .0907 | .0847 | .0862 |
| 30K | .0984 | .0569 | .1118 | .1009 | .1178 | .1382 |

The aggregate view is:

| Sessions | GRU mean | SSM mean | SSM − GRU | SSM wins/draws |
|---:|---:|---:|---:|---:|
| 10K | .0706 | .0602 | -.0104 | 0/3 |
| 12K | .0777 | .0717 | -.0060 | 0/3 |
| 15K | .0870 | .0882 | +.0012 | 3/3 |
| 30K | .1093 | .0987 | -.0106 | 1/3 |

All three draws show a temporary sign reversal between 12K and 15K, but only
draw 456 remains positive at 30K. This non-monotonic behavior does not support
a stable scaling crossover.

### 4.2 Paired inference

At 15K, the SSM-minus-GRU Recall@20 differences are +.0020, +.0002, and
+.0016 for draws 42, 123, and 456. All three query-level paired bootstrap
intervals contain zero. At 30K, the differences are -.0416 (95% CI [-.0502,
-.0331]), -.0109 ([-.0189, -.0027]), and +.0204 ([+.0111, +.0298]). The
30K direction therefore depends on which training sessions are sampled.

These confidence intervals must not be pooled as though queries from different
training draws were independent replications of the model comparison. The
three draw-level estimates are the relevant evidence for robustness to data
sampling.

### 4.3 Optimization sensitivity

Six of nine 30K SSM runs reach Recall@20 of at least .10, whereas three converge
to solutions between .008 and .016. Their training histories show much earlier
best epochs and weaker validation improvement. This supports an optimization-
sensitivity observation, but not a claim about the global loss landscape.

The bounded contractive recurrence changes which initialization seeds succeed,
including one run near .155, but does not improve aggregate accuracy or
robustness. It is therefore retained as a negative diagnostic rather than a
proposed method.

## 5. Discussion

A single 15K draw would suggest that the SSM proxy has overtaken GRU4Rec. The
factorial experiment changes that interpretation: the advantage is small,
query-level inconclusive, and not sustained in two of three draws at 30K.
Reporting only the best SSM seed would hide an additional failure mode because
one third of the 30K initializations converge to substantially weaker
solutions.

The result does not imply that official Mamba4Rec or SIGMA architectures are
inferior to GRU4Rec. It applies only to the local proxy and shared protocol
evaluated here. Rather, it shows that evidence for an architecture-level
scaling claim requires both data-draw replication and model-seed replication.

## 6. Limitations

The primary study uses one public dataset and a fixed test set. The local
pure-PyTorch recurrence is not the complete Mamba4Rec or SIGMA architecture.
The current machine lacks CUDA, `mamba_ssm`, and `causal-conv1d`, so an official
kernel-backed comparison has not been run. A preliminary RetailRocket pipeline
check is retained in the artifact repository but excluded from scientific
comparisons because its capped training set and two-epoch budget are not
competitive. Full-scale official-model experiments on a second public dataset
remain necessary for broader external validity.

## 7. Reproducibility

The primary matrix contains 72 runs. The artifact audit currently covers 136
records because it also includes architecture ablations and contractive-SSM
diagnostics; those records are not counted as extra primary replications.

```bash
PYTHONPATH=. pytest -q \
  sparse_bench/scaling_study/tests/test_pipeline.py

PYTHONPATH=. python -m sparse_bench.scaling_study.cli audit \
  --results sparse_bench/scaling_study/artifacts/runs

PYTHONPATH=. python -m sparse_bench.scaling_study.cli analyze \
  --results sparse_bench/scaling_study/artifacts/runs \
  --out sparse_bench/scaling_study/artifacts/analysis \
  --bootstrap-samples 10000
```

The main outputs are `paired_inference.json`, `crossover.json`,
`variance_decomposition.json`, and `aggregated.csv` under
`sparse_bench/scaling_study/artifacts/analysis/`.

## References

Hidasi et al. Session-based Recommendations with Recurrent Neural Networks.
ICLR, 2016.

Kang and McAuley. Self-Attentive Sequential Recommendation. ICDM, 2018.

Wu et al. Session-Based Recommendation with Graph Neural Networks. AAAI, 2019.

Gu and Dao. Mamba: Linear-Time Sequence Modeling with Selective State Spaces.
COLM, 2024.

Liu et al. Mamba4Rec: Towards Efficient Sequential Recommendation with
Selective State Space Models. arXiv:2403.03900, 2024.

Liu et al. SIGMA: Selective Gated Mamba for Sequential Recommendation. AAAI,
2025.

Ferrari Dacrema et al. Are We Really Making Much Progress? A Warning on
Optimizing Recommender Systems. RecSys, 2019.
