# HID/Diginetica SOTA audit

## Verdict

The current method is **not literature-wide SOTA** and must not be described as
such. The strongest local MGCOT-MPS run is now clearly above HID but remains
below published MGCOT. No venue can provide a 100% acceptance guarantee.

## Exact protocol

- Data: the official `diginetica-2` artifacts distributed by
  [Code4HID](https://github.com/jarviswww/Code4HID).
- Train: 719,470 expanded examples reconstructed from 186,670 sessions.
- Test: 60,858 official examples, untouched during selection.
- Model selection: deterministic 5,000-query leave-last-out validation split.
- Metrics: the zero-based score-index convention in the official HID code.
- Published comparator: GCE-GNN+HID in Table 1 of the
  [AAAI 2026 HID paper](https://arxiv.org/abs/2511.08378).

## Best validation-selected result

| Metric | CEARF + Graph-NARM | HID | Difference |
|---|---:|---:|---:|
| HR@20 | **0.54703** | 0.54220 | **+0.00483** |
| MRR@20 | 0.19115 | 0.19180 | -0.00065 |
| tHR@20 | **0.52385** | 0.51830 | **+0.00555** |
| tMRR@20 | 0.18342 | 0.18370 | -0.00028 |
| tCov@20 | 0.90788 | 0.94210 | -0.03422 |
| Tail@20 | 0.68105 | 0.46670 | +0.21435 |

Artifact: `cearf_graph_narm_hid_results.json`.

This supports a positive HR/tHR result against HID, not a general SOTA claim.
MRR, tail MRR, and tail coverage remain below HID, and MGCOT reports a much
higher 0.6831 HR / 0.2979 MRR on byte-identical public artifacts.

## Root cause found and fixed

The first official run incorrectly filtered items already observed in the
session. The HID next-item protocol permits repeated items: 12,484 of 60,858
test targets (20.51%) already occur in their context, and 5,564 (9.14%) equal
the last context item. The mismatch also removed self-transitions.

Making repeat handling protocol-configurable raised CEARF HR@20 from 0.33547
to 0.49428 before neural fusion. Adding validation-selected recurrence rank
fusion raised the memory result to 0.50807 and the full model to 0.54151.

## Negative results retained

| Variant | Selection basis | Test HR@20 | Test MRR@20 |
|---|---|---:|---:|
| CEARF-N, coarse length router (best) | validation | **0.54151** | **0.18958** |
| Fine-beta length router | validation | 0.53906 | 0.18928 |
| Head/tail contextual router | validation | 0.53963 | 0.18887 |
| OOF listwise ranker | 5-fold OOF | 0.53993 | 0.19156 |
| OOF-selected RRF/LTR blend | 5-fold OOF | 0.53922 | 0.18929 |

Fine routing improved validation but not test, so neither variant should be
cherry-picked. In-batch multi-positive InfoNCE was also tested on validation;
it improved early recall for long sessions but reduced R@20 and was not
promoted to the official test.

The listwise ranker exposed a useful ceiling: the union of the two retrievers
contains the target for 84.38% of test queries, but the learned ranker does not
generalize well enough to select the correct top 20. It improves MRR, tMRR and
tCov relative to fixed RRF, but not total HR. The OOF-selected blend also
improved validation and degraded test, so it is retained as a negative result.

## Strong-backbone reproduction audit

The public MGCOT repository contains the same 719,470/60,858 Diginetica
artifacts and item range as Code4HID, making it a valid same-split comparator.
The Apple-MPS blocker was removed without densifying the graph: COO `A @ H`
was rewritten as algebraically equivalent edge-index `index_add_`, the
hard-coded CUDA/NumPy relation path was made tensor-native, and batch overlap
was vectorized with sparse incidence multiplication. Forward, backward,
entmax, full-softmax, checkpointing, and evaluation all pass end-to-end MPS
smokes. The port lives in `reference_repos/MGCOT`.

### MPS backbone and transfer observations

| Run | HR@20 | MRR@20 | Status |
|---|---:|---:|---|
| NARM, 4 epochs | 0.51921 | 0.18048 | full test |
| NARM, 8 epochs | 0.52972 | 0.18238 | validation-gated resume |
| Graph-NARM | 0.53705 | 0.18585 | validation-gated graph transfer |
| CEARF + Graph-NARM | **0.54703** | 0.19115 | validation-selected fusion |
| MGCOT-MPS + Graph-NARM warm start, 1 epoch, no CL | 0.60640 | **0.23179** | exploratory |
| MGCOT-MPS continuous epoch 2, no CL | **0.61057** | 0.22810 | exploratory |
| MGCOT-MPS, nested-selected 2-epoch low-LR fine-tune | **0.62905** | **0.24223** | strongest local run |
| Published MGCOT | **0.68310** | **0.29790** | paper result |

The transfer runs are explicitly exploratory: multiple configurations were
observed on the official test and therefore cannot be promoted to final paper
numbers. They nevertheless establish feasibility and a large positive effect
from train-only Graph-NARM item transfer. A continuous second epoch raises
HR mainly at ranks 11--20 while top-5 and MRR stagnate; the next root cause is
top-rank calibration, not insufficient epochs.

### Locked low-learning-rate continuation

Resetting the optimizer at learning rate 1e-3 caused catastrophic forgetting
on nested validation (HR/MRR fell from 0.4964/0.1860 to 0.4076/0.1363). A
lower-rate schedule selected on the nested split (1e-4 then 5e-5) instead
improved nested validation monotonically to 0.5196/0.1979. The same two-epoch
schedule was then applied to the full checkpoint in train-only mode, with no
intermediate test evaluation, and the final checkpoint was evaluated once:

| Metric | Before fine-tune | Final checkpoint | Change |
|---|---:|---:|---:|
| HR@20 | 0.60640 | **0.62905** | +0.02266 |
| MRR@20 | 0.23179 | **0.24223** | +0.01044 |
| tHR@20 | 0.57889 | **0.60416** | +0.02527 |
| tMRR@20 | 0.22075 | **0.23254** | +0.01180 |
| candidate recall@120 | 0.82495 | **0.84130** | +0.01635 |

Artifacts: `mgcot_mps_full_ft1e4_2ep.pt`,
`mgcot_full_ft1e4_test_top120.npz`, and
`mgcot_full_ft1e4_results.json`. The nested validation adjacency was inherited
from the public full MGCOT graph, so absence of held-target graph provenance
cannot be proven. This and prior observations of the official test keep the
result exploratory rather than a clean final-paper estimate.

### Residual reranking negative result

A fixed-candidate 18-feature residual reranker passed five-fold OOF validation
(0.5196/0.1979 to 0.5710/0.2184) but degraded the official test to
0.5752/0.2074. The cause is an expert-order reversal: CEARF/Graph-NARM/MGIR is
stronger than the under-trained nested MGCOT on validation, while full MGCOT is
stronger on test. Entropy-gated selective reranking reduced the degradation to
0.6052/0.2309 but did not turn it positive, so both variants are rejected.
Artifacts: `mgcot_residual_ft_hid_results.json` and
`mgcot_safe_residual_hid_results.json`.

## What is still required for a defensible SOTA paper

1. Reproduce or port at least one strong neural backbone under the exact HID
   artifact, preferably GCE-GNN+HID, and add CEARF-N as a module rather than
   comparing only against a published aggregate.
2. Run at least five seeds and report mean, standard deviation, and paired
   bootstrap confidence intervals from saved per-query predictions.
3. Re-run the transfer method with a train-only nested validation split,
   optimizer-state checkpoints, at least five seeds, and a single locked test
   evaluation. Then validate on the other official HID datasets and at least
   one additional public SBR protocol.
4. Include candidate-generation recall, head/tail calibration, latency,
   parameter count, and ablations for repeat policy, recurrence fusion,
   memory, neural residual, and router.
5. Frame the current contribution as protocol-aware cross-evidence fusion with
   a tail-accuracy result, not as a digital twin or a guaranteed-SOTA system.
