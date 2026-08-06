# Rental cross-domain audit

## Verdict

The Diginetica result cannot be presented as a Rental result. MGCOT-MPS was
therefore ported and evaluated separately on the competition-faithful masked
Rental leave-one-out protocol. The port is a positive neural result but is not
the strongest local Rental method. Train-only CEARF is the strongest method in
this controlled comparison. None of these offline numbers establishes a
literature-wide or Kaggle-leaderboard SOTA claim.

## Protocol

- Data: `rental_intent_bench/split_loo_masked`.
- Catalog: 627 products plus padding.
- Test: 2,557 leave-one-out rental queries; previously rented items masked.
- Nested validation: 1,347 targets held from training prefixes.
- Graph: directed adjacency built only from nested/full training sequences.
- Selection: native contrastive weight and epoch chosen on nested HR@6/MRR@6.
- Final evaluation: locked 13-epoch `cw=0` checkpoint evaluated once.

## Results

| Method | Recall@6 | MRR@6 | Status |
|---|---:|---:|---|
| MostPop | 0.22918 | — | local comparator |
| ItemKNN | 0.23074 | — | local comparator |
| MGCOT-MPS | 0.23817 | 0.17837 | positive, not best |
| DT-RRF | 0.26875 | — | prior local result |
| SKNN | 0.26946 | — | prior local result |
| MGCOT–CEARF | 0.28080 | 0.19985 | validation-selected fusion |
| **CEARF** | **0.28471** | **0.20641** | strongest controlled local result |

MGCOT's masked candidate recall@120 is 0.53109; the MGCOT–CEARF union reaches
0.54478. The nested-selected fusion improves over MGCOT but CEARF alone remains
better on the locked test, so MGCOT should not be claimed as the source of the
Rental gain.

Artifacts:

- `mgcot_rental_results.json`
- `mgcot_cearf_rental_results.json`
- `mgcot_rental_full_13ep.pt`
- `mgcot_rental_full_top120.npz`

## Kaggle candidate

`kaggle_submission_cearf_dualtwin_candidate.csv` is an experimental,
schema-validated candidate. It conservatively uses the known DualTwin-MANF
submission as the primary ranking and session-memory consensus as a secondary
view. Its leaderboard score is unknown; it must not replace the known 0.44328
submission as the reported best until Kaggle evaluates it.
