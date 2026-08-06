# Fixed-allocation fusion-operator control

`run_dynamic_beta_fusion_control.py` compares CEARF-N's weighted reciprocal
rank fusion with normalized CombSUM while holding every upstream object fixed:

- the CEARF profile and component ranks;
- the PASGR checkpoint and full-catalogue top-120 ranks;
- the primary query-conditioned `beta_q`.

The script performs no fitting, beta search, validation selection, or
early-stopping. It writes a manifest after both rankings are frozen and before
it directly accesses the target arrays.

## Preconditions

Run the control only after `run_dynamic_beta.py` has completed all requested
domains and seeds. The default command requires all three canonical seeds:

```bash
cd sparse_bench
python run_dynamic_beta_fusion_control.py \
  Video_Games Baby_Products Diginetica_HID \
  --seeds 42 123 456
```

Do not run this command concurrently with the main dynamic-beta experiment.
It executes frozen-checkpoint full-catalogue PASGR inference to recover cosine
scores, so it competes for the same accelerator and memory.

The principal output is
`dynamic_beta_fusion_operator_control.json`. Per-seed frozen manifests,
top-20/rank vectors, and reusable PASGR score caches are written under
`dynamic_beta_fusion_operator_artifacts/`.

## Exact checks

For every query and seed, the script requires:

1. identical query order across memory, neural, and primary-beta artifacts;
2. exact equality between checkpoint-regenerated and persisted PASGR top-120
   item IDs;
3. exact reconstruction of the positive-score prefix of CEARF's final expert;
4. exact equality between recomputed weighted RRF and the persisted primary
   dynamic-beta top-20.

Any mismatch stops the run rather than silently evaluating a different model.
Manifests include SHA-256 hashes for checkpoints, inputs, score caches, beta
vectors, and both frozen top-20 outputs.

## Operator definition

For each expert independently, native scores within its top-120 are min-max
normalized per query. A candidate absent from an expert receives zero from
that expert. A zero-span score vector maps to all zeros because it carries no
within-expert preference. The fixed primary `beta_q` then gives

```text
(1 - beta_q) * minmax(CEARF final score)
  + beta_q * minmax(PASGR cosine score).
```

Ties are broken by ascending item ID, as in the primary cross-expert fusion.

## Scope and limitations

- The control's candidate set is the union of the two full-catalogue top-120
  expert outputs, not the entire item catalogue. Scores outside top-120 were
  not persisted.
- CEARF is already a rank-fused memory expert. Its final native score is the
  frozen profile-weighted component RRF score with the consensus multiplier;
  popularity backfill without an admitted-component score is assigned zero.
  This is not a comparison of pre-ranking transition/session score scales.
- CombSUM does not receive a separately optimized beta. This is intentional:
  the experiment isolates the fusion operator, not the allocation policy.
- PASGR retrieval uses cosine similarity. The recovered values are retrieval
  scores, not calibrated probabilities.

Run the focused tests with:

```bash
cd sparse_bench
python -m unittest test_dynamic_beta_fusion_control.py
```
