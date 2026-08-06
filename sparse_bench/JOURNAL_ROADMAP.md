# CEARF-N journal extension roadmap

This roadmap is deliberately separate from the ADMA short-paper freeze. The
short paper establishes rejection-complete validation gating and a compact
system instantiation. The journal extension must add substantial theory,
datasets, controls, and metadata-confound analysis rather than merely reporting
more seeds of the same experiment.

## Target and positioning

- Primary target: ACM Transactions on Information Systems (verify current JCR
  status and author requirements immediately before submission).
- Backups: Information Processing & Management, then User Modeling and
  User-Adapted Interaction.
- Central claim: statistically guarded configuration rejection and metadata
  confound accounting.
- CEARF-N remains an instantiation, not a universal state-of-the-art claim.

## Frozen short-paper boundary

The following belong to the ADMA submission and are not journal-only novelty:

- finite rejection-complete candidate families with explicit null states;
- CEARF-N v2 and its three matched-seed domains;
- semantic/no-metadata fairness audit;
- NARM-containing expert-family control;
- the provisional Tmall transfer diagnostic;
- the rejected M5 repeat-memory result.

Do not delay the short paper for MiniLM, additional datasets, a new residual,
or a statistically guarded selector.

## Journal work packages

### J1. Statistically guarded selection

Primary extension:

1. Define a paired per-query validation comparison for every candidate against
   the validation-best candidate.
2. Apply a declared multiple-comparison procedure (report both BH-FDR and a
   conservative family-wise sensitivity analysis).
3. Select the least complex candidate that is not detectably worse under the
   declared rule.
4. Compare this rule with maximum-utility selection and the one-standard-error
   heuristic.

Important wording: a failure to reject is not proof of equivalence. Any
proposition must state the exact error-control guarantee and assumptions; do
not claim that the selected candidate is truly “no worse” without an
equivalence/non-inferiority margin.

Required artifacts:

- per-query validation hits and reciprocal ranks for every candidate;
- candidate-family declaration and complexity order created before test access;
- corrected p-values, confidence intervals, and selected candidate;
- validation-to-test decision transfer table.

### J2. Semantic-memory and teacher-quality axis

Teacher family:

- none;
- TF-IDF/SVD;
- one frozen compact sentence encoder.

Apply the same teacher to:

- semantic retrieval memory;
- PASGR initialization;
- NARM semantic initialization.

This produces a matched dose-response study rather than an asymmetric metadata
comparison. Precompute and checksum embeddings; report embedding construction
separately from online retrieval cost.

Implementation prerequisite: batch semantic retrieval. The current
`SemanticMemory.ranking` performs one catalogue matrix-vector product per
query and is not suitable for full journal-scale evaluation.

Stopping rule: semantic memory is a main-method component only if the
statistically guarded selector retains it on at least three of the final public
datasets. Otherwise report it as a rejected candidate.

### J3. Dataset breadth

Retain:

- Amazon Video Games;
- Amazon Baby Products;
- Diginetica;
- Tmall.

Add:

- RetailRocket;
- Yoochoose/RSC15.

Any private industrial dataset is supplementary evidence only. Every central
claim must remain reproducible on public data.

### J4. Baselines and near-miss controls

Required:

- STAN and VSTAN;
- NARM, SR-GNN, GRU4Rec, SASRec, and SIGMA;
- RepeatNet on repeat-consumption protocols;
- a frozen-embedding semantic retrieval baseline;
- the strongest simple expert ensemble selected without CEARF-N.

Literature distinctions to document:

- STAR: training-free semantic and collaborative evidence fusion;
- CSRM: parallel memory modules;
- Jannach and Ludewig: recurrent/neighbourhood hybrid;
- RepeatNet and subsequent repeat-aware systems;
- algorithm selection versus rejection-complete finite-family selection;
- multiple-testing procedures for IR/recommender experiments.

### J5. Stability and audit analyses

- redraw validation splits and report decision frequencies;
- three or more matched training seeds;
- validation-to-test selection transfer;
- effect sizes and paired confidence intervals;
- tuning-budget flip study;
- convergence/failure-rate reporting;
- full-catalogue evaluation fingerprints;
- energy and wall-clock accounting with embedding precomputation separated.

The current component grid was selected once and reused across training seeds;
it cannot support a component-decision stability claim. Journal stability
requires independently redrawn validation sets or independently repeated
selection runs.

## Work order

1. Finish and freeze the ADMA matched expert-family runs.
2. Package the short-paper artifacts and write the 6--8 page manuscript.
3. Implement batched semantic retrieval and artifact checksums.
4. Implement statistically guarded selection with synthetic/unit tests.
5. Add RetailRocket and Yoochoose protocol tests.
6. Run the public-dataset matrix and teacher dose-response.
7. Add journal baselines and stability redraws.
8. Write the journal manuscript around selection guarantees and confound
   accounting.

## Pilot outcome (2026-07-26, seed 42, Video_Games + Baby_Products)

Design, implementation, and pilot for J1 + J2 are complete. Modules:
`casm.py` (contrastive-aligned semantic memory, InfoNCE alignment head,
batched retrieval), `guarded_selection.py` (McNemar + BH-FDR least-complexity
rule, argmax/1-SE comparators), integration behind
`run_cearfn_v2.py --memory-variant {off,raw-semantic,casm}` with the locked
ADMA path verified bit-identical. Frozen spec: `DESIGN_CASM.md`. Results:
`PILOT_RESULTS.md`, `pilot_casm_seed42.json`, `pilot_guarded_audit.json`.

Decisions from the pilot:

- **J1 guarded selection: GO.** The selector reproduced two legacy gates,
  simplified the Baby router gate (continuous detectably worse, BH p=0.004),
  and exposed an argmax pick that failed to transfer to test. Full matrix can
  run on the existing 3-seed rank artifacts without new training. This arm
  now carries the journal narrative.
- **CASM as designed: HOLD.** The validation gate kept both semantic slots
  off in both domains. CASM beats the raw-semantic control on Video_Games
  (+13% relative, McNemar p=0.014) but the gain concentrates in HEAD targets
  (p=0.0002) — the tail/cold hypothesis is unsupported; on the 128-d SVD
  teacher (Baby) alignment is detectably harmful (p=0.018). This matches the
  declared popularity-collapse risk (DESIGN_CASM §6.2).
- **Before any CASM 3-seed matrix**, run the two declared validation-only
  probes: (a) 1/sqrt(freq) pair down-weighting in InfoNCE; (b) matched MiniLM
  teacher on Baby (matrices already built and checksummed in
  `semantic_teacher_artifacts/`). If neither shows a tail/cold signal, report
  CASM as a rejected candidate inside the rejection-complete framing — that is
  a publishable negative result under this framework, not a failure.
- **Protocol correction for the full phase:** the pilot's test vectors were
  read twice (complexity-metadata fix between reads; disclosed in
  PILOT_RESULTS.md). Freeze the corrected audit code before any test access
  in the full matrix.
- Cold-start (train-freq-0) recall is 0 for every variant under exclude_seen
  with gated-off semantic slots; if cold-start is to be claimed at all, it
  needs a dedicated mechanism, not CASM as currently gated.

## Go/no-go rules

- Do not replace the compact GRU residual with Mamba for the journal identity.
- Do not call M5 a method contribution; the ADMA run selected repeat profiles
  on validation but produced essentially no Diginetica test improvement.
- Do not promote a component based on one dataset or one seed.
- Do not claim equivalence from a non-significant superiority test.
- Do not claim CPU-only reproduction without reporting frozen-embedding
  precomputation separately.
- Freeze each experiment matrix before inspecting its test results.

