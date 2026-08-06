# DESIGN_CASM — Contrastive-Aligned Semantic Memory + Statistically Guarded Selection

Design document for the journal extension of CEARF-N (roadmap items J1 + J2,
`JOURNAL_ROADMAP.md`). **Design phase only — no experiment in this document has
been run.** The candidate-family declaration in §5 is frozen before any test
access, per the roadmap requirement ("candidate-family declaration and
complexity order created before test access").

Contributions:

- **(A) CASM** — a fourth training-free-at-inference evidence memory: frozen
  text-teacher item embeddings passed through a small alignment head trained
  with InfoNCE on session co-occurrence pairs, retrieved with one batched
  catalogue matrix multiply, entering the rejection-complete rank-fusion family
  as an optional ranked list with an explicit off state.
- **(B) Guarded selection** — replace argmax validation gating with a declared
  rule: paired per-query tests against the validation-best candidate, BH-FDR
  across the candidate family, select the least complex candidate not
  detectably worse.

---

## §1. Motivation & positioning

### 1.1 Why the current SemanticMemory is unusable at journal scale

`cearf_v3_ext.py` already contains a semantic memory
(`class SemanticMemory`, lines 80–128). Two blockers:

1. **Unbatched retrieval.** `SemanticMemory.ranking()` (line 102) computes one
   catalogue matrix–vector product per query (`sims = self.vectors @ (query /
   norm)`, line 113) inside a Python loop over queries. `JOURNAL_ROADMAP.md`
   (J2) names batch retrieval an explicit implementation prerequisite.
2. **Semantic–behavioral misalignment.** The teacher space (TF-IDF/SVD 128-d or
   e5-small 384-d) orders items by textual similarity, not by transition
   plausibility. Textual neighbours of a video game are often same-franchise
   substitutes, whereas the behavioral neighbourhood contains complements
   (controllers, sequels, points cards). An unaligned semantic list injected
   into RRF therefore votes for the wrong neighbourhood; the validation gate
   correctly kept it OFF in prior smoke runs.

### 1.2 Positioning against the 2023–2026 literature

**vs. LLM-embedding *initialization* work.** The dominant pattern uses text
embeddings to initialize or adapt the item table of a *trained* sequential
model: Harte et al. [1] initialize recommender item embeddings from LLM
representations; SAID [2] aligns item IDs with the LLM space via projection;
LLMEmb [3] contrastively fine-tunes the LLM itself into an embedding generator;
LLM2Rec [4] adds collaborative supervised fine-tuning of the LLM before
embedding extraction; AlphaFuse [5] learns ID embeddings in the null space of
language embeddings; ACE [6] controls anisotropy of LLM embeddings for
sequential rec. In all of these the semantic signal ends up *inside a trained
neural ranker* (our PASGR already does semantic init — that axis is covered).
CASM is different in kind: the aligned embedding is used as a **standalone
retrieval memory** whose ranked list is fused *in rank space* with three other
explicit memories, is **training-free at inference** (one matmul; the trained
artifact is a frozen ≤(384+1)×128 head applied once offline to the catalogue),
and carries a **null state** so the selection gate can reject it. No trained
recommender consumes the embedding.

**vs. STAR-style training-free fusion.** STAR [7] combines frozen LLM semantic
similarity with co-occurrence statistics in *score space* with hand-set mixing
weights, plus an LLM re-ranking pass. CASM differs in three ways: (i) the
semantic–behavioral agreement is *learned* (a small alignment head, InfoNCE on
co-occurrence positives) rather than obtained by score interpolation of two
misaligned geometries; (ii) fusion is weighted RRF over ranked lists, keeping
the rejection-complete family and null states of CEARF-N; (iii) no LLM is in
the serving path.

**vs. LLM-for-long-tail work.** LLM-ESR [8] and successors motivate our
stratified hypothesis — semantic signal helps precisely where collaborative
evidence is sparse — but again route the signal through a trained dual-view
model. CASM tests the same hypothesis in a memory-first, gate-audited setting:
gains should concentrate on tail/cold targets (§4.3), and if they do not
survive the guarded gate, that is a reportable rejection (§6).

**The selection gap.** Statistical rigor work in recsys evaluation is about
*reporting*, not *selection*: Ihemelandu & Ekstrand [9] study multiple-testing
corrections (incl. BH-FDR [10]) when comparing systems in IR/recsys
experiments; Shehzad & Jannach [11] show that untuned baselines make "everyone
a winner"; conformal/risk-control work [12,13,14] gives distribution-free
guarantees on *served output* (calibrated set sizes, bounded exposure to
unwanted items), not on *which pipeline configuration is selected*; FDR-based
model selection exists in the regression/variable-selection literature [15,16]
but has, to our knowledge, not been instantiated as a component-selection rule
inside a recommender pipeline. We found **no prior work that selects
recommender components by "least complex candidate not detectably worse than
the validation best" under an explicit FDR guarantee** — that is the J1 gap
this design fills.

---

## §2. CASM specification

### 2.1 Inputs (existing artifacts, frozen teachers)

| Domain | Teacher artifact | Shape | Loader |
|---|---|---|---|
| Video_Games | `artifacts/Video_Games_e5_small.npy` | (25613, 384) | `run_pasgr_full.semantic_matrix()` (lines 31–42; prefers `*_e5_small.npy`, falls back to `*_pasgr_semantic.npy`, then TF-IDF/SVD) |
| Baby_Products | `artifacts/Baby_Products_pasgr_semantic.npy` | (36014, 128) | same |

Open item: build a 384-d e5/MiniLM teacher for Baby_Products with
`build_minilm_teacher.py` (checksummed, cached) so both pilot domains have a
matched sentence-encoder teacher; the 128-d SVD matrix remains as the
TF-IDF/SVD rung of the J2 teacher ladder. Row 0 is padding; items without text
have zero vectors and are excluded from training pairs and retrieval
(`has_vector` convention, `cearf_v3_ext.py` lines 98–100).

### 2.2 Alignment head

- **Architecture (default):** single linear layer `W ∈ R^{d_t×128}` + bias,
  outputs L2-normalized: `z_i = normalize(W^T t_i + b)` where `t_i` is the
  frozen teacher row (d_t ∈ {128, 384}). Parameter count ≤ 49.3k.
- **Ablation variant:** 2-layer MLP `d_t → 256 (GELU) → 128`, L2-normalized.
  Chosen on validation only; the linear head is the declared default because it
  is the least complex member (ties broken toward linear, §3.4).
- Items without text keep the zero vector (never retrieved, never trained on).

### 2.3 InfoNCE training on behavioral co-occurrence

- **Positive pairs:** ordered pairs (source, target) with distance ≤ 4 within a
  training session, weighted `1/distance` — exactly the transition-memory
  convention (`cearf.py` lines 102–107: `window=4`,
  `transition[source][target] += 1.0/distance`). Pairs are sampled with
  probability proportional to that weight; pairs where either side lacks a
  text vector are dropped.
- **Loss:** symmetric in-batch InfoNCE. For batch {(a_k, b_k)}:
  `L = −½ Σ_k [log softmax_k(z_{a}·z_{b}/τ) + log softmax_k(z_{b}·z_{a}/τ)]`,
  in-batch negatives only (batch 1024 → 1023 negatives per anchor).
- **Hyperparameters (validation-tuned, small declared grid):** τ ∈ {0.05,
  0.07, 0.10}; epochs ∈ {3, 5, 10} over ~2M sampled pairs; Adam lr 1e-3, no
  weight decay; seed fixed per run.
- **Budget:** linear head, batch 1024, MPS or CPU on this machine — the
  forward/backward is a (1024×d_t)·(d_t×128) matmul pair; ≈ minutes per domain
  per config. Offline catalogue projection is one (n_items×d_t)·(d_t×128)
  matmul, cached to `artifacts/<domain>_casm_<teacherhash>.npy` with a config
  fingerprint (mirroring `build_minilm_teacher.py`'s checksum discipline).
- **Leakage rule (nested protocol):** for validation-phase scoring the head is
  trained only on `hold_out_validation_targets(sessions, valid_queries)`
  (`validation_protocol.py` lines 7–29, which strips `<source>_v` targets from
  Diginetica-style rows); for test-phase scoring it is retrained on the full
  `train_sessions`. This mirrors the `tune_index` / `final_index` split in
  `run_cearfn_v2.py` lines 138–144 and the memory-array split at lines 147–154.

### 2.4 Batched retrieval

Query embedding mirrors the existing recency conventions: for the last ≤8
context items (`context_tail=8`), `q = Σ_age exp(−0.35·age)·z_item`, then
L2-normalize — the same decay=0.35/tail=8 as `SemanticMemory.__init__`
(`cearf_v3_ext.py` lines 89–96) and `CEARFIndex._transition_scores`
(`cearf.py` line 125); the session memory uses 0.20 (line 136) — we keep 0.35
as the declared default and do not tune it.

Batched scoring: stack queries into `Q (B×128)`, compute `S = Q @ Z^T` in
chunks of ≤4096 queries (Z = aligned catalogue matrix, ~13–18 MB fp32), mask
`~has_vector` and per-query blocked items (context items when
`exclude_seen=True`, matching `component_rankings`, `cearf.py` lines 163–170),
then `np.argpartition` row-wise for the **top-120** candidate list
(`component_topn=120`, `CEARFConfig` line 32). API:

```python
class CASMemory:
    def rankings_batch(self, contexts: list[Sequence[int]],
                       blocked: list[set[int]]) -> list[list[int]]: ...
    # single-query wrapper keeps the CEARFIndexV3 interface:
    def ranking(self, context, blocked) -> list[int]
```

**The memory interface (what any new memory must expose).** A memory is a
callable producing, per query, a ranked `list[int]` of item ids (>0, blocked
removed, length ≤ `component_topn=120`); an empty list is the null
contribution. It joins fusion via `CEARFIndexV3.component_rankings()`
(`cearf_v3_ext.py` lines 161–169), which appends it to the returned tuple;
`CEARFIndex.fuse_rankings()` (`cearf.py` lines 172–193) zips the profile
weight vector with the tuple of lists — RRF `weight/(20+rank)` with
`rrf_constant=20.0` (line 33) and consensus bonus 0.12 for items on ≥2 lists
(lines 183–185). OFF state = profile weight 0 (list is then skipped at line
178). For array pipelines, `build_memory_arrays` / `load_or_build_memory`
(`run_cearfn_evidence.py` lines 149–184) persist each component as an
`(n_queries × 120) int32` array in an `.npz` keyed by query fingerprint —
CASM adds a `"casm"` key there.

### 2.5 RRF entry: profile extension

`PROFILES_V3` (`cearf_v3_ext.py` lines 51–69) becomes 6-slot
`(transition, session, popularity, semantic_raw, repeat, casm)`; all existing
profiles get casm=0, preserving the containment argument (v2 set ⊂ v3 set ⊂
v4 set) and hence rejection-completeness. New entries (weights follow the
spirit of the existing semantic variants at lines 60–62):

```
casm_only            (0.00, 0.00, 0.00, 0.00, 0.00, 1.00)
balanced_casm        (0.40, 0.35, 0.05, 0.00, 0.00, 0.20)
casm_tail            (0.30, 0.30, 0.05, 0.00, 0.00, 0.35)
raw_semantic control (existing semantic_* profiles, unchanged)
```

The **unaligned control** is the identical retrieval path run on the raw
teacher matrix (i.e., today's `SemanticMemory`, batched) — same top-120, same
profiles with the `semantic_raw` slot. Any CASM–control gap is then
attributable to the alignment head, not to text similarity per se.

`complexity()` (line 72, count of nonzero weights) extends unchanged to 6
slots. `tune_profiles_v3` (lines 176–212) — argmax of
`0.5·R@6 + 0.5·R@20` with simplicity tie-break — remains the *argmax
comparator*; the guarded selector of §3 is the proposed replacement.

---

## §3. Statistically guarded selection (J1)

### 3.1 Candidate family formalization

A **candidate** is a full serving configuration
`c = (profile_short, profile_long, router_family, β-parameters, PASGR on/off)`
drawn from the frozen list in §5. Let `C` be the family, `|C| = m`. Selection
uses only validation queries (deterministic split:
`cearf.make_validation_split`, `cearf.py` lines 56–79 — blake2b-ordered,
fraction 0.10, cap 5000; the additional cap at `run_cearfn_v2.py` lines
134–136; leakage-safe tuning index via `hold_out_validation_targets`).

Per-query outcomes already exist as artifacts: `run_cearfn_v2.py` persists
`valid_*_rank` and `*_rank` uint8 rank-at-20 vectors per candidate per seed in
`{domain}_v2_seed{seed}_ranks.npz` (lines 361–405). The guarded selector reads
these; new candidates (raw-semantic, CASM) must persist the same keys.

### 3.2 Paired per-query statistic

Primary metric: **hit@20** (binary per query; rank vectors → `0 < r ≤ 20`).
Let `c* = argmax_c U_val(c)` with `U = ½(R@6 + R@20)` (utility as in
`router_utility`, `run_cearfn_v2.py` lines 101–103). For every `c ≠ c*` test,
one-sided:

- `H0(c): θ_c ≥ θ_{c*}` (c is not worse) vs `H1(c): θ_c < θ_{c*}`.
- **Test:** exact McNemar / binomial sign test on discordant pairs — with
  `n01 = #{queries: c* hits, c misses}`, `n10 = #{c hits, c* misses}`,
  p-value = `Binom(n10; n01+n10, ½)` lower tail (one-sided). This matches the
  per-seed McNemar already implemented in
  `paired_statistics.cluster_paired_recall` (`paired_statistics.py` lines
  36–44), restricted to validation and made one-sided.
- **Degenerate/discrete cases (declared handling):** if `n01 + n10 = 0`
  (identical hit vectors) set p = 1 (never rejected — c remains eligible).
  The binomial test is exact, so small discordant counts are valid but
  low-powered; we report `n01+n10` per pair, and we do NOT interpret
  non-rejection at low power as evidence of equivalence (see 3.5). With
  multiple seeds, the seed-averaged per-query hit difference and sign-flip
  randomization from `cluster_paired_recall` (lines 22–35) replace the
  single-seed binomial; the pilot is 1 seed, so exact McNemar applies.

### 3.3 Multiplicity: BH-FDR at q = 0.10

Apply Benjamini–Hochberg [10] to the `m−1` p-values (one per non-best
candidate). BH is valid under independence and PRDS; our statistics are
positively correlated across candidates (shared queries, overlapping
components), which is the canonical PRDS-motivated setting; we additionally
report Benjamini–Yekutieli as the conservative sensitivity analysis required
by the roadmap. Rejected `H0(c)` ⇒ `c` is **detectably worse**; the eligible
set is `E = {c : H0(c) not rejected} ∪ {c*}`.

### 3.4 Complexity partial order and tie-breaks

Complexity tuple, compared lexicographically (smaller = simpler):

1. number of active components: nonzero profile slots (union over the two
   regimes) + 1 if PASGR is on (`complexity()`, `cearf_v3_ext.py` line 72,
   extended);
2. router rank: constant β (0) < regime β (1) < bucketed (2) < continuous (3)
   — mirroring the family order in the router selection block,
   `run_cearfn_v2.py` lines 275–296;
3. total number of nonzero profile weights across both regimes.

**Rule:** select `ĉ = argmin_{c ∈ E} complexity(c)`; ties → higher validation
utility; residual ties → earlier position in the declared candidate list
(deterministic). This generalizes the existing tie-break "fewer active
components wins" (`tune_profiles_v3`, lines 200–205).

### 3.5 Guarantee proposition (exact wording for the paper)

> **Proposition (FDR-guarded selection).** Let p_c be valid one-sided p-values
> for H0(c): θ_c ≥ θ_{c*}, computed on validation queries. If the BH procedure
> at level q is applied to {p_c : c ∈ C \ {c*}} and the joint distribution of
> the p-values satisfies PRDS (or independence), then the expected proportion
> of candidates falsely declared detectably worse, among all candidates so
> declared, is at most q.

**What this does NOT claim:** (i) non-rejection is not evidence of
equivalence — eligible candidates are "not shown worse", not "shown no worse";
a non-inferiority margin would be needed for that claim and we do not make it;
(ii) no guarantee attaches to the *test-set* performance of ĉ — the guarantee
is about the validation-stage error of the family of "worse" declarations;
(iii) conditioning on the data-chosen comparator c* makes the p-values valid
conditionally on c* being fixed by the utility ranking; we state this
explicitly rather than claiming post-selection-inference exactness.

### 3.6 Comparison protocol

Three selection rules on identical per-query artifacts: (a) **argmax**
(status quo, `tune_profiles_v3` semantics), (b) **1-SE rule** (simplest
candidate within one standard error of the best, SE via query-level bootstrap
of the utility), (c) **guarded** (§3.2–3.4). Report per domain: selected
candidate, complexity, validation utility, test R@20/NDCG@20 (test read once,
after all three selections are locked), and a validation→test decision-transfer
table (does simpler selection lose anything detectable on test?).

### 3.7 Unit-test plan (synthetic)

New `test_guarded_selection.py`:

1. synthetic hit matrices with planted effect sizes → guarded rule recovers
   the simplest of the truly-equivalent set with frequency consistent with q;
2. BH implementation cross-checked against `statsmodels.stats.multitest`;
3. degenerate pairs: zero discordant queries → p=1, candidate stays eligible;
   all candidates identical → selects minimum-complexity candidate;
4. complexity order: crafted profiles verify the lexicographic tuple and both
   tie-break stages;
5. determinism: same inputs → same selection across runs/orderings;
6. leakage guard: assert the selector never touches keys of `test_queries`
   (fingerprint check as in `load_or_build_memory`,
   `run_cearfn_evidence.py` lines 171–179).

CASM-side tests (`test_casm.py`): alignment-head shapes; zero-vector items
never retrieved; batched vs. single-query ranking equality on random contexts
(`rankings_batch(...) == [ranking(...)  per query]`); co-occurrence sampler
respects window ≤4 / 1/distance weighting and the held-out-target exclusion.

---

## §4. Pilot experimental design (design only — not yet run)

- **Seed:** 42 only. **Domains:** Video_Games, Baby_Products (both
  `exclude_seen=True`, non-repeat protocol; `REPEAT_PROTOCOL_DOMAINS`,
  `run_cearfn_v2.py` line 48).
- **Variants (fixed order):**
  1. CEARF-N v2 locked (memory profiles + gated PASGR + selected router; the
     `selected_rank` artifact from the existing v2 run);
  2. v2 + raw semantic memory (unaligned control; batched retrieval on the
     frozen teacher);
  3. v2 + CASM.
- **Metrics:** full-catalogue R@20 and NDCG@20 (from rank vectors via
  `metrics_from_ranks`), R@6 retained for the utility.
- **Stratification:** target-item train-frequency strata using the existing
  20/60/20 head/torso/tail convention (`run_stratified_evaluation.py`,
  `popularity_strata`, lines 62–80) **plus a cold-start stratum: train
  frequency = 0** (reported separately from tail). Hypothesis to be tested:
  CASM's gain over v2 and over the raw control concentrates in tail + cold;
  head is flat or slightly negative.
- **Protocol:** all tuning (head hyperparameters, profiles, β, router) on
  validation only; test rank vectors computed once per variant after the §5
  family is locked; per-query artifacts persisted in the v2 `.npz` schema for
  the guarded selector and for the paired analysis
  (`cluster_paired_recall`).
- **Compute:** teacher matrices exist (VG) or are one `build_minilm_teacher.py`
  run (Baby); head training ≈ minutes/domain on MPS; retrieval one matmul per
  4096-query chunk; total pilot well under an hour per domain excluding PASGR
  reuse.

## §5. Candidate-family declaration (FROZEN before any experiment)

Family `C` for the journal pilot, per domain, per regime pair (short, long):

| # | Candidate | Active components | Router | Notes |
|---|---|---|---|---|
| 1 | popularity only | pop | — | floor / null-heavy |
| 2 | transition only | trans | — | `PROFILES` v1 |
| 3 | best v1 3-slot profile family | trans+sess+pop | — | 6 profiles of `cearf.PROFILES` (lines 41–48) as one sub-family, argmax within |
| 4 | v2 memory + PASGR, constant β | 3 + neural | constant | β grid `run_cearfn.BETAS` |
| 5 | v2 memory + PASGR, regime β | 3 + neural | regime | status-quo v2 |
| 6 | v2 memory + PASGR, bucketed router | 3 + neural | bucketed | |
| 7 | v2 memory + PASGR, continuous router | 3 + neural | continuous | |
| 8 | 5 + raw semantic memory (semantic_* profiles) | 4 + neural | regime | unaligned control |
| 9 | 5 + CASM (casm_* profiles, §2.5) | 4 + neural | regime | proposed |
| 10 | 9 with bucketed router | 4 + neural | bucketed | interaction check |

Null states: every memory slot may carry weight 0; PASGR may be absent
(candidates 1–3). No candidate outside this table may be evaluated on test in
the pilot; additions require a new frozen declaration *before* their first
test read (they may be validation-explored freely).

## §6. Risks & kill criteria

1. **CASM ≤ raw-semantic control on both pilot domains** (validation utility,
   and guarded test shows no tail/cold gain) → per the roadmap stopping rule,
   CASM is reported as a **rejected candidate** (the rejection-complete framing
   makes this a publishable negative result, not a discarded run) and the
   journal narrative leans on J1 + the teacher-axis study.
2. **Alignment collapses to popularity** (InfoNCE on co-occurrence can learn
   frequency): monitor by checking rank correlation of CASM top-1 lists with
   the popularity list; mitigation = 1/√freq down-weighting in pair sampling
   (same normalization spirit as `_transition_scores`, `cearf.py` line 127–128)
   — declared as an ablation, not silent tuning.
3. **Guarded rule is vacuous** (m−1 small, tests low-powered → everything
   eligible → rule = "pick simplest"): report discordant-pair counts and a
   power note; this is still an honest, declared rule and contrasts with
   argmax's zero guarantee.
4. **Baby_Products teacher mismatch** (128-d SVD vs VG's 384-d e5): build the
   e5 teacher first (§2.1); if metadata coverage is too sparse, report the SVD
   rung only and say so.
5. **Complexity order disputes**: the order in §3.4 is declared here, before
   experiments; reviewers may prefer parameter counts — we report both but
   select by the declared order.

## References

[1] Harte et al., "Leveraging Large Language Models for Sequential
Recommendation" (LLM2X), RecSys 2023. — LLM embeddings as item-embedding init
for trained recommenders; not a fused retrieval memory.
[2] Hu et al., "Enhancing Sequential Recommendation via LLM-based Semantic
Embedding Learning" (SAID), WWW 2024 Companion. — projects item IDs into the
LLM semantic space; init/adapter, not a training-free memory.
[3] Liu et al., "LLMEmb: LLM Can Be a Good Embedding Generator for Sequential
Recommendation", AAAI 2025. — contrastive fine-tuning of the LLM itself; CASM
freezes a small encoder and trains only a ≤49k-param head.
[4] He et al., "LLM2Rec: Large Language Models Are Powerful Embedding Models
for Sequential Recommendation", KDD 2025, arXiv:2506.21579. — collaborative
SFT inside the LLM; embeddings feed trained recommenders via an adapter.
[5] Hu et al., "AlphaFuse: Learn ID Embeddings for Sequential Recommendation
in Null Space of Language Embeddings", SIGIR 2025. — geometric coupling of ID
and language spaces inside a trained model.
[6] "ACE: Anisotropy-Controllable Embedding for LLM-enhanced Sequential
Recommendation", arXiv:2605.29322. — post-processes LLM embedding geometry for
sequential rec; complementary to alignment-by-contrastive-training.
[7] Kraft et al., "STAR: A Simple Training-free Approach for Recommendations
using Large Language Models", arXiv:2410.16458. — training-free score-space
mix of semantic + collaborative similarity with LLM re-ranking; no learned
alignment, no gated null state.
[8] Liu et al., "LLM-ESR: Large Language Models Enhancement for Long-tailed
Sequential Recommendation", NeurIPS 2024 (arXiv:2405.20646 line). — semantic
signal targeted at tail users/items via a trained dual-view model; motivates
our tail/cold stratification.
[9] Ihemelandu & Ekstrand, "Multiple Testing for IR and Recommendation System
Experiments", ECIR 2024. — FDR-aware multiple-comparison procedures for
*evaluating* recsys experiments; closest prior to J1, but reporting-side, not
a selection rule with a complexity order.
[10] Benjamini & Hochberg, "Controlling the False Discovery Rate", JRSS-B
57(1):289–300, 1995. — the FDR procedure we apply across the candidate family.
[11] Shehzad & Jannach, "Everyone's a Winner! On Hyperparameter Tuning of
Recommendation Models", RecSys 2023. — argmax-style tuning inflates claimed
wins; direct motivation for guarded selection.
[12] Angelopoulos, Krauth, Bates, Wang, Jordan, "Recommendation Systems with
Distribution-Free Reliability Guarantees", COPA 2023, arXiv:2207.01609. —
conformal guarantees on served recommendation sets, not on configuration
selection.
[13] Angelopoulos, Bates, Fisch, Lei, Schuster, "Conformal Risk Control",
ICLR 2024. — expected-loss control for monotone losses; serving-time, not
selection-time.
[14] De Toni et al., "You Don't Bring Me Flowers: Mitigating Unwanted
Recommendations Through Conformal Risk Control", RecSys 2025. — bounds
exposure to unwanted items; guarantee object is the slate, not the pipeline
configuration.
[15] Benjamini & Gavrilov, "A simple forward selection procedure based on
false discovery rate control", Ann. Appl. Stat. 2009 (arXiv:0905.2819). —
FDR-penalized variable selection in regression; the selection-by-FDR idea we
transplant to recommender components.
[16] G'Sell, Wager, Chouldechova, Tibshirani, "Sequential Selection Procedures
and False Discovery Rate Control", JRSS-B 2016. — ordered-testing FDR for
model selection; our family is unordered, hence BH/BY, but the framing of
"model selection as multiple testing" is shared.
[17] Wang et al., "Text Embeddings by Weakly-Supervised Contrastive
Pre-training" (E5), arXiv:2212.03533. — the frozen e5-small teacher used for
Video_Games (and to be built for Baby_Products).

---

## Appendix A. Implementation notes (added at implementation time)

No frozen declaration (§2 spec, §3 rule, §5 family) was changed. The
following are implementation clarifications, not deviations:

1. **Modules.** `casm.py` (AlignmentHead, `train_alignment_head`,
   `CASMemory`, `load_or_train_casm` checksummed cache) and
   `guarded_selection.py` (`Candidate`, exact one-sided McNemar +
   sign-flip permutation, BH/BY adjustment, argmax / 1-SE / guarded rules,
   JSON-serialisable audit record). Both import without side effects; torch
   is needed only for the training path.
2. **Raw-semantic control path.** The unaligned control is served by
   `CASMemory.from_teacher(teacher)` (identical batched retrieval code as
   CASM, L2-normalisation only) mounted on the `semantic_raw` slot of
   `CEARFIndexV3`. This supersedes the per-query matvec of the old
   `SemanticMemory` for array pipelines; `SemanticMemory` itself is retained
   untouched for interface compatibility. `CEARFIndexV3` gained a `casm=`
   memory slot and a `component_rankings_batch()` that routes any memory
   exposing `rankings_batch` through the chunked-matmul path.
3. **Six-slot profiles.** `PROFILES_V3` is now
   (transition, session, popularity, semantic_raw, repeat, casm) with all
   legacy profiles zero-padded and exactly the three §2.5 CASM profiles
   added (`casm_only`, `balanced_casm`, `casm_tail`).
4. **Runner integration.** `run_cearfn_v2.py --memory-variant
   {off,raw-semantic,casm}` (default `off` = locked ADMA path; the code in
   that branch is character-identical to the pre-change runner). Note the
   stored `*_v2_seed*_ranks.npz` artifacts from the July v2 runs carry only
   `{bucketed,continuous,regime}_rank` + `test_fingerprint`, so a
   bit-for-bit smoke comparison is possible only against the cached memory
   npz files (`diginetica_hid_{valid,test}_memory.npz`). Smoke-check record:
   rebuilding the Diginetica_HID validation memory with the unchanged
   `cearf.CEARFIndex` + `cearf.tune_profiles` + `build_memory_arrays` path
   reproduced `diginetica_hid_valid_memory.npz` bit-for-bit (fingerprint,
   profiles, and all five arrays `keys/transition/session/popularity/
   selected` exactly equal). Active variants write their own artifacts
   (`<domain>_v2_<variant>_seed<seed>_ranks.npz`, variant-tagged memory
   caches, `"<domain>::<variant>"` results keys) so locked v2 files are
   never overwritten; npz keys are unchanged so the guarded selector reads
   both uniformly. CASM hyperparameters are exposed as `--casm-tau/-epochs/
   -pairs/-seed/--casm-mlp` with defaults at the declared grid midpoints
   (τ=0.07, epochs=5, 2M pairs, seed 42). Per the §2.3 leakage rule the
   head is trained on `tune_sessions` for the validation-phase index and on
   full `train_sessions` for the test-phase index (two cached matrices).
5. **1-SE comparator detail.** The SE in the 1-SE rule is the query-level
   bootstrap SE of the best candidate's mean utility (2000 resamples,
   fixed seed), as §3.6(b) implies; ties inside the 1-SE band use the same
   complexity/utility/position tie-break as the guarded rule.

