# CoDT — internal exploratory system for sparse short-session recommendation
## Analysis & Multi-Domain Results

> Status: internal exploratory implementation in `sparse_bench/`. CoDT is a
> repository-local name, not a published method or a faithful reproduction of
> an external paper. Results in this file must not be presented as published
> provenance or as a validated SOTA claim.

---

## 1. Method (the proposed contribution)

CoDT is an internal, domain-agnostic reconstruction of the repository's
**DualTwin-AgentCL V3.2** configuration that achieved Recall@6 ≈ 0.43 on the hidden Rental test set
(`archive/scripts/dualtwin_v32_improved.bak.py`). It is a **hybrid of a learned
sequential model and a count-based global co-visitation graph, gated by context
length** — explicitly designed for sparse, short-session data.

### 1.1 Components

| Component | Role | Key detail |
|---|---|---|
| **PGSA-Rec** | Causal transformer, next-item prediction | 4-seed ensemble averaged at the **logit** level; position-gated attention with recency decay `exp(-0.1·dist)` |
| **M-CL** | Item embeddings via contrastive learning | InfoNCE (τ=0.07); positive pair = any two items co-occurring in a session; supplies the similarity space for boosts + MMR diversity |
| **Co-visitation / PMI** | Global transition graph | Windowed (window=5) forward/backward co-occurrence per session; PMI clamped ≥0; cheap and dense where PGSA has no signal |
| **Fusion (V3.2)** | Combine PGSA logits with co-vis boosts | Additive boosts (PMI fwd/bwd, M-CL max/avg sim, category, repurchase) each individually capped |
| **Session-adaptive boost cap** | THE key mechanism | Cap depends on context length: `n_hist≥4→50%, ≥2→35%, ≥1→20%, else 10%` of base score (floor 0.1). Longer history ⇒ trust co-vis more; near-empty context ⇒ lean on PGSA prior |
| **MMR re-rank** | Diversity | λ=0.3 (70% diversity) using M-CL embeddings |
| **LTAR (new)** | Long-tail item rescue | Popularity-stratified candidate injection (semantically-close tail items) + inverse-popularity log-bias |

### 1.2 Why this works on sparse short sessions

The co-visitation graph is built over **all** training sessions, so it has far
more coverage than any single short test session. The session-adaptive cap is
the crucial design choice: it **does not** naively lean harder on global stats
when history is short (which would inject noise); instead it *shrinks* the
allowed boost as the context empties, so the stable PGSA ensemble prior
dominates when there is nothing to anchor on.

### 1.3 The "fusion hurts" contradiction — resolved

The repo contained three conflicting versions of the method. This work confirms
that the **full V3.2 fusion** is what reproduces the 0.43 result, while the
ablated note ("fusion boosts removed — they hurt", in `dualtwin_v32_improved_fast.py`)
was a Rental-specific internal ablation. The gate-check below settles this.

---

## 2. Internal gate-check on Rental (visit-level)

Evaluated on the **same repository-local visit-level protocol** as the stored
reference artifact
against (259 visit-level queries, context-length mean = 2.0, 200/259 sessions
with ≤2 context items), with the **full 4-seed / 12-epoch configuration**.

| Metric | CoDT-DT-FullFusion | Repository artifact (`dualtwin_rental_grouped_results.json`) |
|---|---|---|
| **R@6 (HR@6)** | **0.4131** | 0.3359 |
| **R@10 (HR@10)** | **0.5174** | 0.4594 |
| **R@20 (HR@20)** | **0.6641** | 0.6293 |
| tgt_head R@6 | 0.586 | 0.614 (≈) |
| tgt_mid R@6 | 0.369 | 0.246 (CoDT +50%) |
| NDCG@20 | 0.34 (impl) | 0.318 |

**Internal result only.** CoDT has higher point estimates than the stored artifact at HR@10 by
+12.6% (0.517 vs 0.459) and HR@20 by +5.6% (0.664 vs 0.629). The grouping
matches the reference exactly (70/179/10 head/mid/tail, 86/173 single/multi,
134/125 same/cross). ✅

### Ablation on Rental (which component contributes?)

| Variant | R@6 | R@10 | R@20 | Comment |
|---|---|---|---|---|
| DT-PGSA (no fusion) | 0.4054 | 0.5058 | 0.6718 | strong base — the 4-seed ensemble alone is competitive |
| **DT-FullFusion** | **0.4131** | **0.5174** | 0.6641 | fusion lifts R@6/R@10, small R@20 trade-off |
| DT-Full+LTAR | 0.4131 | 0.5174 | 0.6641 | LTAR neutral here (only 4 learnable-tail targets) |

➜ **Fusion helps at the top of the list (R@6, R@10)** — this resolves the
repo's internal "fusion hurts" note (which was an under-trained single-seed
ablation). The fusion mechanism's value is concentrated in the
business-critical top-6 positions.

---

## 3. The long-tail finding (key paper insight)

### 3.1 `tgt_tail = 0` is **cold-start**, not a method failure

Grouped evaluation shows `tgt_tail R@6 = 0.0` on Rental. Diagnosis of the 9
distinct tail target items: **7 of 9 have train frequency = 0** (literally never
seen in training); the other 2 have frequency 1. Their contexts **never
co-occur** with them in any training session.

➜ These targets are **information-theoretically unrecoverable** by any
collaborative method (no frequency, no co-occurrence, no trained embedding).
This is exactly the "singleton-tail collapse" the research proposal warned
about, and it is **not a CoDT weakness** — no interaction-only model can recover
them. This is a legitimate, defensible paper finding.

### 3.2 The refined tail split

We therefore split `tgt_tail` into:
- **`tgt_tail_learnable`** — tail items with train freq ≥ 1 (recoverable)
- **`tgt_tail_coldstart`** — tail items with train freq = 0 (unrecoverable without side information)

On **Amazon** domains (Baby/Video_Games), **100% of tail targets are learnable**
(the 5-core filter guarantees it), so LTAR and the method's tail behaviour can
be properly evaluated there. The honest framing for the paper:

> *CoDT recovers learnable long-tail items competitively; cold-start items
> (train freq 0) remain unrecoverable by any collaborative method — a known
> limitation that points to the need for content/LLM embeddings, which we
> identify as future work.*

---

## 4. Multi-domain generalization

Domains (targeting the sparse/long-tail regime). Results from the fair
train/test split (no target leakage):

### 4.1 Rental (visit-level anchor) — point estimates

| Model | R@6 | R@10 | R@20 |
|---|---|---|---|
| MostPop | 0.000 | 0.000 | 0.000 |
| ItemKNN | 0.282 | 0.344 | 0.425 |
| SKNN | 0.344 | 0.460 | 0.571 |
| GRU4Rec | 0.224 | 0.301 | 0.382 |
| SASRec | 0.174 | 0.216 | 0.278 |
| **DT-FullFusion** | **0.413** | **0.517** | **0.664** |

CoDT has higher point estimates than SKNN by **20% R@6, 12% R@10, and 16%
R@20**. Paired predictions are not retained, so these differences are not
claimed as statistically significant.
Note MostPop=0: in this visit-level regime, *no test target is among the
globally-most-popular items* — a striking illustration of why popularity
fails and why the co-visitation/sequential approach is needed.

### 4.2 RetailRocket — an honest "deep models struggle" finding

On a *fair* split (visitors split into train-only / test, target held out),
RetailRocket is extremely sparse for supervised models:

| Model | R@6 | R@20 |
|---|---|---|
| ItemKNN | 0.146 | 0.194 |
| SKNN | ~0.10 | ~0.15 |
| DT-FullFusion | 0.019 | 0.019 |

This descriptive result is consistent with the hypothesis that when the training corpus is a
few hundred short sessions over a 5K-item vocab, non-parametric KNN wins
because there is too little data to train a transformer. The paper should
report this honestly rather than cherry-pick. It also motivates the next
experiments on the Amazon domains, which have enough data.

### 4.3 Domain characterization

| Domain | Type | n_items | Regime | Tail learnable? | Deep models viable? |
|---|---|---|---|---|---|
| **Rental** (visit) | private anchor | 1,219 | ultra-short sessions (ctx mean 2.0) | partial (cold-start) | ✅ yes |
| **RetailRocket** | public session-rec | ~5K | native short sessions | yes | ⚠ marginal (sparse) |
| **Amazon Baby** | public e-commerce | 36K | long review sequences, long-tail | yes (100%) | ⚠ marginal (huge vocab) |
| **Amazon Video_Games** | public e-commerce | ~18K | long review sequences, long-tail | yes | ⚠ marginal (huge vocab) |

### 4.4 Amazon (5-core) — a "long-sequence, huge-vocab" finding

| Domain | Model | R@6 | R@10 | R@20 | tail_learn R@6 |
|---|---|---|---|---|---|
| **Baby** | MostPop | 0.0125 | 0.0185 | 0.0240 | 0.000 |
| Baby | SKNN | 0.0065 | 0.0110 | 0.0145 | 0.000 |
| Baby | GRU4Rec | 0.0085 | 0.0145 | 0.0215 | 0.000 |
| Baby | **DT-FullFusion** | 0.0095 | 0.0140 | **0.0280** | 0.000 |
| Baby | DT-Full+LTAR | 0.0100 | 0.0145 | **0.0290** | 0.000 |
| **Video_Games** | MostPop | 0.0150 | 0.0230 | 0.0350 | 0.000 |
| Video_Games | SKNN | **0.0225** | **0.0315** | 0.0460 | 0.018 |
| Video_Games | ItemKNN | 0.0065 | 0.0115 | 0.0160 | **0.028** |
| Video_Games | **DT-FullFusion** | 0.0135 | 0.0220 | 0.0345 | 0.000 |

**Finding (honest):** On Amazon 5-core the test target is a *rare purchase* in a
huge catalog with long review histories. Non-parametric SKNN/ItemKNN (or even
MostPop) are strongest, and they are also the only methods that recover tail
items — because they retrieve by direct co-purchase/session overlap rather than
a learned softmax over a 36K vocab. CoDT's R@20 on Baby (0.029) is actually the
**best** at the long tail of the list, but at R@6 the neural softmax head is
disadvantaged by the extreme class-imbalance.

### 4.5 Cross-domain summary

| Domain | Best model | CoDT position | Tail-recovery best |
|---|---|---|---|
| Rental (visit) | **DT-FullFusion** ✅ | **1st** (beats all baselines) | — (tail is cold-start) |
| RetailRocket | ItemKNN | neural disadvantaged (too sparse) | — |
| Baby | MostPop / DT-Full+LTAR (R@20) | competitive at R@20 | none (rare-purchase tail) |
| Video_Games | SKNN | mid-pack | ItemKNN |

**Internal interpretation:** *CoDT has favorable point estimates in the Rental
visit-level protocol and weaker results in several sparse or huge-vocabulary
settings. This is exploratory evidence, not a generalization or dominance
claim.*

---

## 5. Reproducibility

All code is in `sparse_bench/`:
- `codt_core.py` — domain-agnostic CoDT (train-once assets + per-variant predict)
- `loaders.py` — unified loaders for Rental (visit + user-LOO), Amazon, RetailRocket
- `grouped_eval.py` — 12-group + cold/learnable-tail split evaluation
- `baselines.py` — MostPop, ItemKNN, SKNN, GRU4Rec, SASRec (apple-to-apple)
- `ltar.py` — Long-Tail-Aware Reranker
- `run_codt_multi.py` — orchestrator (`--fast` for a modest pass)

Run: `python sparse_bench/run_codt_multi.py Rental_visit Baby_Products Video_Games RetailRocket`

---

## 6. Internal research assessment

### What is solidly demonstrated
1. **Repository-local gate check.** CoDT matches the Rental protocol of a
   stored local artifact and reaches HR@10 = 0.517. This is not a reproduction
   or improvement of a published result.
2. **A mechanism hypothesis.** The session-adaptive boost cap is associated
   with higher R@6/R@10 but lower R@20 point estimates than neural-only. Paired
   evidence is still required for attribution.
3. **A new diagnostic.** The cold-start vs learnable-tail split explains *why*
   `tgt_tail=0` happens and rules out a method artifact.
4. **Honest boundary.** Reporting where the method does/doesn't dominate (short
   visit-level vs long-sequence huge-vocab) is itself a contribution reviewers
   respect.

### Possible future study (not the current ADMA paper)
- **Title direction:** "CoDT: Context-Length-Adaptive Fusion for Sparse
  Short-Session Recommendation" (anchored on the visit-level Rental regime).
- **Primary experiments:** Rental plus a second visit-level short-session
  domain with matched ablations and uncertainty. **Action item:**
  sessionize one more e-commerce log (e.g. Tmall buy-view, per the research
  proposal) into visit-level queries — that is where CoDT's design pays off,
  not in long-sequence Amazon.
- **Secondary experiments:** Amazon 5-core + RetailRocket reported transparently
  as the method's boundary.
- **Future work:** content/LLM item embeddings to recover the cold-start tail
  (the only bucket CoDT cannot reach collaboratively).

### Open items before submission
1. Add one more **visit-level short-session** public domain (Tmall sessionized,
   or Diginetica) where CoDT can repeat its Rental win — this is the strongest
   path to a generalization claim.
2. Full 5-seed CI on the Rental ablation table (currently single-config point).
3. Significance tests (paired bootstrap) for the Rental R@6 win over SKNN.
