# PILOT_RESULTS — CASM + Guarded Selection (seed 42, Video_Games + Baby_Products)

Pilot for the CEARF-N journal extension (`DESIGN_CASM.md` §4). One seed (42),
two Amazon domains, three memory variants each. All selection decisions were
made on validation-only artifacts (`pilot_guarded_audit.json`).

**Protocol deviation (disclosed).** The stored test rank vectors were read
**twice**, not once. The first test read used an initial audit in which the
complexity metadata of the off-variant candidates was hardcoded rather than
derived from the stored tuned profiles. After that read, the metadata was
corrected (a deterministic, validation-side fix — no utilities or p-values
changed) and the audit + test read were regenerated. The correction changed
three selections: Video_Games guarded flipped from `casm_bucketed` (first
read: test R@20 0.14581) to `v2_pasgr_bucketed` (reported below), and the
1-SE pick changed identity in **both** domains (`raw_semantic_regime` →
`v2_pasgr_regime`; test R@20 unchanged at 0.14642 / 0.05708 because the two
candidates' fused rank vectors coincide under the gated-off semantic slots).
Because the fix
happened after test metrics had been observed, the §5 picks reported here
were not frozen before the *first* test read; the full-matrix runs must
freeze the corrected audit code before any test access. Both reads used only
stored rank vectors — no re-tuning against test occurred.

## 1. Setup

| | Video_Games | Baby_Products |
|---|---|---|
| n_items | 25,613 | 36,014 |
| test queries | 94,762 | 150,777 |
| validation queries (capped) | 5,000 | 5,000 |
| teacher | `artifacts/Video_Games_e5_small.npy` (384-d e5-small) | `artifacts/Baby_Products_pasgr_semantic.npy` (128-d TF-IDF/SVD) |
| protocol | exclude_seen=True | exclude_seen=True |

**Teacher substitution note.** The pilot ran with the pre-existing teachers
above. MiniLM (384-d) matrices for both domains were built during the pilot
(`semantic_teacher_artifacts/{video_games,baby_products}_minilm.npy`,
checksummed manifests alongside) after resolving HF network access, but too
late to re-run Baby_Products within budget; they are ready for the full
phase. Per `DESIGN_CASM.md` §6.4 the Baby result below is the TF-IDF/SVD
teacher rung.

CASM hyperparameters: declared grid midpoints (τ=0.07, epochs=5, 2M pairs,
linear head, casm-seed 42). Nested protocol: head trained on
`tune_sessions` for the validation index, full `train_sessions` for the test
index (checksummed caches under `cearfn_v2_pilot_artifacts/casm_cache/`).

### Exact commands

```
# preflight (all green: 26 + 33 tests)
python -m pytest test_casm.py test_guarded_selection.py -q
python -m pytest test_casm.py test_guarded_selection.py test_cearf_v3_ext.py \
    test_validation_protocol.py test_semantic_teacher.py -q

# per domain (Video_Games shown; Baby_Products identical with its own output file)
python run_cearfn_v2.py Video_Games --seeds 42 --memory-variant off \
    --output pilot_cearfn_v2_results.json --artifact-dir cearfn_v2_pilot_artifacts \
    --partial-dir cearfn_v2_pilot_partials
python run_cearfn_v2.py Video_Games --seeds 42 --memory-variant raw-semantic ...
python run_cearfn_v2.py Video_Games --seeds 42 --memory-variant casm ...

# teacher build (after network grant; cache-checked)
python build_minilm_teacher.py Video_Games Baby_Products --device cpu

# analysis (validation-only audit first, then the test read; run twice in this
# pilot — see the two-read protocol deviation disclosed above)
python pilot_guarded_analysis.py    # -> pilot_guarded_audit.json
python pilot_test_read.py           # -> pilot_test_metrics.json
```

The `off` runs reuse the locked v2 memory caches (copied into
`cearfn_v2_pilot_artifacts/`); active variants built their own
variant-tagged caches and rank artifacts
(`<domain>_v2_<variant>_seed42_ranks.npz`).

## 2. Headline result: the validation gate kept both semantic memories OFF

`tune_profiles_v3` (argmax over the frozen 6-slot profile table) selected
profiles with **zero weight on both `semantic_raw` and `casm` slots in both
domains and both regimes** (short: `short_safe` 0.65/0.20/0.15, long:
`session` 1.0). Consequently the fused memory arrays of the raw-semantic and
CASM runs are identical to the locked baseline; the regime-router test rank
vectors are bit-identical across all three variants (verified). Residual
differences below come only from the bucketed/continuous router features,
which see the 6-component tuple.

This reproduces — now under the full journal protocol — the earlier smoke
finding that an unaligned semantic list is rejected by the gate, and extends
it to the aligned CASM list at the declared grid midpoint.

## 3. Variant × domain test metrics (runner-selected router)

### Video_Games (test n=94,762; head/torso/tail = 60,148/25,411/9,203; cold=345)

| variant | router | R@6 | R@10 | R@20 | NDCG@20 | head R@20 | torso | tail | cold |
|---|---|---|---|---|---|---|---|---|---|
| off (locked) | bucketed | 0.07798 | 0.10315 | 0.14511 | 0.06839 | 0.20247 | 0.05549 | 0.01771 | 0.0 |
| raw-semantic | regime | 0.07864 | 0.10424 | 0.14642 | 0.06883 | 0.20430 | 0.05608 | 0.01760 | 0.0 |
| casm | regime | 0.07864 | 0.10424 | 0.14642 | 0.06883 | 0.20430 | 0.05608 | 0.01760 | 0.0 |

raw-semantic and casm rows are identical because the gate zeroed both memory
slots and both runs selected the regime router (identical fused ranks). Their
apparent +0.0013 R@20 over `off` is a **router-selection flip** (regime vs
bucketed under the 6-component features), not a semantic-memory effect.

### Baby_Products (test n=150,777; head/torso/tail = 105,853/33,358/11,566; cold=410)

| variant | router | R@6 | R@10 | R@20 | NDCG@20 | head R@20 | torso | tail | cold |
|---|---|---|---|---|---|---|---|---|---|
| off (locked) | continuous | 0.02360 | 0.03292 | 0.04903 | 0.02196 | 0.06639 | 0.00989 | 0.00303 | 0.0 |
| raw-semantic | continuous | 0.02357 | 0.03290 | 0.04899 | 0.02196 | 0.06633 | 0.00989 | 0.00303 | 0.0 |
| casm | continuous | 0.02353 | 0.03292 | 0.04899 | 0.02196 | 0.06634 | 0.00989 | 0.00294 | 0.0 |

Paired McNemar (variant vs off, test hit@20): VG variants better
(p_off_worse=1.5e-05, again a router flip); Baby variants marginally worse
(raw p=0.046, casm p=0.072) — tiny discordant counts (13–17 of 150k).

**Cold-start (train-freq-0 targets): recall = 0 for every variant in both
domains.** With `exclude_seen=True` and no semantic weight in the fused
profile, unseen items are unreachable; even the standalone semantic
components hit only 1/7 (VG) and 0/10 (Baby) cold validation targets.

## 4. Does CASM beat the raw-semantic control? (component level, validation)

Standalone memory lists (top-20 of each component), validation hit@20:

| domain | component | overall | head | torso | tail | cold |
|---|---|---|---|---|---|---|
| VG | semantic_raw (e5, unaligned) | 0.0490 | 0.0465 | 0.0562 | 0.0475 | 1/7 |
| VG | **casm (aligned)** | **0.0556** | **0.0593** | 0.0506 | 0.0396 | 1/7 |
| VG | transition (reference) | 0.0494 | 0.0509 | 0.0522 | 0.0264 | 0/7 |
| Baby | semantic_raw (SVD, unaligned) | 0.0118 | 0.0111 | 0.0161 | 0.0067 | 0/10 |
| Baby | **casm (aligned)** | 0.0086 | 0.0078 | 0.0121 | 0.0067 | 0/10 |
| Baby | transition (reference) | 0.0070 | 0.0046 | 0.0172 | 0.0033 | 0/10 |

Paired one-sided McNemar (casm vs raw, validation):

| domain | stratum | casm-only hits | raw-only hits | p(raw worse) | p(casm worse) |
|---|---|---|---|---|---|
| VG | overall | 124 | 91 | **0.014** | 0.99 |
| VG | head | 92 | 49 | **0.0002** | 1.00 |
| VG | torso | 27 | 34 | 0.85 | 0.22 |
| VG | tail | 5 | 8 | 0.87 | 0.29 |
| VG | cold | 1 | 1 | 0.75 | 0.75 |
| Baby | overall | 18 | 34 | 0.99 | **0.018** |
| Baby | head | 10 | 22 | 0.99 | **0.025** |
| Baby | torso/tail/cold | — | — | n.s. | n.s. |

**Verdict.** Video_Games: yes overall (+13% relative), but the gain sits
entirely in the **head**; tail and cold show no gain (point estimates go the
wrong way). Baby_Products: **no** — alignment on the 128-d SVD teacher makes
the memory significantly worse. The §4.3 hypothesis (gains concentrate in
tail + cold) is **not supported in either domain**. The head-concentration
pattern on VG is the §6.2 risk signature (InfoNCE on co-occurrence drifting
toward popularity); the declared 1/√freq down-weighting ablation is the
designed response.

## 5. Guarded selection (frozen family, §5) — decision table

Candidate family per domain: 10 candidates (§5 table; per-query validation
hit vectors; exact one-sided McNemar vs validation-best; BH-FDR q=0.10, BY
sensitivity). Full audit in `pilot_guarded_audit.json`.

### Video_Games (comparator = v2_pasgr_bucketed, val utility 0.1358)

| rule | selected | complexity (comp, router, weights) | val utility | test R@20 | test NDCG@20 |
|---|---|---|---|---|---|
| argmax | v2_pasgr_bucketed | (4, 2, 4) | 0.1358 | 0.14511 | 0.06839 |
| 1-SE | v2_pasgr_regime | (4, 1, 4) | 0.1345 | 0.14642 | 0.06883 |
| guarded | v2_pasgr_bucketed | (4, 2, 4) | 0.1358 | 0.14511 | 0.06839 |

BH kept only {v2_pasgr_bucketed, casm_bucketed} eligible (all simpler
candidates detectably worse, including regime at BH-adjusted p=0.063 —
BY sensitivity does NOT reject regime, so the conservative eligible set is
larger). Guarded = argmax here: no simpler candidate survived.

### Baby_Products (comparator = casm_bucketed, val utility 0.0487)

| rule | selected | complexity | val utility | test R@20 | test NDCG@20 |
|---|---|---|---|---|---|
| argmax | casm_bucketed | (4, 2, 4) | 0.0487 | 0.05543 | 0.02625 |
| 1-SE | v2_pasgr_regime | (4, 1, 4) | 0.0468 | 0.05708 | 0.02691 |
| guarded | v2_pasgr_constant_beta | (4, 0, 4) | 0.0457 | 0.05254* | 0.02511* |

Eligible after BH: 6 of 10 candidates (only popularity/transition/v1/
continuous rejected). The guarded rule therefore picked the **simplest
router (constant β)** — strictly simpler than argmax's bucketed pick.
*Constant-β test metrics are approximated by re-fusing the stored top-20
memory/neural lists (the runner does not persist a constant-β test vector);
the validation-selected constant β=0.5 (`pilot_guarded_audit.json`
`extra.constant_beta`, same value in both domains) was fixed on validation
before either test read.

Notably the argmax pick (casm_bucketed) landed on validation utility that
did **not** transfer best to test — 1-SE's simpler regime pick scored higher
on test in both domains — exactly the "everyone's a winner" instability the
guarded rule is designed to expose. Power note: with 5,000 validation
queries the discordant counts between the top candidates are small (6–44),
so non-rejection ≠ equivalence (§3.5); the Baby eligible set is wide because
the tests are low-powered there, and the rule then honestly defaults toward
simplicity.

### Retrospective legacy gates

- **Router gate** (family {regime, bucketed, continuous} on off-variant
  validation ranks): argmax reproduces the runner's picks (VG bucketed);
  guarded picks bucketed on VG and **regime on Baby** (simpler than the
  runner's continuous — which the guarded family analysis flags as
  detectably worse, BH p=0.004; the runner's own nested selector had
  already flipped to continuous on a 1,000-query sub-split, a decision the
  guarded rule would have blocked on the full 5,000).
- **Memory-profile gate** (v1-best vs semantic_*/casm_* profiles): all three
  rules select `v1_best` in both domains — the legacy argmax gating decision
  that kept semantic memories off is reproduced and is also the
  guarded-optimal decision.

## 6. Timing (CPU-only sandbox, per variant run incl. 2× PASGR training)

| run | seconds |
|---|---|
| VG off / raw-semantic / casm | 857 / 896 / 986 |
| Baby off / raw-semantic / casm | 1324 / 1333 / 1165 |

CASM head training itself is ~1–2 min/domain (2M pairs, 5 epochs, linear);
the bulk is PASGR (~7 min ×2 per run) and batched memory construction for
150k test queries (~10–15 min, cached per variant).

## 7. Go/no-go against §6 kill criteria

1. **CASM vs raw control:** worse on Baby (SVD teacher), better-but-head-only
   on VG. The tail/cold hypothesis that motivates CASM is unsupported on
   both pilot domains → per the §6.1 stopping rule the *current* CASM
   configuration is a **publishable rejected candidate**; the full 3-seed
   CASM matrix as designed should NOT be launched.
2. **§6.2 popularity collapse:** the VG head-concentration is the predicted
   signature; the declared 1/√freq pair down-weighting ablation plus the
   now-built MiniLM teacher for Baby are the two cheap validation-only
   probes that could revive CASM without touching test.
3. **J1 (guarded selection): go.** The selector ran end-to-end on real
   per-query artifacts, reproduced two legacy gates, simplified one (Baby
   router), and exposed an argmax pick that failed to transfer — with audit
   records. This arm carries the journal narrative regardless of CASM's fate.

**Recommendation: MIXED.** Proceed to the full matrix for the guarded-
selection study (J1) on the existing 3-seed × 5-domain rank artifacts (no
new training needed). Hold the CASM 3-seed matrix until the two declared
validation-only ablations (freq-down-weighted pairs; matched MiniLM teacher
for Baby) show a tail/cold signal; otherwise report CASM as a rejected
candidate within the rejection-complete framing.

## Artifacts

- `pilot_casm_seed42.json` — all numbers in this report, machine-readable.
- `pilot_guarded_audit.json` — validation-side audit (tests, BH/BY, picks).
- `pilot_test_metrics.json` — final (second) test read after the audit
  correction (variants, strata, decisions); see the deviation disclosure above.
- `cearfn_v2_pilot_artifacts/` — variant-tagged rank/memory npz + CASM caches.
- `pilot_cearfn_v2_results.json`, `pilot_baby_results.json` — runner outputs.
- `semantic_teacher_artifacts/{video_games,baby_products}_minilm.{npy,json}` —
  matched MiniLM teachers, ready for the full phase.
