# CEARF-N multi-seed evidence audit

## Verdict

The new full-catalog evidence suite confirms a reproducible positive result on
Video Games and Baby Products. CEARF-N beats both of its direct constituents
(validation-selected CEARF memory and PASGR) in overall Recall@20 for every one
of five independent seeds on each dataset. All paired 20,000-repetition
bootstrap intervals exclude zero and all exact McNemar tests reject equal hit
rates by a wide margin.

This establishes a new **internal benchmark best on two domains**. It does not
establish literature-wide SOTA because the Amazon preprocessing and evaluation
protocol have not yet been matched to published external baselines.

## Locked protocol

- Explicit Amazon train, validation, and test splits are retained separately.
- Memory profile and neural fusion beta are selected on a deterministic 5,000
  validation-query subset.
- The test labels are not used until profile and beta selection are complete.
- Previously observed targets are excluded consistently; both domains have
  zero target-in-context violations.
- Evaluation is exact full-catalog retrieval, not sampled-candidate ranking.
- Neural seeds are `42, 43, 44, 123, 456`; the complete stochastic pipeline,
  including prototype initialization, changes with the seed.
- Query-level rank artifacts support paired inference without rerunning models.

## Five-seed results

| Dataset | Method | Recall@6 | Recall@10 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|
| Video Games | CEARF | .06932 | .08821 | .11901 | .06009 |
| Video Games | PASGR | .06327±.00057 | .08604±.00082 | .12605±.00053 | .05694±.00031 |
| Video Games | **CEARF-N** | **.07732±.00088** | **.10328±.00086** | **.14577±.00086** | **.06830±.00055** |
| Baby Products | CEARF | .02288 | .02900 | .03879 | .02017 |
| Baby Products | PASGR | .02267±.00022 | .03068±.00024 | .04492±.00028 | .02046±.00018 |
| Baby Products | **CEARF-N** | **.02965±.00028** | **.03836±.00014** | **.05370±.00036** | **.02609±.00009** |

Values after `±` are sample standard deviations over five seeds, not confidence
intervals. Student-t 95% intervals are stored in the JSON artifact.

Relative Recall@20 gains:

- Video Games: +22.5% over CEARF and +15.6% over PASGR.
- Baby Products: +38.4% over CEARF and +19.6% over PASGR.

## Paired inference

For Video Games, the per-seed CEARF-N-minus-CEARF Recall@20 gain ranges from
.02560 to .02753. The weakest lower bound among the five paired 95% bootstrap
intervals is .02414. Against the matched PASGR seed, gains range from .01918 to
.02024 and the weakest lower bound is .01782.

For Baby Products, the per-seed gain over CEARF ranges from .01447 to .01541;
the weakest lower bound is .01363. Gains over matched PASGR range from .00838
to .00916 and the weakest lower bound is .00751.

All ten seeds therefore preserve a strictly positive paired interval against
both direct constituents. Exact p-values and discordant hit counts are retained
per seed in `cearfn_evidence_results.json`.

## Component ablation

| Dataset | Transition R@20 | Popularity R@20 | Session memory R@20 | PASGR R@20 | CEARF-N R@20 |
|---|---:|---:|---:|---:|---:|
| Video Games | .04223 | .03787 | .11901 | .12605 | **.14577** |
| Baby Products | .00735 | .02908 | .03879 | .04492 | **.05370** |

Validation selects pure neighbour-session memory on both Amazon domains. Thus,
the positive result is specifically evidence for rank fusion between session
retrieval and the PASGR neural residual. Transition and popularity are genuine
available components, but they are rejected by validation here and must not be
described as active contributors to these two test results.

## Head/tail audit

Head items are the smallest popularity-sorted catalog subset covering 80% of
training interactions; all remaining items are tail.

| Dataset | Method | Head-target R@20 | Tail-target R@20 |
|---|---|---:|---:|
| Video Games | CEARF | .15477 | .02361 |
| Video Games | PASGR | .16030 | **.03465** |
| Video Games | CEARF-N | **.18749** | .03448 |
| Baby Products | CEARF | .04881 | .00657 |
| Baby Products | PASGR | .05618 | .00869 |
| Baby Products | CEARF-N | **.06751** | **.00929** |

CEARF-N improves tail-target recall over CEARF on both datasets and over PASGR
on Baby Products. On Video Games it is approximately tied with, and slightly
below, PASGR on tail targets. The overall Video Games gain is therefore driven
primarily by head-target ranking. CEARF-N must not be claimed as uniformly
tail-superior.

## Claim boundary and paper wording

Defensible claim:

> Under a leakage-audited unified full-catalog protocol, validation-calibrated
> retrieval--neural rank fusion improves overall Recall@20 over either direct
> constituent in all five seeds on two Amazon long-tail domains, with paired
> confidence intervals excluding zero.

Important limitations:

1. These two Amazon splits contain no short-context validation/test queries, so
   they do not test the short/long context router. They validate beta calibration
   and fusion, not context-conditional routing.
2. Literature-wide SOTA still requires external official implementations under
   byte-identical preprocessing and candidate rules.
3. The head/tail result is mixed on Video Games and should be reported exactly.
4. Five seeds support stability, not a guarantee of venue acceptance.

## Artifacts

- `cearfn_evidence_results.json`: metrics, selection, ablations, groups, paired tests.
- `cearfn_evidence_artifacts/*_memory.npz`: fixed query order and memory rankings.
- `cearfn_evidence_artifacts/*_seed*_ranks.npz`: query-level CEARF-N/PASGR/CEARF ranks.
- `run_cearfn_evidence.py`: reproducible evidence runner.
