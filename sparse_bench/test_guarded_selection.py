import numpy as np
import pytest
from scipy.stats import binom

from guarded_selection import (
    Candidate,
    bh_adjust,
    by_adjust,
    candidate_from_ranks,
    compare_selection_rules,
    guarded_selection,
    mcnemar_one_sided_p,
    select_argmax,
    select_one_se,
    sign_flip_one_sided_p,
)


def make_candidate(name, hits, active=1, router=None, weights=None,
                   position=0):
    hits = tuple(int(x) for x in hits)
    return Candidate(name=name, hits=hits,
                     utilities=tuple(float(x) for x in hits),
                     active_components=active, router=router,
                     total_weights=weights if weights is not None else active,
                     position=position)


def test_guarded_picks_simpler_equal_candidate():
    # (i) complex candidate and simple candidate have IDENTICAL hits →
    # zero discordance → p = 1 → simple one is eligible and selected.
    rng = np.random.default_rng(0)
    hits = rng.integers(0, 2, size=400)
    complex_c = make_candidate("complex", hits, active=4, router="bucketed",
                               weights=8, position=0)
    simple_c = make_candidate("simple", hits, active=1, router="regime",
                              weights=1, position=1)
    audit = guarded_selection([complex_c, simple_c])
    assert audit["selected"] == "simple"
    assert audit["eligible"] == ["complex", "simple"] or \
        set(audit["eligible"]) == {"complex", "simple"}


def test_guarded_rejects_genuinely_worse_simple_candidate():
    # (ii) simple candidate misses on 80 queries the best hits → detectably
    # worse → guarded keeps the (more complex) best.
    n = 500
    best_hits = np.ones(n, dtype=int)
    worse_hits = best_hits.copy()
    worse_hits[:80] = 0
    best = make_candidate("best", best_hits, active=4, router="bucketed",
                          weights=8, position=0)
    worse = make_candidate("worse_simple", worse_hits, active=1,
                           router="regime", weights=1, position=1)
    audit = guarded_selection([best, worse])
    assert audit["comparator"] == "best"
    (test,) = audit["tests"]
    assert test["candidate"] == "worse_simple"
    assert test["detectably_worse_bh"]
    assert audit["selected"] == "best"


def test_all_equal_candidates_selects_simplest():
    # (iii) every candidate identical → all eligible → minimum complexity,
    # ties broken by declared position.
    hits = np.ones(50, dtype=int)
    family = [
        make_candidate("c_bucketed", hits, active=3, router="bucketed",
                       weights=6, position=0),
        make_candidate("c_simple_late", hits, active=1, router="regime",
                       weights=1, position=2),
        make_candidate("c_simple_early", hits, active=1, router="regime",
                       weights=1, position=1),
    ]
    audit = guarded_selection(family)
    assert audit["selected"] == "c_simple_early"
    assert set(audit["eligible"]) == {"c_bucketed", "c_simple_late",
                                      "c_simple_early"}


def test_degenerate_zero_discordance_pvalue_is_one():
    # (iv) zero discordant pairs → declared handling p = 1.
    assert mcnemar_one_sided_p(0, 0) == 1.0
    hits = np.zeros(30, dtype=int)
    a = make_candidate("a", hits, active=2, position=0)
    b = make_candidate("b", hits, active=1, position=1)
    audit = guarded_selection([a, b])
    assert audit["tests"][0]["p_value"] == 1.0
    assert audit["selected"] == "b"


def test_mcnemar_matches_binomial_lower_tail():
    assert mcnemar_one_sided_p(9, 1) == pytest.approx(
        float(binom.cdf(1, 10, 0.5)))
    # symmetric case: candidate at least as good → p large
    assert mcnemar_one_sided_p(2, 8) > 0.5


def test_bh_fdr_monotonicity():
    # (v) BH adjusted p-values are monotone in the raw p ordering and never
    # smaller than the raw p-value.
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, size=25)
    adjusted, rejected = bh_adjust(p, q=0.10)
    order = np.argsort(p)
    sorted_adjusted = np.asarray(adjusted)[order]
    assert np.all(np.diff(sorted_adjusted) >= -1e-12)
    assert np.all(np.asarray(adjusted) >= p - 1e-12)
    # rejections form a prefix of the sorted p-values
    flags = np.asarray(rejected)[order]
    if flags.any():
        last_true = np.max(np.flatnonzero(flags))
        assert flags[:last_true + 1].all()


def test_bh_against_reference_example():
    # Worked example: p = (.01, .04, .03, .005), m=4.
    adjusted, rejected = bh_adjust([0.01, 0.04, 0.03, 0.005], q=0.05)
    assert adjusted == pytest.approx([0.02, 0.04, 0.04, 0.02])
    assert rejected == [True, True, True, True]


def test_by_is_more_conservative_than_bh():
    p = [0.001, 0.01, 0.02, 0.2, 0.9]
    bh_adj, _ = bh_adjust(p)
    by_adj, _ = by_adjust(p)
    assert all(b >= a - 1e-12 for a, b in zip(bh_adj, by_adj))


def test_sign_flip_detects_worse_candidate():
    rng = np.random.default_rng(2)
    diffs = np.zeros(300)
    diffs[:60] = -1.0          # candidate loses on 60 queries
    diffs[60:70] = 1.0         # wins on 10
    p = sign_flip_one_sided_p(diffs, reps=5000, seed=3)
    assert p < 0.01
    assert sign_flip_one_sided_p(np.zeros(10)) == 1.0


def test_argmax_and_one_se_agreement_on_clear_winner():
    # A dominant candidate: argmax picks it; 1-SE also picks it when nothing
    # simpler is within one SE.
    n = 400
    strong = np.ones(n, dtype=int)
    weak = np.zeros(n, dtype=int)
    family = [
        make_candidate("weak_simple", weak, active=1, position=0),
        make_candidate("strong_complex", strong, active=3, router="bucketed",
                       weights=6, position=1),
    ]
    assert select_argmax(family)["selected"] == "strong_complex"
    assert select_one_se(family)["selected"] == "strong_complex"
    audit = guarded_selection(family)
    assert audit["selected"] == "strong_complex"


def test_one_se_prefers_simple_within_band():
    # Simple candidate within one SE of the best → 1-SE picks it while
    # argmax keeps the complex best.
    rng = np.random.default_rng(4)
    base = rng.integers(0, 2, size=500)
    near = base.copy()
    flip = rng.choice(500, size=3, replace=False)
    near[flip] = 1 - near[flip]
    if near.mean() > base.mean():   # ensure "complex" is the argmax winner
        base, near = near, base
    family = [
        make_candidate("complex_best", base, active=4, router="continuous",
                       weights=8, position=0),
        make_candidate("simple_near", near, active=1, router="regime",
                       weights=1, position=1),
    ]
    # make complex the utility argmax
    if select_argmax(family)["selected"] != "complex_best":
        pytest.skip("random draw made simple the argmax; construction issue")
    assert select_one_se(family, seed=5)["selected"] == "simple_near"


def test_determinism_same_inputs_same_selection():
    rng = np.random.default_rng(6)
    family = []
    for position in range(6):
        hits = rng.integers(0, 2, size=200)
        family.append(make_candidate(f"c{position}", hits,
                                     active=1 + position % 3,
                                     router=("regime", "bucketed",
                                             "continuous")[position % 3],
                                     weights=1 + position, position=position))
    first = compare_selection_rules(family, seed=7)
    second = compare_selection_rules(family, seed=7)
    assert first == second


def test_candidate_from_ranks_hits_and_utility():
    ranks = np.asarray([0, 1, 6, 7, 20, 21 % 21], dtype=np.uint8)
    ranks[5] = 0
    candidate = candidate_from_ranks("x", ranks, active_components=2)
    assert candidate.hits == (0, 1, 1, 1, 1, 0)
    # utility = .5*hit6 + .5*hit20
    assert candidate.utilities[1] == pytest.approx(1.0)   # rank 1: both
    assert candidate.utilities[3] == pytest.approx(0.5)   # rank 7: only @20


def test_complexity_lexicographic_order():
    a = Candidate("a", (1,), (1.0,), active_components=2, router="regime",
                  total_weights=9)
    b = Candidate("b", (1,), (1.0,), active_components=2, router="bucketed",
                  total_weights=1)
    c = Candidate("c", (1,), (1.0,), active_components=1,
                  router="continuous", total_weights=99)
    assert c.complexity() < a.complexity() < b.complexity()


def test_tiny_n_and_errors():
    with pytest.raises(ValueError):
        guarded_selection([])
    a = make_candidate("a", [1], active=1, position=0)
    b = make_candidate("b", [0], active=2, position=1)
    audit = guarded_selection([a, b])   # n = 1: valid, low power
    assert audit["selected"] in {"a", "b"}
    mismatched = make_candidate("m", [1, 0], active=1)
    with pytest.raises(ValueError):
        guarded_selection([a, mismatched])
