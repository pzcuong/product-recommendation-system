"""
LTAR — Long-Tail-Aware Reranker (novel contribution).
=====================================================

Problem (measured on Rental): the existing grouped evaluation reports
`tgt_tail = 0.0` recall — long-tail items are *never* recommended because PGSA
and co-visitation both concentrate probability on head items. This kills the
"sparse long-tail recommendation" framing.

LTAR fixes this with two complementary mechanisms, applied after fusion and
*before* MMR, so it composes cleanly with CoDT:

  1. Popularity-stratified candidate injection (CSI).
     For tail items that are semantically close to the context (high M-CL
     similarity to the last context items) but were dropped below the PGSA
     top-k cut, re-inject them into the candidate pool. Concretely: for each
     tail item `j`, keep it if its max M-CL cosine to the last-5 context items
     exceeds a per-tier threshold. This rescues tail items that the neural
     ranker under-scored for lack of training frequency, without polluting the
     list with unrelated tail items.

  2. Inverse-popularity log-bias.
     Add a small additive boost to each candidate scaled by inverse log
     popularity: `bias = beta * (1 - log(1+freq)/log(1+fmax))`. This nudges
     the ranker toward less-popular items when their fused score is close to a
     head competitor, breaking ties in favour of the tail. `beta` is the only
     hyperparameter and is capped so head recall is not sacrificed.

Tiers are 20/60/20 over training item frequency (matching the grouped eval),
so LTAR directly targets the `tgt_tail` bucket.

Reference: research_proposal.md §8.3 (tgt_tail = 0 is the documented weakness).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np


def compute_tiers(item_freq: Counter) -> Tuple[set, set, set]:
    """20/60/20 head/mid/tail split over training item frequency."""
    if not item_freq:
        return set(), set(), set()
    sorted_items = sorted(item_freq.items(), key=lambda x: x[1], reverse=True)
    n = len(sorted_items)
    head = set(i for i, _ in sorted_items[: max(1, int(n * 0.2))])
    tail = set(i for i, _ in sorted_items[max(1, int(n * 0.8)):])
    mid = set(i for i, _ in sorted_items) - head - tail
    return head, mid, tail


def build_inverse_pop_bias(item_freq: Counter, n_items: int, beta: float = 0.08) -> np.ndarray:
    """Per-item additive bias in [0, beta], larger for less-popular items."""
    bias = np.zeros(n_items, dtype=np.float32)
    if not item_freq:
        return bias
    fmax = max(item_freq.values())
    log_fmax = math.log(1 + fmax)
    for item, freq in item_freq.items():
        if 0 < item < n_items:
            bias[item] = beta * (1.0 - math.log(1 + freq) / log_fmax)
    return bias


def ltar_rerank(fused: List[Tuple[int, float]],
                mcl_emb: np.ndarray,
                ctx_items: List[int],
                item_freq: Counter,
                n_items: int,
                head: set, tail: set,
                beta: float = 0.08,
                sim_threshold: float = 0.30,
                max_inject: int = 30,
                tail_pool_size: int = 500) -> List[Tuple[int, float]]:
    """Apply LTAR to a single query's fused candidate list.

    Steps:
      1. Build inverse-pop bias and add it to every fused candidate.
      2. Inject up to `max_inject` tail candidates that have high M-CL
         similarity to the context but were absent from `fused`. The injection
         pool is restricted to the top `tail_pool_size` tail items by max
         similarity to the context (so we never scan the whole vocab).

    Args:
        fused        : list of (item, score) from CoDT fusion.
        mcl_emb      : [n_items, dim] L2-normalized M-CL embeddings (may be None
                       to disable injection but keep the inverse-pop bias).
        ctx_items    : last context items (used for similarity gating).
        beta         : inverse-popularity bias strength.
        sim_threshold: min M-CL cosine to a context item for a tail item to be injected.
        max_inject   : cap on injected tail candidates.
        tail_pool_size: how many tail items to consider for injection.

    Returns:
        new list of (item, score) ready for MMR.
    """
    if not fused and mcl_emb is None:
        return list(fused)

    bias = build_inverse_pop_bias(item_freq, n_items, beta=beta)

    # 1) Apply inverse-pop bias to existing candidates.
    reranked = [(it, sc + float(bias[it])) for it, sc in fused]
    have = set(it for it, _ in reranked)

    # 2) Inject semantically-close tail items absent from the candidate list.
    if mcl_emb is not None and ctx_items and tail:
        ctx_recent = [x for x in ctx_items[-5:] if 0 < x < mcl_emb.shape[0]]
        if ctx_recent:
            ctx_mat = mcl_emb[ctx_recent]                      # [k, dim]
            # Restrict to tail items only and rank by max-sim to context.
            tail_arr = np.fromiter((t for t in tail if 0 < t < mcl_emb.shape[0]),
                                   dtype=np.int64)
            if tail_arr.size:
                tail_mat = mcl_emb[tail_arr]                   # [m, dim]
                sims = tail_mat @ ctx_mat.T                    # [m, k]
                max_sim = sims.max(axis=1)                     # [m]
                # Take a pool, then threshold.
                pool = min(tail_pool_size, tail_arr.size)
                top_pool_idx = np.argpartition(-max_sim, pool - 1)[:pool]
                kept = []
                for idx in top_pool_idx:
                    if max_sim[idx] >= sim_threshold:
                        item = int(tail_arr[idx])
                        if item not in have:
                            kept.append((max_sim[idx], item))
                kept.sort(reverse=True)
                for rank, (sim, item) in enumerate(kept[:max_inject]):
                    # injected candidates get a modest score: relative sim
                    # scaled so they compete with, but don't dominate, head items.
                    score = float(bias[item]) + 0.10 * float(sim) - 0.01 * rank
                    reranked.append((item, score))
    return reranked
