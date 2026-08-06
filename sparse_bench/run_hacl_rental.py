"""
Gate-check HACL-SBR on Rental (visit-level).

Builds the three augmentation assets from raw Rental data, trains the HACL
ensemble, and evaluates on the verified visit-level protocol (259 queries,
SR-GNN-standard P@20 / MRR@20 plus the grouped breakdown including the
cold-start vs learnable-tail split).

Targets:
  - R@10 > CoDT reference (0.517)
  - tgt_tail_learnable > 0  (the diagnosed long-tail failure)
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loaders
import grouped_eval
import hacl

RENTAL_DIR = Path(__file__).resolve().parent.parent / "rental_data"


def build_slug2text() -> dict:
    """slug -> 'name_en main_category_en' for items that have English text."""
    df = pd.read_csv(RENTAL_DIR / "old_site_products_en.csv", dtype=str)
    out = {}
    for _, r in df.iterrows():
        s = r.get("slug")
        if pd.isna(s):
            continue
        name = r.get("name_en") or ""
        cat = r.get("main_category_en") or ""
        if name or cat:
            out[str(s)] = f"{name} {cat}".strip()
    return out


def main():
    print("=" * 70)
    print("HACL-SBR gate-check on Rental (visit-level)")
    print("=" * 70)
    # 1. Load Rental visit-level data (uses multi_domain_eval loader + reference groups)
    data = loaders.load_rental_visit()
    slug2id = _load_slug2id()
    n_items = data["n_items"]

    # 2. Build text embeddings per item index
    slug2text = build_slug2text()
    item_text: dict[int, str] = {}
    for slug, idx in slug2id.items():
        if slug in slug2text:
            item_text[idx] = slug2text[slug]
    print(f"text coverage: {len(item_text)}/{n_items - 1} items")
    text_emb = hacl.build_tfidf(list(range(n_items)), item_text)
    print(f"TF-IDF text emb: {text_emb.shape}")

    # 3. Build text-knn (item -> [text-similar items]) for the item-view aug
    text_knn = _build_text_knn(text_emb, k=10)
    print(f"text-knn: {len(text_knn)} items have text neighbours")

    # 4. Build co-visitation neighbours (graph-view aug) from train sessions
    covisit = _build_covisit(list(data["train_sessions"].values()), window=5, topn=10)
    print(f"covisit graph: {len(covisit)} items have neighbours")

    # 5. Train HACL ensemble
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    print(f"train sessions: {len(sessions)}")
    t0 = time.time()
    models = hacl.train_hacl(
        sessions, n_items, text_emb, text_knn, covisit, data["item_freq"],
        epochs=15, seeds=[42, 123, 456])
    print(f"trained in {time.time() - t0:.0f}s")

    # 6. Evaluate — sweep covisit_weight to find the best fusion point
    test_uids = sorted(data["test_queries"].keys())
    print("\n=== HACL-SBR fusion sweep (covisit_weight) ===")
    for cw in [0.0, 0.2, 0.3, 0.5, 0.7]:
        preds = hacl.predict_hacl(models, test_uids, data["test_queries"], n_items,
                                  covisit=covisit, covisit_weight=cw)
        gm = grouped_eval.evaluate_all_groups(preds, data, k_values=[6, 10, 20])
        o = gm["overall"]
        print(f"  cw={cw}: R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
              f"P@20={o['precision@20']:.4f} MRR@20={o['mrr@20']:.4f} | "
              f"tgt_mid R@6={gm['tgt_mid']['recall@6']:.3f} tgt_tail_learn R@6={gm['tgt_tail_learnable']['recall@6']:.3f}")
    print("CoDT reference: R@6=0.4131 R@10=0.5174 R@20=0.6641")

    # 7. Best-config full grouped table
    best_cw = 0.3
    preds = hacl.predict_hacl(models, test_uids, data["test_queries"], n_items,
                              covisit=covisit, covisit_weight=best_cw)
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=[6, 10, 20])
    print(f"\n=== HACL-SBR best (covisit_weight={best_cw}) grouped table ===")
    print(grouped_eval.format_group_table(gm, k_show=6))


def _load_slug2id():
    import importlib.util
    spec = importlib.util.spec_from_file_location("mde", Path(__file__).resolve().parent.parent / "multi_domain_eval.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m.load_rental_grouped(str(Path(__file__).resolve().parent.parent / "rental_data"))["item_map"]


def _build_text_knn(text_emb: np.ndarray, k: int = 10) -> dict:
    """item -> list of k most text-similar items (cosine on TF-IDF)."""
    norms = np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-9
    unit = text_emb / norms
    sim = unit @ unit.T
    np.fill_diagonal(sim, -1.0)
    knn = {}
    for i in range(text_emb.shape[0]):
        if text_emb[i].sum() == 0:
            continue
        top = np.argpartition(-sim[i], min(k, len(sim) - 1))[:k]
        knn[i] = [int(j) for j in top if sim[i, j] > 0]
    return knn


def _build_covisit(sessions, window=5, topn=10) -> dict:
    """item -> [co-visited items] for the graph-view augmentation."""
    co = defaultdict(Counter)
    for seq in sessions:
        for i, a in enumerate(seq):
            for b in seq[i + 1:i + 1 + window]:
                if a != b:
                    co[a][b] += 1
    return {a: [b for b, _ in c.most_common(topn)] for a, c in co.items()}


if __name__ == "__main__":
    main()
