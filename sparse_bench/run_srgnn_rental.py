"""
Gate-check: SR-GNN (graph) vs CoDT (transformer+fusion) vs baselines on Rental.

Question: does the graph-structured propagation that SR-GNN uses add value on
Rental's visit-level short sessions, where CoDT already reaches R@20=0.664?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loaders
import grouped_eval
import srgnn_model as G
import baselines as B

K_EVAL = [6, 10, 20]


def score(name, preds, data):
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=K_EVAL)
    o = gm["overall"]
    print(f"{name:16} R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f} "
          f"P@20={o['precision@20']:.4f} MRR@20={o['mrr@20']:.4f} | "
          f"tgt_head R@20={gm['tgt_head']['recall@20']:.3f} tgt_mid R@6={gm['tgt_mid']['recall@6']:.3f}")
    return gm


def main():
    print("=" * 70)
    print("SR-GNN vs CoDT vs baselines on Rental (visit-level)")
    print("=" * 70)
    data = loaders.load_rental_visit()
    n_items = data["n_items"]
    test_uids = sorted(data["test_queries"].keys())

    # 1. Non-parametric baselines (fast, no training)
    print("\n--- baselines ---")
    for name in ["MostPop", "ItemKNN", "SKNN"]:
        preds = B.run_nonparametric(name, data["train_sessions"], data["test_queries"], n_items)
        score(name, preds, data)

    # 2. SR-GNN (the graph model)
    print("\n--- SR-GNN (graph) ---")
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]
    print(f"train sessions: {len(sessions)}")
    t0 = time.time()
    models = G.train_srgnn(sessions, n_items, data["item_freq"], epochs=15, seeds=(42, 123, 456))
    print(f"trained in {time.time() - t0:.0f}s")
    preds = G.predict_srgnn(models, test_uids, data["test_queries"], n_items)
    gm_gnn = score("SR-GNN", preds, data)
    print("\nSR-GNN grouped table:")
    print(grouped_eval.format_group_table(gm_gnn, k_show=6))
    print("\nReference: CoDT R@6=0.4131 R@10=0.5174 R@20=0.6641")


if __name__ == "__main__":
    main()
