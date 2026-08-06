"""
HACL-SBR ablations on Rental: isolate each component's contribution.

Ablations (each = full config minus one component):
  full              : text-view + graph-view + adaptive-CL + hard-neg
  -text-view        : drop the item-text augmentation view
  -graph-view       : drop the co-visitation augmentation view
  random-neg        : uniform negatives instead of popularity-stratified hard
  -adaptive-loss    : plain InfoNCE instead of MLP-weighted adaptive loss
  no-CL (rec only)  : lambda_cl = 0 (pure sampled-softmax SASRec)

All use 2 seeds × 12 epochs for tractable total runtime, cw=0.3.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hacl, loaders, grouped_eval, run_hacl_rental as R


def run_variant(name, **kw):
    data = loaders.load_rental_visit()
    slug2id = R._load_slug2id()
    slug2text = R.build_slug2text()
    item_text = {idx: slug2text[s] for s, idx in slug2id.items() if s in slug2text}
    text_emb = hacl.build_tfidf(list(range(data["n_items"])), item_text)
    text_knn = R._build_text_knn(text_emb, 10)
    covisit = R._build_covisit(list(data["train_sessions"].values()), 5, 10)
    sessions = [s for s in data["train_sessions"].values() if len(s) >= 2]

    text_knn_arg = text_knn if kw.get("use_text_view", True) else {}
    covisit_arg = covisit if kw.get("use_graph_view", True) else {}
    models = hacl.train_hacl(
        sessions, data["n_items"], text_emb, text_knn_arg, covisit_arg,
        data["item_freq"], epochs=12, seeds=[42, 123],
        use_text_view=kw.get("use_text_view", True),
        use_graph_view=kw.get("use_graph_view", True),
        hard_neg=kw.get("hard_neg", True),
        use_adaptive=kw.get("use_adaptive", True),
        lambda_cl=kw.get("lambda_cl", 0.5),
    )
    test_uids = sorted(data["test_queries"].keys())
    preds = hacl.predict_hacl(models, test_uids, data["test_queries"], data["n_items"],
                              covisit=covisit, covisit_weight=0.3)
    gm = grouped_eval.evaluate_all_groups(preds, data, k_values=[6, 10, 20])
    o = gm["overall"]
    res = {"R@6": o["recall@6"], "R@10": o["recall@10"], "R@20": o["recall@20"],
           "MRR@20": o["mrr@20"], "tgt_mid_R@6": gm["tgt_mid"]["recall@6"],
           "tgt_head_R@20": gm["tgt_head"]["recall@20"]}
    print(f"{name:18} R@6={res['R@6']:.4f} R@10={res['R@10']:.4f} R@20={res['R@20']:.4f} "
          f"MRR@20={res['MRR@20']:.4f} | tgt_mid R@6={res['tgt_mid_R@6']:.3f}")
    return res


if __name__ == "__main__":
    out = {}
    out["full"]              = run_variant("full")
    out["-text-view"]        = run_variant("-text-view",        use_text_view=False)
    out["-graph-view"]       = run_variant("-graph-view",       use_graph_view=False)
    out["random-neg"]        = run_variant("random-neg",        hard_neg=False)
    out["-adaptive-loss"]    = run_variant("-adaptive-loss",    use_adaptive=False)
    out["no-CL (rec only)"]  = run_variant("no-CL (rec only)",  lambda_cl=0.0)
    Path("sparse_bench/hacl_ablations.json").write_text(json.dumps(out, indent=2))
    print("\nSaved to sparse_bench/hacl_ablations.json")
