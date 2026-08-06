"""Offline SLM semantic teacher with frequency-adaptive behavior fusion."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np


def build_semantic_teacher(item_texts: Mapping[int, str], n_items: int,
                           behavior: np.ndarray, item_freq: Counter,
                           cache_path: Path, model_name: str = "intfloat/e5-small"
                           ) -> np.ndarray:
    if behavior.shape[0] != n_items:
        raise ValueError("behavior teacher vocabulary mismatch")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        semantic = np.load(cache_path)
    else:
        ids = sorted(i for i in item_texts if 0 < i < n_items)
        texts = ["passage: " + item_texts[i] for i in ids]
        raw = None
        # A HF directory can contain configuration without weights. Avoid any
        # network call unless a complete local snapshot is present.
        hf_root = Path.home() / ".cache" / "huggingface" / "hub" / \
            ("models--" + model_name.replace("/", "--")) / "snapshots"
        snapshots = list(hf_root.glob("*")) if hf_root.exists() else []
        complete = next((p for p in snapshots if
                         (p / "model.safetensors").exists() or
                         (p / "pytorch_model.bin").exists()), None)
        if complete is not None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(str(complete), local_files_only=True)
            raw = model.encode(texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True, convert_to_numpy=True)
            print(f"  [semantic] encoder=SLM:{model_name}")
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                         max_features=30000, sublinear_tf=True,
                                         stop_words="english")
            sparse = vectorizer.fit_transform(texts)
            out_dim = min(384, max(2, min(sparse.shape) - 1))
            raw = TruncatedSVD(out_dim, random_state=42).fit_transform(sparse)
            raw /= np.maximum(np.linalg.norm(raw, axis=1, keepdims=True), 1e-8)
            print("  [semantic] encoder=TFIDF-SVD (offline fallback; no SLM weights)")
        semantic = np.zeros((n_items, raw.shape[1]), dtype=np.float32)
        semantic[ids] = raw.astype(np.float32)
        np.save(cache_path, semantic)
    if semantic.shape[0] != n_items:
        raise ValueError("semantic cache vocabulary mismatch")
    dim = behavior.shape[1]
    if semantic.shape[1] != dim:
        rng = np.random.default_rng(42)
        projection = rng.standard_normal((semantic.shape[1], dim)).astype(np.float32)
        projection /= np.sqrt(dim)
        semantic = semantic @ projection
    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    behavior_n, semantic_n = norm(behavior.astype(np.float32)), norm(semantic)
    freq = np.asarray([item_freq.get(i, 0) for i in range(n_items)], dtype=np.float32)
    # Well-observed items trust collaborative behavior; sparse items borrow the
    # SLM prior. Items without metadata automatically remain behavioral.
    w = np.sqrt(freq / (freq + 10.0))[:, None]
    has_text = (np.linalg.norm(semantic, axis=1, keepdims=True) > 0)
    w = np.where(has_text, w, 1.0)
    fused = w * behavior_n + (1.0 - w) * semantic_n
    fused[0] = 0
    return norm(fused).astype(np.float32)


def semantic_retrieve(queries: Mapping[str, Mapping[str, Sequence[int]]],
                      semantic: np.ndarray, top_k: int = 200,
                      recent_items: int = 5, batch_size: int = 128,
                      decay: float = 0.65
                      ) -> Dict[str, List[Tuple[int, float]]]:
    """Retrieve catalog items from metadata embeddings, without target access.

    The query vector is an exponentially recency-weighted centroid of the
    observed item embeddings.  Returning scores as well as ids lets the
    downstream ranker distinguish a confident semantic match from a weak one.
    Computation is batched so a 25K-item Amazon catalog stays memory bounded.
    """
    if semantic.ndim != 2:
        raise ValueError("semantic embeddings must be a 2-D matrix")
    n_items = semantic.shape[0]
    emb = semantic.astype(np.float32, copy=True)
    emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    keys = list(queries)
    output: Dict[str, List[Tuple[int, float]]] = {}
    k = min(max(int(top_k), 0), max(n_items - 1, 0))
    if k == 0:
        return {uid: [] for uid in keys}
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        reps = np.zeros((len(batch_keys), emb.shape[1]), dtype=np.float32)
        contexts: List[List[int]] = []
        for row, uid in enumerate(batch_keys):
            ctx = [int(x) for x in queries[uid].get("context", [])
                   if 0 < int(x) < n_items][-recent_items:]
            contexts.append(ctx)
            if not ctx:
                continue
            # newest item has weight 1; earlier evidence decays geometrically
            weights = np.power(decay, np.arange(len(ctx) - 1, -1, -1)).astype(np.float32)
            reps[row] = (emb[ctx] * weights[:, None]).sum(axis=0) / weights.sum()
        reps /= np.maximum(np.linalg.norm(reps, axis=1, keepdims=True), 1e-8)
        scores = reps @ emb.T
        scores[:, 0] = -np.inf
        for row, (uid, ctx) in enumerate(zip(batch_keys, contexts)):
            if ctx:
                scores[row, list(set(ctx))] = -np.inf
            top = np.argpartition(-scores[row], k - 1)[:k]
            top = top[np.argsort(-scores[row, top])]
            output[uid] = [(int(i), float(scores[row, i])) for i in top
                           if np.isfinite(scores[row, i])]
    return output
