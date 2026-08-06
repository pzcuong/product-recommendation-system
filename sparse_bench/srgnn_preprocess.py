"""
SR-GNN-protocol preprocessing for session-based recommendation benchmarks.

Faithful port of the official CRIPAC-DIG/SR-GNN `datasets/preprocess.py`
(https://github.com/CRIPAC-DIG/SR-GNN/blob/master/datasets/preprocess.py),
which is the canonical preprocessing used by virtually every session-based
recommendation paper (SR-GNN, GCE-GNN, LESSR, CITD, ...). Using this exact
protocol is REQUIRED to compare Recall@20 / MRR@20 against published numbers.

Pipeline:
  1. Sessionize raw events by (visitor + 30-min inactivity gap).
  2. Drop sessions of length < 2.
  3. Item support filter (drop items appearing < `min_support` times; re-filter
     sessions that become length < 2).
  4. Temporal split: sessions whose first timestamp falls in the last
     `test_days` days -> test; the rest -> train.
  5. Build a single contiguous item vocabulary over TRAIN items (1-indexed; 0 = PAD).
  6. process_seqs: TRAIN augmentation — each session [a,b,c,d] becomes the
     prefix examples ([a]->b), ([a,b]->c), ([a,b,c]->d). TEST: one query per
     session = (seq=full[:-1], target=full[-1]); drop test items not in vocab
     (cold items, per SR-GNN's obtian_tes).

Each loader returns a dict with the SAME keys as the rest of sparse_bench:
    {
      "domain": str,
      "n_items": int,            # 0 = PAD
      "train_sessions": dict[int->List[int]],   # augmented? No — raw sessions;
                                                # SeqDataset does next-item augmentation internally.
      "test_queries": dict[str->{"context": List[int], "targets": List[int]}],
      "item_freq": Counter[int],
      "item_categories": {},     # none for these benchmarks
      "visit_counts": {},        # n/a
      "reference_groups": None,
    }

NOTE on augmentation: SR-GNN's `process_seqs` produces explicit (prefix, label)
training pairs. Our SeqDataset (codt_core.py) already generates the same
next-item examples from a session via its sliding window, so we feed it the
*raw sessions* (length >= 2). This is equivalent and avoids materializing
millions of pairs in memory.
"""

from __future__ import annotations

import math
import os
import pickle
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
CACHE_DIR = _HERE / "_srgnn_cache"
CACHE_DIR.mkdir(exist_ok=True)

SESSION_TIMEOUT = 30 * 60  # 30 minutes
MIN_SESSION_LEN = 2
DEFAULT_MIN_SUPPORT = 5


# =============================================================================
# Core SR-GNN steps
# =============================================================================
def sessionize(df: pd.DataFrame,
               visitor_col: str = "visitorid",
               item_col: str = "itemid",
               ts_col: str = "timestamp") -> List[Tuple[float, List[str]]]:
    """Group events into sessions by visitor + 30-min inactivity gap.

    Auto-detects millisecond vs second timestamps. Returns a list of
    (start_timestamp_in_seconds, [raw_itemids]) in time order.
    """
    df = df[[visitor_col, item_col, ts_col]].dropna().copy()
    df[visitor_col] = df[visitor_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    ts_num = pd.to_numeric(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    df["_ts"] = ts_num
    # Auto-detect ms vs s: a 13-digit epoch is milliseconds.
    median_ts = df["_ts"].median()
    if median_ts > 1e12:  # milliseconds
        df["_ts"] = df["_ts"] / 1000.0
    df = df.sort_values([visitor_col, "_ts"]).reset_index(drop=True)

    sessions: List[Tuple[float, List[str]]] = []
    cur_vid = None
    cur_sess: List[str] = []
    last_ts = 0.0
    cur_start = 0.0
    for vid, iid, ts in zip(df[visitor_col].values, df[item_col].values, df["_ts"].values):
        if vid != cur_vid or (ts - last_ts) > SESSION_TIMEOUT:
            if len(cur_sess) >= MIN_SESSION_LEN:
                sessions.append((cur_start, cur_sess))
            cur_sess = []
            cur_vid = vid
            cur_start = ts
        cur_sess.append(iid)
        last_ts = ts
    if len(cur_sess) >= MIN_SESSION_LEN:
        sessions.append((cur_start, cur_sess))
    return sessions


def filter_support(sessions: List[Tuple[float, List[str]]],
                   min_support: int = DEFAULT_MIN_SUPPORT
                   ) -> List[Tuple[float, List[str]]]:
    """Drop items appearing < min_support times, then drop sessions < 2 long."""
    item_cnt: Counter = Counter()
    for _, sess in sessions:
        item_cnt.update(sess)
    valid_items = {it for it, c in item_cnt.items() if c >= min_support}
    out = []
    for ts, sess in sessions:
        f = [it for it in sess if it in valid_items]
        if len(f) >= MIN_SESSION_LEN:
            out.append((ts, f))
    return out


def temporal_split(sessions: List[Tuple[float, List[str]]],
                   test_days: int) -> Tuple[List[List[str]], List[List[str]]]:
    """Sessions starting in the last `test_days` days -> test (SR-GNN protocol)."""
    if not sessions:
        return [], []
    max_ts = max(ts for ts, _ in sessions)
    cutoff = max_ts - test_days * 86400
    train = [s for ts, s in sessions if ts <= cutoff]
    test = [s for ts, s in sessions if ts > cutoff]
    return train, test


def build_vocab(train_sessions: List[List[str]]) -> Tuple[Dict[str, int], int]:
    """1-indexed vocabulary over TRAIN items; 0 reserved for PAD."""
    items = sorted({it for sess in train_sessions for it in sess})
    item2id = {it: i + 1 for i, it in enumerate(items)}
    return item2id, len(items) + 1


def process_seqs(train_sessions: List[List[int]]):
    """SR-GNN augmentation: each session -> list of (prefix, next-label) pairs.

    Returned for transparency/diagnostics; SeqDataset generates the same
    examples internally, so this is not strictly needed for training.
    """
    out_ids, out_labels = [], []
    for seq in train_sessions:
        for i in range(1, len(seq)):
            out_ids.append(seq[:i])
            out_labels.append(seq[i])
    return out_ids, out_labels


def _result(domain: str, n_items: int, train_sessions_ids: List[List[int]],
            test_sessions_ids: List[List[int]], item2id: Dict[str, int]) -> dict:
    """Pack into the sparse_bench unified loader format.

    train_sessions: keyed by integer index, value = full session (SeqDataset
                    will slide over it to make next-item examples).
    test_queries:   keyed by string index, context = session[:-1], target = session[-1].
                    Cold test items (not in train vocab) are dropped per SR-GNN.
    """
    train_dict = {str(i): seq for i, seq in enumerate(train_sessions_ids) if len(seq) >= 2}
    test_dict: Dict[str, dict] = {}
    item_freq: Counter = Counter()
    for seq in train_sessions_ids:
        item_freq.update(seq)
    for i, seq in enumerate(test_sessions_ids):
        if len(seq) < 2:
            continue
        tgt = seq[-1]
        # SR-GNN obtian_tes: drop test targets not in vocab (already mapped, so
        # all are in vocab; but also require target to have appeared in TRAIN so
        # it is learnable — cold targets are not scored).
        if tgt in item_freq:
            test_dict[str(i)] = {"context": list(seq[:-1]), "targets": [tgt]}

    print(f"[{domain}] train_sessions={len(train_dict)} test_queries={len(test_dict)} "
          f"items={n_items} | (cold test targets dropped)")
    return {
        "domain": domain,
        "n_items": n_items,
        "train_sessions": train_dict,
        "test_queries": test_dict,
        "item_freq": item_freq,
        "item_categories": {},
        "visit_counts": {u: 1 for u in train_dict},
        "reference_groups": None,
    }


# =============================================================================
# Per-dataset loaders
# =============================================================================
def _load_or_cache(domain: str, build_fn, item2id_var="item2id"):
    cache = CACHE_DIR / f"{domain}_srgnn.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    result = build_fn()
    with open(cache, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    return result


def load_retailrocket_srgnn(test_days: int = 1, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    """RetailRocket via SR-GNN protocol. events.csv already on disk.

    test_days: sessions starting in the last N days form the test set. The
    original SR-GNN codebase uses 1 day for YooChoose and 7 for Diginetica; for
    RetailRocket the common reproduction window yields ~60K test sessions at
    ~30 days. Tune per-dataset to match published split sizes.
    """
    """RetailRocket via SR-GNN protocol. events.csv already on disk."""
    def build():
        path = REPO / "archive" / "crossdomain_data" / "events.csv"
        df = pd.read_csv(path, usecols=["timestamp", "visitorid", "event", "itemid"],
                         dtype={"visitorid": str, "event": str, "itemid": str})
        df = df.dropna(subset=["visitorid", "itemid"])
        # SR-GNN on RetailRocket uses ALL event types (view is dominant); we
        # keep view + addtocart + transaction as item interactions.
        sessions = sessionize(df)
        sessions = filter_support(sessions, min_support)
        train_raw, test_raw = temporal_split(sessions, test_days)
        item2id, n_items = build_vocab(train_raw)
        train_ids = [[item2id[it] for it in s if it in item2id] for s in train_raw]
        test_ids = [[item2id[it] for it in s if it in item2id] for s in test_raw]
        train_ids = [s for s in train_ids if len(s) >= 2]
        test_ids = [s for s in test_ids if len(s) >= 2]
        return _result("RetailRocket", n_items, train_ids, test_ids, item2id)
    return _load_or_cache("RetailRocket", build)


def load_diginetica(test_days: int = 7, min_support: int = DEFAULT_MIN_SUPPORT) -> dict:
    """Diginetica via SR-GNN protocol. train-item-views.csv already on disk.

    Columns (semicolon-delimited): sessionId;userId;itemId;timeframe;eventdate.
    Diginetica ALREADY provides a session id — we group by it directly (no
    30-min sessionization). SR-GNN standard: test_days=7 (last week)."""
    def build():
        path = _HERE / "_datasets" / "diginetica" / "train-item-views.csv"
        df = pd.read_csv(path, sep=";", dtype=str)
        df = df.dropna(subset=["sessionId", "itemId", "eventdate"])
        # Group items by sessionId (the dataset's native session id).
        sess_to_items = defaultdict(list)
        sess_to_date = {}
        for sid, iid, ed in zip(df["sessionId"], df["itemId"], df["eventdate"]):
            sess_to_items[sid].append(iid)
            sess_to_date[sid] = ed
        # Build (start_date, items) tuples; order items within session by
        # appearance (the file is already session-ordered).
        sessions = []
        for sid, items in sess_to_items.items():
            if len(items) >= MIN_SESSION_LEN:
                sessions.append((sess_to_date[sid], items))
        # date-based temporal split: convert eventdate to ordinal for ordering.
        import datetime as _dt
        def _ord(d):
            try:
                return _dt.date.fromisoformat(str(d)).toordinal()
            except Exception:
                return 0
        sessions.sort(key=lambda x: _ord(x[0]))
        max_ord = _ord(sessions[-1][0]) if sessions else 0
        train_raw = [s for d, s in sessions if _ord(d) <= max_ord - test_days]
        test_raw = [s for d, s in sessions if _ord(d) > max_ord - test_days]
        # item support filter (after split, like SR-GNN)
        train_raw = filter_support([(0, s) for s in train_raw], min_support)
        train_raw = [s for _, s in train_raw]
        item2id, n_items = build_vocab(train_raw)
        train_ids = [[item2id[it] for it in s if it in item2id] for s in train_raw]
        test_ids = [[item2id[it] for it in s if it in item2id] for s in test_raw]
        train_ids = [s for s in train_ids if len(s) >= 2]
        test_ids = [s for s in test_ids if len(s) >= 2]
        return _result("Diginetica", n_items, train_ids, test_ids, item2id)
    return _load_or_cache("Diginetica", build)


if __name__ == "__main__":
    import sys
    dom = sys.argv[1] if len(sys.argv) > 1 else "RetailRocket"
    if dom == "RetailRocket":
        data = load_retailrocket_srgnn()
        print(f"  SOTA reference: SR-GNN R@20=0.5515 MRR@20=0.3256; GCE-GNN R@20=0.6753")
    elif dom == "Diginetica":
        data = load_diginetica()
        print(f"  SOTA reference: SR-GNN R@20=0.5073 MRR@20=0.1759; GCE-GNN R@20=0.5466")
    else:
        raise SystemExit(f"unknown domain {dom}")
    slens = [len(s) for s in data["train_sessions"].values()]
    print(f"  train session len: mean={np.mean(slens):.1f} median={np.median(slens):.0f} "
          f"max={max(slens)}")
    tlens = [len(q["context"]) + 1 for q in data["test_queries"].values()]
    print(f"  test session len:  mean={np.mean(tlens):.1f} median={np.median(tlens):.0f}")
