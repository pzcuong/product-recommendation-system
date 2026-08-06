"""
Unified domain loaders for the sparse short-session benchmark.

Each loader returns a dict with the *same* keys so codt_core can run
unchanged across domains:

    {
      "domain"        : str,
      "n_items"       : int,            # vocab size, index 0 reserved for PAD
      "train_sessions": dict[uid->List[int]],
      "test_queries"  : dict[uid->{"context": List[int], "targets": List[int]}],
      "item_freq"     : Counter[int],
      "item_categories": dict[int->category_id],   # may be empty
      "visit_counts"  : dict[uid->int],            # visits per user (for single_visit grouping)
    }

Datasets:
  - Rental          : private anchor (rental_intent_bench/split_loo_masked/)
  - Amazon          : Baby_Products / Video_Games / Arts_Crafts_and_Sewing (5-core, last_out_w_his)
  - RetailRocket    : archive/crossdomain_data/events.csv (30-min session timeout, LOO)
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Resolve repo root regardless of where the script is run from.
_HERE = Path(__file__).resolve().parent
REPO = _HERE.parent
DATA_DIR = REPO / "archive" / "data"
RENTAL_DIR = REPO / "rental_intent_bench" / "split_loo_masked"
RETAIL_DIR = REPO / "archive" / "crossdomain_data"

MIN_SESSION_LEN = 2  # a context of >=1 item + 1 target


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _result(domain, n_items, train_sessions, test_queries, item_categories=None,
            visit_counts=None, reference_groups=None, valid_queries=None,
            item_texts=None):
    item_freq: Counter = Counter()
    for seq in train_sessions.values():
        item_freq.update(seq)
    return {
        "domain": domain,
        "n_items": n_items,
        "train_sessions": train_sessions,
        "test_queries": test_queries,
        "valid_queries": valid_queries or {},
        "item_freq": item_freq,
        "item_categories": item_categories or {},
        "item_texts": item_texts or {},
        "visit_counts": visit_counts or {u: 1 for u in train_sessions},
        "reference_groups": reference_groups,
    }


# -----------------------------------------------------------------------------
# Rental (anchor)  — read pre-built leave-one-out split.
# -----------------------------------------------------------------------------
def load_rental() -> dict:
    vocab = json.load(open(RENTAL_DIR / "vocab.json"))
    n_items = vocab["n_items"]
    slug2id = vocab["slug2id"]
    train_sessions = {str(k): v for k, v in json.load(open(RENTAL_DIR / "train_seqs.json")).items()}
    raw_q = json.load(open(RENTAL_DIR / "queries.json"))
    test_queries = {str(k): {"context": v["context"], "targets": v["targets"]}
                    for k, v in raw_q.items()}

    # Build slug -> main_category from old_site_products (vocab is slug-keyed).
    item_categories: Dict[int, str] = {}
    item_texts: Dict[int, str] = {}
    rental_root = REPO / "rental_data"
    try:
        import pandas as pd
        old = pd.read_csv(rental_root / "old_site_products.csv",
                          usecols=["slug", "main_category"], dtype=str).dropna()
        slug2cat = dict(zip(old["slug"], old["main_category"]))
        # fallback: new_site_products via old->new id map
        if rental_root.joinpath("old_site_new_site_products.csv").exists() and \
           rental_root.joinpath("new_site_products.csv").exists():
            o2n = pd.read_csv(rental_root / "old_site_new_site_products.csv", dtype=str)
            new = pd.read_csv(rental_root / "new_site_products.csv",
                              usecols=["id", "main_category"], dtype=str).dropna()
            newid2cat = dict(zip(new["id"], new["main_category"]))
            oldid2newid = dict(zip(o2n["old_site_id"], o2n["new_site_id"]))
            # old_site_products with id + slug to bridge
            old_full = pd.read_csv(rental_root / "old_site_products.csv",
                                   usecols=["id", "slug"], dtype=str).dropna()
            for _, r in old_full.iterrows():
                if pd.isna(slug2cat.get(r["slug"])):
                    nid = oldid2newid.get(r["id"])
                    if nid and nid in newid2cat:
                        slug2cat[r["slug"]] = newid2cat[nid]
        for slug, idx in slug2id.items():
            cat = slug2cat.get(slug)
            if cat:
                item_categories[idx] = cat
    except Exception as e:
        print(f"[Rental] category load failed: {e}")

    # visit_counts: number of distinct queries per user is a reasonable proxy;
    # Rental visit_id keys already encode one visit each -> default 1.
    visit_counts = {u: 1 for u in train_sessions}

    print(f"[Rental] users={len(train_sessions)} items={n_items} test={len(test_queries)} "
          f"cats={len(item_categories)}")
    return _result("Rental", n_items, train_sessions, test_queries,
                   item_categories=item_categories, visit_counts=visit_counts)


def load_rental_visit() -> dict:
    """Visit-level Rental (the regime that hit HR@10≈0.43 in the reference work).

    Uses the visit-level loader from multi_domain_eval.py (259 short visit-level
    queries, ctx-len mean 2.0), which is the *exact* evaluation the reference
    .bak file was scored against. We keep the visit_id as the query key and
    build train_sessions from the per-visit product sequences.
    """
    import importlib.util
    repo_root = REPO
    spec_path = repo_root / "multi_domain_eval.py"
    spec = importlib.util.spec_from_file_location("mde_rental_visit", spec_path)
    mde = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mde)
    raw = mde.load_rental_grouped(str(repo_root / "rental_data"))

    n_items = raw["n_items"]
    # train_sessions: dict[vid -> List[int]] (one per visit, the per-visit product seq)
    train_sessions = {str(vid): seq for vid, seq in raw["sessions"].items()}
    # test_queries: context = the visit's prior products, target = last product
    test_queries = {}
    for vid, tgt in raw["test_items"].items():
        ctx = raw["test_contexts"].get(str(vid), raw["test_contexts"].get(vid, []))
        if ctx:  # leave-one-out within the visit
            test_queries[str(vid)] = {"context": list(ctx), "targets": [tgt]}
        else:
            test_queries[str(vid)] = {"context": list(ctx), "targets": [tgt]}

    # Categories: raw["item_map"] is {slug -> idx}. Map idx -> main_category by
    # joining slug -> main_category from the Rental product tables. This must use
    # the SAME item_map the loader produced, so test-query item ids line up.
    item_categories: Dict[int, str] = {}
    try:
        import pandas as pd
        old = pd.read_csv(REPO / "rental_data" / "old_site_products.csv",
                          usecols=["slug", "main_category"], dtype=str).dropna()
        slug2cat = dict(zip(old["slug"], old["main_category"]))
        # fallback via new_site_products through the old->new id bridge
        if (REPO / "rental_data" / "old_site_new_site_products.csv").exists() and \
           (REPO / "rental_data" / "new_site_products.csv").exists():
            o2n = pd.read_csv(REPO / "rental_data" / "old_site_new_site_products.csv", dtype=str)
            new = pd.read_csv(REPO / "rental_data" / "new_site_products.csv",
                              usecols=["id", "main_category"], dtype=str).dropna()
            newid2cat = dict(zip(new["id"], new["main_category"]))
            oldid2newid = dict(zip(o2n["old_site_id"], o2n["new_site_id"]))
            old_full = pd.read_csv(REPO / "rental_data" / "old_site_products.csv",
                                   usecols=["id", "slug"], dtype=str).dropna()
            for _, r in old_full.iterrows():
                if pd.isna(slug2cat.get(r["slug"])):
                    nid = oldid2newid.get(r["id"])
                    if nid and nid in newid2cat:
                        slug2cat[r["slug"]] = newid2cat[nid]
        for slug, idx in raw["item_map"].items():
            cat = slug2cat.get(slug)
            if cat:
                item_categories[idx] = cat
    except Exception as e:
        print(f"[Rental-visit] category load failed: {e}")

    # visit_counts: not needed when reference_groups is supplied (grouping uses
    # the reference's own single/multi assignment). Default to 1.
    visit_counts = {u: 1 for u in train_sessions}

    print(f"[Rental-visit] visits={len(train_sessions)} items={n_items} "
          f"test={len(test_queries)} cats={len(item_categories)}")
    return _result("Rental", n_items, train_sessions, test_queries,
                   item_categories=item_categories, visit_counts=visit_counts,
                   reference_groups=raw["test_groups"])


# -----------------------------------------------------------------------------
# Amazon 5-core (last_out_w_his CSVs)  — Baby / Video_Games / Arts_Crafts.
# -----------------------------------------------------------------------------
def _amazon_csv_prefix(domain_dir: Path) -> str:
    train_files = list((domain_dir / "benchmark" / "5core" / "last_out_w_his").glob("*.train.csv"))
    if not train_files:
        raise FileNotFoundError(f"No train CSV under {domain_dir}")
    return train_files[0].stem.replace(".train", "")


def load_amazon(domain_name: str) -> dict:
    """domain_name: dir under archive/data, e.g. 'amazon_baby', 'Video_Games'."""
    domain_dir = DATA_DIR / domain_name
    csv_dir = domain_dir / "benchmark" / "5core" / "last_out_w_his"
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV dir not found: {csv_dir}")
    prefix = _amazon_csv_prefix(domain_dir)

    # 1) Build vocabulary from all splits.
    all_item_ids: set = set()
    for split in ("train", "valid", "test"):
        p = csv_dir / f"{prefix}.{split}.csv"
        if not p.exists():
            continue
        with open(p) as f:
            next(f)
            for line in f:
                parts = line.strip().split(",", 4)
                if len(parts) >= 2:
                    all_item_ids.add(parts[1])
    item_list = sorted(all_item_ids)
    item2id = {x: i + 1 for i, x in enumerate(item_list)}
    n_items = len(item_list) + 1

    # 2) Train sequences from the train split (timestamp-sorted per user).
    user_rows: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    user_test_targets: Dict[str, List[Tuple[int, str]]] = {}
    with open(csv_dir / f"{prefix}.train.csv") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",", 4)
            if len(parts) < 4:
                continue
            uid, iid, _, ts = parts[0], parts[1], parts[2], int(parts[3])
            user_rows[uid].append((ts, iid))
    train_sessions: Dict[str, List[int]] = {}
    for uid, rows in user_rows.items():
        rows.sort(key=lambda x: x[0])
        seq = [item2id[x[1]] for x in rows if x[1] in item2id]
        if len(seq) >= MIN_SESSION_LEN:
            train_sessions[uid] = seq

    # 3) Validation/test queries. Prefer the explicit history column shipped
    # with the benchmark; falling back to the train sequence preserves support
    # for older exports. Validation is kept separate and is never merged into
    # training, so downstream routers can be calibrated without test leakage.
    def read_queries(split: str) -> Dict[str, dict]:
        queries: Dict[str, dict] = {}
        path = csv_dir / f"{prefix}.{split}.csv"
        if not path.exists():
            return queries
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.strip().split(",", 4)
                if len(parts) < 4:
                    continue
                uid, iid, _, ts = parts[0], parts[1], parts[2], int(parts[3])
                if uid in train_sessions and iid in item2id:
                    raw_history = parts[4].strip().strip('"') if len(parts) >= 5 else ""
                    ctx = [item2id[x] for x in raw_history.split() if x in item2id]
                    if not ctx:
                        ctx = train_sessions[uid]
                    tgt = item2id[iid]
                    if tgt in ctx:
                        continue  # avoid leaking a target already in context
                    queries[uid] = {"context": ctx, "targets": [tgt]}
        return queries

    valid_queries = read_queries("valid")
    test_queries = read_queries("test")

    # 4) Categories from meta jsonl (first element of `categories`, else main_category).
    item_categories: Dict[int, str] = {}
    item_texts: Dict[int, str] = {}
    meta_path = domain_dir / "raw" / "meta_categories" / f"meta_{prefix}.jsonl"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    asin = d.get("parent_asin")
                    if asin not in item2id:
                        continue
                    cats = d.get("categories") or []
                    cat = cats[0] if cats else d.get("main_category")
                    if cat:
                        item_categories[item2id[asin]] = cat
                    details = d.get("details") or {}
                    brand = details.get("Brand") or d.get("store") or ""
                    brand = str(brand).strip()
                    fields = []
                    if brand:
                        # Brand is a strong, low-noise signal that currently is
                        # silently dropped; leading with it biases TF-IDF toward
                        # manufacturer clusters.
                        fields.append(f"Brand: {brand}")
                    fields.append(d.get("title") or "")
                    fields.append(d.get("main_category") or "")
                    fields.extend(str(x) for x in cats[:5])
                    fields.extend(str(x) for x in (d.get("features") or [])[:4])
                    fields.extend(str(x) for x in (d.get("description") or [])[:2])
                    text = " | ".join(x.strip() for x in fields if x and x.strip())
                    if text:
                        item_texts[item2id[asin]] = text[:2000]
        except Exception as e:  # meta is best-effort
            print(f"[Amazon:{domain_name}] meta load failed: {e}")

    # 5) visit_counts: number of distinct purchase-days per user (proxy).
    visit_counts = {u: 1 for u in train_sessions}

    print(f"[Amazon:{prefix}] users={len(train_sessions)} items={n_items} "
          f"test={len(test_queries)} cats={len(item_categories)}")
    return _result(prefix, n_items, train_sessions, test_queries,
                   item_categories=item_categories, visit_counts=visit_counts,
                   valid_queries=valid_queries, item_texts=item_texts)


# -----------------------------------------------------------------------------
# RetailRocket  — sessionize events.csv with a 30-min timeout, LOO target.
# -----------------------------------------------------------------------------
RETAIL_SESSION_TIMEOUT = 30 * 60  # seconds
RETAIL_EVENT_WEIGHTS = {"view": 1.0, "addtocart": 4.0, "transaction": 7.0}
RETAIL_MIN_ITEM_SUPPORT = 5        # drop items with < this many events
RETAIL_MIN_SESSION_LEN = 3         # need >=3 items in a session to keep it
RETAIL_MIN_USER_INTERACTIONS = 2   # keep visitors with at least this many kept-events


def load_retailrocket(max_items: int = 5000) -> dict:
    events_path = RETAIL_DIR / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"RetailRocket events.csv not found at {events_path}")

    import pandas as pd

    df = pd.read_csv(events_path,
                     usecols=["timestamp", "visitorid", "event", "itemid"],
                     dtype={"visitorid": str, "event": str, "itemid": str})
    df = df.dropna(subset=["visitorid", "itemid"]).copy()

    # Filter to items with enough support to keep a learnable tail (avoid singleton
    # collapse — see research_proposal analysis). Then cap to top-N for speed.
    # NOTE: use groupby().size() (not value_counts()) — pandas' value_counts hits
    # an nargsort IndexError on this object column in some pandas builds.
    df = df.dropna(subset=["itemid", "visitorid"])
    df["itemid"] = df["itemid"].astype(str)
    df["visitorid"] = df["visitorid"].astype(str)
    item_counts = df.groupby("itemid").size().sort_values(ascending=False)
    supported = item_counts[item_counts >= RETAIL_MIN_ITEM_SUPPORT].index
    df = df[df["itemid"].isin(supported)]
    if df["itemid"].nunique() > max_items:
        top_items = item_counts.loc[lambda s: s.index.isin(df["itemid"].unique())].head(max_items).index
        df = df[df["itemid"].isin(top_items)]

    # Keep only visitors with enough remaining interactions to form a query.
    vc = df.groupby("visitorid").size()
    keep_visitors = vc[vc >= RETAIL_MIN_USER_INTERACTIONS].index
    df = df[df["visitorid"].isin(keep_visitors)]

    item_list = sorted(df["itemid"].unique())
    item2id = {x: i + 1 for i, x in enumerate(item_list)}
    n_items = len(item_list) + 1

    # Sessionize by (visitor, 30-min timeout gap).
    df = df.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    df["gap"] = df.groupby("visitorid")["timestamp"].diff().fillna(0)
    df["new_session"] = (df["gap"] > RETAIL_SESSION_TIMEOUT) | (df["visitorid"] != df["visitorid"].shift())
    df["sid"] = df["new_session"].cumsum()

    # Build per-visitor sessions (ordered item sequences, dedupe consecutive dupes).
    sessions_by_visitor: Dict[str, List[List[int]]] = defaultdict(list)
    for sid, grp in df.groupby("sid"):
        vid = grp["visitorid"].iloc[0]
        seq = [item2id[x] for x in grp["itemid"].tolist()]
        # collapse consecutive duplicates
        dedup = [seq[0]] + [b for a, b in zip(seq, seq[1:]) if b != a]
        if len(dedup) >= RETAIL_MIN_SESSION_LEN:
            sessions_by_visitor[vid].append(dedup)

    # Split visitors into train-only and test so we have BOTH rich training
    # sequences (from train-only visitors, who contribute their FULL item
    # sequence) AND a clean LOO test set (test visitors: last item held out).
    # This is the standard session-rec protocol and avoids the train-data
    # starvation that a single-sequence-per-user LOO would cause, while keeping
    # the comparison apples-to-apples (no test target leaks into any model).
    rng = random.Random(42)
    all_vids = sorted(sessions_by_visitor.keys())
    rng.shuffle(all_vids)
    n_test = max(1, int(len(all_vids) * 0.5))
    test_vids = set(all_vids[:n_test])

    train_sessions: Dict[str, List[int]] = {}
    test_queries: Dict[str, dict] = {}
    visit_counts: Dict[str, int] = {}
    for vid, sess_list in sessions_by_visitor.items():
        visit_counts[vid] = len(sess_list)
        full = [x for s in sess_list for x in s]
        if len(full) < 2:
            continue
        if vid in test_vids:
            ctx = full[:-1]
            tgt = full[-1]
            if len(ctx) >= MIN_SESSION_LEN:
                train_sessions[vid] = ctx   # prefix only (target held out)
            test_queries[vid] = {"context": ctx, "targets": [tgt]}
        else:
            # train-only visitor: contribute the FULL sequence (no test query)
            if len(full) >= MIN_SESSION_LEN:
                train_sessions[vid] = full

    # Categories: map item -> root category via item_properties + category_tree.
    item_categories = _retailrocket_categories(item2id)

    print(f"[RetailRocket] users={len(train_sessions)} items={n_items} "
          f"test={len(test_queries)} cats={len(item_categories)}")
    return _result("RetailRocket", n_items, train_sessions, test_queries,
                   item_categories=item_categories, visit_counts=visit_counts)


def _retailrocket_categories(item2id: dict) -> Dict[int, int]:
    """item -> root category_id using item_properties(categoryid) + category_tree."""
    cat_path = RETAIL_DIR / "category_tree.csv"
    prop_files = [RETAIL_DIR / "item_properties_part1.csv",
                  RETAIL_DIR / "item_properties_part2.csv"]
    if not cat_path.exists():
        return {}

    import pandas as pd

    # Build category parent pointers; roots have parentid NaN -> treated as self.
    tree = pd.read_csv(cat_path, dtype={"categoryid": str, "parentid": str})
    parent: Dict[str, str] = {}
    for cid, pid in zip(tree["categoryid"], tree["parentid"]):
        if pd.isna(pid) or str(pid) == "nan":
            continue  # root: no parent
        parent[str(cid)] = str(pid)

    def root_of(cid: str) -> str:
        seen = set()
        while cid in parent and cid not in seen:
            seen.add(cid)
            cid = parent[cid]
        return cid

    item_to_cat: Dict[str, str] = {}
    for pf in prop_files:
        if not pf.exists():
            continue
        try:
            pr = pd.read_csv(pf, dtype=str)
        except Exception:
            continue
        pr = pr[pr["property"] == "categoryid"]
        for iid, val in zip(pr["itemid"], pr["value"]):
            if not pd.isna(val):
                item_to_cat[str(iid)] = str(val)

    out: Dict[int, int] = {}
    for raw_item, idx in item2id.items():
        cat = item_to_cat.get(raw_item)
        if cat is None:
            continue
        root = root_of(cat)
        try:
            out[idx] = int(root)
        except ValueError:
            continue
    return out


# -----------------------------------------------------------------------------
# Diginetica (HID official split) — reads the preprocessed Code4HID files.
# -----------------------------------------------------------------------------
HID_DIGI_DIR = REPO / "reference_repos" / "Code4HID" / "datasets" / "diginetica-2"
DIGI_RAW_DIR = REPO / "sparse_bench" / "_datasets" / "diginetica"


def _pickle_sessions_to_dict(train_x, train_y) -> tuple[dict, int]:
    """Reconstruct full train sessions from the HID prefix expansion.

    The HID split expands each session into one (prefix, next-item) example. We
    invert that: two consecutive rows belong to the same session iff the later
    context is a strict prefix of the earlier one.
    """
    sessions: Dict[str, list[int]] = {}
    previous = None
    for row, (context, target) in enumerate(zip(train_x, train_y)):
        ctx = [int(x) for x in context]
        same_block = (previous is not None and len(ctx) < len(previous)
                      and ctx == previous[:len(ctx)])
        if not same_block:
            sessions[f"hid_train_{row}"] = ctx + [int(target)]
        previous = ctx
    n_items = max((max(s) for s in sessions.values() if s), default=0) + 1
    return sessions, n_items


def load_diginetica_hid() -> dict:
    """HID Diginetica with product-name text metadata.

    Reads the official AAAI-2026 HID train/test pickles and the raw product
    catalog (`products.csv`, `product-categories.csv`). Validation queries are
    carved from training via a deterministic leave-last-out so that per-query
    routing can be calibrated without test leakage.
    """
    import pickle
    train_x, train_y = pickle.load(open(HID_DIGI_DIR / "train.txt", "rb"))
    test_x, test_y = pickle.load(open(HID_DIGI_DIR / "test.txt", "rb"))
    train_sessions, n_items = _pickle_sessions_to_dict(train_x, train_y)
    test_queries = {
        f"hid_test_{row}": {"context": [int(x) for x in context],
                            "targets": [int(target)]}
        for row, (context, target) in enumerate(zip(test_x, test_y))
    }
    # Deterministic validation: leave the last in-session item out for every
    # training session of length >= 3 (need >=2 context items + 1 target).
    valid_queries: Dict[str, dict] = {}
    for uid, seq in train_sessions.items():
        if len(seq) < 3:
            continue
        valid_queries[f"{uid}_v"] = {"context": seq[:-1], "targets": [seq[-1]]}

    # Metadata: product name tokens + category id.
    item_categories: Dict[int, str] = {}
    item_texts: Dict[int, str] = {}
    try:
        import pandas as pd
        if (DIGI_RAW_DIR / "products.csv").exists():
            prod = pd.read_csv(
                DIGI_RAW_DIR / "products.csv",
                sep=";", usecols=["itemId", "product.name.tokens"], dtype=str)
            for iid, tokens in zip(prod["itemId"], prod["product.name.tokens"]):
                if pd.isna(tokens):
                    continue
                try:
                    idx = int(iid)
                except (TypeError, ValueError):
                    continue
                if 0 < idx < n_items:
                    item_texts[idx] = " ".join(str(tokens).split(",")[:30])[:500]
        if (DIGI_RAW_DIR / "product-categories.csv").exists():
            cats = pd.read_csv(
                DIGI_RAW_DIR / "product-categories.csv",
                sep=";", usecols=["itemId", "categoryId"], dtype=str)
            for iid, cid in zip(cats["itemId"], cats["categoryId"]):
                try:
                    idx, cat = int(iid), int(cid)
                except (TypeError, ValueError):
                    continue
                if 0 < idx < n_items:
                    item_categories[idx] = str(cat)
    except Exception as e:
        print(f"[Diginetica_HID] metadata load failed: {e}")

    print(f"[Diginetica_HID] sessions={len(train_sessions)} items={n_items} "
          f"test={len(test_queries)} valid={len(valid_queries)} "
          f"texts={len(item_texts)} cats={len(item_categories)}")
    return _result("Diginetica_HID", n_items, train_sessions, test_queries,
                   item_categories=item_categories,
                   valid_queries=valid_queries, item_texts=item_texts)


# -----------------------------------------------------------------------------
# Tmall (COTREC/GCE-GNN preprocessed split).
# -----------------------------------------------------------------------------
TMALL_DIR = REPO / "reference_repos" / "SelfContrastiveLearningRecSys" / \
    "COTREC" / "datasets" / "Tmall"


def load_tmall() -> dict:
    """Tmall session-rec split shipped with the SelfContrastiveLearningRecSys repo.

    Pickle format (COTREC convention): each file is a tuple
    ``(contexts, targets)`` where ``contexts`` is a list of item-id lists and
    ``targets`` is a list of next item ids. ``all_train_seq.txt`` holds the
    un-split training sessions as a flat list of item-id lists. No item
    metadata is available, so `item_categories` and `item_texts` stay empty
    and the semantic teacher falls back to category hashing / random init.
    """
    import pickle
    train_ctx, train_tgt = pickle.load(open(TMALL_DIR / "train.txt", "rb"))
    test_ctx, test_tgt = pickle.load(open(TMALL_DIR / "test.txt", "rb"))

    def as_int_list(s):
        if isinstance(s, np.ndarray):
            return [int(x) for x in s.tolist()]
        return [int(x) for x in s]

    # Full training sessions: context + target, deduplicated and length-filtered.
    train_sessions: Dict[str, list[int]] = {}
    for i, (ctx, tgt) in enumerate(zip(train_ctx, train_tgt)):
        seq = as_int_list(ctx) + [int(tgt)]
        if len(seq) >= MIN_SESSION_LEN:
            train_sessions[f"tmall_train_{i}"] = seq
    n_items = max((max(s) for s in train_sessions.values() if s), default=0) + 1

    test_queries: Dict[str, dict] = {}
    for i, (ctx, tgt) in enumerate(zip(test_ctx, test_tgt)):
        ctx_list = as_int_list(ctx)
        if not ctx_list:
            continue
        test_queries[f"tmall_test_{i}"] = {"context": ctx_list, "targets": [int(tgt)]}

    # Deterministic validation from training (leave-last-out).
    valid_queries: Dict[str, dict] = {}
    for uid, seq in train_sessions.items():
        if len(seq) < 3:
            continue
        valid_queries[f"{uid}_v"] = {"context": seq[:-1], "targets": [seq[-1]]}

    print(f"[Tmall] sessions={len(train_sessions)} items={n_items} "
          f"test={len(test_queries)} valid={len(valid_queries)}")
    return _result("Tmall", n_items, train_sessions, test_queries,
                   valid_queries=valid_queries)


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------
ALL_LOADERS = {
    "Rental": load_rental,
    "Rental_visit": load_rental_visit,
    "Baby_Products": lambda: load_amazon("amazon_baby"),
    "Video_Games": lambda: load_amazon("Video_Games"),
    "Arts_Crafts_and_Sewing": lambda: load_amazon("Arts_Crafts_and_Sewing"),
    "RetailRocket": load_retailrocket,
    "Diginetica_HID": load_diginetica_hid,
    "Tmall": load_tmall,
}


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "Rental"
    loader = ALL_LOADERS[target]
    data = loader()
    ctx_lens = [len(q["context"]) for q in data["test_queries"].values()]
    print(f"  context len: mean={np.mean(ctx_lens):.1f} median={np.median(ctx_lens):.0f} "
          f"min={min(ctx_lens)} max={max(ctx_lens)}")
    cold = sum(1 for l in ctx_lens if l <= 2)
    print(f"  ctx<=2: {cold} ({cold/len(ctx_lens)*100:.1f}%)")
