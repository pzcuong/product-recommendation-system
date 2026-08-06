"""Fast eval: GRU4Rec + SASRec + SSM + ItemKNN on Diginetica subsampled."""
import sys, time, random as _r
sys.path.insert(0, "sparse_bench")
import srgnn_preprocess as sp, ssm_model, grouped_eval, baselines as B
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np

DEV = torch.device("mps")

class DS(Dataset):
    def __init__(self, sqs, n, ml=50):
        self.ex = []
        for seq in sqs.values():
            s = [x for x in seq if 1 <= x < n]
            for i in range(1, len(s)):
                self.ex.append((s[max(0, i-ml):i], s[i]))
    def __len__(self): return len(self.ex)
    def __getitem__(self, i): return self.ex[i]

def collate(b):
    ml = max(len(c) for c, _ in b)
    return (torch.LongTensor([c+[0]*(ml-len(c)) for c, _ in b]),
            torch.LongTensor([len(c) for c, _ in b]),
            torch.LongTensor([t for _, t in b]))

def train_and_eval(ModelClass, cfg, train, n_items, test, uids, epochs=10):
    torch.manual_seed(42)
    model = ModelClass(n_items, **cfg).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(DS(train, n_items), batch_size=512, shuffle=True,
                        collate_fn=collate, drop_last=True)
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for inp, lens, tgt in loader:
            inp, lens, tgt = inp.to(DEV), lens.to(DEV), tgt.to(DEV)
            loss = F.cross_entropy(model(inp, lens), tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()
    train_t = time.time() - t0
    model.eval()
    hits = {6:0, 10:0, 20:0}
    for uid in uids:
        ctx = [x for x in test[uid]["context"] if 1 <= x < n_items][-50:]
        if not ctx: continue
        inp = torch.LongTensor([ctx]).to(DEV)
        ln = torch.LongTensor([len(ctx)]).to(DEV)
        with torch.no_grad():
            sc = model(inp, ln)[0].cpu().numpy()
        sc[0] = -1e9
        for c in set(ctx): sc[c] = -1e9
        tgt = test[uid]["targets"][0]
        top = np.argsort(-sc)
        for k in [6, 10, 20]:
            if tgt in top[:k]: hits[k] += 1
    nq = len(uids)
    r = {k: hits[k]/nq for k in [6, 10, 20]}
    print(f"  time={train_t:.0f}s", end="")
    return r

for dataset_name, loader_fn, max_train, max_test in [
    ("Diginetica", sp.load_diginetica, 15000, 1500),
]:
    print(f"\n=== {dataset_name} ===")
    d = loader_fn()
    keys = list(d["train_sessions"].keys())
    keep = set(_r.Random(42).sample(keys, max_train)) if len(keys) > max_train else set(keys)
    train = {k: v for k, v in d["train_sessions"].items() if k in keep}
    freq = Counter()
    for s in train.values(): freq.update(s)
    keys2 = list(d["test_queries"].keys())
    keep2 = set(_r.Random(0).sample(keys2, max_test)) if len(keys2) > max_test else set(keys2)
    test = {k: v for k, v in d["test_queries"].items() if k in keep2}
    uids = sorted(test.keys())
    n = d["n_items"]
    sessions = [s for s in train.values() if len(s) >= 2]
    print(f"  {len(sessions)} sessions, {n} items, {len(uids)} test")

    # GRU4Rec
    print("  GRU4Rec:", end="")
    r = train_and_eval(B.GRU4Rec, {"emb": 64, "h": 64, "L": 1, "dp": 0.5},
                       train, n, test, uids)
    print(f" R@6={r[6]:.4f} R@10={r[10]:.4f} R@20={r[20]:.4f}")

    # SASRec
    print("  SASRec:", end="")
    r = train_and_eval(B.SASRec, {"emb": 64, "hds": 2, "L": 1, "dp": 0.5},
                       train, n, test, uids)
    print(f" R@6={r[6]:.4f} R@10={r[10]:.4f} R@20={r[20]:.4f}")

    # SSM
    print("  SSM:", end="")
    t0 = time.time()
    models = ssm_model.train_ssm(sessions, n, epochs=6, seeds=(42, 123, 456, 789),
                                 embed_dim=64)
    print(f" time={time.time()-t0:.0f}s", end="")
    for m in models: m.eval()
    DEV2 = next(m.parameters()).device
    hits = {6:0, 10:0, 20:0}
    for uid in uids:
        ctx = [x for x in test[uid]["context"] if 1 <= x < n][-50:]
        if not ctx: continue
        inp = torch.LongTensor([ctx]).to(DEV2)
        ln = torch.LongTensor([len(ctx)]).to(DEV2)
        with torch.no_grad():
            sc = torch.zeros(1, n, device=DEV2)
            for m in models: sc += m.score_all(inp, ln)
        sc = sc[0].cpu().numpy()
        sc[0] = -1e9
        for c in set(ctx): sc[c] = -1e9
        tgt = test[uid]["targets"][0]
        top = np.argsort(-sc)
        for k in [6, 10, 20]:
            if tgt in top[:k]: hits[k] += 1
    nq = len(uids)
    print(f" R@6={hits[6]/nq:.4f} R@10={hits[10]/nq:.4f} R@20={hits[20]/nq:.4f}")

    # ItemKNN
    data_dict = {
        "train_sessions": train, "test_queries": test, "n_items": n,
        "item_freq": freq, "item_categories": {},
    }
    preds = B.run_nonparametric("ItemKNN", train, test, n)
    gm = grouped_eval.evaluate_all_groups(preds, data_dict, k_values=[6, 10, 20])
    o = gm["overall"]
    print(f"  ItemKNN: R@6={o['recall@6']:.4f} R@10={o['recall@10']:.4f} R@20={o['recall@20']:.4f}")
