#!/usr/bin/env python3
"""Export MGCOT top-k items and logits without retraining."""
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import scipy.sparse
import torch


ROOT = Path(__file__).resolve().parents[1]
MGCOT = ROOT / "reference_repos" / "MGCOT"
sys.path.insert(0, str(MGCOT))
from model import SessionGraph, forward, trans_to_cuda  # noqa: E402
from recommender import GraphRecommender  # noqa: E402
from utils import Data  # noqa: E402


def edge_adjacency(path):
    matrix = scipy.sparse.load_npz(path).tocoo()
    device = (torch.device("cuda") if torch.cuda.is_available() else
              torch.device("mps") if torch.backends.mps.is_available() else
              torch.device("cpu"))
    return tuple(tensor.to(device) for tensor in (
        torch.as_tensor(matrix.col, dtype=torch.long),
        torch.as_tensor(matrix.row, dtype=torch.long),
        torch.as_tensor(matrix.data, dtype=torch.float32),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--topk", type=int, default=120)
    parser.add_argument("--n-node", type=int, default=43098)
    parser.add_argument("--dataset", default="diginetica")
    args = parser.parse_args()
    dataset_dir = args.data_root / args.dataset
    test_tuple = pickle.load(open(dataset_dir / "test.txt", "rb"))
    train_tuple = pickle.load(open(dataset_dir / "train.txt", "rb"))
    test_data = Data(test_tuple, shuffle=False)
    opt = SimpleNamespace(
        dataset=args.dataset, batchSize=args.batch_size, hiddenSize=100,
        step=1, nonhybrid=False, num_attention_heads=5, neighbor_n=3,
        w_ne=1.7, gama=1.7, lr=1e-3, l2=1e-5, lr_dc_step=5,
        lr_dc=0.1,
    )
    graph = trans_to_cuda(GraphRecommender(
        opt, args.n_node, edge_adjacency(dataset_dir / "adj_global.npz"),
        len_session=max(map(len, train_tuple[0])),
        n_train_sessions=len(train_tuple[0])))
    model = trans_to_cuda(SessionGraph(opt, args.n_node, graph))
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    item_rows, score_rows, targets = [], [], []
    slices = test_data.generate_batch(args.batch_size)
    with torch.no_grad():
        for step, indices in enumerate(slices, 1):
            batch_targets, scores, _ = forward(model, indices, test_data)
            values, items = torch.topk(scores, args.topk, dim=1)
            item_rows.append((items + 1).cpu().numpy().astype(np.int32))
            score_rows.append(values.cpu().numpy().astype(np.float32))
            targets.append(np.asarray(batch_targets, dtype=np.int32))
            if step % 25 == 0:
                print(f"[MGCOT-EXPORT] batch={step}/{len(slices)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, items=np.concatenate(item_rows),
        scores=np.concatenate(score_rows), targets=np.concatenate(targets))
    print({"rows": sum(len(x) for x in targets), "topk": args.topk,
           "output": str(args.output)}, flush=True)


if __name__ == "__main__":
    main()
