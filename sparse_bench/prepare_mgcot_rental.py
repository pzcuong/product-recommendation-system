#!/usr/bin/env python3
"""Create leakage-safe full and nested Rental data for the MGCOT port."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import scipy.sparse


def expand(sequences):
    contexts, targets = [], []
    for sequence in sequences:
        for position in range(1, len(sequence)):
            contexts.append(sequence[:position])
            targets.append(sequence[position])
    return contexts, targets


def adjacency(sequences, n_items):
    rows, cols, values = [], [], []
    counts = {}
    for sequence in sequences:
        for left, right in zip(sequence, sequence[1:]):
            counts[(int(left), int(right))] = counts.get(
                (int(left), int(right)), 0.0) + 1.0
    for item in range(n_items):
        counts[(item, item)] = counts.get((item, item), 0.0) + 1.0
    for (left, right), value in counts.items():
        rows.append(left); cols.append(right); values.append(value)
    raw = scipy.sparse.csr_matrix(
        (np.asarray(values, np.float32), (rows, cols)),
        shape=(n_items, n_items))
    degree = np.asarray(raw.sum(axis=1)).ravel()
    inv = np.zeros_like(degree, dtype=np.float32)
    inv[degree > 0] = degree[degree > 0] ** -0.5
    normalizer = scipy.sparse.diags(inv)
    return (normalizer @ raw @ normalizer).tocsr()


def write_dataset(root, name, train_sequences, queries, n_items):
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    train = expand(train_sequences)
    test = (
        [query["context"] for query in queries.values()],
        [query["targets"][0] for query in queries.values()],
    )
    with open(folder / "train.txt", "wb") as handle:
        pickle.dump(train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(folder / "test.txt", "wb") as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)
    scipy.sparse.save_npz(folder / "adj_global.npz",
                          adjacency(train_sequences, n_items))
    (folder / "metadata.json").write_text(json.dumps({
        "n_items": n_items, "train_examples": len(train[0]),
        "test_queries": len(test[0]), "uids": list(queries),
    }, indent=2))
    print(name, json.loads((folder / "metadata.json").read_text()), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path,
                        default=Path("rental_intent_bench/split_loo_masked"))
    parser.add_argument("--output", type=Path,
                        default=Path("sparse_bench/mgcot_rental"))
    args = parser.parse_args()
    vocab = json.loads((args.split / "vocab.json").read_text())
    sequences_by_uid = json.loads((args.split / "train_seqs.json").read_text())
    official_queries = json.loads((args.split / "queries.json").read_text())
    full_sequences = [list(map(int, sequence))
                      for sequence in sequences_by_uid.values()]
    write_dataset(args.output / "full", "rental", full_sequences,
                  official_queries, int(vocab["n_items"]))

    nested_sequences, nested_queries = [], {}
    for uid, sequence0 in sequences_by_uid.items():
        sequence = list(map(int, sequence0))
        if len(sequence) >= 2:
            nested_queries[str(uid)] = {
                "context": sequence[:-1], "targets": [sequence[-1]]}
            nested_sequences.append(sequence[:-1])
        else:
            nested_sequences.append(sequence)
    write_dataset(args.output / "nested", "rental", nested_sequences,
                  nested_queries, int(vocab["n_items"]))


if __name__ == "__main__":
    main()
