from collections import Counter

import numpy as np

from mgcot_residual_ranker import build_matrix, predict, ResidualRanker
from run_mgcot_residual_hid import npz_predictions


def test_zero_residual_preserves_mgcot_order():
    model = ResidualRanker()
    x = np.zeros((1, 3, 18), dtype=np.float32)
    x[0, :, 0] = [2, 1, 0]
    import torch
    scores = model(torch.from_numpy(x)).detach().numpy()
    assert scores.argsort(1)[0, ::-1].tolist() == [0, 1, 2]


def test_matrix_finds_target_column():
    items = np.array([[3, 2, 1]], dtype=np.int32)
    scores = np.array([[3.0, 2.0, 1.0]], dtype=np.float32)
    queries = {"q": {"context": [1], "targets": [2]}}
    ranks = {"q": [3, 2, 1]}
    matrix = build_matrix(items, scores, queries, ranks, ranks, ranks,
                          Counter({1: 1, 2: 1, 3: 1}), set())
    assert matrix.target_columns.tolist() == [1]


def test_npz_predictions_materializes_items_once():
    class CountingBundle:
        def __init__(self):
            self.calls = 0

        def __getitem__(self, key):
            assert key == "items"
            self.calls += 1
            return np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32)

    bundle = CountingBundle()
    assert npz_predictions(bundle, ["a", "b"], 2) == {
        "a": [3, 2], "b": [6, 5],
    }
    assert bundle.calls == 1


def test_entropy_gate_preserves_baseline_for_certain_query():
    items = np.array([[3, 2, 1]], dtype=np.int32)
    matrix = type("Matrix", (), {
        "x": np.zeros((1, 3, 18), dtype=np.float32),
        "items": items,
        "uids": ["q"],
    })()
    matrix.x[0, :, 0] = [5.0, 0.0, -2.0]
    model = ResidualRanker()
    with __import__("torch").no_grad():
        model.residual.weight.zero_()
        model.residual.bias.zero_()
        model.residual.weight[0, 1] = -100.0
    # threshold > 1 changes no query, regardless of residual scores.
    assert predict(model, matrix, entropy_threshold=1.01) == {"q": [3, 2, 1]}
