import numpy as np

from prepare_mgcot_rental import adjacency, expand
from evaluate_mgcot_rental import metrics


def test_expand_prefixes():
    assert expand([[1, 2, 3]]) == ([[1], [1, 2]], [2, 3])


def test_adjacency_is_train_only_shape_and_finite():
    graph = adjacency([[1, 2, 3], [1, 2]], 5)
    assert graph.shape == (5, 5)
    assert np.isfinite(graph.data).all()
    assert graph[1, 2] > 0
    assert graph[3, 1] == 0


def test_rental_metrics_single_target():
    queries = {"a": {"targets": [2]}, "b": {"targets": [4]}}
    result = metrics({"a": [1, 2], "b": [4, 3]}, queries, 2)
    assert result["HR@2"] == 1.0
    assert result["MRR@2"] == 0.75
