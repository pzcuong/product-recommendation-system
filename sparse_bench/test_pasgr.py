import numpy as np
from collections import Counter

from sparse_bench.pasgr import (PASGRConfig, build_prototype_graph_embeddings,
                                predict_pasgr, train_pasgr)


def _fixture():
    sessions = {
        "a": [1, 2, 3, 4],
        "b": [1, 2, 5, 4],
        "c": [6, 2, 3, 7],
        "d": [6, 5, 3, 7],
    }
    freq = Counter(x for sequence in sessions.values() for x in sequence)
    rng = np.random.default_rng(4)
    semantic = rng.normal(size=(10, 12)).astype(np.float32)
    semantic[0] = 0
    return sessions, freq, semantic


def test_prototype_graph_builder_is_finite_and_vocab_aligned():
    sessions, freq, semantic = _fixture()
    config = PASGRConfig(dim=8, prototypes=3, epochs=1)
    embedding, assignment, neighbours = build_prototype_graph_embeddings(
        sessions, 10, freq, semantic, config)
    assert embedding.shape == (10, 8)
    assert assignment.shape == (10,)
    assert np.isfinite(embedding).all()
    assert np.allclose(embedding[0], 0)
    assert 2 in neighbours[1]


def test_pasgr_train_predict_contract_and_seen_filtering():
    sessions, freq, semantic = _fixture()
    config = PASGRConfig(dim=8, prototypes=3, epochs=2, batch_size=4,
                         hard_negatives=4, top_k=5, seed=7)
    model = train_pasgr(sessions, 10, freq, semantic, config, device="cpu")
    queries = {"q": {"context": [1, 2], "targets": [3]}}
    ranking = predict_pasgr(model, queries, 10)["q"]
    assert len(ranking) == 5
    assert 0 not in ranking
    assert 1 not in ranking
    assert 2 not in ranking
    assert len(set(ranking)) == len(ranking)
