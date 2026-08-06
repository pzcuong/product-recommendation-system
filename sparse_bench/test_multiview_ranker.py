from sparse_bench.multiview_ranker import (apply_rank_fusion, evaluate_rank_fusion,
                                           fit_rank_fusion)


def test_validation_fusion_learns_complementary_semantic_view():
    queries = {str(i): {"context": [1], "targets": [10 + i]} for i in range(8)}
    sknn = {str(i): ([10 + i, 99] if i < 4 else [99, 98]) for i in range(8)}
    semantic = {str(i): ([97, 96] if i < 4 else [10 + i, 97]) for i in range(8)}
    pop = {str(i): [99, 98] for i in range(8)}
    views = {"SKNN": sknn, "MostPop": pop, "Semantic": semantic,
             "DualTwin": sknn}
    policy = fit_rank_fusion(views, queries)
    result = apply_rank_fusion(policy, views, queries)
    assert policy["weights"]["Semantic"] > 0
    assert all(10 + i in result[str(i)][:6] for i in range(8))


def test_frozen_policy_is_scored_on_disjoint_gate():
    experts = {"SKNN": {"g": [1, 2]}, "Semantic": {"g": [3, 4]}}
    policy = {"weights": {"SKNN": 1.0, "Semantic": 1.0}, "rrf_k": 10}
    score = evaluate_rank_fusion(
        policy, experts, {"g": {"context": [9], "targets": [1]}})
    assert score == {"utility": 5.0, "recall@6": 1.0, "n": 1}


def test_pasgr_is_a_tunable_view():
    queries = {str(i): {"targets": [20 + i]} for i in range(6)}
    views = {
        "SKNN": {str(i): [99] for i in range(6)},
        "PASGR": {str(i): [20 + i, 98] for i in range(6)},
    }
    policy = fit_rank_fusion(views, queries)
    assert policy["weights"]["PASGR"] > 0
