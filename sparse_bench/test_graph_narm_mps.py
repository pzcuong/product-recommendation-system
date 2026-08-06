import torch

from graph_narm_mps import GraphNARM
from narm_mps import NARMConfig


def test_edge_index_matches_sparse_matrix():
    config = NARMConfig(n_items=5, dim=4, dropout=0.0)
    src = torch.tensor([0, 1, 2, 3])
    dst = torch.tensor([1, 2, 2, 4])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.0])
    model = GraphNARM(config, src, dst, weight)
    h = model.graph_linear(model.item.weight)
    actual = torch.zeros_like(h).index_add(
        0, dst, h[src] * weight[:, None])
    matrix = torch.sparse_coo_tensor(
        torch.stack([dst, src]), weight, (5, 5))
    expected = torch.sparse.mm(matrix, h)
    assert torch.allclose(actual, expected)


def test_graph_narm_logits_and_gradients():
    config = NARMConfig(n_items=8, dim=4, dropout=0.0)
    nodes = torch.arange(8)
    model = GraphNARM(config, nodes, nodes, torch.ones(8))
    contexts = torch.tensor([[1, 2], [3, 0]])
    lengths = torch.tensor([2, 1])
    logits = model.logits(contexts, lengths)
    assert logits.shape == (2, 8)
    logits.sum().backward()
    assert model.graph_linear.weight.grad is not None
