import sys
from pathlib import Path

import torch


MGCOT = Path(__file__).resolve().parents[1] / "reference_repos" / "MGCOT"
sys.path.insert(0, str(MGCOT))
from model import hard_top_rank_loss, sampled_scl_loss  # noqa: E402


def test_hard_rank_loss_rewards_target_above_negatives():
    target = torch.tensor([0])
    bad = torch.tensor([[0.0, 3.0, 2.0, 1.0]], requires_grad=True)
    good = torch.tensor([[4.0, 3.0, 2.0, 1.0]], requires_grad=True)
    bad_loss = hard_top_rank_loss(bad, target, k=3)
    good_loss = hard_top_rank_loss(good, target, k=3)
    assert good_loss < bad_loss
    bad_loss.backward()
    assert bad.grad[0, 0] < 0
    assert bad.grad[0, 1] > 0


def test_scl_loss_is_finite_and_has_gradient():
    embedding = torch.randn(33, 8, requires_grad=True)
    loss = sampled_scl_loss(embedding, sample_size=16, temperature=0.1)
    assert torch.isfinite(loss)
    loss.backward()
    assert embedding.grad is not None


def test_scl_dedicated_generator_is_reproducible():
    embedding = torch.randn(33, 8)
    a = sampled_scl_loss(
        embedding, 16, 0.1, torch.Generator().manual_seed(7))
    b = sampled_scl_loss(
        embedding, 16, 0.1, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
