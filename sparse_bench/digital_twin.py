"""Stateful dual digital twin for sequential recommendation.

The two twins have deliberately separate responsibilities:

* UserTwin assimilates observations into a posterior belief state.
* EnvironmentTwin models how that state evolves after a recommendation and
  predicts the next interaction/reward distribution.

At serving time each candidate is an intervention.  We clone the synchronized
belief, apply the intervention and roll the learned world model forward.  This
is materially different from calling a sequential encoder a "digital twin":
the twin has identity/state, observation assimilation, learned dynamics,
counterfactual branches and uncertainty-aware planning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TwinConfig:
    dim: int = 96
    rollout_horizon: int = 3
    discount: float = 0.85
    uncertainty_penalty: float = 0.15
    counterfactual_weight: float = 0.30
    transition_weight: float = 0.20
    reward_weight: float = 0.15
    confidence_floor: float = 0.08
    rerank_pool: int = 80
    temperature: float = 1.0
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 1e-3
    seed: int = 42
    transfer_weight: float = 0.05
    uncertainty_weight: float = 0.10
    initial_logit_scale: float = 10.0


@dataclass
class TwinState:
    """A versioned posterior belief belonging to one real-world entity."""
    entity_id: str
    belief: torch.Tensor
    observations: int
    version: int = 1

    def branch(self) -> "TwinState":
        return TwinState(self.entity_id, self.belief.clone(), self.observations,
                         self.version)


class UserTwin(nn.Module):
    """Bayesian-style observation assimilation with a learned uncertainty gate."""
    def __init__(self, n_items: int, dim: int):
        super().__init__()
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.assimilator = nn.GRUCell(dim, dim)
        self.prior = nn.Parameter(torch.zeros(dim))
        self.log_variance = nn.Sequential(nn.Linear(dim, dim), nn.Softplus())

    def synchronize(self, entity_id: str, observations: Sequence[int],
                    device: torch.device) -> TwinState:
        belief = self.prior.to(device).unsqueeze(0)
        count = 0
        for item in observations:
            if item <= 0 or item >= self.item.num_embeddings:
                continue
            belief = self.assimilator(self.item(torch.tensor([item], device=device)), belief)
            count += 1
        return TwinState(str(entity_id), belief.squeeze(0), count)

    def uncertainty(self, belief: torch.Tensor) -> torch.Tensor:
        return self.log_variance(belief).mean(-1)


class EnvironmentTwin(nn.Module):
    """Action-conditioned world model p(s[t+1], y[t+1] | s[t], do(a[t]))."""
    def __init__(self, item_embedding: nn.Embedding, dim: int):
        super().__init__()
        self.item = item_embedding
        self.transition = nn.GRUCell(dim, dim)
        self.state_projection = nn.Linear(dim, dim)
        self.reward = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))

    def intervene(self, belief: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_emb = self.item(action)
        next_belief = self.transition(action_emb, belief)
        reward = self.reward(torch.cat([belief, action_emb], -1)).squeeze(-1)
        return next_belief, reward

    def logits(self, belief: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.state_projection(belief), dim=-1)
        items = F.normalize(self.item.weight, dim=-1)
        return z @ items.t()


class DualDigitalTwin(nn.Module):
    """Coupled user/environment twins with counterfactual model-predictive ranking."""
    def __init__(self, n_items: int, config: TwinConfig = TwinConfig(),
                 teacher_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        self.config = config
        self.user = UserTwin(n_items, config.dim)
        self.register_buffer("teacher_embeddings", torch.empty(0), persistent=False)
        if teacher_embeddings is not None:
            teacher = teacher_embeddings.detach().float()
            if teacher.shape != self.user.item.weight.shape:
                raise ValueError(
                    f"teacher shape {tuple(teacher.shape)} != twin item shape "
                    f"{tuple(self.user.item.weight.shape)}")
            teacher = F.normalize(teacher, dim=-1)
            with torch.no_grad():
                self.user.item.weight.copy_(teacher)
                self.user.item.weight[0].zero_()
            self.teacher_embeddings = teacher.clone()
        self.environment = EnvironmentTwin(self.user.item, config.dim)
        # Cosine logits without a scale live in [-1, 1], which is too narrow
        # for one-positive/many-negative next-item discrimination.  Learning
        # the scale is the standard contrastive remedy and keeps the geometry
        # inherited from the semantic teacher intact.
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(config.initial_logit_scale))))

    def sequence_loss(self, sequence: Sequence[int], device: torch.device) -> torch.Tensor:
        state = self.user.prior.to(device).unsqueeze(0)
        losses: List[torch.Tensor] = []
        valid = [x for x in sequence if 0 < x < self.user.item.num_embeddings]
        for current, target in zip(valid[:-1], valid[1:]):
            current_t = torch.tensor([current], device=device)
            # Train the same observation-assimilation operator used by
            # synchronize() at inference. Previously this GRU was never in the
            # loss graph, so every deployed user belief was effectively random.
            state = self.user.assimilator(self.user.item(current_t), state)
            target_t = torch.tensor([target], device=device)
            # The observed next item is a positive intervention. Random unseen
            # actions are explicit negatives, preventing the degenerate reward
            # head that labels every historical action as positive.
            n_neg = min(64, max(4, self.user.item.num_embeddings - 2))
            # Never present the positive item as a negative label.  Sampling a
            # few extra values is cheaper than a catalog-sized mask.
            negative = torch.randint(1, self.user.item.num_embeddings,
                                     (n_neg + 8,), device=device)
            negative = negative[negative != target_t.item()].unique()[:n_neg]
            while negative.numel() < n_neg:
                extra = torch.randint(1, self.user.item.num_embeddings,
                                      (n_neg,), device=device)
                negative = torch.cat([negative, extra[extra != target_t.item()]]).unique()[:n_neg]
            pos_emb = self.user.item(target_t)
            neg_emb = self.user.item(negative)
            pos_reward = self.environment.reward(torch.cat([state, pos_emb], -1)).squeeze(-1)
            state_rep = state.expand(negative.numel(), -1)
            neg_reward = self.environment.reward(torch.cat([state_rep, neg_emb], -1)).squeeze(-1)
            reward_loss = F.softplus(-(pos_reward.unsqueeze(-1) - neg_reward)).mean()
            # Sampled normalized softmax scales independently of catalog size;
            # full-vocabulary CE per transition made large-domain experiments
            # needlessly quadratic in sessions × items.
            query = F.normalize(self.environment.state_projection(state), dim=-1)
            pos_logit = (query * F.normalize(pos_emb, dim=-1)).sum(-1, keepdim=True)
            neg_logits = query @ F.normalize(neg_emb, dim=-1).t()
            scale = self.logit_scale.exp().clamp(1.0, 100.0)
            sampled_logits = scale * torch.cat([pos_logit, neg_logits], dim=-1)
            sampled_target = torch.zeros(1, dtype=torch.long, device=device)
            # A digital twin needs its interventional and observational state
            # updates to describe the same system. Align do(target)'s predicted
            # state with the state obtained when that target is later observed.
            intervention_state, _ = self.environment.intervene(state, target_t)
            observation_state = self.user.assimilator(self.user.item(target_t), state)
            transition_loss = (1.0 - F.cosine_similarity(
                intervention_state, observation_state, dim=-1)).mean()
            # Calibrate epistemic risk to an observable model-consistency
            # residual. Previously the uncertainty head was never optimized,
            # yet its random output changed candidate ranks at inference.
            transition_residual = (1.0 - F.cosine_similarity(
                intervention_state, observation_state, dim=-1)).detach()
            predicted_uncertainty = self.user.uncertainty(intervention_state)
            uncertainty_loss = F.smooth_l1_loss(predicted_uncertainty,
                                                transition_residual)
            transfer_loss = state.sum() * 0.0
            if self.teacher_embeddings.numel():
                ids = torch.cat([target_t, negative]).unique()
                learned = F.normalize(self.user.item(ids), dim=-1)
                teacher = self.teacher_embeddings[ids]
                transfer_loss = (1.0 - (learned * teacher).sum(-1)).mean()
            losses.append(F.cross_entropy(sampled_logits / self.config.temperature, sampled_target) +
                          self.config.reward_weight * reward_loss +
                          self.config.transition_weight * transition_loss +
                          self.config.uncertainty_weight * uncertainty_loss +
                          self.config.transfer_weight * transfer_loss)
        return torch.stack(losses).mean() if losses else state.sum() * 0.0

    @torch.no_grad()
    def counterfactual_value(self, state: TwinState, candidate: int,
                             current_log_probs: Optional[torch.Tensor] = None) -> float:
        device = state.belief.device
        belief = state.belief.unsqueeze(0)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        if current_log_probs is None:
            current_logits = scale * self.environment.logits(belief) / self.config.temperature
            current_logits[:, 0] = -torch.inf
            current_log_probs = F.log_softmax(current_logits, -1).squeeze(0)
        immediate_likelihood = current_log_probs[candidate]
        action = torch.tensor([candidate], device=device)
        belief, immediate = self.environment.intervene(belief, action)
        value = immediate_likelihood + self.config.reward_weight * F.logsigmoid(immediate)
        discount = self.config.discount
        uncertainties = [self.user.uncertainty(belief)]
        horizon = self.config.rollout_horizon if self.user.item.num_embeddings <= 10000 else 1
        for _ in range(max(0, horizon - 1)):
            logits = scale * self.environment.logits(belief)
            logits[:, 0] = -torch.inf
            probs = torch.softmax(logits, -1)
            # Expected embedding makes the rollout deterministic and differentiable in spirit.
            expected_action = probs @ self.user.item.weight
            belief = self.environment.transition(expected_action, belief)
            expected_reward = self.environment.reward(torch.cat([belief, expected_action], -1)).squeeze(-1)
            # A good intervention should lead to a confident, coherent future
            # state, not merely a high unconstrained reward prediction.
            entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(-1)
            normalized_entropy = entropy / np.log(max(self.user.item.num_embeddings - 1, 2))
            value = value + self.config.transition_weight * discount * (
                torch.log(probs.max(-1).values.clamp_min(1e-9)) +
                self.config.reward_weight * F.logsigmoid(expected_reward))
            discount *= self.config.discount
            uncertainties.append(self.user.uncertainty(belief) + normalized_entropy)
        risk = torch.stack(uncertainties).mean()
        return float((value - self.config.uncertainty_penalty * risk).item())

    @torch.no_grad()
    def rerank(self, entity_id: str, context: Sequence[int],
               candidates: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
        device = next(self.parameters()).device
        state = self.user.synchronize(entity_id, context, device)
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda x: x[1], reverse=True)
        budget = self.config.rerank_pool
        if self.user.item.num_embeddings > 10000:
            budget = min(budget, 20)
        pool_n = min(len(ordered), max(2, budget))
        pool = ordered[:pool_n]
        base = np.asarray([s for _, s in pool], dtype=np.float64)
        base_z = (base - np.median(base)) / max(float(np.std(base)), 1e-6)
        # The pre-intervention catalog distribution is shared by every branch.
        # Computing it once per entity rather than once per candidate removes
        # the dominant O(pool * catalog) serving cost without approximation.
        belief = state.belief.unsqueeze(0)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        current_logits = scale * self.environment.logits(belief) / self.config.temperature
        current_logits[:, 0] = -torch.inf
        current_log_probs = F.log_softmax(current_logits, -1).squeeze(0)
        cf = np.asarray([self.counterfactual_value(state.branch(), int(item),
                                                   current_log_probs)
                         for item, _ in pool], dtype=np.float64)
        cf_z = (cf - np.median(cf)) / max(float(np.std(cf)), 1e-6)

        # Gate the twin by context evidence and agreement with the production
        # ranker. Anti-correlated/uncertain twins automatically fall back.
        context_conf = 1.0 - np.exp(-state.observations / 3.0)
        agreement = float(np.corrcoef(base_z, cf_z)[0, 1]) if pool_n > 2 else 0.0
        if not np.isfinite(agreement):
            agreement = 0.0
        gate = context_conf * max(self.config.confidence_floor, agreement)
        weight = self.config.counterfactual_weight * min(gate, 1.0)
        scores = base_z + weight * np.clip(cf_z, -3.0, 3.0)
        reranked = [(int(item), float(score)) for (item, _), score in zip(pool, scores)]
        reranked.sort(key=lambda x: x[1], reverse=True)
        # Retain the untouched tail so the method is a genuine safe reranker.
        floor = reranked[-1][1] - 1.0
        reranked.extend((int(item), floor - i) for i, (item, _) in enumerate(ordered[pool_n:], 1))
        return reranked


def train_digital_twin(train_sessions: Iterable[Sequence[int]], n_items: int,
                       config: TwinConfig = TwinConfig(), device: Optional[str] = None,
                       teacher_embeddings: Optional[np.ndarray] = None
                       ) -> DualDigitalTwin:
    """Fit the world model on observed trajectories; deterministic by default."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu"))
    teacher = (torch.as_tensor(teacher_embeddings, dtype=torch.float32, device=dev)
               if teacher_embeddings is not None else None)
    model = DualDigitalTwin(n_items, config, teacher_embeddings=teacher).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    sessions = [list(s) for s in train_sessions if len(s) >= 2]
    generator = np.random.default_rng(config.seed)
    model.train()
    for epoch in range(config.epochs):
        generator.shuffle(sessions)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for sequence in sessions:
            loss = model.sequence_loss(sequence, dev) / config.batch_size
            loss.backward()
            pending += 1
            if pending == config.batch_size:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); pending = 0
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        print(f"  [DigitalTwin] epoch {epoch + 1}/{config.epochs}", flush=True)
    model.eval()
    return model
