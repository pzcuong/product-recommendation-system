"""Prototype-Aligned Semantic Graph Retrieval (PASGR).

PASGR is a lightweight sequential recommender for sparse, long-tail domains.
It deliberately uses an LLM/SLM only as an offline item-semantic teacher:

1. head-item semantic prototypes transport stable structure to tail items;
2. directed transition aggregation injects collaborative/session evidence;
3. a small GRU is trained with semantic and graph hard negatives;
4. inference is exact full-catalog retrieval, with no target/test leakage.

The implementation is self-contained so it can be evaluated as a genuine
model rather than as a renamed post-hoc ensemble.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class PASGRConfig:
    dim: int = 64
    prototypes: int = 64
    max_seq: int = 50
    epochs: int = 8
    batch_size: int = 256
    hard_negatives: int = 32
    learning_rate: float = 1e-3
    contrastive_weight: float = 0.15
    inbatch_weight: float = 0.0
    graph_weight: float = 0.35
    prototype_temperature: float = 0.10
    prototype_transport: bool = True
    seed: int = 42
    top_k: int = 200


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def build_prototype_graph_embeddings(
        sessions: Mapping[str, Sequence[int]], n_items: int,
        item_freq: Counter, semantic: Optional[np.ndarray],
        config: PASGRConfig) -> tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """Build prototype-aligned semantic+transition item representations.

    Returns the fused item matrix, the nearest prototype assignment used for
    hard negatives, and directed graph-neighbour arrays.
    """
    rng = np.random.default_rng(config.seed)
    if semantic is None or semantic.shape[0] != n_items:
        base = rng.standard_normal((n_items, config.dim)).astype(np.float32)
        has_semantic = np.zeros(n_items, dtype=bool)
    else:
        raw = semantic.astype(np.float32, copy=False)
        has_semantic = np.linalg.norm(raw, axis=1) > 0
        out_dim = min(config.dim, raw.shape[1], max(2, int(has_semantic.sum()) - 1))
        if raw.shape[1] == out_dim:
            base = raw.copy()
        elif has_semantic.sum() > out_dim:
            svd = TruncatedSVD(out_dim, random_state=config.seed)
            base = np.zeros((n_items, out_dim), dtype=np.float32)
            base[has_semantic] = svd.fit_transform(raw[has_semantic]).astype(np.float32)
        else:
            base = rng.standard_normal((n_items, out_dim)).astype(np.float32)
        if out_dim < config.dim:
            base = np.pad(base, ((0, 0), (0, config.dim - out_dim)))
    base = _normalize(base.astype(np.float32))
    base[0] = 0

    freq = np.asarray([item_freq.get(i, 0) for i in range(n_items)], dtype=np.float32)
    observed = np.flatnonzero((freq > 0) & has_semantic)
    if observed.size < 2:
        observed = np.flatnonzero(freq > 0)
    n_proto = min(config.prototypes, max(1, observed.size))
    # Fit prototypes on the better-observed half; tail items then inherit a
    # global semantic anchor instead of a noisy independent ID vector.
    if observed.size:
        threshold = np.quantile(freq[observed], 0.50)
        head = observed[freq[observed] >= threshold]
        if head.size < n_proto:
            head = observed
        kmeans = MiniBatchKMeans(n_clusters=min(n_proto, head.size),
                                 batch_size=1024, n_init=3,
                                 random_state=config.seed)
        kmeans.fit(base[head])
        centers = _normalize(kmeans.cluster_centers_.astype(np.float32))
        similarity = base @ centers.T / max(config.prototype_temperature, 1e-3)
        similarity -= similarity.max(axis=1, keepdims=True)
        membership = np.exp(similarity)
        membership /= np.maximum(membership.sum(axis=1, keepdims=True), 1e-8)
        transported = membership @ centers
        assignment = membership.argmax(axis=1).astype(np.int64)
        if config.prototype_transport:
            trust = np.sqrt(freq / (freq + 10.0))[:, None]
            aligned = trust * base + (1.0 - trust) * transported
        else:
            aligned = base
    else:
        aligned = base
        assignment = np.zeros(n_items, dtype=np.int64)

    # Directed transition graph; retain counts for sampling genuinely hard
    # graph negatives at training time.
    neighbours: Dict[int, Counter] = defaultdict(Counter)
    graph_sum = np.zeros_like(aligned, dtype=np.float32)
    graph_count = np.zeros(n_items, dtype=np.float32)
    for sequence in sessions.values():
        valid = [int(x) for x in sequence if 0 < int(x) < n_items]
        for left, right in zip(valid[:-1], valid[1:]):
            neighbours[left][right] += 1
            neighbours[right][left] += 1
            graph_sum[left] += aligned[right]
            graph_sum[right] += aligned[left]
            graph_count[left] += 1
            graph_count[right] += 1
    graph = graph_sum / np.maximum(graph_count[:, None], 1.0)
    graph = _normalize(graph)
    fused = _normalize(aligned + config.graph_weight * graph)
    fused[0] = 0
    graph_arrays = {item: np.asarray([x for x, _ in counts.most_common(64)],
                                     dtype=np.int64)
                    for item, counts in neighbours.items()}
    return fused.astype(np.float32), assignment, graph_arrays


class _PrefixDataset(Dataset):
    def __init__(self, sessions: Mapping[str, Sequence[int]], n_items: int,
                 max_seq: int):
        self.sequences: list[list[int]] = []
        self.examples: list[tuple[int, int]] = []
        for sequence in sessions.values():
            valid = [int(x) for x in sequence if 0 < int(x) < n_items]
            if len(valid) < 2:
                continue
            sequence_id = len(self.sequences)
            self.sequences.append(valid)
            for index in range(1, len(valid)):
                self.examples.append((sequence_id, index))
        self.max_seq = max_seq

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        sequence_id, position = self.examples[index]
        sequence = self.sequences[sequence_id]
        return (sequence[max(0, position - self.max_seq):position], sequence[position])


def _collate(batch):
    length = max(len(context) for context, _ in batch)
    contexts = torch.zeros(len(batch), length, dtype=torch.long)
    lengths = torch.empty(len(batch), dtype=torch.long)
    targets = torch.empty(len(batch), dtype=torch.long)
    for row, (context, target) in enumerate(batch):
        contexts[row, :len(context)] = torch.as_tensor(context)
        lengths[row] = len(context)
        targets[row] = target
    return contexts, lengths, targets


class PASGRModel(nn.Module):
    def __init__(self, initial_embeddings: np.ndarray, config: PASGRConfig):
        super().__init__()
        n_items, dim = initial_embeddings.shape
        self.config = config
        self.item = nn.Embedding(n_items, dim, padding_idx=0)
        self.item.weight.data.copy_(torch.from_numpy(initial_embeddings))
        self.encoder = nn.GRU(dim, dim, batch_first=True)
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(10.0))))
        self.register_buffer("semantic_teacher",
                             torch.from_numpy(initial_embeddings.copy()), persistent=False)

    def encode(self, contexts: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(self.item(contexts))
        rows = torch.arange(contexts.size(0), device=contexts.device)
        state = encoded[rows, (lengths - 1).clamp_min(0)]
        return F.normalize(self.query(state), dim=-1)

    def candidate_logits(self, query: torch.Tensor,
                         candidates: torch.Tensor) -> torch.Tensor:
        candidate_emb = F.normalize(self.item(candidates), dim=-1)
        scale = self.logit_scale.exp().clamp(1.0, 100.0)
        return scale * torch.einsum("bd,bkd->bk", query, candidate_emb)


def _sample_hard_negatives(targets: np.ndarray, contexts: np.ndarray,
                           assignment: np.ndarray,
                           prototype_members: Dict[int, np.ndarray],
                           graph_neighbours: Dict[int, np.ndarray],
                           n_items: int, count: int,
                           rng: np.random.Generator) -> np.ndarray:
    negatives = np.empty((len(targets), count), dtype=np.int64)
    for row, target in enumerate(targets):
        blocked = {int(target), 0, *[int(x) for x in contexts[row] if x > 0]}
        pool: list[int] = []
        semantic_pool = prototype_members.get(int(assignment[target]), np.empty(0, dtype=np.int64))
        if semantic_pool.size:
            pool.extend(int(x) for x in rng.choice(
                semantic_pool, size=min(count, semantic_pool.size), replace=False)
                        if int(x) not in blocked)
        for source in contexts[row, -5:]:
            if source <= 0:
                continue
            graph_pool = graph_neighbours.get(int(source), np.empty(0, dtype=np.int64))
            pool.extend(int(x) for x in graph_pool[:count] if int(x) not in blocked)
        pool = list(dict.fromkeys(pool))
        while len(pool) < count:
            item = int(rng.integers(1, n_items))
            if item not in blocked and item not in pool:
                pool.append(item)
        negatives[row] = np.asarray(pool[:count])
    return negatives


def train_pasgr(train_sessions: Mapping[str, Sequence[int]], n_items: int,
                item_freq: Counter, semantic_embeddings: Optional[np.ndarray],
                config: PASGRConfig = PASGRConfig(),
                device: Optional[str] = None,
                prepared_assets: Optional[tuple[np.ndarray, np.ndarray,
                                                Dict[int, np.ndarray]]] = None
                ) -> PASGRModel:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                  "mps" if torch.backends.mps.is_available() else "cpu"))
    if prepared_assets is None:
        initial, assignment, graph_neighbours = build_prototype_graph_embeddings(
            train_sessions, n_items, item_freq, semantic_embeddings, config)
    else:
        initial, assignment, graph_neighbours = prepared_assets
    model = PASGRModel(initial, config).to(dev)
    dataset = _PrefixDataset(train_sessions, n_items, config.max_seq)
    if not len(dataset):
        return model.eval()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        collate_fn=_collate, drop_last=False)
    prototype_members = {cluster: np.flatnonzero(assignment == cluster)
                         for cluster in np.unique(assignment)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=1e-4)
    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        seen = 0
        for contexts, lengths, targets in loader:
            context_np = contexts.numpy()
            target_np = targets.numpy()
            negatives = _sample_hard_negatives(
                target_np, context_np, assignment, prototype_members,
                graph_neighbours, n_items, config.hard_negatives, rng)
            candidates = torch.cat([targets[:, None], torch.from_numpy(negatives)], dim=1)
            contexts, lengths = contexts.to(dev), lengths.to(dev)
            candidates = candidates.to(dev)
            query = model.encode(contexts, lengths)
            logits = model.candidate_logits(query, candidates)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=dev)
            rank_loss = F.cross_entropy(logits, labels)
            # Full-catalog inference is much harder than discrimination among
            # a few sampled candidates. In-batch retrieval supplies hundreds
            # of observed negatives per update. Repeated targets are treated
            # as additional positives instead of false negatives.
            inbatch_loss = torch.zeros((), device=dev)
            if config.inbatch_weight > 0 and targets.numel() > 1:
                target_ids = targets.to(dev)
                target_emb = F.normalize(model.item(target_ids), dim=-1)
                scale = model.logit_scale.exp().clamp(1.0, 100.0)
                batch_logits = scale * (query @ target_emb.T)
                positive = target_ids[:, None].eq(target_ids[None, :])
                positive_logits = batch_logits.masked_fill(~positive, -torch.inf)
                inbatch_loss = -(
                    torch.logsumexp(positive_logits, dim=1)
                    - torch.logsumexp(batch_logits, dim=1)
                ).mean()
            # Align the behavioral state with a recency-aware semantic view;
            # gradients update the sequence encoder, not the frozen teacher.
            positions = torch.arange(contexts.size(1), device=dev)[None, :]
            mask = positions < lengths[:, None]
            recency = torch.exp(-0.35 * (lengths[:, None] - 1 - positions).clamp_min(0)) * mask
            teacher = model.semantic_teacher[contexts]
            semantic_state = (teacher * recency[..., None]).sum(1) / \
                recency.sum(1, keepdim=True).clamp_min(1e-8)
            contrastive = (1.0 - F.cosine_similarity(
                query, F.normalize(semantic_state, dim=-1), dim=-1)).mean()
            loss = (rank_loss + config.inbatch_weight * inbatch_loss
                    + config.contrastive_weight * contrastive)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * contexts.size(0)
            seen += contexts.size(0)
        print(f"  [PASGR] epoch {epoch + 1}/{config.epochs} loss={total / max(seen, 1):.4f}",
              flush=True)
    return model.eval()


@torch.no_grad()
def predict_pasgr_array(model: PASGRModel,
                        queries: Mapping[str, Mapping[str, Sequence[int]]],
                        n_items: int, top_k: Optional[int] = None,
                        exclude_seen: bool = True
                        ) -> tuple[list[str], np.ndarray]:
    """Return full-catalog rankings in a compact, deterministic array.

    The array form avoids the large Python-object overhead of retaining up to
    120 integer candidates for every query during multi-seed evidence runs.
    Row order is the lexicographically sorted string query id.
    """
    device = next(model.parameters()).device
    keys = sorted(queries)
    k = int(top_k or model.config.top_k)
    width = min(k, n_items - 1)
    result = np.empty((len(keys), width), dtype=np.int32)
    catalog = F.normalize(model.item.weight, dim=-1)
    batch_size = 256
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        sequences = [[int(x) for x in queries[uid].get("context", [])
                      if 0 < int(x) < n_items][-model.config.max_seq:]
                     for uid in batch_keys]
        max_len = max([len(x) for x in sequences] + [1])
        contexts = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        lengths = torch.ones(len(sequences), dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            if sequence:
                contexts[row, :len(sequence)] = torch.as_tensor(sequence, device=device)
                lengths[row] = len(sequence)
        query = model.encode(contexts, lengths)
        scores = query @ catalog.T
        scores[:, 0] = -torch.inf
        if exclude_seen:
            for row, sequence in enumerate(sequences):
                if sequence:
                    scores[row, torch.as_tensor(list(set(sequence)), device=device)] = -torch.inf
        top = torch.topk(scores, k=width, dim=-1).indices.cpu().numpy()
        result[start:start + len(batch_keys)] = top.astype(np.int32, copy=False)
    return keys, result


@torch.no_grad()
def predict_pasgr(model: PASGRModel,
                  queries: Mapping[str, Mapping[str, Sequence[int]]],
                  n_items: int, top_k: Optional[int] = None,
                  exclude_seen: bool = True) -> Dict[str, list[int]]:
    keys, rankings = predict_pasgr_array(
        model, queries, n_items, top_k=top_k, exclude_seen=exclude_seen)
    return {uid: [int(x) for x in row] for uid, row in zip(keys, rankings)}


@torch.no_grad()
def evaluate_pasgr(model: PASGRModel,
                   queries: Mapping[str, Mapping[str, Sequence[int]]],
                   n_items: int, k_values: Sequence[int] = (6, 10, 20),
                   batch_size: int = 256) -> Dict[str, float]:
    """Stream full-catalog evaluation without retaining prediction lists."""
    device = next(model.parameters()).device
    keys = sorted(queries)
    max_k = min(max(k_values), n_items - 1)
    hits = {int(k): 0 for k in k_values}
    ndcg = {int(k): 0.0 for k in k_values}
    catalog = F.normalize(model.item.weight, dim=-1)
    for start in range(0, len(keys), batch_size):
        batch_keys = keys[start:start + batch_size]
        sequences = [[int(x) for x in queries[uid].get("context", [])
                      if 0 < int(x) < n_items][-model.config.max_seq:]
                     for uid in batch_keys]
        max_len = max([len(x) for x in sequences] + [1])
        contexts = torch.zeros(len(sequences), max_len, dtype=torch.long, device=device)
        lengths = torch.ones(len(sequences), dtype=torch.long, device=device)
        for row, sequence in enumerate(sequences):
            if sequence:
                contexts[row, :len(sequence)] = torch.as_tensor(sequence, device=device)
                lengths[row] = len(sequence)
        scores = model.encode(contexts, lengths) @ catalog.T
        scores[:, 0] = -torch.inf
        for row, sequence in enumerate(sequences):
            if sequence:
                scores[row, torch.as_tensor(list(set(sequence)), device=device)] = -torch.inf
        ranking = torch.topk(scores, k=max_k, dim=-1).indices.cpu().numpy()
        for row, uid in enumerate(batch_keys):
            target = set(int(x) for x in queries[uid].get("targets", []))
            rank = next((idx for idx, item in enumerate(ranking[row], 1)
                         if int(item) in target), None)
            if rank is None:
                continue
            for k in k_values:
                if rank <= k:
                    hits[int(k)] += 1
                    ndcg[int(k)] += 1.0 / np.log2(rank + 1)
    count = max(len(keys), 1)
    output: Dict[str, float] = {}
    for k in k_values:
        output[f"recall@{k}"] = hits[int(k)] / count
        output[f"ndcg@{k}"] = ndcg[int(k)] / count
    output["n"] = len(keys)
    return output


def run_pasgr(train_sessions: Mapping[str, Sequence[int]],
              queries: Mapping[str, Mapping[str, Sequence[int]]], n_items: int,
              item_freq: Counter, semantic_embeddings: Optional[np.ndarray],
              config: PASGRConfig = PASGRConfig(),
              device: Optional[str] = None) -> Dict[str, list[int]]:
    model = train_pasgr(train_sessions, n_items, item_freq, semantic_embeddings,
                        config=config, device=device)
    return predict_pasgr(model, queries, n_items)
