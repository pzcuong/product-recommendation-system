"""CASM — Contrastive-Aligned Semantic Memory (DESIGN_CASM.md §2).

A fourth training-free-at-inference evidence memory for CEARF-N:

  * ``AlignmentHead``          — small torch head (linear by default, optional
    2-layer MLP ablation) mapping frozen teacher rows (d_t ∈ {128, 384}) to a
    128-d L2-normalised behavioural space.
  * ``train_alignment_head``   — symmetric in-batch InfoNCE on session
    co-occurrence positive pairs (window ≤ 4, weight 1/distance — the
    transition-memory convention of ``cearf.CEARFIndex``). Deterministic given
    the seed. Returns the aligned catalogue matrix (float32, L2-normalised).
  * ``CASMemory``              — batched retrieval memory. Query embedding is
    the recency-decayed mean of the last ≤ 8 context item embeddings
    (decay 0.35, matching ``SemanticMemory``); scoring is one chunked
    catalogue matmul; per-query top-120 lists. The single-query ``ranking``
    wrapper keeps the ``CEARFIndexV3`` memory interface.
  * Raw-semantic control       — ``CASMemory.from_teacher(teacher)`` runs the
    identical retrieval path on the frozen, un-aligned teacher matrix so any
    CASM–control gap is attributable to the alignment head.
  * ``load_or_train_casm``     — checksummed on-disk cache of the aligned
    matrix (mirrors ``build_minilm_teacher.py`` conventions).

Importing this module has no side effects. Torch is required only for the
training path; ``CASMemory`` itself is numpy-only at inference.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ALIGN_DIM = 128           # d of the aligned space (DESIGN_CASM.md §2.2)
PAIR_WINDOW = 4           # co-occurrence window (§2.3, transition convention)
DEFAULT_TAU = 0.07        # middle of the declared grid {0.05, 0.07, 0.10}
DEFAULT_EPOCHS = 5        # middle of the declared grid {3, 5, 10}
DEFAULT_BATCH = 1024
DEFAULT_LR = 1e-3
DEFAULT_PAIRS = 2_000_000
QUERY_DECAY = 0.35        # matches SemanticMemory / _transition_scores
CONTEXT_TAIL = 8
TOPN = 120                # component_topn (CEARFConfig)


def resolve_device(device: str | None = None):
    """MPS if available, else CUDA, else CPU. Explicit strings pass through."""
    import torch
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_alignment_head_class():
    import torch
    from torch import nn

    class AlignmentHead(nn.Module):
        """Teacher-space → 128-d aligned space, L2-normalised outputs.

        Linear by default (≤ 49.3k parameters); ``mlp=True`` selects the
        declared ablation variant d_t → 256 (GELU) → 128 (§2.2).
        """

        def __init__(self, d_in: int, d_out: int = ALIGN_DIM, mlp: bool = False):
            super().__init__()
            if mlp:
                self.net = nn.Sequential(
                    nn.Linear(d_in, 256), nn.GELU(), nn.Linear(256, d_out))
            else:
                self.net = nn.Linear(d_in, d_out)

        def forward(self, x):
            return torch.nn.functional.normalize(self.net(x), dim=-1, eps=1e-8)

    return AlignmentHead


def __getattr__(name: str):
    # Lazy export so `import casm` works without torch installed.
    if name == "AlignmentHead":
        return _make_alignment_head_class()
    raise AttributeError(f"module 'casm' has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Co-occurrence positive pairs (§2.3).
# ---------------------------------------------------------------------------
def enumerate_cooccurrence_pairs(
    sessions: Mapping[str, Sequence[int]], n_items: int,
    has_vector: np.ndarray, window: int = PAIR_WINDOW,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All ordered (source, target) pairs with distance ≤ window, 1/distance
    weighted — exactly the transition-memory convention (cearf.py lines
    102–107). Self-pairs and pairs where either side lacks a text vector are
    dropped.
    """
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    for seq0 in sessions.values():
        seq = [int(x) for x in seq0 if 0 < int(x) < n_items]
        for right in range(1, len(seq)):
            target = seq[right]
            if not has_vector[target]:
                continue
            for distance in range(1, min(window, right) + 1):
                source = seq[right - distance]
                if source == target or not has_vector[source]:
                    continue
                sources.append(source)
                targets.append(target)
                weights.append(1.0 / distance)
    return (np.asarray(sources, dtype=np.int64),
            np.asarray(targets, dtype=np.int64),
            np.asarray(weights, dtype=np.float64))


def sample_cooccurrence_pairs(
    sessions: Mapping[str, Sequence[int]], n_items: int,
    has_vector: np.ndarray, n_pairs: int = DEFAULT_PAIRS,
    window: int = PAIR_WINDOW, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample ``n_pairs`` positive pairs with probability ∝ 1/distance."""
    sources, targets, weights = enumerate_cooccurrence_pairs(
        sessions, n_items, has_vector, window)
    if not len(sources):
        raise ValueError("no eligible co-occurrence pairs "
                         "(sessions empty or items lack text vectors)")
    rng = np.random.default_rng(seed)
    picks = rng.choice(len(sources), size=int(n_pairs), replace=True,
                       p=weights / weights.sum())
    return sources[picks], targets[picks]


def teacher_has_vector(teacher: np.ndarray) -> np.ndarray:
    """Rows with a usable text vector; row 0 (padding) is always excluded."""
    norms = np.linalg.norm(np.asarray(teacher, dtype=np.float32), axis=1)
    has_vector = norms >= 1e-8
    if len(has_vector):
        has_vector[0] = False
    return has_vector


# ---------------------------------------------------------------------------
# InfoNCE training (§2.3).
# ---------------------------------------------------------------------------
def train_alignment_head(
    sessions: Mapping[str, Sequence[int]], teacher_matrix: np.ndarray, *,
    d_out: int = ALIGN_DIM, mlp: bool = False, tau: float = DEFAULT_TAU,
    epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH,
    n_pairs: int = DEFAULT_PAIRS, lr: float = DEFAULT_LR, seed: int = 42,
    device: str | None = None, verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """Train the alignment head; return (aligned catalogue matrix, info).

    The aligned matrix is (n_items, d_out) float32 with L2-normalised rows;
    items without a text vector (and row 0) stay zero and are therefore never
    retrieved. ``info`` carries the per-epoch mean loss and the exact config
    (used for the cache fingerprint). Deterministic given ``seed`` on a fixed
    device.
    """
    import torch
    import torch.nn.functional as F

    teacher = np.asarray(teacher_matrix, dtype=np.float32)
    n_items, d_in = teacher.shape
    has_vector = teacher_has_vector(teacher)
    src, tgt = sample_cooccurrence_pairs(
        sessions, n_items, has_vector, n_pairs=n_pairs, seed=seed)

    dev = resolve_device(device)
    torch.manual_seed(seed)
    AlignmentHead = _make_alignment_head_class()
    head = AlignmentHead(d_in, d_out, mlp=mlp).to(dev)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    teacher_t = torch.from_numpy(teacher).to(dev)
    src_t = torch.from_numpy(src)
    tgt_t = torch.from_numpy(tgt)

    rng = np.random.default_rng(seed)
    loss_per_epoch: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(src))
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            if len(rows) < 2:      # InfoNCE needs in-batch negatives
                continue
            idx = torch.from_numpy(rows)
            za = head(teacher_t[src_t[idx]])
            zb = head(teacher_t[tgt_t[idx]])
            logits = (za @ zb.T) / tau
            labels = torch.arange(len(rows), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.T, labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        loss_per_epoch.append(total / max(batches, 1))
        if verbose:
            print(f"[CASM] epoch {epoch + 1}/{epochs} "
                  f"loss={loss_per_epoch[-1]:.4f}", flush=True)

    aligned = np.zeros((n_items, d_out), dtype=np.float32)
    with torch.no_grad():
        rows = np.flatnonzero(has_vector)
        for start in range(0, len(rows), 8192):
            chunk = rows[start:start + 8192]
            z = head(teacher_t[torch.from_numpy(chunk)])
            aligned[chunk] = z.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(aligned, axis=1, keepdims=True)
    aligned = aligned / np.maximum(norms, 1e-8)
    aligned[~has_vector] = 0.0

    info = {
        "loss_per_epoch": loss_per_epoch,
        "n_pairs": int(len(src)),
        "config": {"d_out": int(d_out), "mlp": bool(mlp), "tau": float(tau),
                   "epochs": int(epochs), "batch_size": int(batch_size),
                   "n_pairs": int(n_pairs), "lr": float(lr),
                   "seed": int(seed), "window": PAIR_WINDOW,
                   "device": str(dev)},
    }
    return aligned, info


# ---------------------------------------------------------------------------
# Checksummed cache (mirrors build_minilm_teacher.py).
# ---------------------------------------------------------------------------
def sessions_fingerprint(sessions: Mapping[str, Sequence[int]]) -> str:
    digest = hashlib.sha256()
    for key in sorted(sessions, key=str):
        digest.update(str(key).encode())
        digest.update(b"|")
        digest.update(" ".join(str(int(x)) for x in sessions[key]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def casm_fingerprint(teacher_matrix: np.ndarray,
                     sessions: Mapping[str, Sequence[int]],
                     config: dict) -> str:
    digest = hashlib.sha256()
    teacher = np.ascontiguousarray(np.asarray(teacher_matrix, dtype=np.float32))
    digest.update(hashlib.sha256(teacher.tobytes()).hexdigest().encode())
    digest.update(sessions_fingerprint(sessions).encode())
    keys = {k: v for k, v in config.items() if k != "device"}
    digest.update(json.dumps(keys, sort_keys=True).encode())
    return digest.hexdigest()


def load_or_train_casm(
    cache_dir: Path, tag: str, sessions: Mapping[str, Sequence[int]],
    teacher_matrix: np.ndarray, **train_kwargs,
) -> np.ndarray:
    """Load a cached aligned matrix or train + cache it.

    Cache layout: ``<cache_dir>/<tag>_casm_<fingerprint12>.npy`` plus a JSON
    manifest with the full fingerprint, training config, final loss, and the
    sha256 of the matrix file.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    probe_config = {
        "d_out": int(train_kwargs.get("d_out", ALIGN_DIM)),
        "mlp": bool(train_kwargs.get("mlp", False)),
        "tau": float(train_kwargs.get("tau", DEFAULT_TAU)),
        "epochs": int(train_kwargs.get("epochs", DEFAULT_EPOCHS)),
        "batch_size": int(train_kwargs.get("batch_size", DEFAULT_BATCH)),
        "n_pairs": int(train_kwargs.get("n_pairs", DEFAULT_PAIRS)),
        "lr": float(train_kwargs.get("lr", DEFAULT_LR)),
        "seed": int(train_kwargs.get("seed", 42)),
        "window": PAIR_WINDOW,
    }
    fingerprint = casm_fingerprint(teacher_matrix, sessions, probe_config)
    matrix_path = cache_dir / f"{tag}_casm_{fingerprint[:12]}.npy"
    manifest_path = matrix_path.with_suffix(".json")
    if matrix_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        matrix_sha = hashlib.sha256(matrix_path.read_bytes()).hexdigest()
        if (manifest.get("fingerprint") == fingerprint
                and manifest.get("matrix_sha256") == matrix_sha):
            print(f"[CASM] cache hit: {matrix_path}", flush=True)
            return np.load(matrix_path)

    aligned, info = train_alignment_head(sessions, teacher_matrix,
                                         **train_kwargs)
    np.save(matrix_path, aligned)
    manifest = {
        "tag": tag,
        "fingerprint": fingerprint,
        "config": probe_config,
        "n_pairs_sampled": info["n_pairs"],
        "loss_per_epoch": info["loss_per_epoch"],
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "matrix": str(matrix_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[CASM] trained + cached {matrix_path}", flush=True)
    return aligned


# ---------------------------------------------------------------------------
# Batched retrieval memory (§2.4).
# ---------------------------------------------------------------------------
class CASMemory:
    """Batched cosine retrieval over an (aligned or raw) item matrix.

    Query = L2-normalised recency-decayed sum of the last ≤ ``context_tail``
    context item embeddings (decay per SemanticMemory). ``rankings_batch``
    scores all queries with chunked catalogue matmuls; ``ranking`` is the
    single-query wrapper required by ``CEARFIndexV3``. Rows without a vector
    (including padding row 0) are never retrieved.
    """

    def __init__(self, item_vectors: np.ndarray, topn: int = TOPN,
                 decay: float = QUERY_DECAY, context_tail: int = CONTEXT_TAIL,
                 chunk_size: int = 4096):
        vectors = np.asarray(item_vectors, dtype=np.float32).copy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-8)
        self.topn = int(topn)
        self.decay = float(decay)
        self.context_tail = int(context_tail)
        self.chunk_size = int(chunk_size)
        self.n_items = self.vectors.shape[0]
        self.has_vector = (norms[:, 0] >= 1e-8)
        if self.n_items:
            self.has_vector[0] = False

    @classmethod
    def from_teacher(cls, teacher_matrix: np.ndarray, **kwargs) -> "CASMemory":
        """Raw-semantic control: identical retrieval path on the frozen,
        un-aligned teacher matrix (L2-normalised only)."""
        return cls(teacher_matrix, **kwargs)

    def _query_vector(self, context: Sequence[int]) -> np.ndarray | None:
        tail = [int(x) for x in context
                if 0 < int(x) < self.n_items][-self.context_tail:]
        if not tail:
            return None
        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for age, item in enumerate(reversed(tail)):
            query += math.exp(-self.decay * age) * self.vectors[item]
        norm = float(np.linalg.norm(query))
        if norm < 1e-8:
            return None
        return query / norm

    def rankings_batch(self, contexts: Sequence[Sequence[int]],
                       blocked: Sequence[set[int]]) -> list[list[int]]:
        if len(contexts) != len(blocked):
            raise ValueError("contexts and blocked must have equal length")
        output: list[list[int]] = [[] for _ in contexts]
        queries: list[np.ndarray] = []
        rows: list[int] = []
        for position, context in enumerate(contexts):
            vector = self._query_vector(context)
            if vector is not None:
                queries.append(vector)
                rows.append(position)
        if not queries:
            return output
        matrix = np.stack(queries)
        wanted = min(self.topn, self.n_items)
        for start in range(0, len(rows), self.chunk_size):
            stop = min(start + self.chunk_size, len(rows))
            sims = matrix[start:stop] @ self.vectors.T
            sims[:, ~self.has_vector] = -np.inf
            for local, row in enumerate(rows[start:stop]):
                mask = [b for b in blocked[row] if 0 <= b < self.n_items]
                if mask:
                    sims[local, mask] = -np.inf
            if wanted < sims.shape[1]:
                part = np.argpartition(-sims, wanted - 1, axis=1)[:, :wanted]
            else:
                part = np.tile(np.arange(sims.shape[1]), (sims.shape[0], 1))
            for local, row in enumerate(rows[start:stop]):
                scores = sims[local, part[local]]
                order = part[local][np.argsort(-scores, kind="stable")]
                output[row] = [int(x) for x in order
                               if np.isfinite(sims[local, x])][:self.topn]
        return output

    def ranking(self, context: Sequence[int], blocked: set[int]) -> list[int]:
        """Single-query wrapper (CEARFIndexV3 memory interface parity)."""
        return self.rankings_batch([context], [blocked])[0]
