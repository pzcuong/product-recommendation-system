"""CEARF v3 extensions — hai memory training-free mới + hook nâng teacher.

Thêm vào cạnh cearf.py (không sửa file gốc):

  * M4 ``SemanticMemory``  — truy hồi item theo cosine trên embedding văn bản
    (TF-IDF-SVD hiện tại hoặc MiniLM); training-free; OFF ở mọi profile cũ.
  * M5 ``repeat_ranking``  — bằng chứng repeat/recency cho protocol
    repeat-consumption (Diginetica_HID, Tmall); OFF khi exclude_seen=True.
  * ``PROFILES_V3``        — profile 5 trọng số (transition, session,
    popularity, semantic, repeat). Mọi profile cũ được giữ nguyên với hai
    slot mới = 0, nên tập cấu hình v3 CHỨA tập v2 → tính rejection-complete
    được bảo toàn.
  * ``tune_profiles_v3``   — như cearf.tune_profiles nhưng tie-break đúng
    chuẩn bài báo: hòa điểm → ít thành phần hoạt động hơn (thay cho
    tie-break theo tên profile trong bản gốc).
  * ``minilm_matrix``      — encode text item bằng sentence-transformers
    (all-MiniLM-L6-v2, 384d) làm teacher thay/đặt cạnh TF-IDF-SVD.

Cách chạy nhanh (memory-side, không cần train):

    import cearf, loaders, numpy as np
    from cearf_v3_ext import (SemanticMemory, CEARFIndexV3,
                              PROFILES_V3, tune_profiles_v3)
    data  = loaders.ALL_LOADERS["Video_Games"]()
    S     = np.load("semantic_vg.npy")          # (n_items, d) — TF-IDF-SVD cũ
    index = CEARFIndexV3(data["train_sessions"], data["n_items"],
                         semantic=SemanticMemory(S),
                         use_repeat=False)       # True cho Diginetica_HID/Tmall
    profiles, report = tune_profiles_v3(index, data["valid_queries"])
    preds = index.predict(data["test_queries"], profiles)

Lưu ý protocol: với Diginetica_HID/Tmall dùng CEARFConfig(exclude_seen=False)
như pipeline hiện tại; M5 tự tắt khi exclude_seen=True (fuse sẽ block các
item context nên list repeat vô nghĩa ở chế độ đó).
"""
from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

import cearf
from cearf import CEARFConfig, CEARFIndex, recall_at


# ---------------------------------------------------------------------------
# Profile 6 trọng số: (transition, session, popularity, semantic_raw, repeat,
# casm) — DESIGN_CASM.md §2.5. Mọi profile cũ giữ nguyên với casm = 0, nên
# tập cấu hình v3 ⊂ v4 → tính rejection-complete được bảo toàn.
# ---------------------------------------------------------------------------
PROFILES_V3: dict[str, tuple[float, float, float, float, float, float]] = {
    # --- kế thừa nguyên trạng (slot mới = 0) ---
    "transition":         (1.00, 0.00, 0.00, 0.00, 0.00, 0.00),
    "session":            (0.00, 1.00, 0.00, 0.00, 0.00, 0.00),
    "balanced":           (0.50, 0.40, 0.10, 0.00, 0.00, 0.00),
    "transition_session": (0.60, 0.40, 0.00, 0.00, 0.00, 0.00),
    "session_transition": (0.35, 0.60, 0.05, 0.00, 0.00, 0.00),
    "short_safe":         (0.65, 0.20, 0.15, 0.00, 0.00, 0.00),
    # --- biến thể có semantic (Amazon là nơi kỳ vọng) ---
    "semantic_only":      (0.00, 0.00, 0.00, 1.00, 0.00, 0.00),
    "balanced_semantic":  (0.40, 0.35, 0.05, 0.20, 0.00, 0.00),
    "semantic_tail":      (0.30, 0.30, 0.05, 0.35, 0.00, 0.00),
    # --- biến thể có repeat (protocol repeat-consumption) ---
    "repeat_only":        (0.00, 0.00, 0.00, 0.00, 1.00, 0.00),
    "balanced_repeat":    (0.40, 0.35, 0.05, 0.00, 0.20, 0.00),
    "repeat_heavy":       (0.30, 0.30, 0.00, 0.00, 0.40, 0.00),
    # --- đủ cả hai ---
    "semantic_repeat":    (0.30, 0.30, 0.00, 0.20, 0.20, 0.00),
    # --- CASM (DESIGN_CASM.md §5, khai báo đóng băng — không thêm ngoài) ---
    "casm_only":          (0.00, 0.00, 0.00, 0.00, 0.00, 1.00),
    "balanced_casm":      (0.40, 0.35, 0.05, 0.00, 0.00, 0.20),
    "casm_tail":          (0.30, 0.30, 0.05, 0.00, 0.00, 0.35),
}


def complexity(profile: Sequence[float]) -> int:
    """Số thành phần đang hoạt động — thước đo 'đơn giản hơn' cho tie-break."""
    return sum(1 for w in profile if w > 0)


# ---------------------------------------------------------------------------
# M4 — Semantic kNN memory (training-free).
# ---------------------------------------------------------------------------
class SemanticMemory:
    """Truy hồi theo cosine trên embedding item; query = tổng có decay của
    embedding các item trong context (mới nhất nặng nhất).

    ``item_vectors``: (n_items, d); hàng 0 là padding, không dùng. Ma trận sẽ
    được L2-normalise tại chỗ. Với 25–43k item × 128–384d, mỗi truy vấn là
    một matmul — vài mili giây trên CPU.
    """

    def __init__(self, item_vectors: np.ndarray, topn: int = 120,
                 decay: float = 0.35, context_tail: int = 8):
        vectors = np.asarray(item_vectors, dtype=np.float32).copy()
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = vectors / np.maximum(norms, 1e-8)
        self.topn = int(topn)
        self.decay = float(decay)
        self.context_tail = int(context_tail)
        self.n_items = self.vectors.shape[0]
        self.has_vector = (norms[:, 0] >= 1e-8)
        if self.n_items:
            self.has_vector[0] = False

    def ranking(self, context: Sequence[int], blocked: set[int]) -> list[int]:
        tail = [int(x) for x in context
                if 0 < int(x) < self.n_items][-self.context_tail:]
        if not tail:
            return []
        query = np.zeros(self.vectors.shape[1], dtype=np.float32)
        for age, item in enumerate(reversed(tail)):
            query += math.exp(-self.decay * age) * self.vectors[item]
        norm = float(np.linalg.norm(query))
        if norm < 1e-8:          # item không có text (fallback hash = 0-vector)
            return []
        sims = self.vectors @ (query / norm)
        sims[~self.has_vector] = -np.inf
        wanted = min(self.topn + len(blocked) + 1,
                     int(self.has_vector.sum()))
        if wanted <= 0:
            return []
        part = np.argpartition(-sims, wanted - 1)[:wanted]
        order = part[np.argsort(-sims[part])]
        output: list[int] = []
        for item in order:
            item = int(item)
            if item > 0 and item not in blocked:
                output.append(item)
                if len(output) >= self.topn:
                    break
        return output


# ---------------------------------------------------------------------------
# M5 — Repeat/recency memory (chỉ có nghĩa khi exclude_seen=False).
# ---------------------------------------------------------------------------
def repeat_ranking(context: Sequence[int], n_items: int,
                   topn: int = 120, decay: float = 0.5) -> list[int]:
    """Item đã xuất hiện trong context, chấm theo recency + số lần lặp."""
    scores: dict[int, float] = {}
    tail = [int(x) for x in context if 0 < int(x) < n_items]
    for age, item in enumerate(reversed(tail)):
        scores[item] = scores.get(item, 0.0) + math.exp(-decay * age)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [item for item, _ in ranked[:topn]]


# ---------------------------------------------------------------------------
# Index v3 — trả 5 danh sách; fuse_rankings gốc zip theo profile nên chạy
# nguyên trạng với profile 5 trọng số.
# ---------------------------------------------------------------------------
class CEARFIndexV3(CEARFIndex):
    def __init__(self, sessions: Mapping[str, Sequence[int]], n_items: int,
                 config: CEARFConfig | None = None,
                 semantic: SemanticMemory | None = None,
                 use_repeat: bool = False,
                 casm=None):
        super().__init__(sessions, n_items, config or CEARFConfig())
        self.semantic = semantic
        self.use_repeat = bool(use_repeat)
        self.casm = casm
        if semantic is not None and semantic.n_items != int(n_items):
            raise ValueError(
                f"semantic rows ({semantic.n_items}) != n_items ({n_items})")
        if casm is not None and casm.n_items != int(n_items):
            raise ValueError(
                f"casm rows ({casm.n_items}) != n_items ({n_items})")

    def _blocked(self, context: Sequence[int]) -> set[int]:
        return ({int(x) for x in context if int(x) > 0}
                if self.config.exclude_seen else set())

    def component_rankings(self, context: Sequence[int]):  # type: ignore[override]
        transition, session, popularity = super().component_rankings(context)
        blocked = self._blocked(context)
        semantic = (self.semantic.ranking(context, blocked)
                    if self.semantic is not None else [])
        repeat = (repeat_ranking(context, self.n_items)
                  if (self.use_repeat and not self.config.exclude_seen) else [])
        casm = (self.casm.ranking(context, blocked)
                if self.casm is not None else [])
        return transition, session, popularity, semantic, repeat, casm

    def component_rankings_batch(self, contexts: Sequence[Sequence[int]]):
        """Per-query 6-tuples; slot semantic/CASM dùng rankings_batch (một
        loạt matmul chunked cho mọi truy vấn) khi memory hỗ trợ, thay vì một
        matvec mỗi truy vấn."""
        def batch_lists(memory):
            if memory is None:
                return [[] for _ in contexts]
            if hasattr(memory, "rankings_batch"):
                return memory.rankings_batch(list(contexts), blocked)
            return [memory.ranking(ctx, block)
                    for ctx, block in zip(contexts, blocked)]

        blocked = [self._blocked(ctx) for ctx in contexts]
        semantic_lists = batch_lists(self.semantic)
        casm_lists = batch_lists(self.casm)
        output = []
        for ctx, semantic, casm in zip(contexts, semantic_lists, casm_lists):
            transition, session, popularity = CEARFIndex.component_rankings(
                self, ctx)
            repeat = (repeat_ranking(ctx, self.n_items)
                      if (self.use_repeat and not self.config.exclude_seen)
                      else [])
            output.append((transition, session, popularity, semantic,
                           repeat, casm))
        return output


# ---------------------------------------------------------------------------
# Tuner v3 — logic như cearf.tune_profiles, nhưng tie-break theo ĐỘ ĐƠN GIẢN
# (ít thành phần hoạt động hơn thắng khi hòa), đúng như bài tuyên bố.
# ---------------------------------------------------------------------------
def tune_profiles_v3(index: CEARFIndexV3,
                     validation: Mapping[str, Mapping[str, Sequence[int]]],
                     profiles: Mapping[str, tuple] = PROFILES_V3):
    chosen: dict[str, tuple] = {}
    report: dict = {}
    for regime in ("short", "long"):
        subset = {str(uid): q for uid, q in validation.items()
                  if (len(q.get("context", [])) <= index.config.short_context)
                  == (regime == "short")}
        if not subset:
            name = "short_safe" if regime == "short" else "session_transition"
            chosen[regime] = profiles[name]
            report[regime] = {"profile": name, "n": 0, "score": None}
            continue
        cached = {uid: index.component_rankings(q.get("context", []))
                  for uid, q in subset.items()}
        best = None
        for name, profile in profiles.items():
            predictions = {uid: index.fuse_rankings(
                q.get("context", []), cached[uid], profile, 20)
                for uid, q in subset.items()}
            r6 = recall_at(predictions, subset, 6)
            r20 = recall_at(predictions, subset, 20)
            score = 0.5 * r6 + 0.5 * r20
            # Hòa điểm (score, r20, r6 đều bằng) → -complexity lớn hơn nghĩa
            # là ÍT thành phần hơn → bản đơn giản thắng. Tên chỉ là nhãn cuối
            # cùng để deterministic.
            candidate = (score, r20, r6, -complexity(profile), name, profile)
            if best is None or candidate[:5] > best[:5]:
                best = candidate
        assert best is not None
        chosen[regime] = best[-1]
        report[regime] = {"profile": best[-2], "n": len(subset),
                          "score": best[0], "recall@6": best[2],
                          "recall@20": best[1],
                          "active_components": complexity(best[-1])}
    return chosen, report


# ---------------------------------------------------------------------------
# Teacher nâng cấp — encode text item bằng MiniLM (hoặc E5-small).
# Dùng thay/đặt cạnh semantic_matrix (TF-IDF-SVD) hiện tại: cùng shape
# (n_items, d), hàng 0 để 0. Chạy một lần, cache ra .npy.
# ---------------------------------------------------------------------------
def minilm_matrix(texts: Sequence[str], n_items: int,
                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                  batch_size: int = 256, device: str | None = None) -> np.ndarray:
    """texts[i] là văn bản của item i (chuỗi rỗng nếu không có metadata).

    Trả về (n_items, 384) float32, L2-normalised; item không text → vector 0
    (SemanticMemory sẽ bỏ qua chúng một cách tự nhiên).
    """
    from sentence_transformers import SentenceTransformer  # pip install sentence-transformers
    if device is None:
        try:
            import torch
            device = ("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
        except Exception:
            device = "cpu"
    model = SentenceTransformer(model_name, device=device)
    matrix = np.zeros((n_items, model.get_sentence_embedding_dimension()),
                      dtype=np.float32)
    has_text = [i for i, t in enumerate(texts) if i > 0 and t and t.strip()]
    if has_text:
        emb = model.encode([texts[i] for i in has_text],
                           batch_size=batch_size,
                           normalize_embeddings=True,
                           show_progress_bar=True)
        matrix[has_text] = np.asarray(emb, dtype=np.float32)
    return matrix


# ---------------------------------------------------------------------------
# Ghi chú thí nghiệm (đúng thứ tự nên chạy):
# 1. Diginetica_HID + Tmall: CEARFIndexV3(use_repeat=True, semantic=None)
#    → tune_profiles_v3 → so paired với v2. Kỳ vọng: gate chọn profile có
#    repeat; nếu không chọn → báo cáo như một rejection.
# 2. Video_Games + Baby: semantic=SemanticMemory(TFIDF_SVD) trước (cùng
#    teacher cũ, chỉ thêm đường memory) → tách được "giá trị của semantic
#    kNN" khỏi "giá trị của teacher mới".
# 3. Trục teacher: thay S bằng minilm_matrix(...) cho CẢ SemanticMemory lẫn
#    init PASGR, và chạy fairness NARM+MiniLM trên 2 dataset Amazon.
# 4. Mọi bảng mới đi kèm: cột val-utility vs test (gap), và dòng trong bảng
#    decision-stability cho các quyết định mới.
# ---------------------------------------------------------------------------
