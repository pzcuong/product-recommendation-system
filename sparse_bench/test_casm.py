import numpy as np
import pytest

from casm import (
    CASMemory,
    enumerate_cooccurrence_pairs,
    load_or_train_casm,
    sample_cooccurrence_pairs,
    teacher_has_vector,
    train_alignment_head,
)

torch = pytest.importorskip("torch")


def toy_teacher(n_items=30, dim=16, seed=0, zero_rows=(5, 6)):
    rng = np.random.default_rng(seed)
    teacher = rng.normal(size=(n_items, dim)).astype(np.float32)
    teacher[0] = 0.0
    for row in zero_rows:
        teacher[row] = 0.0
    return teacher


def toy_sessions(n_items=30, n_sessions=120, seed=1):
    # Two disjoint co-occurrence blocks so alignment has structure to learn:
    # items 1..14 co-occur, items 15..29 co-occur.
    rng = np.random.default_rng(seed)
    sessions = {}
    for row in range(n_sessions):
        block = (list(range(1, 15)) if row % 2 == 0
                 else list(range(15, n_items)))
        sessions[f"s{row}"] = list(rng.choice(block, size=6, replace=False))
    return sessions


def test_cooccurrence_pairs_window_and_weights():
    teacher = toy_teacher()
    has_vector = teacher_has_vector(teacher)
    sessions = {"a": [1, 2, 3, 4, 5, 6, 7]}   # 5, 6 have zero vectors
    src, tgt, w = enumerate_cooccurrence_pairs(sessions, 30, has_vector,
                                               window=4)
    # no pair may involve zero-vector items
    assert not set(src.tolist()) & {5, 6}
    assert not set(tgt.tolist()) & {5, 6}
    # adjacent pair (1, 2) has weight 1.0; distance-2 pair (1, 3) has 0.5
    pairs = {(int(s), int(t)): float(x) for s, t, x in zip(src, tgt, w)}
    assert pairs[(1, 2)] == 1.0
    assert pairs[(1, 3)] == 0.5
    # window: (1, 7) would be distance 6 — but 7 is within window of 3,4;
    # distance from 1 to 7 is 6 > 4 so the pair must be absent
    assert (1, 7) not in pairs
    # ordered pairs only within window
    assert all(0 < s < 30 and 0 < t < 30 for s, t in pairs)


def test_pair_sampler_deterministic():
    teacher = toy_teacher()
    sessions = toy_sessions()
    has_vector = teacher_has_vector(teacher)
    a = sample_cooccurrence_pairs(sessions, 30, has_vector, n_pairs=1000,
                                  seed=9)
    b = sample_cooccurrence_pairs(sessions, 30, has_vector, n_pairs=1000,
                                  seed=9)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_alignment_output_shape_dtype_norms():
    teacher = toy_teacher()
    sessions = toy_sessions()
    aligned, info = train_alignment_head(
        sessions, teacher, d_out=8, epochs=2, batch_size=64, n_pairs=2000,
        seed=3, device="cpu")
    assert aligned.shape == (30, 8)
    assert aligned.dtype == np.float32
    norms = np.linalg.norm(aligned, axis=1)
    has_vector = teacher_has_vector(teacher)
    assert np.allclose(norms[has_vector], 1.0, atol=1e-5)
    assert np.allclose(norms[~has_vector], 0.0)
    assert len(info["loss_per_epoch"]) == 2


def test_alignment_deterministic_given_seed():
    teacher = toy_teacher()
    sessions = toy_sessions()
    kwargs = dict(d_out=8, epochs=1, batch_size=64, n_pairs=1000, seed=11,
                  device="cpu")
    a, _ = train_alignment_head(sessions, teacher, **kwargs)
    b, _ = train_alignment_head(sessions, teacher, **kwargs)
    assert np.array_equal(a, b)


def test_alignment_loss_decreases_on_toy_data():
    teacher = toy_teacher(n_items=40, dim=12, zero_rows=())
    sessions = toy_sessions(n_items=40, n_sessions=200)
    _, info = train_alignment_head(
        sessions, teacher, d_out=8, epochs=6, batch_size=128, n_pairs=8000,
        seed=5, device="cpu")
    losses = info["loss_per_epoch"]
    assert losses[-1] < losses[0]


def test_raw_and_aligned_rankings_differ():
    teacher = toy_teacher()
    sessions = toy_sessions()
    aligned, _ = train_alignment_head(
        sessions, teacher, d_out=8, epochs=4, batch_size=64, n_pairs=4000,
        seed=7, device="cpu")
    raw_memory = CASMemory.from_teacher(teacher, topn=10)
    casm_memory = CASMemory(aligned, topn=10)
    contexts = [[1, 2, 3], [15, 16], [4, 8, 12]]
    blocked = [set(c) for c in contexts]
    raw_lists = raw_memory.rankings_batch(contexts, blocked)
    casm_lists = casm_memory.rankings_batch(contexts, blocked)
    assert raw_lists != casm_lists


def test_batched_equals_single_query():
    rng = np.random.default_rng(13)
    vectors = rng.normal(size=(50, 12)).astype(np.float32)
    vectors[0] = 0.0
    vectors[7] = 0.0
    memory = CASMemory(vectors, topn=15, chunk_size=2)  # force chunking
    contexts = [list(rng.integers(1, 50, size=rng.integers(1, 12)))
                for _ in range(9)]
    contexts.append([])            # empty context → null contribution
    contexts.append([7])           # only a zero-vector item → null
    blocked = [set(ctx) for ctx in contexts]
    batch = memory.rankings_batch(contexts, blocked)
    single = [memory.ranking(ctx, blk) for ctx, blk in zip(contexts, blocked)]
    assert batch == single
    assert batch[-2] == [] and batch[-1] == []


def test_blocked_items_and_padding_never_returned():
    rng = np.random.default_rng(17)
    vectors = rng.normal(size=(40, 8)).astype(np.float32)
    vectors[0] = 0.0
    vectors[3] = 0.0                      # no text vector
    memory = CASMemory(vectors, topn=40)
    context = [1, 2]
    blocked = {1, 2, 10, 11}
    output = memory.ranking(context, blocked)
    assert output
    assert not set(output) & blocked
    assert 0 not in output
    assert 3 not in output
    assert len(output) <= memory.topn


def test_topn_cap():
    rng = np.random.default_rng(19)
    vectors = rng.normal(size=(200, 8)).astype(np.float32)
    vectors[0] = 0.0
    memory = CASMemory(vectors, topn=120)
    output = memory.ranking([5, 9], set())
    assert len(output) == 120
    assert len(set(output)) == 120


def test_cache_roundtrip(tmp_path):
    teacher = toy_teacher()
    sessions = toy_sessions()
    kwargs = dict(d_out=8, epochs=1, batch_size=64, n_pairs=500, seed=23,
                  device="cpu")
    first = load_or_train_casm(tmp_path, "toy", sessions, teacher, **kwargs)
    second = load_or_train_casm(tmp_path, "toy", sessions, teacher, **kwargs)
    assert np.array_equal(first, second)
    manifests = list(tmp_path.glob("toy_casm_*.json"))
    matrices = list(tmp_path.glob("toy_casm_*.npy"))
    assert len(manifests) == 1 and len(matrices) == 1
    # different config → different fingerprint → new artifact
    load_or_train_casm(tmp_path, "toy", sessions, teacher,
                       **{**kwargs, "seed": 24})
    assert len(list(tmp_path.glob("toy_casm_*.npy"))) == 2


def test_casm_slot_joins_component_rankings():
    import cearf
    from cearf_v3_ext import CEARFIndexV3
    rng = np.random.default_rng(29)
    vectors = rng.normal(size=(10, 6)).astype(np.float32)
    vectors[0] = 0.0
    sessions = {"a": [1, 2, 3], "b": [2, 3, 4], "c": [5, 6, 7]}
    index = CEARFIndexV3(sessions, 10, cearf.CEARFConfig(),
                         casm=CASMemory(vectors, topn=5))
    components = index.component_rankings([1, 2])
    assert len(components) == 6
    casm_list = components[5]
    assert casm_list and not set(casm_list) & {1, 2}
    # batched variant returns the same tuples
    batched = index.component_rankings_batch([[1, 2], [5]])
    assert batched[0] == index.component_rankings([1, 2])
    assert batched[1] == index.component_rankings([5])
