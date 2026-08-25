# SPDX-License-Identifier: Apache-2.0
"""Property and edge coverage for the batched prefill paths.

The batched reflection scatter and the memoized/tabulated router seed base are
pure optimizations: they must reproduce the per-row forms EXACTLY, for every
shape the scheduler can hand them. ``mixed_test_prefill_scatter`` pins the happy
path against verbatim references; this file adds the shapes that break naive
batching — masked-only batches, a single group, permuted mappings that reuse a
warm memo, cache eviction mid-round — plus a guard on the cost SHAPE, since a
correct-but-per-row implementation would pass every value assertion.
"""
import random

import pytest
import torch

from vllm.poc import native as native_mod
from vllm.poc.native import PoCNativeState
from vllm.poc.gpu_random import route_base_seed, _seed_from_string

HIDDEN, LAYERS = 8, 4
CPU_DEV = torch.device("cpu")
HASHES = [f"{i:02x}" * 32 for i in range(5)]


def _state(rows):
    st = PoCNativeState.__new__(PoCNativeState)
    st.hidden_size, st.num_layers = HIDDEN, LAYERS
    st.device = torch.device("cpu")
    st.vectors_t, st.vectors = PoCNativeState.alloc_vectors(
        LAYERS, rows, HIDDEN, torch.device("cpu"), torch.bfloat16)
    st._hash_cache, st._stack_cache, st._last_refl_key = {}, {}, None
    st._route_base = [torch.zeros(rows, dtype=torch.int64)
                      for _ in range(LAYERS)]
    st._seed_cache, st._base_key, st._last_route_key = {}, None, None
    st.route_step = torch.zeros(rows, dtype=torch.int64)
    return st


def _ref_scatter(st, hashes, nonces):
    """The per-row form, verbatim."""
    for buf in st.vectors:
        buf.zero_()
    for row, (bh, nz) in enumerate(zip(hashes, nonces)):
        if bh is None:
            continue
        vs = st._vectors_for(bh, nz)
        for i, buf in enumerate(st.vectors):
            buf[row].copy_(vs[i].to(buf.dtype))


def _ref_base(hashes, nonces, layer):
    """The per-row hashing form, verbatim."""
    return [(_seed_from_string(route_base_seed(bh, nz, layer))
             if bh is not None else 0) for bh, nz in zip(hashes, nonces)]


def _mapping(rows, masked_p=0.15, seed=0):
    rnd = random.Random(seed)
    hashes, nonces = [], []
    for _ in range(rows):
        if rnd.random() < masked_p:
            hashes.append(None); nonces.append(None); continue
        hashes.append(rnd.choice(HASHES))
        nonces.append(rnd.choice([None, 0, 1, 7, 4242, rnd.randint(0, 999)]))
    return hashes, nonces


def _assert_equivalent(rows, hashes, nonces):
    ref, got = _state(rows), _state(rows)
    _ref_scatter(ref, hashes, nonces)
    got.set_row_block_hashes(list(hashes), list(nonces))
    for i in range(LAYERS):
        assert torch.equal(got.vectors[i], ref.vectors[i]), f"vectors[{i}]"
    got.set_routing(list(hashes), list(nonces), [0] * rows)
    for i in range(LAYERS):
        assert got._route_base[i][:rows].tolist() == _ref_base(hashes, nonces, i), \
            f"route_base[{i}]"


@pytest.mark.parametrize("rows", [1, 2, 3, 17, 512, 2048])
def test_equivalent_for_random_mappings(rows):
    """Prefill rows are TOKENS: thousands per chunk over a few dozen nonces, in
    scheduler order. Batching must not depend on that order."""
    _assert_equivalent(rows, *_mapping(rows, seed=rows))


def test_all_rows_masked():
    """A chunk can be entirely chat rows: nothing is written, nothing crashes."""
    rows = 12
    _assert_equivalent(rows, [None] * rows, [None] * rows)


def test_single_group_whole_batch():
    """One nonce spanning every row — the degenerate group, where a per-group
    write covers the entire buffer at once."""
    rows = 64
    _assert_equivalent(rows, [HASHES[0]] * rows, [7] * rows)


def test_same_hash_many_nonces_are_distinct_groups():
    """Per-nonce reflection: one block_hash, several nonces. Grouping on the hash
    alone would give every row the first nonce's vector."""
    rows = 24
    hashes = [HASHES[1]] * rows
    nonces = [i % 6 for i in range(rows)]
    _assert_equivalent(rows, hashes, nonces)
    st = _state(rows)
    st.set_row_block_hashes(list(hashes), list(nonces))
    distinct = {tuple(st.vectors[0][r].tolist()) for r in range(rows)}
    assert len(distinct) == 6, "per-nonce vectors collapsed onto one another"


def test_permuted_mapping_reuses_memo_without_stale_values():
    """The memo is keyed by (hash, nonce, layer), the buffer by row. A second
    call with the SAME nonces in a DIFFERENT order must re-expand, not re-serve
    the previous row order."""
    rows = 40
    hashes, nonces = _mapping(rows, seed=5)
    st = _state(rows)
    st.set_routing(list(hashes), list(nonces), [0] * rows)
    warm = len(st._seed_cache)

    order = list(range(rows))[::-1]
    h2 = [hashes[i] for i in order]
    n2 = [nonces[i] for i in order]
    st.set_routing(list(h2), list(n2), [0] * rows)

    for i in range(LAYERS):
        assert st._route_base[i][:rows].tolist() == _ref_base(h2, n2, i)
    assert len(st._seed_cache) == warm, "permutation should not add memo entries"


def test_values_survive_cache_eviction():
    """The memo is cleared wholesale past its cap. Correctness must not depend on
    the cache being warm — only speed may."""
    rows = 24
    hashes, nonces = _mapping(rows, seed=9)
    st = _state(rows)
    st._seed_cache = {("filler", i, 0): i for i in range(262145)}  # over the cap
    st.set_routing(list(hashes), list(nonces), [0] * rows)
    assert len(st._seed_cache) < 262145, "cap did not trigger"
    for i in range(LAYERS):
        assert st._route_base[i][:rows].tolist() == _ref_base(hashes, nonces, i)


def test_unchanged_step_mapping_is_skipped():
    """_last_route_key must suppress the refresh when nothing changed, and must
    NOT suppress it when a step advances."""
    rows = 8
    hashes, nonces = _mapping(rows, seed=3)
    st = _state(rows)
    st.set_routing(list(hashes), list(nonces), [0] * rows)
    st.route_step.fill_(999)                    # would be overwritten by a refresh
    st.set_routing(list(hashes), list(nonces), [0] * rows)
    assert st.route_step[0].item() == 999, "refresh re-ran on an unchanged key"
    st.set_routing(list(hashes), list(nonces), [1] * rows)
    assert st.route_step[0].item() != 999, "refresh skipped after a step change"


def test_upload_count_tracks_groups_not_rows(monkeypatch):
    """Cost-shape guard. Every value assertion above also passes for a per-row
    implementation — this is the one that fails if the batching is undone: host
    uploads must scale with unique (hash, nonce) groups, not with row count."""
    calls = []
    real = native_mod.pinned_to_device
    monkeypatch.setattr(native_mod, "pinned_to_device",
                        lambda v, d, dev: calls.append(len(v)) or real(v, d, dev))

    rows = 600
    hashes = [HASHES[i % 3] for i in range(rows)]      # exactly 3 groups
    nonces = [None] * rows
    st = _state(rows)
    st.set_row_block_hashes(list(hashes), list(nonces))

    assert len(calls) == 3, (
        f"{len(calls)} uploads for 3 groups over {rows} rows — the scatter is "
        "per row again")


# --------------------------------------------------- contiguous buffer (items 6/7)
def test_per_layer_views_alias_the_contiguous_buffer():
    """The layer wrappers capture ``state.vectors[i]`` at ATTACH time. If those stop
    being views into the one buffer, the scatter writes into tensors nothing reads —
    silent-correctness, no crash."""
    st = _state(4)
    st.vectors_t[2, 1, :] = 5
    assert st.vectors[2][1].eq(5).all(), "layer view detached from the buffer"
    st.vectors[0][3] = 7
    assert st.vectors_t[0, 3].eq(7).all(), "write through the view is not visible"


def test_buffer_is_zeroed_in_one_kernel(monkeypatch):
    """Was one zero_ per layer (62 kernel launches per prefill chunk on M2)."""
    calls = []
    real = torch.Tensor.zero_
    monkeypatch.setattr(torch.Tensor, "zero_",
                        lambda self: calls.append(tuple(self.shape)) or real(self))
    st = _state(8)
    st.set_row_block_hashes([HASHES[0]] * 8, [None] * 8)
    assert len(calls) == 1, f"{len(calls)} zero kernels — one per layer again?"
    assert calls[0][0] == LAYERS, "zeroed something other than the whole buffer"


def test_one_write_per_group_regardless_of_layer_count(monkeypatch):
    """Was n_groups x n_layers indexed writes. The layer count must not appear in
    the op count at all — that factor is 62 on the production model."""
    writes = []
    real = torch.Tensor.__setitem__
    monkeypatch.setattr(torch.Tensor, "__setitem__",
                        lambda s, k, v: writes.append(1) or real(s, k, v))
    rows = 90
    hashes = [HASHES[i % 3] for i in range(rows)]      # exactly 3 groups
    st = _state(rows)
    st.set_row_block_hashes(hashes, [None] * rows)
    assert len(writes) == 3, (
        f"{len(writes)} writes for 3 groups over {LAYERS} layers — the scatter is "
        "per layer again")


def test_stack_cache_is_dropped_with_the_vector_cache():
    """The stacked form is derived from the per-layer vectors; if the two caches
    can diverge, a cleared hash cache would serve stale stacks forever."""
    st = _state(4)
    st.set_row_block_hashes([HASHES[0]] * 4, [None] * 4)
    assert st._stack_cache, "nothing cached"
    st._hash_cache.clear(); st._stack_cache.clear()
    st._last_refl_key = None
    st.set_row_block_hashes([HASHES[0]] * 4, [None] * 4)
    ref = _state(4)
    _ref_scatter(ref, [HASHES[0]] * 4, [None] * 4)
    assert torch.equal(st.vectors_t, ref.vectors_t)


# ------------------------------------------------------- TP replication contract
# Real TP needs multiple GPUs, but the property TP depends on is testable without
# NCCL: these buffers are REPLICATED across ranks, not sharded, so every rank must
# independently compute bit-identical state from the same mapping. If any value
# picked up a rank-, device- or iteration-order dependence, ranks would diverge and
# the forward would produce per-rank artifacts — which _assert_replicated_across_tp
# only catches when someone remembers to set VLLM_POC_DEBUG_TP=1 on a TP box.
def test_reflection_state_is_rank_independent():
    rows = 96
    hashes, nonces = _mapping(rows, seed=21)
    rank0, rank1 = _state(rows), _state(rows)          # two independent "ranks"
    rank0.set_row_block_hashes(list(hashes), list(nonces))
    rank1.set_row_block_hashes(list(hashes), list(nonces))
    assert torch.equal(rank0.vectors_t, rank1.vectors_t), \
        "two ranks computed different reflection buffers"


def test_router_base_is_rank_independent():
    rows = 96
    hashes, nonces = _mapping(rows, seed=22)
    rank0, rank1 = _state(rows), _state(rows)
    for st in (rank0, rank1):
        st.set_routing(list(hashes), list(nonces), [3] * rows)
    for i in range(LAYERS):
        assert torch.equal(rank0._route_base[i], rank1._route_base[i]), \
            f"two ranks computed different router bases at layer {i}"
    assert torch.equal(rank0.route_step, rank1.route_step)


def test_rank_independence_holds_when_groups_are_discovered_in_another_order():
    """Ranks see the same rows, but nothing guarantees a dict iterates in the same
    order if the mapping were built differently. Values must not depend on it."""
    rows = 60
    hashes, nonces = _mapping(rows, seed=23)
    rank0 = _state(rows)
    rank0.set_row_block_hashes(list(hashes), list(nonces))

    order = list(range(rows))[::-1]                    # same rows, reversed order
    rank1 = _state(rows)
    rank1.set_row_block_hashes([hashes[i] for i in order], [nonces[i] for i in order])
    for row, src in enumerate(order):
        assert torch.equal(rank1.vectors_t[:, row], rank0.vectors_t[:, src]), \
            f"row {row} depends on discovery order"


# --------------------------------------------------- production-scale layer count
def test_layout_holds_at_production_layer_count():
    """The layout math is shape-generic, but the production model has 62 layers and
    thousands of prefill rows per chunk — the case the fix exists for. Values are
    checked against the per-row reference, on CPU."""
    layers, rows = 62, 1024
    st = PoCNativeState.__new__(PoCNativeState)
    st.hidden_size, st.num_layers, st.device = HIDDEN, layers, CPU_DEV
    st.vectors_t, st.vectors = PoCNativeState.alloc_vectors(
        layers, rows, HIDDEN, CPU_DEV, torch.bfloat16)
    st._hash_cache, st._stack_cache, st._last_refl_key = {}, {}, None

    hashes, nonces = _mapping(rows, seed=62)
    st.set_row_block_hashes(list(hashes), list(nonces))

    for row, (bh, nz) in enumerate(zip(hashes, nonces)):
        if bh is None:
            assert not st.vectors_t[:, row].any(), f"masked row {row} written"
            continue
        vs = st._vectors_for(bh, nz)
        for i in range(layers):
            assert torch.equal(st.vectors_t[i, row], vs[i].to(torch.bfloat16)), \
                f"layer {i} row {row}"
