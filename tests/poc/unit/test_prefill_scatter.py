# SPDX-License-Identifier: Apache-2.0
"""Reflection scatter is batched per (block_hash, nonce) group; router seed
bases are memoized per (block_hash, nonce, layer).

Perf regression guard for the two prefill hot spots. The decode-side keys
(``_last_refl_key`` / ``_base_key``) hide both costs for a stable batch, but in
PREFILL the rows are tokens: the mapping changes on every chunk, so

* ``set_row_block_hashes`` re-ran a per-(row, layer) ``copy_`` scatter —
  num_layers x B separate device copies of one hidden row each — on every
  chunk (nsys: prefill forwards at ~23% GPU busy from this alone), and
* the ``_route_base`` rebuild hashed num_layers x n_rows host strings per
  chunk, though the value depends only on (block_hash, nonce, layer).

The batched/memoized forms must produce the SAME buffers bit-for-bit — same
values, same dtype cast, every (row, layer) cell written exactly once. Pinned
against the reference (pre-optimization) implementations with mixed groups:
shared hashes, per-nonce reflection rows, and masked (None) rows.
"""
import torch

from vllm.poc.native import PoCNativeState
from vllm.poc.gpu_random import route_base_seed, _seed_from_string

HIDDEN = 16
LAYERS = 3
ROWS = 12

BH_A = "aa" * 32
BH_B = "bb" * 32


def _state():
    st = PoCNativeState.__new__(PoCNativeState)
    st.hidden_size = HIDDEN
    st.num_layers = LAYERS
    st.device = torch.device("cpu")
    st.vectors_t, st.vectors = PoCNativeState.alloc_vectors(
        LAYERS, ROWS, HIDDEN, torch.device("cpu"), torch.bfloat16)
    st._hash_cache, st._stack_cache = {}, {}
    st._last_refl_key = None
    st._route_base = [torch.zeros(ROWS, dtype=torch.int64)
                      for _ in range(LAYERS)]
    st._seed_cache = {}
    st._base_key = None
    st._last_route_key = None
    st.route_step = torch.zeros(ROWS, dtype=torch.int64)
    return st


# Rows exercise every grouping case: repeated (bh, None), per-nonce seeds
# (bh, nonce), a hash appearing with several nonces, and masked rows.
ROW_HASHES = [BH_A, BH_A, BH_B, None, BH_A, BH_B, BH_B, None,
              BH_A, BH_B, BH_A, BH_B]
ROW_NONCES = [None, None, 7, None, 3, 7, None, None, 3, 9, None, 7]


def _reference_scatter(st):
    """The pre-optimization per-row loop, verbatim."""
    for buf in st.vectors:
        buf.zero_()
    for row, (bh, nz) in enumerate(zip(ROW_HASHES, ROW_NONCES)):
        if bh is None:
            continue
        vs = st._vectors_for(bh, nz)
        for i, buf in enumerate(st.vectors):
            buf[row].copy_(vs[i].to(buf.dtype))


def test_batched_reflection_scatter_bit_identical():
    ref = _state()
    _reference_scatter(ref)

    got = _state()
    got.set_row_block_hashes(list(ROW_HASHES), list(ROW_NONCES))

    for i in range(LAYERS):
        assert torch.equal(got.vectors[i], ref.vectors[i]), f"layer {i} differs"
    # masked rows stay zero
    for row, bh in enumerate(ROW_HASHES):
        if bh is None:
            assert not got.vectors[0][row].any()


def test_scatter_skips_unchanged_mapping():
    st = _state()
    st.set_row_block_hashes(list(ROW_HASHES), list(ROW_NONCES))
    snap = [buf.clone() for buf in st.vectors]
    st.vectors[0][0] += 1  # would be zeroed if the scatter re-ran
    st.set_row_block_hashes(list(ROW_HASHES), list(ROW_NONCES))
    assert not torch.equal(st.vectors[0], snap[0]), "scatter re-ran on same key"


def test_memoized_seed_base_bit_identical():
    st = _state()
    st.set_routing(list(ROW_HASHES), list(ROW_NONCES),
                   [0] * len(ROW_HASHES))

    for i in range(LAYERS):
        expect = [(_seed_from_string(route_base_seed(bh, nz, i))
                   if bh is not None else 0)
                  for bh, nz in zip(ROW_HASHES, ROW_NONCES)]
        assert st._route_base[i][:len(expect)].tolist() == expect, \
            f"layer {i} base differs from direct hashing"
    # ONE memo entry per unique (bh, nonce) — holding the per-layer bases — so the
    # cap counts nonces and the table build does n_unique lookups, not n_unique x n_layers
    uniq = {(bh, nz) for bh, nz in zip(ROW_HASHES, ROW_NONCES)
            if bh is not None}
    cached_nonces = {(bh, nz) for bh, scope in st._seed_cache.items() for nz in scope}
    assert cached_nonces == uniq
    assert all(len(v) == LAYERS
               for scope in st._seed_cache.values() for v in scope.values())
