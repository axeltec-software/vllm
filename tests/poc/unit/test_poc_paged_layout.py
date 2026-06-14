"""poc_paged_layout: slot_mapping/block_table from per-request block ids.

Foundation for dynamic (manager-paged) PoC KV: the attention metadata must work
with NON-contiguous block ids, and must stay byte-identical to the old contiguous
layout when blocks happen to be contiguous (the static-reserved-slot case).
"""
import math

from vllm.poc.mixed_decode import poc_paged_layout


def _contiguous(bases, blocks_per_seq):
    return [list(range(b, b + blocks_per_seq)) for b in bases]


def test_contiguous_matches_old_base_formula():
    # Old layout: slot = (base + t//block) * block + t%block. New must match.
    seq_len, block_size = 8, 4               # 2 blocks/seq
    bps = math.ceil(seq_len / block_size)
    bases = [0, 5]                            # contiguous runs [0,1] and [5,6]
    block_ids = _contiguous(bases, bps)
    slot, bt = poc_paged_layout(block_ids, seq_len, block_size)
    expected = []
    for base in bases:
        for t in range(seq_len):
            expected.append((base + t // block_size) * block_size + t % block_size)
    assert slot == expected
    assert bt == block_ids


def test_non_contiguous_blocks_paged_correctly():
    # Manager-paged: a request's blocks are scattered, not a run.
    seq_len, block_size = 8, 4               # 2 blocks/seq
    block_ids = [[9, 2]]                      # token 0-3 -> block 9, token 4-7 -> block 2
    slot, _ = poc_paged_layout(block_ids, seq_len, block_size)
    assert slot == [9*4+0, 9*4+1, 9*4+2, 9*4+3, 2*4+0, 2*4+1, 2*4+2, 2*4+3]


def test_block_table_is_block_ids():
    slot, bt = poc_paged_layout([[3, 7], [1, 4]], 8, 4)
    assert bt == [[3, 7], [1, 4]]
    assert len(slot) == 2 * 8                 # batch * seq_len


def test_block_ids_tp_broadcast_roundtrip():
    # Dynamic KV ships paged block_ids to non-driver TP ranks as a 2D tensor.
    # Rectangular list[list[int]] (all reqs share seq_len+max_tokens) must survive
    # the tensor encode/decode AND yield the identical paged layout on every rank.
    import torch
    block_ids = [[5, 9, 2, 7], [11, 0, 3, 8], [1, 6, 4, 10]]   # 3 reqs x 4 blocks
    decoded = torch.tensor(block_ids, dtype=torch.long).tolist()
    assert decoded == block_ids
    s_driver, _ = poc_paged_layout(block_ids, seq_len=8, block_size=2)
    s_rank, _ = poc_paged_layout(decoded, seq_len=8, block_size=2)
    assert s_driver == s_rank          # no rank divergence -> no KV corruption


def test_partial_last_block():
    # seq_len not a multiple of block_size: last block partially filled.
    seq_len, block_size = 5, 4               # ceil(5/4)=2 blocks
    block_ids = [[6, 11]]                     # tokens 0-3 -> 6, token 4 -> 11
    slot, _ = poc_paged_layout(block_ids, seq_len, block_size)
    assert slot == [24, 25, 26, 27, 44]      # 6*4+{0..3}, 11*4+0
