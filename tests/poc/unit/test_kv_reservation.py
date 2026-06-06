"""Unit tests for the PoC KV-cache block reservation.

Covers the consolidation's reservation port:
  * the sizing math in vllm/poc/reservation.py, and
  * the BlockPool carve that pops blocks [0, poc_reserved_blocks) out of the
    chat free pool so PoC and chat KV never collide.

The math tests need only the standard library + reservation.py. The BlockPool
tests need the full vllm import (run in the build/CI env).
"""
import math

import pytest

from vllm.poc.reservation import poc_blocks_needed, poc_reserved_blocks


class _FakeCacheConfig:
    """Minimal duck-typed stand-in for CacheConfig."""

    def __init__(self, batch, seq_len, max_tokens):
        self.poc_max_batch_size = batch
        self.poc_seq_len = seq_len
        self.poc_max_tokens = max_tokens


class TestReservationMath:
    def test_formula_matches_physical_layout(self):
        # batch * (ceil((seq_len + max_tokens) / block) + 1)
        assert poc_blocks_needed(1, 256, 256, 16) == 1 * (math.ceil(512 / 16) + 1)
        assert poc_blocks_needed(1, 256, 256, 16) == 33
        assert poc_blocks_needed(32, 256, 256, 16) == 32 * (32 + 1) == 1056

    def test_prefill_only_max_tokens_zero(self):
        # max_tokens=0 -> ceil(seq/block)+1 blocks per sequence
        assert poc_blocks_needed(4, 256, 0, 16) == 4 * (16 + 1) == 68

    def test_partial_block_rounds_up(self):
        # ceil(1/16) == 1, plus the +1 decode slack
        assert poc_blocks_needed(1, 1, 0, 16) == 2
        # 257 tokens with block 16 -> ceil(257/16)=17, +1 = 18
        assert poc_blocks_needed(1, 257, 0, 16) == 18

    def test_scales_linearly_with_batch(self):
        one = poc_blocks_needed(1, 256, 256, 16)
        assert poc_blocks_needed(10, 256, 256, 16) == 10 * one

    def test_reserved_blocks_reads_config(self):
        cfg = _FakeCacheConfig(batch=8, seq_len=128, max_tokens=64)
        assert poc_reserved_blocks(cfg, 16) == poc_blocks_needed(8, 128, 64, 16)


# ---------------------------------------------------------------------------
# BlockPool carve — requires the full vllm import (build/CI env).
# ---------------------------------------------------------------------------
def _make_block_pool(num_gpu_blocks, poc_reserved, block_size=16):
    from vllm.v1.core.block_pool import BlockPool
    return BlockPool(
        num_gpu_blocks,
        False,            # enable_caching
        block_size,       # hash_block_size
        poc_reserved_blocks=poc_reserved,
    )


class TestBlockPoolCarve:
    def test_reserves_low_block_ids(self):
        total, reserved = 64, 7
        pool = _make_block_pool(total, reserved)
        # reserved ids are exactly [0, reserved): block 0 is the null block,
        # blocks 1..reserved-1 are popped from the free queue.
        assert sorted(pool.poc_reserved_block_ids) == list(range(reserved))

    def test_chat_free_pool_excludes_reserved(self):
        total, reserved = 64, 7
        pool = _make_block_pool(total, reserved)
        drained = []
        q = pool.free_block_queue
        while q.num_free_blocks > 0:
            drained.append(q.popleft().block_id)
        # chat never sees a reserved (or the null) block ...
        assert set(range(reserved)).isdisjoint(drained)
        # ... and gets exactly the remaining blocks.
        assert len(drained) == total - reserved

    def test_zero_reservation_only_removes_null(self):
        total = 32
        pool = _make_block_pool(total, 0)
        # With no PoC reservation, only the null block (id 0) is held back.
        assert pool.poc_reserved_block_ids == [0]
        assert pool.free_block_queue.num_free_blocks == total - 1

    def test_default_is_backward_compatible(self):
        # Default poc_reserved_blocks=0 path leaves the normal pool intact.
        from vllm.v1.core.block_pool import BlockPool
        pool = BlockPool(16, False, 16)
        assert pool.poc_reserved_block_ids == [0]
        assert pool.free_block_queue.num_free_blocks == 15
