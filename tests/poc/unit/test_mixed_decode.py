"""Unit tests for step-driven mixed decode-PoC bookkeeping (Phase 2).

Pure logic, no GPU. Verifies the reserved-slot layout tiles the reservation
exactly and the slot pool allocates/frees correctly.
"""
from vllm.poc.mixed_decode import (
    poc_per_slot_blocks,
    poc_slot_block_ids,
    PoCMixedDecodeManager,
)
from vllm.poc.reservation import poc_blocks_needed


def test_per_slot_blocks_matches_reservation_per_seq():
    # reservation per-seq sizing == per-slot blocks
    assert poc_per_slot_blocks(256, 256, 16) == 33  # ceil(512/16)+1
    assert poc_per_slot_blocks(256, 0, 16) == 17    # ceil(256/16)+1
    assert poc_per_slot_blocks(100, 50, 16) == 11   # ceil(150/16)=10, +1


def test_slots_tile_the_reservation_exactly():
    seq_len, max_tokens, block, batch = 256, 256, 16, 32
    per_slot = poc_per_slot_blocks(seq_len, max_tokens, block)
    # union of all slots' blocks == [0, poc_reserved_blocks), no gaps/overlap
    all_blocks = []
    for slot in range(batch):
        all_blocks.extend(poc_slot_block_ids(slot, seq_len, max_tokens, block))
    reserved = poc_blocks_needed(batch, seq_len, max_tokens, block)
    assert all_blocks == list(range(reserved))
    assert len(all_blocks) == batch * per_slot == reserved


def test_slot_block_ids_contiguous_and_offset():
    ids0 = poc_slot_block_ids(0, 256, 256, 16)
    ids1 = poc_slot_block_ids(1, 256, 256, 16)
    assert ids0 == list(range(0, 33))
    assert ids1 == list(range(33, 66))


def test_manager_allocate_free_reuse():
    mgr = PoCMixedDecodeManager(poc_max_batch_size=2)
    a = mgr.allocate("r0", nonce=10, seq_len=256, max_tokens=256)
    b = mgr.allocate("r1", nonce=11, seq_len=256, max_tokens=256)
    assert a.slot == 0 and b.slot == 1
    # pool exhausted -> None (scheduler caps to poc_max_batch_size, defensive)
    assert mgr.allocate("r2", nonce=12, seq_len=256, max_tokens=256) is None
    # idempotent for same req
    assert mgr.allocate("r0", nonce=10, seq_len=256, max_tokens=256) is a
    # free returns the slot to the pool
    mgr.free("r0")
    c = mgr.allocate("r2", nonce=12, seq_len=256, max_tokens=256)
    assert c.slot == 0
    assert mgr.get("r1") is b and mgr.get("r0") is None


def test_decode_state_defaults():
    mgr = PoCMixedDecodeManager(poc_max_batch_size=1)
    st = mgr.allocate("r", nonce=7, seq_len=256, max_tokens=256)
    assert st.step == 0 and st.prev_k == -1 and st.k_points_steps == []
    assert st.nonce == 7 and st.max_tokens == 256


def test_apply_poc_kv_skip_pads_prefill_only_keeps_decode():
    import torch
    from types import SimpleNamespace
    from vllm.poc.mixed_decode import (
        apply_poc_kv_skip, PoCDecodeState,
    )
    n = 6
    layer = SimpleNamespace(slot_mapping=torch.arange(n, dtype=torch.long))
    attn = {"layer0": layer}
    # positions 0..2 = prefill-only PoC (decode_state None) -> must be PADded;
    # position 3 = decode-PoC (decode_state set) -> must keep its real slot;
    # positions 4,5 = chat (not in poc_metadata) -> untouched.
    poc_metadata = [
        {"start_idx": 0, "length": 3, "decode_state": None},
        {"start_idx": 3, "length": 1,
         "decode_state": PoCDecodeState(nonce=1, slot=0, seq_len=256, max_tokens=8)},
    ]
    apply_poc_kv_skip(attn, poc_metadata, n, torch.device("cpu"))
    sm = layer.slot_mapping.tolist()
    assert sm[0:3] == [-1, -1, -1]      # prefill-only PoC -> PAD
    assert sm[3] == 3                   # decode-PoC -> kept (real reserved slot)
    assert sm[4:6] == [4, 5]            # chat -> untouched


def test_apply_poc_kv_skip_noop_when_no_prefill_only():
    import torch
    from types import SimpleNamespace
    from vllm.poc.mixed_decode import apply_poc_kv_skip, PoCDecodeState
    layer = SimpleNamespace(slot_mapping=torch.arange(2, dtype=torch.long))
    poc_metadata = [
        {"start_idx": 0, "length": 1,
         "decode_state": PoCDecodeState(nonce=1, slot=0, seq_len=256, max_tokens=8)},
    ]
    apply_poc_kv_skip({"l": layer}, poc_metadata, 2, torch.device("cpu"))
    assert layer.slot_mapping.tolist() == [0, 1]  # nothing PADded
