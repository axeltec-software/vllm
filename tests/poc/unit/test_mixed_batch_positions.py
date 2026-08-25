# SPDX-License-Identifier: Apache-2.0
"""Decode-row positions/mask are written in ONE batched op, not per row.

Perf regression guard: the builder used to assign ``unified_positions[offset]``
and ``poc_position_mask[offset]`` inside the per-request loop. Each scalar
assignment into a device tensor is its own H2D copy (~25 us/row/step), so the
cost grew linearly with the nonce batch and dominated the step at batch 1024
(36.5 ms/step of the 63 ms step; artifacts unchanged, pure host overhead).

The batched form must place the SAME values at the SAME offsets — this test
pins that mapping with several concurrent nonces (never trust one row: the
scheduler batches them).
"""
from types import SimpleNamespace

import pytest
import torch

from vllm.poc.mixed_decode import build_unified_mixed_batch_inputs

HIDDEN = 8


class _Native:
    """Stands in for the attached native state: records the decode chain and
    keeps the builder on the in-graph embedding path."""

    def __init__(self):
        self.embed_base = object()
        self.chain = None

    def set_decode_chain(self, **kw):
        self.chain = kw


def _params(nonce, seq_len):
    return SimpleNamespace(
        nonce=nonce,
        seq_len=seq_len,
        public_key="ca" * 32,
        block_hash="de" * 32,
    )


def _state():
    return SimpleNamespace(
        base_seeds=torch.zeros(1, dtype=torch.int64),
        prev_k_t=torch.zeros(1, dtype=torch.int64),
    )


def _runner(req_ids, native):
    mgr = {rid: _state() for rid in req_ids}
    return SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: HIDDEN),
        dtype=torch.float32,
        device=torch.device("cpu"),
        _poc_native=native,
        _poc_mixed_decode_mgr=SimpleNamespace(get=mgr.get),
    )


def _decode_batch(computed_tokens, seq_len=64):
    """One decode row per nonce, in scheduler order."""
    req_ids = [f"poc-{i}" for i in range(len(computed_tokens))]
    native = _Native()
    runner = _runner(req_ids, native)
    sched = SimpleNamespace(
        num_scheduled_tokens={rid: 1 for rid in req_ids},
    )
    views = {
        rid: SimpleNamespace(
            poc_params=_params(nonce=i, seq_len=seq_len),
            num_computed_tokens=ct,
        )
        for i, (rid, ct) in enumerate(zip(req_ids, computed_tokens))
    }
    embeds, positions, mask, meta = build_unified_mixed_batch_inputs(
        runner, sched, None, None,
        torch.zeros(len(req_ids), dtype=torch.long),
        set(req_ids), len(req_ids), (len(req_ids), req_ids), views,
    )
    return native, positions, mask, meta


@pytest.mark.parametrize("n_rows", [2, 5, 32])
def test_every_decode_row_gets_its_own_position(n_rows):
    seq_len = 64
    # distinct per row: a batched write that lost the offset mapping would
    # smear one row's position across the batch and still be "shaped" right
    computed = [seq_len + i for i in range(n_rows)]
    _, positions, mask, meta = _decode_batch(computed)

    assert positions.tolist() == computed
    assert mask.tolist() == [True] * n_rows
    assert [m["decode_step"] for m in meta] == [i + 1 for i in range(n_rows)]


def test_decode_chain_offsets_match_row_order():
    seq_len = 64
    computed = [seq_len + 3, seq_len + 1, seq_len + 7]
    native, positions, _, meta = _decode_batch(computed)

    assert native.chain is not None, "decode chain must be published once"
    assert native.chain["offs"].tolist() == [m["start_idx"] for m in meta]
    assert native.chain["step"].tolist() == [1 + 3, 1 + 1, 1 + 7]
    assert positions.tolist() == computed


def test_no_decode_rows_leaves_mask_clear():
    """A step with no PoC decode rows must not touch positions/mask at all."""
    native = _Native()
    runner = _runner([], native)
    sched = SimpleNamespace(num_scheduled_tokens={"chat-0": 2})
    positions_in = torch.tensor([5, 6], dtype=torch.long)
    _, positions, mask, meta = build_unified_mixed_batch_inputs(
        runner, sched, None, None, positions_in,
        set(), 2, (1, ["chat-0"]), {},
    )
    assert mask.tolist() == [False, False]
    assert positions.tolist() == [5, 6]
    assert meta == []


def test_per_row_device_writes_do_not_scale_with_batch(monkeypatch):
    """The guard that pins the fix itself: the number of element-wise tensor
    writes must be independent of the nonce batch. Per-row ``t[i] = v`` is a
    separate H2D copy each; the batched index_copy_/index_fill_ is one."""
    counts = {}
    real_setitem = torch.Tensor.__setitem__

    def counting_setitem(self, key, value):
        counts[n_rows] = counts.get(n_rows, 0) + 1
        return real_setitem(self, key, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", counting_setitem)
    for n_rows in (2, 64):
        counts[n_rows] = 0
        _decode_batch([64 + i for i in range(n_rows)])

    assert counts[2] == counts[64], (
        f"element-wise writes scale with batch: {counts} — the decode-row "
        "positions/mask must be written in one batched op")


def _mixed_batch():
    """Realistic mixed step: chat rows and PoC decode rows in one forward, in
    scheduler order. The batched write must land ONLY on the PoC offsets — a
    wrong index tensor would silently overwrite a chat row's position (RoPE)
    and corrupt that chat request while leaving PoC artifacts intact."""
    req_ids = ["chat-0", "poc-0", "chat-1", "poc-1"]
    tokens = {"chat-0": 2, "poc-0": 1, "chat-1": 1, "poc-1": 1}
    native = _Native()
    mgr = {"poc-0": _state(), "poc-1": _state()}
    runner = SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: HIDDEN),
        dtype=torch.float32,
        device=torch.device("cpu"),
        _poc_native=native,
        _poc_mixed_decode_mgr=SimpleNamespace(get=mgr.get),
    )
    sched = SimpleNamespace(num_scheduled_tokens=tokens)
    views = {
        "poc-0": SimpleNamespace(poc_params=_params(0, 64),
                                 num_computed_tokens=70),
        "poc-1": SimpleNamespace(poc_params=_params(1, 64),
                                 num_computed_tokens=81),
    }
    total = sum(tokens.values())
    chat_positions = torch.tensor([11, 12, 99, 13, 99], dtype=torch.long)[:total]
    chat_embeds = torch.arange(total * HIDDEN, dtype=torch.float32).reshape(
        total, HIDDEN)
    return build_unified_mixed_batch_inputs(
        runner, sched, None, chat_embeds, chat_positions,
        {"poc-0", "poc-1"}, total, (len(req_ids), req_ids), views,
    )


def test_mixed_batch_writes_land_only_on_poc_rows():
    embeds, positions, mask, meta = _mixed_batch()
    # order: chat-0 (offsets 0,1), poc-0 (2), chat-1 (3), poc-1 (4)
    assert mask.tolist() == [False, False, True, False, True]
    assert positions.tolist() == [11, 12, 70, 13, 81]
    assert [(m["req_id"], m["start_idx"]) for m in meta] == [
        ("poc-0", 2), ("poc-1", 4)]


def test_mixed_batch_leaves_chat_embeddings_untouched():
    """Chat rows must come out of a mixed step bit-identical to stock vLLM."""
    embeds, _, _, _ = _mixed_batch()
    expected = torch.arange(5 * HIDDEN, dtype=torch.float32).reshape(5, HIDDEN)
    for row in (0, 1, 3):
        assert torch.equal(embeds[row], expected[row]), f"chat row {row} altered"
