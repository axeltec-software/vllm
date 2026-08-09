"""Decode-chain marks must be cleared on EVERY forward — regression for the stale-row
bug (PR #6).

`embed_prev_k` is a PERSISTENT buffer the captured cudagraph reads whole; a row is
synthesized as a decode input iff its prev_k >= 0. The old code only wrote the first
`n` rows and only when there were decode jobs, so a mark left by a LARGER previous
batch, a cudagraph-padding row, or a forward with no decode rows after one that had
them, survived and got synthesized as a PHANTOM decode input -> corrupted trajectory.

None of our other tests vary the decode-batch composition across consecutive calls
(single nonce or steady batch), so the leak was never exercised. These do: they SHRINK
the batch and drop to zero decode rows, and assert stale marks are gone. CPU-only.
They fail on the pre-PR#6 set_decode_chain.
"""
import torch

from vllm.poc.native import PoCNativeState


def _state():
    return PoCNativeState(num_layers=2, hidden_size=8, max_tokens=4,
                          device=torch.device("cpu"), dtype=torch.float32)


def _t(xs):
    return torch.tensor(xs, dtype=torch.int64)


def test_init_all_non_decode():
    st = _state()
    assert st.embed_prev_k.tolist() == [-1, -1, -1, -1]


def test_shrinking_batch_clears_stale_decode_rows():
    st = _state()
    # forward A: rows 0,1,2 are decode
    st.set_decode_chain(offs=_t([0, 1, 2]), base=_t([10, 11, 12]),
                        prev_k=_t([5, 6, 7]), step=_t([1, 1, 1]))
    assert st.embed_prev_k.tolist() == [5, 6, 7, -1]
    # forward B: batch SHRINKS to only row 0 -> rows 1,2 must NOT remain decode
    st.set_decode_chain(offs=_t([0]), base=_t([20]), prev_k=_t([9]), step=_t([2]))
    assert st.embed_prev_k.tolist() == [9, -1, -1, -1], "stale rows synthesized as decode"
    assert st.embed_base[0].item() == 20 and st.embed_step[0].item() == 2


def test_forward_with_no_decode_rows_clears_everything():
    st = _state()
    st.set_decode_chain(offs=_t([0, 1]), base=_t([1, 2]), prev_k=_t([3, 4]), step=_t([1, 1]))
    assert (st.embed_prev_k[:2] >= 0).all()
    # a forward with NO decode rows (e.g. all-chat, or cudagraph padding only)
    st.set_decode_chain()
    assert st.embed_prev_k.tolist() == [-1, -1, -1, -1], "no-decode forward left stale marks"


def test_slot_reused_decode_then_prefill_is_not_synthesized():
    """The exact production trigger. A row index is reused: decode in forward T, then a
    NEW nonce prefilling in forward T+1. Prefill rows must have prev_k=-1 so the embed
    wrapper uses the pre-filled embed, NOT the decode synth. Old code left the slot's
    prev_k>=0 (stale from T) -> the prefill row was wrongly synthesized as decode ->
    corrupted, batch-history-dependent trajectory (invisible to consensus tests because
    prover and validator corrupt identically)."""
    st = _state()
    # forward T: row 0 is a decode row
    st.set_decode_chain(offs=_t([0]), base=_t([100]), prev_k=_t([7]), step=_t([3]))
    assert st.embed_prev_k[0].item() == 7
    # forward T+1: row 0 now holds a NEW nonce doing PREFILL -> not in the decode set.
    # A prefill-only forward calls set_decode_chain with no decode rows.
    st.set_decode_chain()
    assert st.embed_prev_k[0].item() == -1, \
        "reused slot still marked decode -> prefill row synthesized as decode (the bug)"


def test_only_prev_k_gates_synthesis():
    # prev_k alone decides decode-vs-not; base/step at non-decode rows are irrelevant
    # (never read when prev_k<0), so clearing prev_k fully is sufficient and correct.
    st = _state()
    st.set_decode_chain(offs=_t([2]), base=_t([99]), prev_k=_t([0]), step=_t([7]))
    assert st.embed_prev_k.tolist() == [-1, -1, 0, -1]  # only row 2 is a decode row
