"""Unit tests for per-nonce Householder reflection seeding (PoCNativeState).

Per-block (default): every nonce of a block reflects with ONE draw seeded by
block_hash — the whole block shares a single measurement instrument. Per-nonce
(opt-in, PoCParams.per_nonce_reflection): each row's vectors are seeded by
(block_hash, nonce), so every nonce measures with its own independent draw and
block statistics average over n independent instruments.

Non-circular like the B1 stress test: buffers are compared to independently
generated Householder vectors from the documented seed strings, never to
self-produced values. Pure CPU.
"""
import torch

from vllm.poc.gpu_random import generate_householder_vector
from vllm.poc.native import PoCNativeState
from vllm.poc.poc_params import PoCParams

DEV = torch.device("cpu")
NL, HS, MT = 4, 16, 8
BH = "deadbeef" * 8


def _truth(bh, layer, nonce=None):
    """Ground-truth vectors from the documented seed-string contract."""
    suffix = "" if nonce is None else f"_nonce{nonce}"
    return generate_householder_vector(
        f"{bh}{suffix}_layer_{layer}_householder", HS, DEV).to(torch.float32)


def _assert_buffers(st, rows):
    """rows: list of (block_hash | None, refl_nonce | None) per row."""
    for row, (bh, nz) in enumerate(rows):
        for layer in range(NL):
            got = st.vectors[layer][row]
            if bh is None:
                assert torch.equal(got, torch.zeros(HS, device=DEV)), \
                    f"row {row} layer {layer}: None row not zero"
            else:
                assert torch.allclose(got, _truth(bh, layer, nz), atol=1e-6), \
                    f"row {row} layer {layer}: wrong vectors for ({bh}, {nz})"


def test_per_block_seed_string_unchanged():
    """The default (per-block) scheme must keep the LEGACY seed string — no
    suffix — or every existing chain breaks. Omitting the argument and passing
    all-None must both hit the legacy path."""
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    st.set_row_block_hashes([BH, BH])
    _assert_buffers(st, [(BH, None), (BH, None)])
    st2 = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    st2.set_row_block_hashes([BH, BH], [None, None])
    for layer in range(NL):
        assert torch.equal(st.vectors[layer], st2.vectors[layer])


def test_per_nonce_rows_get_independent_draws():
    """Same block_hash, different nonces -> DIFFERENT reflection vectors per row
    (each nonce its own instrument); same nonce -> identical; all unit-norm."""
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    st.set_row_block_hashes([BH, BH, BH], [7, 8, 7])
    _assert_buffers(st, [(BH, 7), (BH, 8), (BH, 7)])
    for layer in range(NL):
        v7, v8 = st.vectors[layer][0], st.vectors[layer][1]
        assert not torch.allclose(v7, v8), "nonces 7/8 share a draw"
        assert torch.equal(v7, st.vectors[layer][2]), "same nonce differs"
        assert torch.allclose(v7.norm(), torch.tensor(1.0), atol=1e-5)
    # per-nonce differs from the per-block draw of the same block_hash
    assert not torch.allclose(st.vectors[0][0], _truth(BH, 0, None))


def test_mixed_batch_per_block_and_per_nonce():
    """Rows of a per-block request and a per-nonce request coexist in one
    forward, each seeded by its own scheme (None rows stay zero/masked)."""
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    rows = [(BH, None), (BH, 3), (None, None), ("0xbbb", 3)]
    st.set_row_block_hashes([r[0] for r in rows], [r[1] for r in rows])
    _assert_buffers(st, rows)


def test_skip_key_covers_nonces():
    """Changing ONLY the reflection nonces (same row_hashes) must rescatter —
    the B1 skip key has to include the nonce mapping."""
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    st.set_row_block_hashes([BH, BH], [1, 2])
    _assert_buffers(st, [(BH, 1), (BH, 2)])
    st.set_row_block_hashes([BH, BH], [2, 1])       # same hashes, swapped nonces
    _assert_buffers(st, [(BH, 2), (BH, 1)])
    st.set_row_block_hashes([BH, BH])               # back to per-block
    _assert_buffers(st, [(BH, None), (BH, None)])


def test_cache_bound_keeps_buffers_correct():
    """Blowing past the cache cap must only cost regeneration, never correctness."""
    st = PoCNativeState(NL, HS, MT, DEV, torch.float32)
    st._HASH_CACHE_MAX = 4
    for start in range(0, 24, 2):
        nzs = [start, start + 1]
        st.set_row_block_hashes([BH, BH], nzs)
        _assert_buffers(st, [(BH, nzs[0]), (BH, nzs[1])])
    assert len(st._hash_cache) <= 4


def test_poc_params_carries_and_clones_flag():
    p = PoCParams(block_hash=BH, public_key="pk", block_height=1, nonce=5)
    assert p.per_nonce_reflection is False          # default: legacy behavior
    p2 = PoCParams(block_hash=BH, public_key="pk", block_height=1, nonce=5,
                   per_nonce_reflection=True)
    assert p2.clone().per_nonce_reflection is True
