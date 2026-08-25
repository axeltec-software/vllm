# SPDX-License-Identifier: Apache-2.0
"""Every PoC cache key must contain every field its value depends on.

This generalises a shipped regression. The decode-chain cache was keyed on the
nonce tuple while the cached value was ``decode_base_seeds(block_hash,
public_key, nonce)``; a validator alternating rounds over the same nonce range
therefore served round N-1's seeds to round N. Step 0 still matched (the prefill
snap is computed elsewhere) and every later step chained from the wrong base, so
honest separability collapsed from 7.2% to 93.4% — with no crash, no short
trajectory, and nothing in the artifacts to see.

The tests that existed did not catch it because every fixture hard-coded ONE
block hash: they varied the nonce, the row order, the step and preemption — each
dimension I had thought about — and held constant the one dimension the key was
missing.

So the rule these tests encode is mechanical rather than imaginative: for each
cache, vary each input of the value function INDEPENDENTLY and require the cached
result to change. A key that drops a field fails here without anyone having to
guess which field it will be.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm.poc.native import PoCNativeState
from vllm.poc.gpu_random import (_seed_from_string, decode_base_seeds,
                                 route_base_seed)

HIDDEN, LAYERS, ROWS = 8, 4, 6
CPU = torch.device("cpu")
BH_A, BH_B = "aa" * 32, "bb" * 32
PK_A, PK_B = "11" * 32, "22" * 32


def _state():
    st = PoCNativeState.__new__(PoCNativeState)
    st.hidden_size, st.num_layers, st.device = HIDDEN, LAYERS, CPU
    st.vectors_t, st.vectors = PoCNativeState.alloc_vectors(
        LAYERS, ROWS, HIDDEN, CPU, torch.bfloat16)
    st._hash_cache, st._stack_cache, st._last_refl_key = {}, {}, None
    st._route_base = [torch.zeros(ROWS, dtype=torch.int64) for _ in range(LAYERS)]
    st._seed_cache, st._base_key, st._last_route_key = {}, None, None
    st.route_step = torch.zeros(ROWS, dtype=torch.int64)
    return st


# ------------------------------------------------- reflection vectors (_hash_cache)
# value = generate_householder_vector(f"{block_hash}[_nonce{n}]_layer_{i}_...")
#         -> depends on (block_hash, nonce, layer); NOT on public_key.
def _reflect(block_hash, nonce):
    st = _state()
    st.set_row_block_hashes([block_hash] * ROWS, [nonce] * ROWS)
    return st.vectors_t.clone()


def test_reflection_depends_on_block_hash():
    assert not torch.equal(_reflect(BH_A, None), _reflect(BH_B, None))


def test_reflection_depends_on_nonce_when_per_nonce_seeded():
    assert not torch.equal(_reflect(BH_A, 1), _reflect(BH_A, 2))
    assert not torch.equal(_reflect(BH_A, None), _reflect(BH_A, 1))


def test_reflection_differs_per_layer():
    """A key that collapsed the layer would give every layer the same vector."""
    v = _reflect(BH_A, 7)
    for i in range(1, LAYERS):
        assert not torch.equal(v[0, 0], v[i, 0]), f"layer {i} == layer 0"


def test_stack_cache_agrees_with_the_list_form():
    """_stack_cache is a second view of _hash_cache; if the two could disagree the
    scatter would write values the rest of the code never sees."""
    st = _state()
    for bh, nz in ((BH_A, None), (BH_A, 3), (BH_B, 3)):
        stacked = st._stacked_vectors_for(bh, nz)
        listed = st._vectors_for(bh, nz)
        assert torch.equal(stacked, torch.stack(listed))


# ------------------------------------------------------ router seed bases (_seed_cache)
# value = _seed_from_string(route_base_seed(block_hash, nonce, layer))
#         -> depends on (block_hash, nonce, layer); NOT on public_key.
def _routing(block_hash, nonce):
    st = _state()
    st.set_routing([block_hash] * ROWS, [nonce] * ROWS, [0] * ROWS)
    return [b.clone() for b in st._route_base]


@pytest.mark.parametrize("a,b", [((BH_A, 1), (BH_B, 1)),      # block_hash varies
                                 ((BH_A, 1), (BH_A, 2))])     # nonce varies
def test_router_base_depends_on_each_key_field(a, b):
    ra, rb = _routing(*a), _routing(*b)
    assert any(not torch.equal(x, y) for x, y in zip(ra, rb)), \
        f"router bases identical for {a} and {b}"


def test_router_base_differs_per_layer():
    r = _routing(BH_A, 5)
    vals = {int(b[0]) for b in r}
    assert len(vals) == LAYERS, "layers share a router base"


def test_router_base_matches_direct_hashing():
    r = _routing(BH_A, 9)
    for i, buf in enumerate(r):
        assert int(buf[0]) == _seed_from_string(route_base_seed(BH_A, 9, i))


# ------------------------------------------------------------- decode chain seeds
# value = decode_base_seeds(block_hash, public_key, nonce)
#         -> depends on ALL THREE. This is the field set the shipped key dropped.
@pytest.mark.parametrize("a,b", [
    ((BH_A, PK_A, 5), (BH_B, PK_A, 5)),      # block_hash varies  <- the regression
    ((BH_A, PK_A, 5), (BH_A, PK_B, 5)),      # public_key varies
    ((BH_A, PK_A, 5), (BH_A, PK_A, 6)),      # nonce varies
])
def test_decode_base_seed_depends_on_each_key_field(a, b):
    sa = decode_base_seeds(a[0], a[1], [a[2]], CPU)
    sb = decode_base_seeds(b[0], b[1], [b[2]], CPU)
    assert not torch.equal(sa, sb), f"decode seeds identical for {a} and {b}"


def test_no_two_distinct_identities_collide():
    """Sweep the whole small identity space: distinct (hash, key, nonce) triples
    must give distinct seeds. A truncated key shows up here as a collision."""
    seen = {}
    for bh in (BH_A, BH_B):
        for pk in (PK_A, PK_B):
            for nonce in range(6):
                v = int(decode_base_seeds(bh, pk, [nonce], CPU)[0])
                prev = seen.get(v)
                assert prev is None, f"{(bh[:4], pk[:4], nonce)} collides with {prev}"
                seen[v] = (bh[:4], pk[:4], nonce)


# --------------------------------------------------- lm-head / sampler row filter
# Tier 2 changed how these two build their row index (blocking torch.tensor ->
# pinned_to_device). The index is what decides WHICH rows reach the LM head and
# where sampled tokens land, so the behaviour is worth pinning, not just the
# construction: PoC rows must never reach the LM head, and must come back as zero.
class _Model:
    def __init__(self):
        self.seen = None

    def compute_logits(self, hidden):
        self.seen = hidden.clone()
        return hidden * 10


def _bridge_with(rows_total, poc_ids):
    from vllm.poc.runner_bridge import PoCRunnerBridge

    req_ids = [f"r{i}" for i in range(rows_total)]
    model = _Model()
    runner = SimpleNamespace(
        device=CPU, model=model,
        input_batch=SimpleNamespace(req_ids=req_ids, num_reqs=rows_total),
    )
    b = PoCRunnerBridge(runner)
    b._step = {"poc_req_ids": {req_ids[i] for i in poc_ids}}
    return b, model, req_ids


def test_lm_head_sees_chat_rows_only():
    """PoC rows score hidden states directly; running the LM head for them is both
    wasted work and a different code path than the artifact depends on."""
    b, model, _ = _bridge_with(5, poc_ids=[1, 3])
    hidden = torch.arange(5 * 4, dtype=torch.float32).reshape(5, 4)

    out = b.compute_logits(hidden)

    assert model.seen.shape[0] == 3, "wrong number of rows reached the LM head"
    assert torch.equal(model.seen, hidden[[0, 2, 4]]), "wrong rows reached the LM head"
    assert torch.equal(out, hidden[[0, 2, 4]] * 10)


def test_lm_head_is_skipped_when_every_row_is_poc():
    b, model, _ = _bridge_with(3, poc_ids=[0, 1, 2])
    assert b.compute_logits(torch.zeros(3, 4)) is None
    assert model.seen is None, "LM head ran for a PoC-only batch"


def test_sampled_tokens_scatter_back_to_their_own_rows():
    """The sampler runs on chat rows only, then results are scattered into a
    full-width tensor. A wrong index puts another request's token in a row."""
    b, _, _ = _bridge_with(5, poc_ids=[1, 3])
    b._step["num_reqs_snapshot"] = 5
    b.runner.sampler = lambda logits, sampling_metadata: SimpleNamespace(
        sampled_token_ids=torch.tensor([[7], [8], [9]], dtype=torch.int32),
        logprobs_tensors=None)

    import vllm.poc.mixed_decode as md
    orig = md.slice_sampling_metadata
    md.slice_sampling_metadata = lambda sm, rows, device: sm
    try:
        out = b.sample_chat_rows(torch.zeros(3, 2), SimpleNamespace())
    finally:
        md.slice_sampling_metadata = orig

    ids = out.sampled_token_ids
    assert ids.shape[0] == 5
    assert [int(ids[i]) for i in (0, 2, 4)] == [7, 8, 9], "chat tokens landed wrong"
    assert [int(ids[i]) for i in (1, 3)] == [0, 0], "PoC slots not left at zero"
