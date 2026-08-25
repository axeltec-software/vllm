# SPDX-License-Identifier: Apache-2.0
"""Properties the PoC row bookkeeping must hold for ANY mapping.

The example-based tests pin specific shapes; these pin the invariants, over
randomised mappings. They are the ones that would survive a rewrite of the
batching strategy — each says something about what the code MEANS, not how it is
currently written:

  * a row's buffers depend only on ITS OWN (block_hash, nonce) — no cross-row
    contamination, whatever the batch around it looks like;
  * masked rows are never written;
  * the result does not depend on the order rows arrive in, only on the mapping;
  * a rebuild fully replaces the previous mapping — no stale rows survive;
  * two independent ranks compute bit-identical state (TP replication).

Randomised with fixed seeds so a failure is reproducible.
"""
import random

import pytest
import torch

from vllm.poc import native as native_mod

from vllm.poc.native import PoCNativeState
from vllm.poc.gpu_random import route_base_seed, _seed_from_string

HIDDEN = 8
CPU = torch.device("cpu")
HASHES = [f"{i:02x}" * 32 for i in range(6)]
NONCES = [None, 0, 1, 7, 41, 900]


def _state(rows, layers=4):
    st = PoCNativeState.__new__(PoCNativeState)
    st.hidden_size, st.num_layers, st.device = HIDDEN, layers, CPU
    st.vectors_t, st.vectors = PoCNativeState.alloc_vectors(
        layers, rows, HIDDEN, CPU, torch.bfloat16)
    st._hash_cache, st._stack_cache, st._last_refl_key = {}, {}, None
    st._route_base = [torch.zeros(rows, dtype=torch.int64) for _ in range(layers)]
    st._seed_cache, st._base_key, st._last_route_key = {}, None, None
    st.route_step = torch.zeros(rows, dtype=torch.int64)
    return st


def _mapping(rows, seed, masked=0.2):
    rnd = random.Random(seed)
    hashes, nonces = [], []
    for _ in range(rows):
        if rnd.random() < masked:
            hashes.append(None); nonces.append(None)
        else:
            hashes.append(rnd.choice(HASHES)); nonces.append(rnd.choice(NONCES))
    return hashes, nonces


def _scatter(st, hashes, nonces):
    st.set_row_block_hashes(list(hashes), list(nonces))


SEEDS = [1, 2, 3, 5, 8, 13]


# -------------------------------------------------------------- row provenance
@pytest.mark.parametrize("seed", SEEDS)
def test_each_row_holds_exactly_its_own_key(seed):
    """buffer[layer, row] is a function of (block_hash, nonce, layer) ALONE."""
    rows = 48
    hashes, nonces = _mapping(rows, seed)
    st = _state(rows)
    _scatter(st, hashes, nonces)
    for row, (bh, nz) in enumerate(zip(hashes, nonces)):
        if bh is None:
            continue
        want = st._vectors_for(bh, nz)
        for i in range(st.num_layers):
            assert torch.equal(st.vectors_t[i, row], want[i].to(torch.bfloat16))


@pytest.mark.parametrize("seed", SEEDS)
def test_masked_rows_are_never_written(seed):
    rows = 48
    hashes, nonces = _mapping(rows, seed, masked=0.5)
    st = _state(rows)
    _scatter(st, hashes, nonces)
    for row, bh in enumerate(hashes):
        if bh is None:
            assert not st.vectors_t[:, row].any(), f"masked row {row} written"


@pytest.mark.parametrize("seed", SEEDS)
def test_changing_one_row_leaves_every_other_row_alone(seed):
    """No cross-row contamination: the batch around a row must not change it."""
    rows = 32
    hashes, nonces = _mapping(rows, seed, masked=0.0)
    st_a, st_b = _state(rows), _state(rows)
    _scatter(st_a, hashes, nonces)

    victim = seed % rows
    h2 = list(hashes)
    h2[victim] = HASHES[(HASHES.index(hashes[victim]) + 1) % len(HASHES)]
    _scatter(st_b, h2, nonces)

    for row in range(rows):
        same = torch.equal(st_a.vectors_t[:, row], st_b.vectors_t[:, row])
        assert same == (row != victim), f"row {row}: unexpected change"


# ------------------------------------------------------------ order invariance
@pytest.mark.parametrize("seed", SEEDS)
def test_result_depends_on_the_mapping_not_the_row_order(seed):
    rows = 40
    hashes, nonces = _mapping(rows, seed)
    order = list(range(rows))
    random.Random(seed + 99).shuffle(order)

    st_a, st_b = _state(rows), _state(rows)
    _scatter(st_a, hashes, nonces)
    _scatter(st_b, [hashes[i] for i in order], [nonces[i] for i in order])
    for row, src in enumerate(order):
        assert torch.equal(st_b.vectors_t[:, row], st_a.vectors_t[:, src])


# --------------------------------------------------------------- full rebuild
@pytest.mark.parametrize("seed", SEEDS)
def test_a_rebuild_leaves_nothing_of_the_previous_mapping(seed):
    """Rows that were written and are now masked must go back to zero, or a
    finished nonce keeps reflecting into a new request's row."""
    rows = 36
    first, n1 = _mapping(rows, seed, masked=0.0)
    second, n2 = _mapping(rows, seed + 500, masked=0.6)
    st = _state(rows)
    _scatter(st, first, n1)
    _scatter(st, second, n2)

    fresh = _state(rows)
    _scatter(fresh, second, n2)
    assert torch.equal(st.vectors_t, fresh.vectors_t), "stale rows survived a rebuild"


# ------------------------------------------------------- routing base provenance
@pytest.mark.parametrize("seed", SEEDS)
def test_router_base_matches_direct_hashing_for_every_row(seed):
    rows = 40
    hashes, nonces = _mapping(rows, seed)
    st = _state(rows)
    st.set_routing(list(hashes), list(nonces), [seed % 7] * rows)
    for i in range(st.num_layers):
        want = [(_seed_from_string(route_base_seed(bh, nz, i)) if bh is not None else 0)
                for bh, nz in zip(hashes, nonces)]
        assert st._route_base[i][:rows].tolist() == want


# ------------------------------------------------------------ TP replication
@pytest.mark.parametrize("seed", SEEDS)
def test_two_ranks_compute_identical_state(seed):
    """TP replicates these buffers rather than sharding them, so ranks must agree
    bit-for-bit from the same mapping — no device, rank or ordering dependence."""
    rows = 64
    hashes, nonces = _mapping(rows, seed)
    steps = [seed % 5] * rows
    rank0, rank1 = _state(rows), _state(rows)
    for st in (rank0, rank1):
        _scatter(st, hashes, nonces)
        st.set_routing(list(hashes), list(nonces), steps)

    assert torch.equal(rank0.vectors_t, rank1.vectors_t)
    assert torch.equal(rank0.route_step, rank1.route_step)
    for i in range(rank0.num_layers):
        assert torch.equal(rank0._route_base[i], rank1._route_base[i])


@pytest.mark.parametrize("seed", SEEDS)
def test_repeating_a_mapping_is_idempotent(seed):
    rows = 24
    hashes, nonces = _mapping(rows, seed)
    st = _state(rows)
    _scatter(st, hashes, nonces)
    snapshot = st.vectors_t.clone()
    st._last_refl_key = None                    # force the work to run again
    _scatter(st, hashes, nonces)
    assert torch.equal(st.vectors_t, snapshot)


@pytest.mark.parametrize("seed", SEEDS)
def test_memo_keeps_only_the_hashes_of_the_current_mapping(seed):
    """A round brings a new block_hash, and the memo key contains the hash — so
    entries for the previous round can never be hit again. They are dropped rather
    than retained behind a size cap, and the values are unaffected either way."""
    rows = 32
    st = _state(rows)
    st.set_routing([HASHES[0]] * rows, list(range(rows)), [0] * rows)
    assert set(st._seed_cache) == {HASHES[0]}

    st.set_routing([HASHES[1]] * rows, list(range(rows)), [0] * rows)
    # both scopes are kept: rows carry their own hash, so two rounds can interleave
    # step to step and dropping the absent one would re-derive it every swap
    assert set(st._seed_cache) == {HASHES[0], HASHES[1]}

    # the old hash coming back is simply re-derived — same values as a cold state
    st.set_routing([HASHES[0]] * rows, list(range(rows)), [0] * rows)
    fresh = _state(rows)
    fresh.set_routing([HASHES[0]] * rows, list(range(rows)), [0] * rows)
    for i in range(st.num_layers):
        assert torch.equal(st._route_base[i], fresh._route_base[i])


def test_two_live_hashes_in_one_mapping_are_both_kept():
    """Rows carry their OWN block_hash, so a batch can legitimately mix two rounds.
    Eviction must key off what this mapping contains, not off 'the hash changed'."""
    rows = 24
    hashes = [HASHES[i % 2] for i in range(rows)]
    st = _state(rows)
    st.set_routing(hashes, list(range(rows)), [0] * rows)
    assert set(st._seed_cache) == {HASHES[0], HASHES[1]}


def test_memo_is_bounded_by_scopes_not_by_nonce_count():
    """Garbage accumulates one scope per finished round, so the bound is in ROUNDS.
    The oldest scope goes; the current one never does, whatever the layer count."""
    from vllm.poc.native import _SEED_CACHE_MAX_SCOPES
    rows = 8
    st = _state(rows)
    hashes = [f"{i:02x}" * 32 for i in range(_SEED_CACHE_MAX_SCOPES + 3)]
    for h in hashes:
        st.set_routing([h] * rows, list(range(rows)), [0] * rows)
        assert h in st._seed_cache, "current round evicted itself"
    assert len(st._seed_cache) <= _SEED_CACHE_MAX_SCOPES
    assert hashes[0] not in st._seed_cache, "oldest scope never evicted"


def test_interleaved_blocks_do_not_re_derive_each_other(monkeypatch):
    """The thrash guard. Rows carry their own block_hash, so the scheduler can
    alternate between two rounds step to step. Both scopes must stay warm — an
    eviction rule based on "absent from this mapping" would re-hash a whole round
    on every swap (62 layers x 512 nonces = ~19 ms each time)."""
    rows = 16
    st = _state(rows)
    a, b = HASHES[0], HASHES[1]
    st.set_routing([a] * rows, list(range(rows)), [0] * rows)
    st.set_routing([b] * rows, list(range(rows)), [0] * rows)

    calls = []
    real = native_mod._seed_from_string
    monkeypatch.setattr(native_mod, "_seed_from_string",
                        lambda s: calls.append(s) or real(s))

    for step, h in enumerate([a, b, a, b, a], start=1):     # alternate rounds
        st.set_routing([h] * rows, list(range(rows)), [step] * rows)

    assert not calls, f"{len(calls)} seeds re-derived while swapping between blocks"
    assert set(st._seed_cache) == {a, b}
