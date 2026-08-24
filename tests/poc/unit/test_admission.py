# SPDX-License-Identifier: Apache-2.0
"""PoCAdmission: the whole chat<->PoC mixing policy, unit-tested without an engine.

Covers the invariants the mixed design rests on:
  * zero impact on the pure-chat path (inactive => every method is identity);
  * chat and PoC share a step only while BOTH decode (uniform-decode shape, so
    the step lands on a captured cudagraph rung);
  * poc_share splits the token budget so PoC cannot starve chat;
  * poc_max_batch_size caps PoC rows per step;
  * PoC KV goes through the SHARED KVCacheManager footprint.
"""
from types import SimpleNamespace

import pytest

from vllm.poc.admission import PoCAdmission


def _poc_params(seq_len=64, max_tokens=16):
    return SimpleNamespace(seq_len=seq_len, max_tokens=max_tokens)


def _req(poc=False, computed=0, prompt=64, seq_len=64, max_tokens=16):
    return SimpleNamespace(
        poc_params=_poc_params(seq_len, max_tokens) if poc else None,
        num_computed_tokens=computed,
        num_prompt_tokens=prompt,
    )


def _sched(running=(), waiting=(), max_batch=8, share=0.5,
           num_gpu_blocks=0, block_size=16, max_num_seqs=256,
           poc_seq_len=64, poc_max_tokens=16):
    return SimpleNamespace(
        running=list(running),
        waiting=list(waiting),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        cache_config=SimpleNamespace(
            poc_max_batch_size=max_batch, poc_share=share,
            num_gpu_blocks=num_gpu_blocks, block_size=block_size,
            poc_seq_len=poc_seq_len, poc_max_tokens=poc_max_tokens,
        ),
    )


# ---------------------------------------------------------- chat path is clean
def test_inactive_without_poc_requests():
    a = PoCAdmission(_sched(running=[_req(), _req()]), 1024)
    assert a.active is False


def test_inactive_is_pure_identity():
    """No PoC in the step => admission must not alter chat scheduling at all."""
    chat = _req(computed=5)
    a = PoCAdmission(_sched(running=[chat]), 1024)
    assert a.skip(chat) is False
    assert a.num_tokens(chat, 7) == 7
    assert a.alloc_tokens(chat, 7) == 7
    assert a.over_budget(chat, 10**9) is False
    a.note_scheduled(chat, 7)  # no-op, must not raise


def test_active_with_a_poc_request():
    a = PoCAdmission(_sched(running=[_req(poc=True, computed=1)]), 1024)
    assert a.active is True


# ------------------------------------------------- uniform-shape mixing policy
def test_poc_prefill_defers_chat():
    """A PoC prefill must run isolated: chat is deferred that step."""
    chat, poc_req = _req(computed=3), _req(poc=True, computed=0)
    a = PoCAdmission(_sched(running=[chat, poc_req]), 1024)
    assert a.skip(chat) is True


def test_poc_prefill_excludes_poc_decode_rows():
    """Never mix a PoC prefill row with PoC decode rows (keeps shape uniform)."""
    prefill, decoding = _req(poc=True, computed=0), _req(poc=True, computed=5)
    a = PoCAdmission(_sched(running=[prefill, decoding]), 1024)
    assert a.skip(prefill) is False
    assert a.skip(decoding) is True


def test_both_decoding_mix_together():
    """The point of the design: chat decode + PoC decode share one forward."""
    chat, poc_req = _req(computed=10, prompt=8), _req(poc=True, computed=5)
    a = PoCAdmission(_sched(running=[chat, poc_req]), 1024)
    assert a.skip(chat) is False
    assert a.skip(poc_req) is False


# --------------------------------------------------------------- budget & caps
def test_poc_share_caps_poc_tokens():
    poc_req = _req(poc=True, computed=5)
    a = PoCAdmission(_sched(running=[poc_req], share=0.5), 100)  # budget 50
    assert a.over_budget(poc_req, 40) is False
    a.note_scheduled(poc_req, 40)
    assert a.over_budget(poc_req, 20) is True  # 40+20 > 50


def test_poc_share_never_limits_chat():
    chat, poc_req = _req(computed=10, prompt=8), _req(poc=True, computed=5)
    a = PoCAdmission(_sched(running=[chat, poc_req], share=0.1), 100)
    assert a.over_budget(chat, 10**6) is False


def test_max_batch_caps_poc_rows():
    poc_req = _req(poc=True, computed=5)
    a = PoCAdmission(_sched(running=[poc_req], max_batch=2), 1024)
    assert a.skip(poc_req) is False
    a.note_scheduled(poc_req, 1)
    a.note_scheduled(poc_req, 1)
    assert a.skip(poc_req) is True  # cap reached


# ------------------------------------------------ token count & shared-KV alloc
def test_decode_row_is_one_token_prefill_is_seq_len():
    a = PoCAdmission(_sched(running=[_req(poc=True, computed=5)]), 1024)
    assert a.num_tokens(_req(poc=True, computed=5), 99) == 1
    prefill = _req(poc=True, computed=0)
    assert a.num_tokens(prefill, 99) == prefill.poc_params.seq_len


def test_mixed_path_allocates_one_step_of_kv():
    """max_tokens>0 (mixed decode) allocates per step through the shared manager."""
    a = PoCAdmission(_sched(running=[_req(poc=True, computed=5)]), 1024)
    assert a.alloc_tokens(_req(poc=True, computed=5), 1) == 1


def test_prefill_only_path_allocates_whole_footprint_upfront():
    """max_tokens==0 runs its loop in one step, so it reserves seq_len upfront."""
    pure = _req(poc=True, computed=0, seq_len=64, max_tokens=0)
    a = PoCAdmission(_sched(running=[pure]), 1024)
    assert a.alloc_tokens(pure, 64) == 64


@pytest.mark.parametrize("share,budget,expect", [(0.0, 100, 0), (1.0, 100, 100)])
def test_share_extremes(share, budget, expect):
    poc_req = _req(poc=True, computed=5)
    a = PoCAdmission(_sched(running=[poc_req], share=share), budget)
    assert a.over_budget(poc_req, expect + 1) is True
    if expect:
        assert a.over_budget(poc_req, expect) is False


# ------------------------------------------------- KV-derived batch cap
# Regression: AUTO used to copy max_num_seqs (a scheduler knob with no memory
# awareness). On 1xB300 that admitted 704 nonces x 512 tokens against a
# 302672-token pool -> livelock, GPU 0%, 12-minute hang. The cap must come from
# what the KV pool physically holds.
from vllm.poc.mixed_decode import poc_kv_capacity, resolve_poc_max_batch_size

B300_BLOCKS, B300_BLOCK = 18917, 16          # 302672 tokens
B300_SEQ, B300_MAXTOK = 512, 0


def test_kv_capacity_matches_pool_arithmetic():
    assert poc_kv_capacity(B300_BLOCKS, B300_BLOCK, B300_SEQ, B300_MAXTOK) == 591


def test_kv_capacity_counts_the_decode_budget_too():
    # a nonce holds prefill + its whole trajectory
    assert poc_kv_capacity(100, 16, 128, 128) == 6      # 1600 // 256
    assert poc_kv_capacity(100, 16, 128, 0) == 12       # 1600 // 128


def test_kv_capacity_unknown_pool_is_zero():
    assert poc_kv_capacity(0, 16, 512, 0) == 0
    assert poc_kv_capacity(None, 16, 512, 0) == 0
    assert poc_kv_capacity(100, 0, 512, 0) == 0


def test_auto_cap_is_clamped_by_kv_not_max_num_seqs():
    """The B300 hang, as a unit test."""
    kv = poc_kv_capacity(B300_BLOCKS, B300_BLOCK, B300_SEQ, B300_MAXTOK)
    assert resolve_poc_max_batch_size(0, 704, kv) == 591
    assert resolve_poc_max_batch_size(0, 704, kv) * B300_SEQ <= B300_BLOCKS * B300_BLOCK


def test_auto_cap_uses_concurrency_when_it_is_the_tighter_bound():
    assert resolve_poc_max_batch_size(0, 64, 591) == 64


def test_auto_cap_falls_back_when_pool_unknown():
    assert resolve_poc_max_batch_size(0, 704, 0) == 704


def test_explicit_cap_is_honored_verbatim():
    assert resolve_poc_max_batch_size(128, 704, 591) == 128


def test_auto_cap_via_admission_fits_the_pool():
    """End-to-end through PoCAdmission: AUTO (0) must never admit more nonces
    than the KV pool holds, whatever max_num_seqs says."""
    a = PoCAdmission(
        _sched(running=[_req(poc=True, computed=5)], max_batch=0,
               num_gpu_blocks=B300_BLOCKS, block_size=B300_BLOCK,
               max_num_seqs=704, poc_seq_len=B300_SEQ,
               poc_max_tokens=B300_MAXTOK),
        1024,
    )
    per_nonce = B300_SEQ + B300_MAXTOK
    assert a._max_batch * per_nonce <= B300_BLOCKS * B300_BLOCK
    assert a._max_batch < 704                # not the naive scheduler knob


def test_decode_manager_pool_never_empty_under_auto():
    """Regression: after the cap became lazily-resolved, get_decode_manager
    sized its slot pool from the RAW config value (0 under AUTO) -> empty pool
    -> every allocate failed -> decode state never existed -> prefill emitted a
    pure-path artifact and the chain never ran (empty k_points_steps)."""
    from vllm.poc.mixed_decode import get_decode_manager

    runner = SimpleNamespace(
        cache_config=SimpleNamespace(
            poc_max_batch_size=0, poc_seq_len=64, poc_max_tokens=8,
            num_gpu_blocks=0, block_size=16),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=256)),
    )
    mgr = get_decode_manager(runner)
    st = mgr.allocate("poc-x", nonce=1, seq_len=64, max_tokens=8)
    assert st is not None, "AUTO cap must yield a usable slot pool"
    assert len(mgr._free_slots) > 0 or st is not None
