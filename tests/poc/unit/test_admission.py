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


def _sched(running=(), waiting=(), max_batch=8, share=0.5):
    return SimpleNamespace(
        running=list(running),
        waiting=list(waiting),
        cache_config=SimpleNamespace(
            poc_max_batch_size=max_batch, poc_share=share
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
