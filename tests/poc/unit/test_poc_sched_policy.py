"""Unit tests for the pure PoC scheduling-policy helpers extracted from the
scheduler into vllm.poc.mixed_decode (keeps the core vLLM footprint thin).

These encode the contract the scheduler relies on:
  - validation recompute + prefill-only run the PURE (exclusive) path;
  - decode GENERATION runs step-driven mixed (seq_len prefill, then 1 token/step);
  - dynamic-KV allocation reserves the whole seq_len+max_tokens upfront for the
    pure path (it runs the whole decode loop in one step) but only one step's
    tokens for the mixed path.
"""
from vllm.poc.mixed_decode import (
    poc_is_pure_path, poc_step_num_tokens, poc_alloc_footprint,
)
from vllm.poc.poc_params import PoCParams


def _params(*, max_tokens, validation, seq_len=256):
    return PoCParams(
        block_hash="0xabc", public_key="0xpub", block_height=1, nonce=0,
        seq_len=seq_len, max_tokens=max_tokens,
        inference_k_points_steps=[1, 2, 3] if validation else None,
    )


def test_pure_path_classification():
    # prefill-only (max_tokens == 0) -> pure
    assert poc_is_pure_path(_params(max_tokens=0, validation=False))
    # validation recompute -> pure even with decode steps
    assert poc_is_pure_path(_params(max_tokens=8, validation=True))
    # decode GENERATION -> mixed (not pure)
    assert not poc_is_pure_path(_params(max_tokens=8, validation=False))


def test_step_num_tokens_mixed_generation():
    pp = _params(max_tokens=8, validation=False, seq_len=256)
    # first step (nothing computed) = prefill of seq_len
    assert poc_step_num_tokens(pp, 0) == 256
    # every later step = a single decode token
    assert poc_step_num_tokens(pp, 256) == 1
    assert poc_step_num_tokens(pp, 300) == 1


def test_step_num_tokens_pure_paths():
    # validation + prefill-only run their whole forward in one seq_len step,
    # regardless of how many tokens are already computed.
    val = _params(max_tokens=8, validation=True, seq_len=128)
    assert poc_step_num_tokens(val, 0) == 128
    assert poc_step_num_tokens(val, 64) == 128
    prefill_only = _params(max_tokens=0, validation=False, seq_len=128)
    assert poc_step_num_tokens(prefill_only, 0) == 128


def test_alloc_footprint_pure_reserves_full_trajectory():
    pp = _params(max_tokens=8, validation=True, seq_len=256)
    # pure path runs the whole decode loop in one step -> reserve seq_len+max_tokens
    assert poc_alloc_footprint(pp, num_new_tokens=256) == 256 + 8


def test_alloc_footprint_mixed_reserves_one_step():
    pp = _params(max_tokens=8, validation=False, seq_len=256)
    # mixed path reserves only this step's tokens (prefill: seq_len; decode: 1)
    assert poc_alloc_footprint(pp, num_new_tokens=256) == 256
    assert poc_alloc_footprint(pp, num_new_tokens=1) == 1
