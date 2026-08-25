# SPDX-License-Identifier: Apache-2.0
"""Cudagraph capture ceiling on a PoC node.

vLLM's default ceiling is ``min(max_num_seqs * 2, 512)`` — tuned for chat, where
the top rung is rarely occupied. PoC is the opposite workload: one row per nonce,
at the machine's full concurrency, for all 256 decode steps of every job. With
``--max-num-seqs 1024`` the whole steady state therefore sat ABOVE the ceiling and
ran eager, silently (measured: ~1.7% on a 1-layer model; more layers, more launches
to skip). The ceiling must follow the resolved PoC batch instead.
"""
import pytest

from vllm.poc.mixed_decode import poc_cudagraph_capture_size

VLLM_DEFAULT = 512  # min(max_num_seqs * 2, 512) for any max_num_seqs >= 256


def test_auto_batch_follows_max_num_seqs_past_the_vllm_ceiling():
    # 0 = AUTO -> poc batch is max_num_seqs; 1024 rows must be captured
    assert poc_cudagraph_capture_size(VLLM_DEFAULT, 0, 1024) == 1024


def test_explicit_poc_batch_wins_over_max_num_seqs():
    """poc_max_batch_size > 0 caps PoC rows per step, so that — not the engine's
    concurrency limit — is the largest decode batch that can occur."""
    assert poc_cudagraph_capture_size(VLLM_DEFAULT, 640, 1024) == 640


def test_never_shrinks_below_the_vllm_default():
    """A small PoC batch must not cost chat its captured rungs: the node still
    serves chat at max_num_seqs concurrency."""
    assert poc_cudagraph_capture_size(VLLM_DEFAULT, 64, 256) == VLLM_DEFAULT
    assert poc_cudagraph_capture_size(VLLM_DEFAULT, 0, 128) == VLLM_DEFAULT


@pytest.mark.parametrize("max_num_seqs", [128, 256, 512])
def test_small_machines_are_unchanged(max_num_seqs):
    """Below the ceiling vLLM's own rule already covers the batch — the PoC rule
    must be a no-op there, not a behaviour change for every deployment."""
    vllm_default = min(max_num_seqs * 2, 512)
    assert poc_cudagraph_capture_size(
        vllm_default, 0, max_num_seqs) == vllm_default


def test_speculative_decode_query_len_is_counted():
    """With speculative tokens each sequence contributes decode_query_len rows to
    the forward, so the captured shape is rows, not sequences."""
    assert poc_cudagraph_capture_size(VLLM_DEFAULT, 0, 512, 2) == 1024
