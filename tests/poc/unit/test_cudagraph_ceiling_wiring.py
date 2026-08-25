# SPDX-License-Identifier: Apache-2.0
"""The PoC ceiling rule must actually be WIRED into VllmConfig, not just exist.

``poc_cudagraph_capture_size`` is unit-tested on its own; a pure function nobody
calls would still pass those tests while every server kept capturing to 512 and
running the PoC steady state eager. This drives the real
``VllmConfig._set_cudagraph_sizes`` over a minimal stand-in config — no model, no
network, no GPU — and checks the capture list that comes out of it.
"""
from types import SimpleNamespace

import pytest

from vllm.config import CompilationConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode


def _config(max_num_seqs, poc_max_batch_size=0, max_num_batched_tokens=16384):
    """Smallest object `_set_cudagraph_sizes` needs: it reads model/compilation/
    scheduler/cache/parallel config and writes the capture sizes back."""
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.FULL),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
        cache_config=SimpleNamespace(poc_max_batch_size=poc_max_batch_size),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        num_speculative_tokens=0,
        performance_mode=None,
    )


def _resolve(cfg):
    VllmConfig._set_cudagraph_sizes(cfg)
    return cfg.compilation_config.max_cudagraph_capture_size


def test_large_concurrency_captures_the_poc_steady_state():
    """--max-num-seqs 1024 means PoC decode sits at 1024 rows for the whole job;
    vLLM's own rule would stop at 512 and run every one of those steps eager."""
    assert _resolve(_config(max_num_seqs=1024)) == 1024


def test_explicit_poc_batch_is_the_captured_shape():
    assert _resolve(_config(max_num_seqs=1024, poc_max_batch_size=640)) == 640


def test_ordinary_deployment_is_unchanged():
    """Default concurrency must behave exactly like stock vLLM — this rule is a
    no-op below the 512 ceiling, not a behaviour change for every server."""
    assert _resolve(_config(max_num_seqs=256)) == 512
    assert _resolve(_config(max_num_seqs=64)) == 128


def test_token_budget_still_clamps_the_ceiling():
    """A shape that cannot fit in a step must not be captured: the graph would
    never be replayed, and capture costs boot time and memory."""
    assert _resolve(
        _config(max_num_seqs=1024, max_num_batched_tokens=768)) == 768


def test_explicit_user_ceiling_still_wins():
    """An operator who pins the ceiling keeps it — the PoC rule only fills in the
    default (this is the escape hatch for tight-memory boxes)."""
    cfg = _config(max_num_seqs=1024)
    cfg.compilation_config.max_cudagraph_capture_size = 256
    assert _resolve(cfg) == 256


@pytest.mark.parametrize("max_num_seqs", [256, 1024])
def test_capture_list_ends_at_the_ceiling(max_num_seqs):
    """The rung list and the ceiling must agree, or vLLM's own consistency
    assertion downstream trips."""
    cfg = _config(max_num_seqs=max_num_seqs)
    ceiling = _resolve(cfg)
    sizes = cfg.compilation_config.cudagraph_capture_sizes
    assert sizes[-1] == ceiling
    assert sizes == sorted(set(sizes))
