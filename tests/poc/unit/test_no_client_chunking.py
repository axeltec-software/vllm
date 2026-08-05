"""PoC nonce submission must NOT be chunked client-side — pure logic, no GPU/server.

The HTTP layer used to default `batch_size` to a hardcoded 32 and await each chunk
SEQUENTIALLY, so at most 32 nonces were ever in flight no matter what the engine could
serve. That throttled PoC to 32 concurrent sequences on every machine while inference
scaled to hundreds — and it sat ABOVE the engine, so the scheduler's own
poc_max_batch_size (auto-scaled to max_num_seqs) could never bind.

Now 0 = AUTO = submit everything in one shot and let vLLM do the batching. This guards
the chunk arithmetic: one chunk when AUTO, and no ZeroDivisionError for any nonce count.
"""
import inspect
import sys
from pathlib import Path

import pytest

from vllm.poc.config import PoCConfig
from vllm.poc.routes import POC_BATCH_SIZE_DEFAULT

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmarks" / "poc"))


def _chunks(total, batch_size):
    """The submission arithmetic used by routes/_process_job."""
    step = batch_size or total or 1
    return (total + step - 1) // step, step


def test_default_is_auto_no_chunking():
    # The hardcoded 32 is gone; 0 means "let the engine batch".
    assert POC_BATCH_SIZE_DEFAULT == 0


@pytest.mark.parametrize("total", [1, 32, 64, 128, 305, 1000])
def test_auto_submits_everything_in_one_chunk(total):
    n, step = _chunks(total, 0)
    assert n == 1, f"AUTO must submit all {total} nonces at once, got {n} chunks"
    assert step == total


@pytest.mark.parametrize("total", [0, 1, 64, 128])
@pytest.mark.parametrize("bs", [0, 1, 32])
def test_never_divides_by_zero(total, bs):
    n, step = _chunks(total, bs)          # must not raise
    assert step >= 1 and n >= 0


def test_explicit_batch_size_still_chunks():
    # An operator can still deliberately cap concurrency.
    assert _chunks(128, 32)[0] == 4
    assert _chunks(64, 64)[0] == 1


# --- the CLIENT side -------------------------------------------------------------
# The gap that let the throttle survive: the server default was fixed to 0, but
# benchmarks/poc/poc_validation.request_generate still defaulted batch_size=32 and SENT
# it in the payload, overriding the server's AUTO. So `collect.py --nonces 64` ran as two
# sequential waves of 32 — in BOTH generate and validate mode — and reported a depressed
# nonces/s. Server-side tests alone cannot catch this; the client default needs its own
# guard.

def test_benchmark_client_does_not_send_a_batch_cap():
    from poc_validation import request_generate
    default = inspect.signature(request_generate).parameters["batch_size"].default
    assert default == 0, (
        f"request_generate defaults batch_size={default} and sends it in the payload, "
        f"which overrides the server's AUTO and re-imposes client-side chunking")


def test_poc_config_dataclass_does_not_carry_a_batch_cap():
    # Dead today (never constructed) but exported — keep it from drifting back to 32
    # and silently becoming a cap if anything ever starts using it.
    assert PoCConfig.batch_size == 0


# --- continuous mining round ----------------------------------------------------
# The mining loop pulls a round of nonces per iteration. It must size that round from
# the ENGINE, not a constant, or a big machine mines 32-nonce rounds forever.

class _Cfg:
    def __init__(self, poc_cap=None, max_num_seqs=None):
        if poc_cap is not None:
            self.cache_config = type("C", (), {"poc_max_batch_size": poc_cap})()
        if max_num_seqs is not None:
            self.scheduler_config = type("S", (), {"max_num_seqs": max_num_seqs})()


class _Engine:
    def __init__(self, cfg=None):
        if cfg is not None:
            self.vllm_config = cfg


def test_mining_round_uses_engine_capacity():
    from vllm.poc.routes import resolve_mining_round
    # a 256-seq machine mines 256-nonce rounds, not 32
    assert resolve_mining_round(0, _Engine(_Cfg(poc_cap=256, max_num_seqs=256))) == 256
    assert resolve_mining_round(0, _Engine(_Cfg(poc_cap=1024, max_num_seqs=1024))) == 1024


def test_mining_round_falls_back_to_max_num_seqs_not_a_constant():
    from vllm.poc.routes import resolve_mining_round
    # cache_config unreadable -> use the engine's own concurrency limit, NOT 32
    assert resolve_mining_round(0, _Engine(_Cfg(max_num_seqs=512))) == 512


def test_mining_round_honors_an_explicit_value():
    from vllm.poc.routes import resolve_mining_round
    assert resolve_mining_round(64, _Engine(_Cfg(poc_cap=256, max_num_seqs=256))) == 64


def test_mining_round_last_resort_is_reached_only_with_no_engine_config():
    from vllm.poc.routes import resolve_mining_round
    assert resolve_mining_round(0, _Engine()) == 32          # nothing readable at all
    assert resolve_mining_round(0, _Engine(_Cfg())) == 32
