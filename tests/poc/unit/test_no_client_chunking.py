"""PoC nonce submission must NOT be chunked client-side — pure logic, no GPU/server.

The HTTP layer used to default `batch_size` to a hardcoded 32 and await each chunk
SEQUENTIALLY, so at most 32 nonces were ever in flight no matter what the engine could
serve. That throttled PoC to 32 concurrent sequences on every machine while inference
scaled to hundreds — and it sat ABOVE the engine, so the scheduler's own
poc_max_batch_size (auto-scaled to max_num_seqs) could never bind.

Now 0 = AUTO = submit everything in one shot and let vLLM do the batching. This guards
the chunk arithmetic: one chunk when AUTO, and no ZeroDivisionError for any nonce count.
"""
import pytest

from vllm.poc.routes import POC_BATCH_SIZE_DEFAULT


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
