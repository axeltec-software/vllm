"""Regression: PoC cudagraph (default; server not --enforce-eager) survives chat.

The minimal co-existence requirement: pure chat runs under vLLM's cudagraph AND
pure PoC runs under Irene's manual cudagraph, and the two co-exist without chat
corrupting the PoC graph.

Bug (root-causes/poc-cudagraph-chat-corrupts-it-via-embeds-buf-aliasing-defer): a
chat request shifted the allocator so the captured PoC graph's activations aliased
its static embeds_buf -> replay overwrote its own input -> all-NaN, permanent.

Fix (Path A): on all-NaN, drop ALL PoC graph state (recapture fresh) + retry the
request once. Transparent self-heal that converges after a few resets then stays
clean. This test asserts that after a chat poison the PoC request still returns a
full batch of artifacts (>=2 nonces, the standing multi-nonce rule).
"""
import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
CHAT_URL = "/v1/chat/completions"
TIMEOUT = 300
NONCES = list(range(8))  # padded to poc_max_batch_size internally; >=2 nonces


def _poc(url: str) -> int:
    body = poc_request_body("0xsurvive", NONCES, MODEL, wait=True, max_tokens=256)
    r = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return len(r.json().get("artifacts", []))


def _chat(url: str) -> None:
    body = {"model": MODEL, "messages": [{"role": "user", "content": "Write 200 words."}],
            "max_tokens": 200}
    httpx.post(f"{url}{CHAT_URL}", json=body, timeout=TIMEOUT)


@pytest.mark.integration
def test_poc_cudagraph_survives_chat():
    """With cudagraph ON, PoC stays valid through repeated chat interleaving
    (Path A reset+retry self-heals the chat-induced graph corruption)."""
    # Irene's pure-PoC cudagraph runs only on the STATIC path (under dynamic KV the
    # pure path is eager on paged blocks), so force static to exercise it + Path A.
    with PoCTestServer(MODEL, BASE_ARGS + ["--no-poc-dynamic-kv"]) as srv:
        url = srv.url_root
        assert _poc(url) == len(NONCES), "PoC cudagraph broken in isolation"
        # Interleave chat + PoC; the reset+retry must keep every PoC batch full.
        results = []
        for _ in range(6):
            _chat(url)
            results.append(_poc(url))
    # Converges: the tail must be clean full batches (allow a couple warmup resets).
    assert results[-3:] == [len(NONCES)] * 3, (
        f"PoC did not stay valid through chat co-existence: {results}")
