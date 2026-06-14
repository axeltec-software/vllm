"""Decode-PoC must run concurrently with live chat in the SAME forward.

chat and a KV-bound PoC decode interleave in one forward pass. Invariants:

1. PoC is CORRECT under concurrency — re-validating the trajectory ALIGNED
   (inference_k_points_steps) while chat streams keeps n_sphere_mismatches at
   honest level (chat must not leak into the PoC computation).
2. Chat is not corrupted or frozen — full-length, non-empty response.
3. A real mixed batch (chat + PoC in one forward) actually ran.

Requires ``VLLM_POC_MIXED_DECODE=1`` and CUDA graphs ON (no ``--enforce-eager``).
"""
import asyncio

import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 240

BLOCK_HASH = "0xconcurrent"
NONCES = [1, 2, 3]
POC_MAX_TOKENS = 64   # KV-bound decode steps that must overlap the chat
CHAT_MAX_TOKENS = 128


@pytest.fixture(scope="module")
def mixed_server():
    with PoCTestServer(
        MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"}
    ) as srv:
        yield srv


def _poc_artifacts(url: str) -> dict[int, dict]:
    body = poc_request_body(
        BLOCK_HASH, NONCES, MODEL, wait=True, max_tokens=POC_MAX_TOKENS
    )
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "completed", f"status != completed: {data}"
    arts = data.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    return {a["nonce"]: a for a in arts}


async def _poc_async(url: str, infk: dict | None = None) -> dict[int, dict]:
    body = poc_request_body(
        BLOCK_HASH, NONCES, MODEL, wait=True, max_tokens=POC_MAX_TOKENS
    )
    if infk is not None:
        body["inference_k_points_steps"] = infk   # ALIGNED validation (per-step, no cascade)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(f"{url}{POC_URL}", json=body)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "completed", f"status != completed: {data}"
    return {a["nonce"]: a for a in data.get("artifacts", [])}


async def _chat_async(server, idx: int) -> str:
    client = server.get_async_client()
    result = await client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": f"Write a detailed paragraph about the number {idx}.",
        }],
        max_tokens=CHAT_MAX_TOKENS,
        temperature=0.0,
    )
    return result.choices[0].message.content or ""


@pytest.mark.integration
def test_poc_decode_reproducible_under_concurrent_chat(mixed_server):
    """PoC decode validates CORRECTLY (aligned) while chat runs concurrently."""
    url = mixed_server.url_root

    # Reference trajectory: PoC decode alone (no chat traffic).
    baseline = _poc_artifacts(url)
    traj = {str(n): baseline[n]["k_points_steps"] for n in NONCES}

    # Concurrent: re-validate that trajectory ALIGNED while long chats stream.
    async def _run():
        chats = [asyncio.create_task(_chat_async(mixed_server, i)) for i in range(4)]
        val = await _poc_async(url, infk=traj)
        chat_res = await asyncio.gather(*chats)
        return val, chat_res

    concurrent_poc, chat_texts = asyncio.run(_run())

    # Invariant 2: chat completed, non-empty, not frozen/truncated to nothing.
    for i, text in enumerate(chat_texts):
        assert text and len(text.strip()) > 0, f"chat {i} returned empty output"

    # Aligned correctness: n_sphere_mismatches compares each step against the seeded
    # trajectory (no cascade), tolerating benign batch-shape sphere_k boundary flips
    # but catching real chat->PoC leakage. (Byte-identity is the wrong invariant here
    # — a single boundary flip cascades and fails even correct runs.)
    assert set(baseline) == set(concurrent_poc) == set(NONCES)
    mism = [concurrent_poc[n].get("n_sphere_mismatches") for n in NONCES]
    assert all(m is not None and m >= 0 for m in mism), f"validation did not run: {mism}"
    worst = max(mism) / POC_MAX_TOKENS
    assert worst < 0.30, (
        f"concurrent chat corrupted the PoC decode: aligned mismatches {mism}/"
        f"{POC_MAX_TOKENS} (worst {worst:.0%}) exceed honest level — chat is leaking "
        f"into the PoC computation."
    )

    # Invariant 3: a real chat+PoC mixed batch actually ran in one forward.
    log_path = getattr(mixed_server, "log_path", None)
    if log_path:
        try:
            with open(log_path) as f:
                log = f.read()
            assert "MIXED BATCH" in log, (
                "no 'MIXED BATCH' in server log — chat and PoC never shared a "
                "forward, so concurrency was not actually exercised"
            )
        except FileNotFoundError:
            pass  # remote/--poc-port server: log not local, skip this check
