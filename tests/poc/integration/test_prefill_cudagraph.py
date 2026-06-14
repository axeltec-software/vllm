"""Regression: the step-driven decode-PoC PREFILL step runs in a CUDA graph
(per-N, dedicated stable buffers) and co-exists with sustained chat WITHOUT
crashing the engine or corrupting the PoC computation.

History: the mixed path used to push PoC prefill through a manual graph keyed by
num_tokens that baked non-persistent FlashInfer prefill-plan buffer addresses;
under sustained load a replay read freed memory -> CUDA illegal memory access ->
EngineCore died -> all later chat/PoC errored (gsm8k collapsed to ~0). Fix:
prefill steps are scheduler-exclusive (uniform N-prefill) and graphed by
poc_graphed_prefill_batch with dedicated use_cuda_graph buffers re-planned to the
current reserved slots each replay.

Contract (GPU, NOT --enforce-eager):
1. NO ENGINE DEATH under sustained concurrent chat + repeated decode-PoC.
2. The prefill graph is actually used (server log "POC_GRAPH prefill captured").
3. PoC stays correct (aligned n_sphere_mismatches at honest level), proving the
   per-slot re-plan writes/reads the right reserved blocks (no KV corruption).
4. Chat keeps making progress (no starvation/hang).
"""
import asyncio

import httpx
import pytest

from tests.poc._server import (
    CANONICAL_MODEL as MODEL,
    DEFAULT_SERVER_ARGS as BASE_ARGS,
    PoCTestServer,
)

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 180
NONCES = [0, 1, 2, 3, 4, 5, 6, 7]   # >1 -> several prefills batch into one step
POC_MAX_TOKENS = 96
N_CHATS = 12
CHAT_MAX_TOKENS = 128
MAX_MISMATCH_FRAC = 0.30            # honest aligned mismatch ~6%; corruption is tens of %


@pytest.fixture(scope="module")
def graph_server():
    with PoCTestServer(
        MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"},
    ) as srv:
        yield srv


def _poc_body(infk=None):
    body = {
        "block_hash": "deadbeef" * 8, "block_height": 100, "public_key": "cafebabe" * 8,
        "node_id": 0, "node_count": 1, "nonces": NONCES,
        "params": {"model": MODEL, "seq_len": 256, "k_dim": 12, "max_tokens": POC_MAX_TOKENS},
        "wait": True,
    }
    if infk is not None:
        body["inference_k_points_steps"] = infk
    return body


async def _poc(client, url, infk=None):
    r = await client.post(f"{url}{POC_URL}", json=_poc_body(infk))
    r.raise_for_status()
    d = r.json()
    assert d.get("status") == "completed", d
    arts = d.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    return arts


async def _chat(server, idx):
    client = server.get_async_client()
    r = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"Write a detailed paragraph about topic {idx}."}],
        max_tokens=CHAT_MAX_TOKENS, temperature=0.0,
    )
    return r.choices[0].message.content or ""


@pytest.mark.integration
def test_prefill_cudagraph_no_crash_correct(graph_server):
    url = graph_server.url_root

    async def _run():
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # reference trajectory (also forces a fresh prefill batch -> capture)
            gen = await _poc(client, url)
            traj = {str(a["nonce"]): a["k_points_steps"] for a in gen}
            # several rounds so prefill graphs are captured AND replayed under chat
            val = None
            for _ in range(3):
                chats = [asyncio.create_task(_chat(graph_server, i)) for i in range(N_CHATS)]
                val = await asyncio.wait_for(_poc(client, url, infk=traj), timeout=TIMEOUT)
                await asyncio.gather(*chats)
            return val

    try:
        val = asyncio.run(_run())
    except (asyncio.TimeoutError, httpx.TimeoutException) as e:
        pytest.fail(f"prefill-cudagraph PoC hung/engine-died under chat: {type(e).__name__}")

    # Invariant 2: prefill graph actually used.
    log_path = getattr(graph_server, "log_path", None)
    if log_path:
        try:
            with open(log_path) as f:
                log = f.read()
        except FileNotFoundError:
            log = ""
        assert "EngineDeadError" not in log and "illegal memory" not in log, \
            "engine crashed during sustained prefill+chat"
        assert "POC_GRAPH prefill captured" in log, \
            "prefill graph was never captured (prefill ran eager, not graphed)"

    # Invariant 3: PoC correct (aligned mismatch at honest level -> no KV corruption).
    mism = [a.get("n_sphere_mismatches") for a in val]
    assert all(m is not None and m >= 0 for m in mism), f"validation did not run: {mism}"
    worst = max(mism) / POC_MAX_TOKENS
    assert worst < MAX_MISMATCH_FRAC, (
        f"prefill-cudagraph CORRUPTED PoC: aligned mismatches {mism}/{POC_MAX_TOKENS} "
        f"(worst {worst:.0%}) exceed honest level {MAX_MISMATCH_FRAC:.0%} — the per-slot "
        f"re-plan wrote/read the wrong reserved blocks."
    )
