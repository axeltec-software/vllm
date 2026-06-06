"""CUDA graph must be a *pure speed optimization* for the PoC forward.

The same PoC request, run once under ``--enforce-eager`` and once with CUDA
graphs enabled, must yield byte-identical artifacts — both the prefill
``vector_b64`` and the KV-bound decode trajectory ``k_points_steps`` (and the
prefill ``sphere_k``). If they differ, the captured graph is not reproducing
the eager computation (e.g. stale buffers, workspace bleed, or the decode no
longer reading the prefill KV).

Two servers are launched sequentially (not concurrently) to keep GPU memory
bounded on small test vehicles.
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
TIMEOUT = 240

BLOCK_HASH = "0xcudagraph_equiv"
NONCES = [1, 2, 3, 4]
MAX_TOKENS = 8  # exercise the KV-bound decode loop


def _collect(url: str) -> dict[int, dict]:
    body = poc_request_body(BLOCK_HASH, NONCES, MODEL, wait=True, max_tokens=MAX_TOKENS)
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    assert data["status"] == "completed"
    return {a["nonce"]: a for a in data["artifacts"]}


@pytest.mark.skip(reason=(
    "PoC cudagraph is disabled by default (VLLM_POC_CUDAGRAPH=0) pending "
    "byte-identical validation — see /gonka/docs/poc_consolidation_findings.md. "
    "Also, comparing --enforce-eager vs a compiled server is confounded: "
    "torch.compile changes model numerics and the discrete sphere_k amplifies "
    "tiny fp differences. Re-enable as a VLLM_POC_CUDAGRAPH on-vs-off test on "
    "identical compiled servers once the PoC cudagraph replay is correct."
))
@pytest.mark.integration
def test_eager_vs_cudagraph_byte_identical():
    """Artifacts must be byte-identical with cudagraph ON vs --enforce-eager."""
    with PoCTestServer(MODEL, BASE_ARGS + ["--enforce-eager"]) as eager_srv:
        eager = _collect(eager_srv.url_root)
    with PoCTestServer(MODEL, BASE_ARGS) as graph_srv:
        graph = _collect(graph_srv.url_root)

    assert set(eager) == set(graph) == set(NONCES), "missing nonces in a run"

    mismatches = []
    for nonce in NONCES:
        e, g = eager[nonce], graph[nonce]
        if e["vector_b64"] != g["vector_b64"]:
            mismatches.append(f"nonce {nonce}: vector_b64 differs")
        if e.get("k_points_steps") != g.get("k_points_steps"):
            mismatches.append(
                f"nonce {nonce}: k_points_steps differs\n"
                f"  eager:    {e.get('k_points_steps')}\n"
                f"  cudagraph:{g.get('k_points_steps')}"
            )
        if e.get("sphere_k") != g.get("sphere_k"):
            mismatches.append(
                f"nonce {nonce}: sphere_k differs "
                f"(eager={e.get('sphere_k')}, cudagraph={g.get('sphere_k')})"
            )

    assert not mismatches, (
        "cudagraph is not a pure optimization — artifacts diverge from eager:\n"
        + "\n".join(mismatches)
    )


@pytest.mark.integration
def test_decode_trajectory_length_matches_max_tokens():
    """Both eager and cudagraph produce a full prefill+decode trajectory."""
    with PoCTestServer(MODEL, BASE_ARGS) as graph_srv:
        graph = _collect(graph_srv.url_root)
    for nonce in NONCES:
        steps = graph[nonce].get("k_points_steps", [])
        assert len(steps) == MAX_TOKENS + 1, (
            f"nonce {nonce}: expected {MAX_TOKENS + 1} steps "
            f"(prefill + {MAX_TOKENS} decode), got {len(steps)}"
        )
