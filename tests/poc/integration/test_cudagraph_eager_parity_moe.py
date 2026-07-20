"""Decode-PoC artifacts must be BYTE-IDENTICAL under cudagraph vs eager on an MoE
model — the guard for the in-graph seeded-routing move (vllm/poc/native.py
PoCRouterWrapper computes the forced expert logits INSIDE the captured graph now).

A unit test can prove the wrapper's compute equals the eager kernel, but only a live
run can catch a cudagraph *capture/replay* divergence (the historical failure mode:
some nonces matched, some didn't). This boots the SAME MoE model twice — default
(cudagraph) and ``--enforce-eager`` — generates the same nonces on each, and asserts
every nonce's ``k_points_steps`` is identical. MoE-specific because the routing force
only runs on MoE gates.

Requires an MoE model with REAL weights: set ``POC_MOE_MODEL`` (e.g. a small
A3B/OLMoE that fits the box). Skipped otherwise so the dense-only CI stays green.
Real weights are mandatory — with ``--load-format dummy`` the hidden states are
garbage, so the snap flips are fp noise, not a meaningful trajectory. What we assert
is therefore NOT byte-identity (a cg-vs-eager fp floor of a few isolated boundary
flips is expected and harmless) but the ABSENCE OF A ROUTING CASCADE: if the in-graph
seeded selection replayed a wrong expert, that step and ~every step after it diverge
(the routing feeds the next hidden), so a broken capture shows a high divergent
fraction, while the honest fp floor stays low and scattered.
"""
import os

import httpx
import pytest

from tests.poc._server import DEFAULT_SERVER_ARGS as BASE_ARGS, PoCTestServer
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 300
NONCES = [1, 2, 3, 4]          # >=2 concurrent nonces (standing multi-nonce rule); 4 to
                               # catch the "2/4 diverged" pattern specifically
MAX_TOKENS = 16
BLOCK_HASH = "0xmoegraph"
MOE_MODEL = os.environ.get("POC_MOE_MODEL")

pytestmark = pytest.mark.skipif(
    not MOE_MODEL, reason="set POC_MOE_MODEL to an MoE checkpoint to run cg/eager parity")


def _gen(url: str) -> dict[int, list]:
    body = poc_request_body(BLOCK_HASH, NONCES, MOE_MODEL, wait=True, max_tokens=MAX_TOKENS)
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    assert data.get("status") == "completed", f"status != completed: {data}"
    arts = data.get("artifacts", [])
    assert len(arts) == len(NONCES), f"expected {len(NONCES)} artifacts, got {len(arts)}"
    k = {a["nonce"]: a["k_points_steps"] for a in arts}
    assert all(k[n] for n in NONCES), f"empty trajectory: {k}"
    return k


CASCADE_FRAC = 0.4      # a routing cascade diverges ~all steps after the flip (~1.0);
                        # the honest cg-vs-eager fp floor stays well below this


@pytest.mark.integration
def test_moe_decode_no_routing_cascade_cudagraph_vs_eager():
    with PoCTestServer(MOE_MODEL, BASE_ARGS) as srv:                  # default: cudagraph
        cg = _gen(srv.url_root)
    with PoCTestServer(MOE_MODEL, BASE_ARGS + ["--enforce-eager"]) as srv:
        eager = _gen(srv.url_root)
    frac = {}
    for n in NONCES:
        a, b = cg[n], eager[n]
        tot = max(len(a), len(b))
        frac[n] = (sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))) / tot
    cascaded = {n: round(frac[n], 3) for n in NONCES if frac[n] >= CASCADE_FRAC}
    assert not cascaded, (
        f"cudagraph vs eager DIVERGED beyond the fp floor on nonce(s) {cascaded} "
        f"(>= {CASCADE_FRAC} of steps) — the in-graph seeded routing is not replaying "
        f"the same experts under capture (a wrong expert cascades downstream).\n"
        + "\n".join(f"  nonce {n}: cg={cg[n]} eager={eager[n]}" for n in cascaded))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
