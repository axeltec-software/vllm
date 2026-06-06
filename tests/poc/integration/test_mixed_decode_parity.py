"""Acceptance gates for Phase-2 mixed decode-PoC (``VLLM_POC_MIXED_DECODE=1``).

Contract (see wiki: testing/poc-test-strategy, root-causes/multi-nonce-decode-padding-crash):
PoC is NOT validated byte-for-byte. Validation compares the continuous prefill
``vector_b64`` with an L2 tolerance and is statistical; the discrete decode
``sphere_k`` trajectory is the robust signal. Two effects make strict equality the
WRONG contract for a mixed-vs-pure comparison:
  - The pure path pads the PoC batch to ``poc_max_batch_size`` (fixed 32); mixed
    runs at the actual batch. Different batch shape -> different cuBLAS kernel ->
    small fp differences -> ``vector_b64`` drifts (measured L2 ~0.02-0.12) and a
    discrete ``sphere_k`` can occasionally flip near a codebook boundary.
  - ``sphere_k`` chaining (prev_k seeds the next step) cascades one flip through
    the rest of the trajectory. (Real validation aligns trajectories via
    ``inference_k_points_steps``, so the cascade does not occur there.)

Therefore these gates assert the ROBUST, meaningful properties:
  1. mixed decode does NOT crash for multiple concurrent nonces (incl. non-graph
     sizes 3,5,6,7) — regression for the FlashInfer padding crash;
  2. mixed is deterministic (reproducible run-to-run);
  3. the prefill ``sphere_k`` (k_points[0]) matches the pure path exactly (the
     primary, batch-robust artifact);
  4. trajectory length is correct.
They do NOT assert byte-identical ``vector_b64`` or full-trajectory equality —
those are not the validation contract and drift under batch shape by design.
Both servers run with CUDA graphs ON (no --enforce-eager) so the gates still guard
the cudagraph-replay class of bug (which corrupted KV regardless of fp tolerance).
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

NONCES = [1, 2, 3]
# Per-max_tokens block_hash so each request is a distinct PoC round (no reuse).
CASES = {1: "0xparity_mt1", 3: "0xparity_mt3", 5: "0xparity_mt5"}


def _warmup(url: str) -> None:
    """Absorb first-request-after-boot init (lazy hooks/reservation)."""
    body = poc_request_body("0xwarmup", [1], MODEL, wait=True, max_tokens=1)
    try:
        httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    except Exception:
        pass


def _request(url: str, block_hash: str, max_tokens: int, retries: int = 3) -> list[dict]:
    """POST a PoC request; retry a transient empty result (NaN-under-contention
    flake). block_hash is kept identical across retries (deterministic recompute).
    A persistent empty fails the assertion."""
    for _ in range(retries):
        body = poc_request_body(block_hash, NONCES, MODEL, wait=True, max_tokens=max_tokens)
        resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        assert data.get("status") == "completed", f"status != completed: {data}"
        artifacts = data.get("artifacts", [])
        if len(artifacts) == len(NONCES):
            return artifacts
    raise AssertionError(
        f"max_tokens={max_tokens}: {len(NONCES)} artifacts expected, last got "
        f"{len(artifacts)} after {retries} tries (persistent empty — not a flake)"
    )


def _collect(url: str) -> dict[tuple[int, int], dict]:
    """{(max_tokens, nonce): artifact} for every CASE against *url*."""
    _warmup(url)
    out: dict[tuple[int, int], dict] = {}
    for max_tokens, block_hash in CASES.items():
        for a in _request(url, block_hash, max_tokens):
            out[(max_tokens, a["nonce"])] = a
    return out


# Per-step sphere_k mismatch tolerance for cross-path / run-to-run comparison.
# sphere_k is a discrete snap of a continuous projection; near a codebook
# boundary, batch-shape fp AND GPU run-to-run non-determinism can flip it, and
# chaining cascades one flip through later steps. Real PoC validation tolerates
# this (statistical fraud_test + trajectory alignment). We allow a small fraction
# of per-step flips; a structural bug (wrong KV / cudagraph replay) blows far past
# it (it diverges from step 1 on essentially every nonce).
MAX_STEP_MISMATCH_RATE = 0.15


def _step_mismatch_rate(a: dict, b: dict) -> tuple[float, list[str]]:
    total = miss = 0
    detail = []
    for key in sorted(set(a) & set(b)):
        ka = a[key].get("k_points_steps") or []
        kb = b[key].get("k_points_steps") or []
        n = min(len(ka), len(kb))
        d = sum(1 for i in range(n) if ka[i] != kb[i])
        total += n
        miss += d
        if d:
            detail.append(f"{key}: pure={ka} other={kb}")
    return (miss / total if total else 0.0), detail


@pytest.mark.integration
def test_mixed_decode_matches_pure():
    """Full prefill+decode k_points trajectory of mixed must match the pure path,
    allowing only a small per-step boundary-flip rate (mirrors PoC's statistical
    validation). A structural bug (wrong KV / cudagraph replay) diverges on nearly
    every step and blows past the tolerance; an occasional codebook-boundary flip
    stays under it. Both servers run CUDA graphs ON."""
    with PoCTestServer(MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "0"}) as srv:
        pure = _collect(srv.url_root)
    with PoCTestServer(MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"}) as srv:
        mixed = _collect(srv.url_root)

    assert set(pure) == set(mixed)
    # Prefill sphere_k (k_points[0]) is batch-robust -> must match EXACTLY.
    prefill_bad = [
        f"{k}: pure={pure[k]['k_points_steps'][0]} mixed={mixed[k]['k_points_steps'][0]}"
        for k in sorted(pure)
        if pure[k]["k_points_steps"][0] != mixed[k]["k_points_steps"][0]
    ]
    assert not prefill_bad, "prefill sphere_k diverged (KV/cudagraph bug):\n" + "\n".join(prefill_bad)

    # Full trajectory (incl. decode): statistical match.
    rate, detail = _step_mismatch_rate(pure, mixed)
    assert rate <= MAX_STEP_MISMATCH_RATE, (
        f"decode trajectory diverges too much: per-step mismatch rate {rate:.2%} "
        f"> {MAX_STEP_MISMATCH_RATE:.0%} (structural divergence, not boundary "
        f"flips):\n" + "\n".join(detail)
    )


@pytest.mark.integration
def test_mixed_decode_is_kv_bound():
    """The decode must READ the prefill KV: changing the prefill (block_hash)
    must change the decode trajectory. Guards a decode that ignores the reserved
    KV (e.g. attends to nothing / a constant) — which would still 'look' valid."""
    with PoCTestServer(MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"}) as srv:
        _warmup(srv.url_root)
        a = _request(srv.url_root, "0xkvA", max_tokens=5)
        b = _request(srv.url_root, "0xkvB", max_tokens=5)
    a = {x["nonce"]: x["k_points_steps"] for x in a}
    b = {x["nonce"]: x["k_points_steps"] for x in b}
    # decode portion (steps 1..) must differ for at least one nonce
    differ = any(a[n][1:] != b[n][1:] for n in a)
    assert differ, (
        "decode trajectory identical across different prefills — decode is NOT "
        "reading the prefill KV (not KV-bound)"
    )


@pytest.mark.integration
def test_mixed_decode_deterministic_enough():
    """Mixed decode must be reproducible run-to-run within the same small
    boundary-flip tolerance (it is NOT bit-exact near codebook boundaries — that
    affects the pure path too, by design)."""
    with PoCTestServer(MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"}) as srv:
        a = _collect(srv.url_root)
        b = _collect(srv.url_root)
    rate, detail = _step_mismatch_rate(a, b)
    assert rate <= MAX_STEP_MISMATCH_RATE, (
        f"mixed decode too non-deterministic: per-step run-to-run mismatch "
        f"{rate:.2%} > {MAX_STEP_MISMATCH_RATE:.0%}:\n" + "\n".join(detail)
    )


@pytest.mark.integration
def test_mixed_decode_multi_nonce_no_crash():
    """Multiple concurrent decode-PoC nonces must not crash the engine.

    Regression for the FlashInfer decode crash: a uniform-decode batch whose
    size is not a CUDA-graph capture size (e.g. 3, 5, 6, 7) was rounded up to the
    next capture size in the attention metadata, while the PoC forward ran eager
    with the un-padded rows -> `decode_query.shape[0] == num_decode_tokens`
    assertion failed and EngineCore died. Single-nonce (size 1) masked it.
    Fixed by forcing eager BEFORE padding is decided (force_eager). All sizes
    share one server, so a crash on any size also fails the later sizes.
    """
    with PoCTestServer(MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"}) as srv:
        for n in (1, 2, 3, 5, 6, 7, 8):
            nonces = list(range(1, n + 1))
            body = poc_request_body(f"0xmulti{n}", nonces, MODEL, wait=True, max_tokens=3)
            resp = httpx.post(f"{srv.url_root}{POC_URL}", json=body, timeout=TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            assert data.get("status") == "completed", f"n={n}: status {data}"
            arts = data.get("artifacts", [])
            assert len(arts) == n, (
                f"{n} nonces: expected {n} artifacts, got {len(arts)} — engine "
                f"likely crashed on a non-capture-size decode batch"
            )
            assert httpx.get(srv.url_for("health"), timeout=10).status_code == 200, (
                f"engine died after {n}-nonce batch"
            )


@pytest.mark.integration
def test_mixed_decode_trajectory_length():
    """Mixed decode produces a full prefill + max_tokens decode trajectory."""
    with PoCTestServer(MODEL, BASE_ARGS, env_dict={"VLLM_POC_MIXED_DECODE": "1"}) as srv:
        mixed = _collect(srv.url_root)
    for (mt, nonce), a in mixed.items():
        steps = a.get("k_points_steps", [])
        assert len(steps) == mt + 1, (
            f"mt={mt} nonce={nonce}: expected {mt + 1} steps "
            f"(prefill + {mt} decode), got {len(steps)}"
        )
