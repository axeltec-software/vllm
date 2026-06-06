import math
import collections
import pytest
import httpx

from tests.poc._server import CANONICAL_MODEL as MODEL, DEFAULT_SERVER_ARGS as SERVER_ARGS, open_poc_server
from tests.poc.utils import poc_request_body

POC_URL = "/api/v1/pow/generate"
TIMEOUT = 120

SPHERE_POINTS = 16


@pytest.fixture(scope="module")
def server(request):
    with open_poc_server(request, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def url(server):
    return server.url_root


def _poc_with_decode(url: str, block_hash: str, nonces: list[int], max_tokens: int) -> dict:
    body = poc_request_body(
        block_hash, nonces, MODEL,
        wait=True,
        max_tokens=max_tokens,
    )
    resp = httpx.post(f"{url}{POC_URL}", json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    # Guard against vacuous passes: a completed request MUST return exactly one
    # artifact per nonce. Previously an empty `artifacts` list slipped through
    # the per-artifact loops below and "passed" silently.
    assert data.get("status") == "completed", f"status != completed: {data}"
    artifacts = data.get("artifacts", [])
    assert len(artifacts) == len(nonces), (
        f"expected {len(nonces)} artifacts, got {len(artifacts)} "
        f"(empty/partial artifacts == decode-PoC produced nothing): {data}"
    )
    return data


def _shannon_entropy(values: list[int], n_bins: int) -> float:
    """Shannon entropy (bits) of a discrete distribution over [0, n_bins)."""
    counts = collections.Counter(values)
    total = len(values)
    entropy = 0.0
    for c in counts.values():
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


@pytest.mark.integration
class TestDecodeStepStructure:
    def test_kpoints_steps_length(self, url):
        """With max_tokens=N, k_points_steps has N+1 entries (prefill + N decode)."""
        max_tokens = 5
        data = _poc_with_decode(url, "0xdecode_length", [1], max_tokens)
        assert data["status"] == "completed"
        artifact = data["artifacts"][0]
        steps = artifact.get("k_points_steps", [])
        assert len(steps) == max_tokens + 1, (
            f"Expected {max_tokens + 1} steps (prefill + {max_tokens} decode), "
            f"got {len(steps)}"
        )

    def test_kpoints_steps_all_in_range(self, url):
        """All k_points_steps values are in [0, SPHERE_POINTS)."""
        data = _poc_with_decode(url, "0xdecode_range", list(range(4)), max_tokens=8)
        for artifact in data["artifacts"]:
            steps = artifact.get("k_points_steps", [])
            for step_idx, k in enumerate(steps):
                assert 0 <= k < SPHERE_POINTS, (
                    f"Artifact nonce={artifact['nonce']} step {step_idx}: "
                    f"k={k} out of [0, {SPHERE_POINTS})"
                )

    def test_sphere_k_present_with_decode(self, url):
        """Prefill sphere_k is present, in range, and equals k_points_steps[0].

        The consolidated API exposes both the scalar ``sphere_k`` (prefill
        codebook index) and the per-step ``k_points_steps``
        ([prefill_k, decode1_k, ...]). sphere_k is the fundamental decode-PoC
        artifact and must equal k_points_steps[0].
        """
        data = _poc_with_decode(url, "0xdecode_sphere_k", [1, 2], max_tokens=3)
        for artifact in data["artifacts"]:
            steps = artifact.get("k_points_steps", [])
            assert steps, f"Artifact nonce={artifact['nonce']} missing k_points_steps"
            assert 0 <= steps[0] < SPHERE_POINTS
            sphere_k = artifact.get("sphere_k")
            assert sphere_k is not None, \
                f"Artifact nonce={artifact['nonce']} missing scalar sphere_k"
            assert sphere_k == steps[0], (
                f"sphere_k ({sphere_k}) must equal k_points_steps[0] ({steps[0]})"
            )


@pytest.mark.integration
class TestDecodeEntropy:
    def test_kpoints_steps_not_constant(self, url):
        """With 20 nonces × 10 decode steps, k values are not all the same."""
        n_nonces, max_tokens = 20, 10
        data = _poc_with_decode(url, "0xentropy_test", list(range(n_nonces)), max_tokens)
        all_k: list[int] = []
        for artifact in data["artifacts"]:
            all_k.extend(artifact.get("k_points_steps", []))
        assert len(set(all_k)) > 1, \
            "All k values are identical — decode is not producing varied sphere indices"

    def test_kpoints_steps_entropy_above_threshold(self, url):
        """Shannon entropy of k distribution exceeds 1.0 bit (not near-constant)."""
        n_nonces, max_tokens = 20, 10
        data = _poc_with_decode(url, "0xentropy_bits", list(range(n_nonces)), max_tokens)
        all_k: list[int] = []
        for artifact in data["artifacts"]:
            all_k.extend(artifact.get("k_points_steps", []))
        entropy = _shannon_entropy(all_k, SPHERE_POINTS)
        assert entropy > 1.0, (
            f"Entropy of k distribution is {entropy:.2f} bits — "
            f"expected > 1.0 (uniform would be {math.log2(SPHERE_POINTS):.2f})"
        )

    def test_distinct_nonces_distinct_kpoints(self, url):
        """Different nonces produce different k_points_steps sequences."""
        data = _poc_with_decode(url, "0xkpoints_nonce_diff", list(range(5)), max_tokens=5)
        sequences = [
            tuple(a.get("k_points_steps", []))
            for a in data["artifacts"]
        ]
        assert len(set(sequences)) > 1, \
            "All nonces produced identical k_points_steps — nonces are not seeding differently"


@pytest.mark.integration
class TestDecodeDeterminism:
    def test_same_request_same_kpoints(self, url):
        """Identical request returns identical k_points_steps on both calls."""
        nonces = [7, 8, 9]
        max_tokens = 5
        data1 = _poc_with_decode(url, "0xdecode_repro", nonces, max_tokens)
        data2 = _poc_with_decode(url, "0xdecode_repro", nonces, max_tokens)
        for a1, a2 in zip(data1["artifacts"], data2["artifacts"]):
            assert a1["nonce"] == a2["nonce"]
            assert a1.get("k_points_steps") == a2.get("k_points_steps"), (
                f"Nonce {a1['nonce']}: k_points_steps changed between identical calls\n"
                f"Call 1: {a1.get('k_points_steps')}\n"
                f"Call 2: {a2.get('k_points_steps')}"
            )

    def test_same_request_same_final_vector(self, url):
        """Decode produces the same final vector on repeated calls."""
        nonces = [42]
        max_tokens = 3
        data1 = _poc_with_decode(url, "0xdecode_vector_repro", nonces, max_tokens)
        data2 = _poc_with_decode(url, "0xdecode_vector_repro", nonces, max_tokens)
        v1 = data1["artifacts"][0]["vector_b64"]
        v2 = data2["artifacts"][0]["vector_b64"]
        assert v1 == v2, "Decode must produce the same final vector for identical inputs"
