# SPDX-License-Identifier: Apache-2.0
"""Live decode-PoC on the 0.25 fork: every nonce gets a FULL trajectory.

Guards both bugs found in bring-up:
  * decode chain must engage (k_points non-empty) — pre_step ordering;
  * EVERY nonce of a multi-nonce round gets prefill + max_tokens points —
    the 0.20 multi-nonce artifact-drop class (standing rule: test >= 2).

Skips when no server on :18299 (launch: VLLM_USE_V2_MODEL_RUNNER=0
vllm serve Qwen/Qwen2.5-0.5B-Instruct --port 18299 ...).
"""
import httpx
import pytest

BASE = "http://127.0.0.1:18299"
PARAMS = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "seq_len": 64,
          "k_dim": 12, "max_tokens": 8}


def _server_up():
    try:
        return httpx.get(f"{BASE}/health", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="no live server")


def _round(nonces):
    r = httpx.post(f"{BASE}/api/v1/pow/generate", json={
        "block_hash": "TESTBLOCK", "block_height": 1, "public_key": "pk",
        "node_id": 0, "node_count": 1, "nonces": nonces,
        "batch_size": len(nonces), "wait": True, "params": PARAMS,
    }, timeout=300)
    assert r.status_code == 200, r.text
    return {a["nonce"]: a for a in r.json()["artifacts"]}


def test_every_nonce_full_trajectory():
    nonces = [0, 1, 2, 3]
    arts = _round(nonces)
    assert set(arts) == set(nonces)
    want = PARAMS["max_tokens"] + 1  # prefill + decode steps
    for n in nonces:
        ks = arts[n]["k_points_steps"]
        assert len(ks) == want, f"nonce {n}: {len(ks)}/{want} k-points {ks}"
        assert all(0 <= k < 16 for k in ks)
        assert arts[n]["n_nan_steps"] == 0


def test_trajectories_deterministic_across_rounds():
    a = _round([0, 1])
    b = _round([0, 1])
    for n in (0, 1):
        assert a[n]["k_points_steps"] == b[n]["k_points_steps"]
