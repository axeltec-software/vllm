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
PARAMS = {"model": "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16", "seq_len": 64,
          "k_dim": 12, "max_tokens": 8}


def _server_up():
    try:
        if httpx.get(f"{BASE}/health", timeout=5).status_code != 200:
            return False
        # Generic (non-golden) tests run against whatever model is served.
        served = httpx.get(f"{BASE}/v1/models", timeout=5).json()["data"]
        PARAMS["model"] = served[0]["id"]
        return True
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


GOLDEN = {"block_hash": "PARITY", "nonce": 7,
          "model": "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16",
          "k_points": [1, 15, 13, 14, 0, 13, 4, 5, 1]}
# Cross-version consensus constant, bit-identical across the 0.20 and 0.25
# branches (seq_len=64 max_tokens=8 route_window=16). Any change here is a
# CONSENSUS change, not a refactor.

GOLDEN_MOE = {"block_hash": "PARITY", "nonce": 7,
              "model": "allenai/OLMoE-1B-7B-0924-Instruct",
              "k_points": [12, 0, 15, 6, 12, 11, 7, 9, 10]}
# MoE consensus constant (OLMoE 64-expert top-8): all 16 routers seeded,
# contiguous-run pick + SELECTION-OVERRIDE scheme (seeded ids and
# ladder-softmax weights written over the engine selection).
# Guards the silent-detach class: unseeded routing shifts a large fraction
# of steps, so any discovery regression breaks this exactly.


def _assert_golden(golden, model):
    served = [m["id"] for m in
              httpx.get(f"{BASE}/v1/models", timeout=10).json().get("data", [])]
    if model not in served:
        pytest.skip(f"live server is not serving {model}")
    r = httpx.post(f"{BASE}/api/v1/pow/generate", json={
        "block_hash": golden["block_hash"], "block_height": 1,
        "public_key": "pk", "node_id": 0, "node_count": 1,
        "nonces": [golden["nonce"]], "batch_size": 1, "wait": True,
        "params": {**PARAMS, "model": model},
    }, timeout=300)
    assert r.status_code == 200, r.text
    art = r.json()["artifacts"][0]
    assert art["k_points_steps"] == golden["k_points"], (
        "trajectory diverged from the cross-version golden — "
        "the consensus computation changed")


def test_golden_trajectory_cross_version_parity():
    _assert_golden(GOLDEN, GOLDEN["model"])


def test_golden_trajectory_moe_seeded_routing():
    """Runs only when the live server serves the MoE reference model."""
    _assert_golden(GOLDEN_MOE, GOLDEN_MOE["model"])


# Dummy-weight architecture goldens (1-layer local models, --load-format
# dummy): pin the full pipeline per architecture — flat sigmoid 256-expert
# (MiniMax) and grouped top-k (DeepSeek family; the 1-layer config carries
# n_group=8 topk_group=4). Each runs only when that model is being served.
GOLDEN_M2_DUMMY = {"block_hash": "PARITY", "nonce": 7,
                   "model": "minimax-m2-1layer",
                   "k_points": [0, 15, 12, 15, 4, 12, 13, 14, 11]}
GOLDEN_DS_GROUPED_DUMMY = {"block_hash": "PARITY", "nonce": 7,
                           "model": "dsv2lite-1layer",
                           "k_points": [13, 3, 2, 15, 15, 12, 5, 5, 0]}


def test_golden_trajectory_minimax_architecture():
    _assert_golden(GOLDEN_M2_DUMMY, GOLDEN_M2_DUMMY["model"])


def test_golden_trajectory_deepseek_grouped_architecture():
    _assert_golden(GOLDEN_DS_GROUPED_DUMMY, GOLDEN_DS_GROUPED_DUMMY["model"])
