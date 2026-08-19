"""Cross-version consensus golden: this exact trajectory is produced
bit-identically by this branch and poc-decode-0.25 (2026-08-19, prod 7B
w8a16, seq_len=64 max_tokens=8 route_window=16). Any change here is a
CONSENSUS change, not a refactor. Skips without a live server on :18298."""
import httpx
import pytest

BASE = "http://127.0.0.1:18298"
GOLDEN = [1, 15, 13, 14, 0, 13, 4, 5, 1]


def _up():
    try:
        return httpx.get(f"{BASE}/health", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="no live server")


def test_golden_trajectory_cross_version_parity():
    r = httpx.post(f"{BASE}/api/v1/pow/generate", json={
        "block_hash": "PARITY", "block_height": 1, "public_key": "pk",
        "node_id": 0, "node_count": 1, "nonces": [7], "batch_size": 1,
        "wait": True,
        "params": {"model": "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16",
                   "seq_len": 64, "k_dim": 12, "max_tokens": 8}}, timeout=300)
    assert r.status_code == 200, r.text
    assert r.json()["artifacts"][0]["k_points_steps"] == GOLDEN
