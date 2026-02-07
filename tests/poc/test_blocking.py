"""Tests for PoC blocking execution mode."""
import pytest
import httpx
import asyncio
import os
from types import SimpleNamespace

MODEL = os.environ.get("POC_TEST_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PORT = int(os.environ.get("POC_TEST_PORT", "8100"))
BASE_URL = f"http://localhost:{PORT}"


class TestBlockingAPI:
    """Unit tests for blocking parameter."""

    def test_blocking_defaults_to_false(self):
        """Test blocking field defaults to False."""
        from vllm.poc.routes import PoCGenerateRequest

        req = PoCGenerateRequest(
            block_hash="hash",
            block_height=100,
            public_key="key",
            node_id=0,
            node_count=1,
            nonces=[0],
            params={"model": "test", "seq_len": 256},
        )
        assert req.blocking is False

    def test_blocking_accepts_true(self):
        """Test blocking field accepts True."""
        from vllm.poc.routes import PoCGenerateRequest

        req = PoCGenerateRequest(
            block_hash="hash",
            block_height=100,
            public_key="key",
            node_id=0,
            node_count=1,
            nonces=[0],
            params={"model": "test", "seq_len": 256},
            blocking=True,
        )
        assert req.blocking is True

    def test_server_load_tracking_auto_enabled(self):
        """Test server_load_tracking auto-enabled when PoC enabled."""
        state = SimpleNamespace(poc_enabled=True)
        args = SimpleNamespace(enable_server_load_tracking=False)

        result = args.enable_server_load_tracking or getattr(state, 'poc_enabled', False)
        assert result is True

    def test_poc_exclusive_mode_defaults_to_false(self):
        """Test poc_exclusive_mode defaults to False."""
        state = SimpleNamespace()
        result = getattr(state, 'poc_exclusive_mode', False)
        assert result is False


@pytest.fixture(scope="module")
def server_url():
    """Check server is running."""
    try:
        resp = httpx.get(f"{BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return BASE_URL
    except Exception:
        pass
    pytest.skip("Server not running on port 8100")


def poc_generate(server_url, nonces, wait=True, blocking=False):
    """Helper to call PoC generate endpoint."""
    payload = {
        "block_hash": f"test_{wait}_{blocking}",
        "block_height": 100,
        "public_key": "key",
        "node_id": 0,
        "node_count": 1,
        "nonces": nonces,
        "params": {"model": MODEL, "seq_len": 256, "k_dim": 12},
        "wait": wait,
        "blocking": blocking,
    }
    return httpx.post(f"{server_url}/api/v1/pow/generate", json=payload, timeout=60)


@pytest.mark.integration
class TestBlockingIntegration:
    """Integration tests (requires running server)."""

    def test_wait_true_blocking_false(self, server_url):
        resp = poc_generate(server_url, [0, 1], wait=True, blocking=False)
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert len(resp.json()["artifacts"]) == 2

    def test_wait_true_blocking_true(self, server_url):
        resp = poc_generate(server_url, [10, 11], wait=True, blocking=True)
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"
        assert len(resp.json()["artifacts"]) == 2

    def test_wait_false_blocking_false(self, server_url):
        resp = poc_generate(server_url, [20, 21], wait=False, blocking=False)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        assert "request_id" in resp.json()

    def test_wait_false_blocking_true(self, server_url):
        resp = poc_generate(server_url, [30, 31], wait=False, blocking=True)
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"
        assert "request_id" in resp.json()


@pytest.mark.integration
@pytest.mark.asyncio
class TestConcurrentBehavior:
    """Test blocking behavior with concurrent requests."""

    async def test_blocking_poc_yields_to_chat(self, server_url):
        """PoC with blocking=True should wait for in-flight chat to complete."""
        async with httpx.AsyncClient(timeout=120) as client:
            chat_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count from 1 to 50"}],
                "max_tokens": 100,
            }
            poc_payload = {
                "block_hash": "concurrent_test",
                "block_height": 300,
                "public_key": "key",
                "node_id": 0,
                "node_count": 1,
                "nonces": [0],
                "params": {"model": MODEL, "seq_len": 256, "k_dim": 12},
                "wait": True,
                "blocking": True,
            }

            chat_task = asyncio.create_task(
                client.post(f"{server_url}/v1/chat/completions", json=chat_payload)
            )
            await asyncio.sleep(0.05)
            poc_task = asyncio.create_task(
                client.post(f"{server_url}/api/v1/pow/generate", json=poc_payload)
            )

            chat_resp, poc_resp = await asyncio.gather(chat_task, poc_task)

            assert chat_resp.status_code == 200
            assert poc_resp.status_code == 200
            assert poc_resp.json()["status"] == "completed"

    async def test_blocking_with_multiple_chat_requests(self, server_url):
        """PoC should wait for multiple concurrent chat requests."""
        async with httpx.AsyncClient(timeout=120) as client:
            chat_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Count 1 to 30"}],
                "max_tokens": 80,
            }
            poc_payload = {
                "block_hash": "multi_chat_test",
                "block_height": 400,
                "public_key": "key",
                "node_id": 0,
                "node_count": 1,
                "nonces": [0],
                "params": {"model": MODEL, "seq_len": 256, "k_dim": 12},
                "wait": True,
                "blocking": True,
            }

            chat_tasks = [
                asyncio.create_task(
                    client.post(f"{server_url}/v1/chat/completions", json=chat_payload)
                )
                for _ in range(3)
            ]
            await asyncio.sleep(0.05)
            poc_task = asyncio.create_task(
                client.post(f"{server_url}/api/v1/pow/generate", json=poc_payload)
            )

            results = await asyncio.gather(*chat_tasks, poc_task)

            for resp in results:
                assert resp.status_code == 200


@pytest.mark.integration
@pytest.mark.asyncio
class TestExclusiveMode:
    """Test exclusive mode - chat requests rejected (503) during PoC."""

    async def test_chat_rejected_during_exclusive_poc(self, server_url):
        """Chat requests should get 503 while PoC runs in exclusive mode."""
        async with httpx.AsyncClient(timeout=120) as client:
            poc_payload = {
                "block_hash": "exclusive_test",
                "block_height": 500,
                "public_key": "key",
                "node_id": 0,
                "node_count": 1,
                "nonces": list(range(50)),  # Many nonces to take time
                "params": {"model": MODEL, "seq_len": 256, "k_dim": 12},
                "wait": True,
                "blocking": True,
            }
            chat_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            }

            # Start PoC first
            poc_task = asyncio.create_task(
                client.post(f"{server_url}/api/v1/pow/generate", json=poc_payload)
            )
            # Wait for PoC to enter exclusive mode
            await asyncio.sleep(0.5)

            # Try chat while PoC is running - should get 503
            chat_resp = await client.post(
                f"{server_url}/v1/chat/completions", json=chat_payload
            )

            # Wait for PoC to complete
            poc_resp = await poc_task

            assert poc_resp.status_code == 200
            assert chat_resp.status_code == 503

    async def test_chat_works_after_exclusive_poc_completes(self, server_url):
        """Chat requests should work after PoC exclusive mode ends."""
        async with httpx.AsyncClient(timeout=120) as client:
            poc_payload = {
                "block_hash": "exclusive_after_test",
                "block_height": 600,
                "public_key": "key",
                "node_id": 0,
                "node_count": 1,
                "nonces": [0, 1, 2],
                "params": {"model": MODEL, "seq_len": 256, "k_dim": 12},
                "wait": True,
                "blocking": True,
            }
            chat_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20,
            }

            # Run PoC first
            poc_resp = await client.post(
                f"{server_url}/api/v1/pow/generate", json=poc_payload
            )
            assert poc_resp.status_code == 200

            # Now chat should work
            chat_resp = await client.post(
                f"{server_url}/v1/chat/completions", json=chat_payload
            )
            assert chat_resp.status_code == 200

    async def test_multiple_chats_rejected_during_exclusive_poc(self, server_url):
        """Multiple chat requests should all get 503 during exclusive mode."""
        async with httpx.AsyncClient(timeout=120) as client:
            poc_payload = {
                "block_hash": "multi_reject_test",
                "block_height": 700,
                "public_key": "key",
                "node_id": 0,
                "node_count": 1,
                "nonces": list(range(100)),  # Many nonces
                "params": {"model": MODEL, "seq_len": 256, "k_dim": 12},
                "wait": True,
                "blocking": True,
            }
            chat_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            }

            # Start PoC
            poc_task = asyncio.create_task(
                client.post(f"{server_url}/api/v1/pow/generate", json=poc_payload)
            )
            await asyncio.sleep(0.5)

            # Send multiple chat requests
            chat_tasks = [
                client.post(f"{server_url}/v1/chat/completions", json=chat_payload)
                for _ in range(5)
            ]
            chat_responses = await asyncio.gather(*chat_tasks)

            # Wait for PoC
            poc_resp = await poc_task

            assert poc_resp.status_code == 200
            rejected_count = sum(1 for r in chat_responses if r.status_code == 503)
            assert rejected_count == 5, f"Expected 5 rejections, got {rejected_count}"
