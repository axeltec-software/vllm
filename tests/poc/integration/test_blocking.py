"""Integration tests for the PoC blocking execution mode."""

import asyncio

import httpx
import pytest

from tests.poc._server import PoCTestServer
from tests.poc.utils import poc_request_body

MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
SERVER_ARGS = ["--gpu-memory-utilization", "0.9", "--max-model-len", "4096"]


@pytest.fixture(scope="module")
def server():
    with PoCTestServer(MODEL, SERVER_ARGS) as srv:
        yield srv


@pytest.fixture(scope="module")
def server_url(server):
    return server.url_root


@pytest.fixture(scope="module")
def model_name():
    return MODEL


@pytest.fixture(scope="module")
def async_client(server):
    return server.get_async_client()


def _poc_generate(
    server_url: str,
    model: str,
    nonces: list[int],
    *,
    wait: bool = True,
    blocking: bool = False,
) -> httpx.Response:
    """Send a single PoC generate request and return the raw response."""
    return httpx.post(
        f"{server_url}/api/v1/pow/generate",
        json=poc_request_body(
            f"test_{wait}_{blocking}",
            nonces,
            model,
            wait=wait,
            blocking=blocking,
        ),
        timeout=60,
    )


@pytest.mark.integration
class TestBlockingIntegration:
    def test_wait_true_blocking_false(self, server_url, model_name):
        """wait=True, blocking=False → synchronous response with all artifacts."""
        resp = _poc_generate(server_url, model_name, [0, 1], wait=True, blocking=False)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["artifacts"]) == 2

    def test_wait_true_blocking_true(self, server_url, model_name):
        """wait=True, blocking=True → synchronous response with all artifacts."""
        resp = _poc_generate(server_url, model_name, [10, 11], wait=True, blocking=True)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert len(data["artifacts"]) == 2

    def test_wait_false_blocking_false(self, server_url, model_name):
        """wait=False, blocking=False → immediate queued response."""
        resp = _poc_generate(server_url, model_name, [20, 21], wait=False, blocking=False)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "request_id" in data

    def test_wait_false_blocking_true(self, server_url, model_name):
        """wait=False, blocking=True → immediate queued response."""
        resp = _poc_generate(server_url, model_name, [30, 31], wait=False, blocking=True)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued"
        assert "request_id" in data


@pytest.mark.integration
@pytest.mark.asyncio
class TestExclusiveMode:
    """Exclusive mode rejects chat (503) while PoC is running."""

    async def test_chat_rejected_during_exclusive_poc(self, server_url, model_name):
        """Chat requests must get 503 while PoC runs in exclusive mode."""
        async with httpx.AsyncClient(timeout=120) as client:
            poc_payload = poc_request_body(
                "exclusive_test",
                list(range(50)),
                model_name,
                block_height=500,
                wait=True,
                blocking=True,
            )
            chat_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10,
            }

            poc_task = asyncio.create_task(
                client.post(f"{server_url}/api/v1/pow/generate", json=poc_payload)
            )
            await asyncio.sleep(0.5)

            chat_resp = await client.post(
                f"{server_url}/v1/chat/completions", json=chat_payload
            )
            poc_resp = await poc_task

        assert poc_resp.status_code == 200
        assert chat_resp.status_code == 503

    async def test_chat_works_after_exclusive_poc_completes(self, server_url, model_name, async_client):
        """Chat must work normally after exclusive-mode PoC finishes."""
        async with httpx.AsyncClient(timeout=120) as http_client:
            poc_payload = poc_request_body(
                "exclusive_after_test",
                [0, 1, 2],
                model_name,
                block_height=600,
                wait=True,
                blocking=True,
            )
            poc_resp = await http_client.post(
                f"{server_url}/api/v1/pow/generate", json=poc_payload
            )
            assert poc_resp.status_code == 200

        result = await async_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=20,
        )
        assert result.choices[0].message.content

    async def test_multiple_chats_rejected_during_exclusive_poc(self, server_url, model_name):
        """All concurrent chat requests must be rejected (503) during exclusive mode."""
        async with httpx.AsyncClient(timeout=120) as client:
            poc_payload = poc_request_body(
                "multi_reject_test",
                list(range(100)),
                model_name,
                block_height=700,
                wait=True,
                blocking=True,
            )
            chat_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            }

            poc_task = asyncio.create_task(
                client.post(f"{server_url}/api/v1/pow/generate", json=poc_payload)
            )
            await asyncio.sleep(0.5)

            chat_responses = await asyncio.gather(*[
                client.post(f"{server_url}/v1/chat/completions", json=chat_payload)
                for _ in range(5)
            ])
            poc_resp = await poc_task

        assert poc_resp.status_code == 200
        rejected = sum(1 for r in chat_responses if r.status_code == 503)
        assert rejected == 5, f"Expected 5 rejections, got {rejected}"
