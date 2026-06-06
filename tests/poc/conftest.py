"""Shared pytest fixtures and configuration for PoC tests.

Server fixture strategy
-----------------------
Pass ``--poc-port PORT`` to connect to an already-running vLLM server.
Omit it to have pytest auto-launch a server via ``PoCTestServer``
(a self-contained helper defined in ``tests.poc/server.py``).

``--poc-model`` selects the HuggingFace model (defaults to a small Qwen model).
"""

from typing import Generator

import httpx
import pytest

from tests.poc._server import PoCTestServer

DEFAULT_MODEL = "RedHatAI/Qwen2.5-7B-Instruct-quantized.w8a16"
DEFAULT_SERVER_ARGS = [
    "--gpu-memory-utilization",
    "0.9",
    "--max-model-len",
    "4096",
]
DEFAULT_MAX_WAIT = 300


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--poc-model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model to serve (default: {DEFAULT_MODEL})",
    )
    parser.addoption(
        "--poc-port",
        type=int,
        default=None,
        help="Port of an already-running vLLM server; skips auto-launch.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require a running vLLM server",
    )


@pytest.fixture(scope="session")
def model_name(request: pytest.FixtureRequest) -> str:
    """HuggingFace model identifier used for the test session."""
    return request.config.getoption("--poc-model")


@pytest.fixture(scope="session")
def poc_server(request: pytest.FixtureRequest, model_name: str) -> Generator:
    """Start or connect to a vLLM server for the entire test session.

    Yields a ``PoCTestServer`` (or a compatible shim for existing servers)
    with ``url_root``, ``url_for()``, ``get_client()``, and
    ``get_async_client()`` attributes.
    """
    port: int | None = request.config.getoption("--poc-port")

    if port is not None:
        base = f"http://localhost:{port}"
        try:
            resp = httpx.get(f"{base}/health", timeout=5)
            if resp.status_code != 200:
                pytest.skip(
                    f"Server on port {port} is unhealthy (HTTP {resp.status_code})"
                )
        except Exception as exc:
            pytest.skip(f"Cannot connect to server on port {port}: {exc}")

        class _ExistingServer:
            url_root = base
            DUMMY_API_KEY = PoCTestServer.DUMMY_API_KEY

            def url_for(self, *parts: str) -> str:
                return self.url_root + "/" + "/".join(parts)

            def get_client(self, **kwargs):
                import openai
                kwargs.setdefault("timeout", 600)
                return openai.OpenAI(
                    base_url=self.url_for("v1"),
                    api_key=self.DUMMY_API_KEY,
                    max_retries=0,
                    **kwargs,
                )

            def get_async_client(self, **kwargs):
                import openai
                kwargs.setdefault("timeout", 600)
                return openai.AsyncOpenAI(
                    base_url=self.url_for("v1"),
                    api_key=self.DUMMY_API_KEY,
                    max_retries=0,
                    **kwargs,
                )

        yield _ExistingServer()
    else:
        with PoCTestServer(
            model_name, DEFAULT_SERVER_ARGS, max_wait_seconds=DEFAULT_MAX_WAIT
        ) as srv:
            yield srv


@pytest.fixture(scope="session")
def server_url(poc_server) -> str:
    """Base URL of the running vLLM server (e.g. ``http://127.0.0.1:8100``)."""
    return poc_server.url_root


@pytest.fixture(scope="session")
def client(poc_server):
    """Synchronous OpenAI client pointed at the PoC server."""
    return poc_server.get_client()


@pytest.fixture(scope="session")
def async_client(poc_server):
    """Asynchronous OpenAI client pointed at the PoC server."""
    return poc_server.get_async_client()
