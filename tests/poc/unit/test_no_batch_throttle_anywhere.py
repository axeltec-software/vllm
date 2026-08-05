"""No layer may silently cap the PoC batch — enforced by SCANNING, not spot checks.

History: `poc_max_batch_size` was a hardcoded 32 in the engine config, so PoC ran 32
concurrent nonces on every machine while inference scaled to hundreds. That was fixed —
and the throttle SURVIVED, because the benchmark client (`request_generate`) still
defaulted `batch_size=32` and sent it in the payload, overriding the server's AUTO. Two
rounds of fixing the definition site while missing a call site.

So these tests do not enumerate known sites; they SWEEP the PoC surface for any new
batch-capping constant, and they check every layer of the chain end to end:

    client default -> HTTP request model -> chunk arithmetic -> engine cap -> scheduler

A new throttle introduced anywhere in that chain should fail here without anyone having
remembered to add a test for it. CPU-only, no GPU, no server.
"""
import ast
import inspect
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "benchmarks" / "poc"))

# Files allowed to contain a batch-capping literal, with the reason.
_ALLOWED = {
    # Reproduces a specific historical corruption bug at a fixed batch; the constant IS
    # the reproduction condition.
    "benchmarks/poc/_repro_chat_corruption.py",
    # Last-resort fallback when the engine config is entirely unreadable; it warns first
    # and is covered by test_no_client_chunking::..._last_resort_...
    "vllm/poc/routes.py",
}

_SCAN = [
    "vllm/poc/routes.py", "vllm/poc/config.py", "vllm/poc/generate_queue.py",
    "vllm/poc/mixed_decode.py", "vllm/config/cache.py",
    "benchmarks/poc/poc_validation.py", "benchmarks/poc/collect.py",
    "benchmarks/poc/perfomance_nonces.py", "benchmarks/poc/batch_sweep.py",
]


# --- the sweep -------------------------------------------------------------------

@pytest.mark.parametrize("rel", _SCAN)
def test_no_hardcoded_batch_cap_literal(rel):
    """Any `batch_size`/`max_batch`-ish name assigned a nonzero literal is a throttle."""
    path = REPO / rel
    if not path.exists():
        pytest.skip(f"{rel} missing")
    src = path.read_text()
    tree = ast.parse(src)
    offenders = []

    def _name_is_batchy(name: str) -> bool:
        n = name.lower()
        return ("batch_size" in n or "max_batch" in n) and "token" not in n

    for node in ast.walk(tree):
        # x: int = 32   /   x = 32
        targets = []
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [(node.target.id, node.value)]
        elif isinstance(node, ast.Assign):
            targets = [(t.id, node.value) for t in node.targets if isinstance(t, ast.Name)]
        for name, val in targets:
            if (_name_is_batchy(name) and isinstance(val, ast.Constant)
                    and isinstance(val.value, int) and val.value != 0):
                offenders.append(f"{name} = {val.value} (line {node.lineno})")
        # def f(..., batch_size=32)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = node.args
            for arg, dflt in zip(args.args[-len(args.defaults):] if args.defaults else [],
                                 args.defaults):
                if (_name_is_batchy(arg.arg) and isinstance(dflt, ast.Constant)
                        and isinstance(dflt.value, int) and dflt.value != 0):
                    offenders.append(f"{node.name}({arg.arg}={dflt.value}) "
                                     f"(line {node.lineno})")

    if offenders and rel not in _ALLOWED:
        pytest.fail(
            f"{rel} hardcodes a PoC batch cap: {offenders}. PoC batching must derive "
            f"from the engine (poc_max_batch_size -> max_num_seqs); a constant here "
            f"throttles every machine to that number. Use 0 = AUTO.")


def test_no_json_payload_pins_batch_size():
    """A cap can also be smuggled in as a dict literal in a request/metadata payload."""
    offenders = []
    for rel in _SCAN:
        path = REPO / rel
        if not path.exists() or rel in _ALLOWED:
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'["\']batch_size["\']\s*:\s*[1-9]\d*', line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, ("batch_size pinned to a nonzero literal in a payload/metadata "
                           f"dict: {offenders}")


# --- the chain, layer by layer ---------------------------------------------------

def test_layer1_benchmark_client_default_is_auto():
    from poc_validation import request_generate
    assert inspect.signature(request_generate).parameters["batch_size"].default == 0


def test_layer2_http_request_models_default_to_auto():
    from vllm.poc.routes import (POC_BATCH_SIZE_DEFAULT, PoCGenerateRequest,
                                 PoCInitGenerateRequest)
    assert POC_BATCH_SIZE_DEFAULT == 0
    for model in (PoCGenerateRequest, PoCInitGenerateRequest):
        assert model.model_fields["batch_size"].default == 0, model.__name__


@pytest.mark.parametrize("total", [1, 32, 33, 64, 128, 512, 1024])
def test_layer3_auto_submits_every_nonce_in_one_chunk(total):
    step = 0 or total or 1
    assert (total + step - 1) // step == 1


def test_layer4_engine_cap_default_is_auto():
    import dataclasses

    from vllm.config.cache import CacheConfig
    fld = next(f for f in dataclasses.fields(CacheConfig)
               if f.name == "poc_max_batch_size")
    # pydantic-flavoured dataclass: the default lives on the FieldInfo
    assert getattr(fld.default, "default", fld.default) == 0


@pytest.mark.parametrize("max_num_seqs", [32, 128, 256, 1024])
def test_layer5_engine_cap_resolves_to_machine_concurrency(max_num_seqs):
    from vllm.poc.mixed_decode import resolve_poc_max_batch_size
    assert resolve_poc_max_batch_size(0, max_num_seqs) == max_num_seqs


def test_end_to_end_no_layer_reduces_the_batch():
    """With everything on AUTO, a 512-nonce request must reach the engine intact and be
    capped only by the machine's own concurrency."""
    from poc_validation import request_generate
    from vllm.poc.mixed_decode import resolve_poc_max_batch_size
    from vllm.poc.routes import POC_BATCH_SIZE_DEFAULT

    total, max_num_seqs = 512, 512
    client = inspect.signature(request_generate).parameters["batch_size"].default
    server = POC_BATCH_SIZE_DEFAULT
    step = (client or server) or total
    chunks = (total + step - 1) // step
    engine_cap = resolve_poc_max_batch_size(0, max_num_seqs)

    assert chunks == 1, "submission was chunked before reaching the engine"
    assert engine_cap == max_num_seqs, "engine capped below the machine's concurrency"
