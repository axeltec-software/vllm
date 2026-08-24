# SPDX-License-Identifier: Apache-2.0
"""The transforms must be ATTACHED, and failure must be LOUD.

Root cause of the 19.08 incident: bridge.load() called attach_native_poc with a
short argument list inside a try/except that logged and continued with
native=None. The forward then skipped seeded embeds, Householder reflections and
seeded routing — yet trajectories still came out full-length, deterministic, and
separating honest from fraud, so every existing test passed. Artifacts looked
fine and were NOT the consensus computation.

These tests fail if PoC can ever run without its transforms.
"""
import inspect
from types import SimpleNamespace

import pytest

from vllm.poc import native
from vllm.poc.runner_bridge import PoCRunnerBridge


def _runner():
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(poc_max_tokens=8, poc_route_window=16)),
        model_config=SimpleNamespace(get_hidden_size=lambda: 128),
        max_num_tokens=1024,
        device="cpu",
        dtype="float16",
    )


def _model(with_layers=True):
    inner = SimpleNamespace(layers=[object(), object()]) if with_layers \
        else SimpleNamespace()
    return SimpleNamespace(model=inner)


def test_load_passes_every_required_argument(monkeypatch):
    """The exact defect: attach was called without hidden_size/device/dtype."""
    seen = {}

    def fake_attach(model, layers, embed_owner, max_tokens, hidden_size,
                    device, dtype, route_window=16):
        seen.update(hidden_size=hidden_size, device=device, dtype=dtype,
                    max_tokens=max_tokens, route_window=route_window,
                    n_layers=len(layers))
        return "attached"

    monkeypatch.setattr(native, "attach_native_poc", fake_attach)
    b = PoCRunnerBridge(_runner())
    b.load(_model())
    assert b.native == "attached"
    assert seen == dict(hidden_size=128, device="cpu", dtype="float16",
                        max_tokens=1024, route_window=16, n_layers=2)


def test_bridge_call_matches_attach_signature():
    """Static guard: every non-default parameter of attach_native_poc must be
    supplied by bridge.load — catches drift without booting a model."""
    sig = inspect.signature(native.attach_native_poc)
    required = [n for n, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty]
    src = inspect.getsource(PoCRunnerBridge.load)
    assert "attach_native_poc(" in src
    # positional call: count the arguments handed over
    call = src.split("attach_native_poc(", 1)[1]
    depth, args, cur = 1, [], ""
    for ch in call:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append(cur)
                break
        if depth == 1 and ch == ",":
            args.append(cur)
            cur = ""
        else:
            cur += ch
    supplied = [a for a in args if a.strip()]
    assert len(supplied) >= len(required), (
        f"bridge.load supplies {len(supplied)} args, "
        f"attach_native_poc requires {len(required)}: {required}")


def test_attach_failure_is_loud_not_silent(monkeypatch):
    """No silent fallback: a failed attach must raise, never leave native=None
    and let the server serve non-consensus artifacts."""
    def boom(*a, **k):
        raise RuntimeError("kernel unavailable")

    monkeypatch.setattr(native, "attach_native_poc", boom)
    b = PoCRunnerBridge(_runner())
    with pytest.raises(RuntimeError):
        b.load(_model())


def test_model_without_layers_raises(monkeypatch):
    b = PoCRunnerBridge(_runner())
    with pytest.raises(RuntimeError, match="no decoder layer list"):
        b.load(_model(with_layers=False))


def test_pre_forward_never_runs_without_native(monkeypatch):
    """Defence in depth: if native is somehow absent, the PoC step must not
    quietly proceed with un-transformed inputs."""
    b = PoCRunnerBridge(_runner())
    b.native = None
    b._step = {"poc_req_ids": {"poc-a"}, "poc_requests": [],
               "poc_metadata": None, "poc_position_mask": None}
    with pytest.raises(RuntimeError, match="native transforms"):
        b.pre_forward(SimpleNamespace(), None, 0)


# ------------------------------------------------------- MoE expert meta
def test_experts_meta_reads_020_and_025_shapes():
    """0.25 moved FusedMoE's top_k/global_num_experts into moe_config
    (experts_per_token/num_experts). Reading only the old names silently
    skips every router — seeded routing is consensus, so the reader must
    cover both shapes and hard-fail on an unknown one."""
    from types import SimpleNamespace
    from vllm.poc.native import _experts_meta

    legacy = SimpleNamespace(top_k=8, global_num_experts=64)
    assert _experts_meta(legacy) == (64, 8)

    modern = SimpleNamespace(
        moe_config=SimpleNamespace(num_experts=64, experts_per_token=8))
    assert _experts_meta(modern) == (64, 8)

    with pytest.raises(RuntimeError, match="extend _experts_meta"):
        _experts_meta(SimpleNamespace())
