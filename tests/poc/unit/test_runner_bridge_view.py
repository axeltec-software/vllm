# SPDX-License-Identifier: Apache-2.0
"""Bridge request view: sourced from SchedulerOutput, not runner internals.

Regression: pre_step read ``runner.requests`` (a V1-only dict). On the V2
runner — which keeps per-request state as columnar tensors — that raised
AttributeError and PoC produced no artifacts. Both runners receive the same
SchedulerOutput, so the view is built from that and works on either.
"""
from types import SimpleNamespace

import pytest

from vllm.poc.runner_bridge import PoCRunnerBridge, _PoCRequestView


def _params(nonce, seq_len=64, max_tokens=8):
    return SimpleNamespace(nonce=nonce, seq_len=seq_len, max_tokens=max_tokens)


def _sched_output(new=(), cached=(), finished=(), poc_ids=None):
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, poc_params=p, num_computed_tokens=n)
            for r, p, n in new
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[r for r, _ in cached],
            num_computed_tokens=[n for _, n in cached],
        ),
        finished_req_ids=set(finished),
        poc_req_ids=poc_ids,
    )


def _bridge():
    # runner is only touched once PoC rows exist; setup is stubbed per test
    return PoCRunnerBridge(SimpleNamespace())


def test_no_poc_rows_is_inactive_and_touches_no_runner_state():
    b = _bridge()
    b.pre_step(_sched_output(poc_ids=None))
    assert b.mixed_active() is False


def test_view_is_built_from_new_requests(monkeypatch):
    import vllm.poc.mixed_decode as md
    monkeypatch.setattr(md, "setup_decode_poc", lambda runner, reqs: True)
    b = _bridge()
    b.pre_step(_sched_output(
        new=[("poc-a", _params(7), 0)], poc_ids={"poc-a"}))
    view = b._reqs["poc-a"]
    assert isinstance(view, _PoCRequestView)
    assert view.poc_params.nonce == 7
    assert view.num_computed_tokens == 0


def test_cached_step_advances_computed_tokens(monkeypatch):
    """The decode step index comes from num_computed_tokens; it must track the
    cached-request update every step, not stay at its prefill value."""
    import vllm.poc.mixed_decode as md
    monkeypatch.setattr(md, "setup_decode_poc", lambda runner, reqs: True)
    b = _bridge()
    b.pre_step(_sched_output(new=[("poc-a", _params(1), 0)], poc_ids={"poc-a"}))
    b.pre_step(_sched_output(cached=[("poc-a", 64)], poc_ids={"poc-a"}))
    assert b._reqs["poc-a"].num_computed_tokens == 64
    b.pre_step(_sched_output(cached=[("poc-a", 65)], poc_ids={"poc-a"}))
    assert b._reqs["poc-a"].num_computed_tokens == 65


def test_rows_are_ordered_by_nonce_not_set_iteration(monkeypatch):
    seen = {}
    import vllm.poc.mixed_decode as md
    monkeypatch.setattr(md, "setup_decode_poc",
                        lambda runner, reqs: seen.setdefault("r", list(reqs)))
    b = _bridge()
    b.pre_step(_sched_output(
        new=[("c", _params(9), 0), ("a", _params(2), 0), ("b", _params(5), 0)],
        poc_ids={"a", "b", "c"}))
    assert [r.poc_params.nonce for r in seen["r"]] == [2, 5, 9]


def test_finished_requests_are_dropped(monkeypatch):
    import vllm.poc.mixed_decode as md
    monkeypatch.setattr(md, "setup_decode_poc", lambda runner, reqs: True)
    b = _bridge()
    b.pre_step(_sched_output(new=[("poc-a", _params(1), 0)], poc_ids={"poc-a"}))
    assert "poc-a" in b._reqs
    b.pre_step(_sched_output(finished=["poc-a"], poc_ids={"poc-a"}))
    assert "poc-a" not in b._reqs


def test_chat_requests_never_enter_the_view(monkeypatch):
    import vllm.poc.mixed_decode as md
    monkeypatch.setattr(md, "setup_decode_poc", lambda runner, reqs: True)
    b = _bridge()
    b.pre_step(_sched_output(
        new=[("chat-1", None, 0), ("poc-a", _params(3), 0)],
        poc_ids={"poc-a"}))
    assert set(b._reqs) == {"poc-a"}


@pytest.mark.parametrize("attr", ["requests", "input_batch", "req_states"])
def test_view_needs_no_runner_attribute(attr, monkeypatch):
    """The V2 break, pinned: building the view must not read runner internals
    (V1 has `requests`, V2 has `req_states` — the bridge uses neither)."""
    import vllm.poc.mixed_decode as md
    monkeypatch.setattr(md, "setup_decode_poc", lambda runner, reqs: True)
    b = PoCRunnerBridge(SimpleNamespace())      # runner with NO attributes
    b.pre_step(_sched_output(new=[("poc-a", _params(4), 0)], poc_ids={"poc-a"}))
    assert not hasattr(b.runner, attr)
    assert b._reqs["poc-a"].poc_params.nonce == 4


def test_bridge_passes_request_view_to_batch_builder(monkeypatch):
    """The emission path must receive the runner-agnostic view: it used to read
    runner.requests (V1-only), which produced no artifacts on V2."""
    import vllm.poc.mixed_decode as md
    seen = {}

    monkeypatch.setattr(md, "setup_decode_poc", lambda runner, reqs: True)

    def fake_build(runner, so, ids_a, ids_b, positions, poc_ids, ntok,
                   batch_view=None, req_views=None):
        seen["req_views"] = req_views
        seen["batch_view"] = batch_view
        return (None, None, None, [])

    monkeypatch.setattr(md, "build_unified_mixed_batch_inputs", fake_build)
    b = PoCRunnerBridge(SimpleNamespace())
    b.native = object()                      # attached (guard requires it)
    so = _sched_output(new=[("poc-a", _params(1), 0)], poc_ids={"poc-a"})
    b.pre_step(so)
    b.pre_forward(so, None, 0, batch_view=(1, ["poc-a"]))
    assert seen["req_views"] is b._reqs
    assert "poc-a" in seen["req_views"]


def test_v2_prompt_logprob_slot_is_neutralized_for_poc():
    """V2 reuses slot columns; a PoC request (no sampling params) must clear
    the previous occupant's prompt-logprobs flags, or compute_prompt_logprobs
    KeyErrors on the PoC request id (the V2 engine-death we hit live)."""
    import numpy as np
    from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker
    from vllm.sampling_params import SamplingParams

    w = PromptLogprobsWorker(4)
    chat = SamplingParams(prompt_logprobs=5, max_tokens=1)
    w.add_request("chat-1", 2, chat)              # slot 2 now flagged
    assert w.uses_prompt_logprobs[2]
    w.remove_request("chat-1")
    w.clear_slot(2)                               # PoC takes slot 2
    assert not w.uses_prompt_logprobs[2]
    assert w.num_prompt_logprobs[2] == 0
    assert "poc-x" not in w.in_progress_prompt_logprobs
