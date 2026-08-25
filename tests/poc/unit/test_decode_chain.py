# SPDX-License-Identifier: Apache-2.0
"""The per-step decode chain is assembled batched, and only when it changes.

``base_seeds`` is constant per nonce and ``prev_k`` for step N+1 is exactly what
step N's snap produced for the SAME rows in the SAME order — yet both were rebuilt
every step with ``torch.cat`` over B one-element tensors, which is O(B) host work
on the critical path (the class of bug that cost 25 us/row/step in the batch
builder, and 1292 ms of prefill forward in the reflection scatter).

The reuse is only valid while the row set is unchanged and no row is teacher-forced
from a reference trajectory, so the tests below pin BOTH the equivalence and the
invalidation rules — a cache that never invalidates would pass a pure value test on
a static batch and silently corrupt a moving one.
"""
from types import SimpleNamespace

import pytest
import torch

from vllm.poc import mixed_decode as rt
from vllm.poc.gpu_random import decode_base_seeds

HIDDEN = 256          # >= SPHERE_DIM: the eager tail picks SPHERE_DIM coordinates
BH, PK = "de" * 32, "ca" * 32
CPU = torch.device("cpu")


class _Native:
    """Captures what the builder publishes to the in-model transforms."""

    def __init__(self):
        self.embed_base = object()
        self.chain = None

    def set_decode_chain(self, **kw):
        self.chain = kw


def _state(nonce, reference=None):
    return SimpleNamespace(
        base_seeds=decode_base_seeds(BH, PK, [nonce], CPU),
        prev_k_t=torch.zeros(1, dtype=torch.int64),
        reference_t=(torch.tensor(reference, dtype=torch.int64)
                     if reference is not None else None),
        reference=reference,
        k_steps_t=[], margin_steps_t=[], q_steps_t=[], max_tokens=64,
    )


def _runner(states, native=None):
    return SimpleNamespace(
        model_config=SimpleNamespace(get_hidden_size=lambda: HIDDEN),
        dtype=torch.float32, device=CPU,
        _poc_native=native,
        _poc_mixed_decode_mgr=SimpleNamespace(get=states.get),
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(poc_vector_artifacts=False)),
    )


def _params(nonce):
    return SimpleNamespace(nonce=nonce, seq_len=64, max_tokens=64, debug=False,
                           block_hash=BH, public_key=PK, k_dim=12,
                           per_nonce_reflection=False)


def _build(runner, states, nonces, computed):
    """One builder call for a decode step over `nonces`, in that row order."""
    req_ids = [f"poc-{n}" for n in nonces]
    sched = SimpleNamespace(num_scheduled_tokens={r: 1 for r in req_ids})
    views = {f"poc-{n}": SimpleNamespace(poc_params=_params(n),
                                         num_computed_tokens=computed)
             for n in nonces}
    return rt.build_unified_mixed_batch_inputs(
        runner, sched, None, None, torch.zeros(len(nonces), dtype=torch.long),
        set(req_ids), len(nonces), (len(nonces), req_ids), views)


def _extract(runner, states, nonces, step):
    """One snap/extract call for the same rows, driving the chain publication."""
    metas = [{"type": "poc", "req_id": f"poc-{n}", "start_idx": i, "length": 1,
              "poc_params": _params(n), "decode_state": states[f"poc-{n}"],
              "decode_step": step} for i, n in enumerate(nonces)]
    rt.process_poc_outputs_from_hidden(
        runner, torch.randn(len(nonces), HIDDEN), metas)
    return metas


def _fixture(nonces, references=None):
    states = {f"poc-{n}": _state(n, (references or {}).get(n)) for n in nonces}
    native = _Native()
    return states, native, _runner(states, native)


def _expected_chain(states, nonces):
    """The per-nonce form the batched assembly must reproduce."""
    base = torch.cat([states[f"poc-{n}"].base_seeds for n in nonces])
    prev = torch.cat([states[f"poc-{n}"].prev_k_t for n in nonces])
    return base, prev


# ------------------------------------------------------------------ equivalence
def test_published_chain_matches_the_per_nonce_form():
    nonces = [5, 9, 2, 7]
    states, native, runner = _fixture(nonces)
    _build(runner, states, nonces, computed=64)

    base, prev = _expected_chain(states, nonces)
    assert torch.equal(native.chain["base"], base)
    assert torch.equal(native.chain["prev_k"], prev)
    assert native.chain["offs"].tolist() == list(range(len(nonces)))


def test_reused_chain_still_matches_after_a_step():
    """Second step over the same rows takes the cached path — the values must
    still equal the per-nonce cat, now chained from the previous snap."""
    nonces = [1, 2, 3]
    states, native, runner = _fixture(nonces)
    _build(runner, states, nonces, computed=64)
    _extract(runner, states, nonces, step=1)
    _build(runner, states, nonces, computed=65)

    base, prev = _expected_chain(states, nonces)
    assert torch.equal(native.chain["base"], base)
    assert torch.equal(native.chain["prev_k"], prev), \
        "reused prev_k diverged from the per-nonce chain"


def test_extract_publishes_exactly_the_next_prev_k():
    nonces = [4, 8]
    states, native, runner = _fixture(nonces)
    _build(runner, states, nonces, computed=64)
    _extract(runner, states, nonces, step=1)

    _, prev = _expected_chain(states, nonces)
    assert torch.equal(runner._poc_chain["prev"], prev)
    # keyed on rows AND the step they will be on next: a new round reusing the
    # same nonces restarts at step 1 and must not inherit this vector
    assert runner._poc_chain["prev_key"] == (tuple(nonces), (2,) * len(nonces))


# ------------------------------------------------------------------ cost shape
def _count_cats(monkeypatch):
    calls = []
    real = torch.cat
    monkeypatch.setattr(torch, "cat",
                        lambda ts, *a, **k: calls.append(len(ts)) or real(ts, *a, **k))
    return calls


def test_steady_state_stops_rebuilding_the_chain(monkeypatch):
    """THE guard: on an unchanged row set the builder must stop catting per-nonce
    tensors. Every value assertion above also passes for the per-step rebuild."""
    nonces = list(range(12))
    states, native, runner = _fixture(nonces)
    _build(runner, states, nonces, computed=64)
    _extract(runner, states, nonces, step=1)

    calls = _count_cats(monkeypatch)
    for step in range(2, 6):
        _build(runner, states, nonces, computed=63 + step)
        _extract(runner, states, nonces, step=step)
    per_row = [n for n in calls if n == len(nonces)]
    assert not per_row, (
        f"{len(per_row)} per-nonce cats over 4 steady steps — the chain is being "
        "rebuilt every step again")


# ------------------------------------------------------------------ invalidation
@pytest.mark.parametrize("second", [
    [1, 2, 3, 4],          # a nonce joined
    [1, 2],                # a nonce finished
    [3, 2, 1],             # same set, different row order
])
def test_row_set_change_rebuilds_and_stays_correct(second):
    first = [1, 2, 3]
    states = {f"poc-{n}": _state(n) for n in set(first) | set(second)}
    native = _Native()
    runner = _runner(states, native)

    _build(runner, states, first, computed=64)
    _extract(runner, states, first, step=1)
    _build(runner, states, second, computed=65)

    base, prev = _expected_chain(states, second)
    assert torch.equal(native.chain["base"], base), "stale base after a row change"
    assert torch.equal(native.chain["prev_k"], prev), "stale prev_k after a row change"


def test_teacher_forced_rows_are_not_published():
    """Validation chains from the REFERENCE, not from this step's k. Publishing
    k_all would feed the next step a free-running chain and silently break the
    aligned compare."""
    nonces = [1, 2]
    states, native, runner = _fixture(nonces, references={2: [3, 4, 5, 6]})
    _build(runner, states, nonces, computed=64)
    _extract(runner, states, nonces, step=1)

    assert runner._poc_chain.get("prev") is None, \
        "published a batched prev_k while a row was teacher-forced"

    _build(runner, states, nonces, computed=65)
    _, prev = _expected_chain(states, nonces)
    assert torch.equal(native.chain["prev_k"], prev)
    assert states["poc-2"].prev_k_t.item() == 4, "reference row did not chain from the reference"


def test_a_new_round_does_not_inherit_the_previous_round_chain():
    """Regression: the cache key was the nonce tuple alone. A new round reuses the
    SAME nonces, so the key matched and the fresh round silently continued the old
    round's chain from its last step — the cross-round determinism test caught it
    live. The key carries the decode step for exactly this reason."""
    nonces = [0, 1]
    states, native, runner = _fixture(nonces)
    for step in range(1, 4):                       # round 1
        _build(runner, states, nonces, computed=63 + step)
        _extract(runner, states, nonces, step=step)
    stale = runner._poc_chain["prev"].clone()

    fresh = {f"poc-{n}": _state(n) for n in nonces}   # round 2: new requests
    runner._poc_mixed_decode_mgr = SimpleNamespace(get=fresh.get)
    _build(runner, fresh, nonces, computed=64)        # back to decode step 1

    base, prev = _expected_chain(fresh, nonces)
    assert torch.equal(native.chain["prev_k"], prev), "new round inherited a stale chain"
    assert not torch.equal(native.chain["prev_k"], stale) or torch.equal(prev, stale)


def test_preempted_and_restarted_row_does_not_reuse_the_chain():
    """vLLM can preempt a running request and restart it: num_computed_tokens resets,
    so the row comes back at an EARLIER decode step with the same nonce set. The key
    carries the step for exactly this — reusing the later step's prev_k would splice
    two halves of different chains together and the trajectory would be nonsense that
    still looks well-formed."""
    nonces = [3, 4]
    states, native, runner = _fixture(nonces)
    for step in (1, 2, 3):
        _build(runner, states, nonces, computed=63 + step)
        _extract(runner, states, nonces, step=step)
    late = runner._poc_chain["prev"].clone()

    for n in nonces:                                   # restart from prefill
        states[f"poc-{n}"].prev_k_t = torch.zeros(1, dtype=torch.int64)
    _build(runner, states, nonces, computed=64)        # back at decode step 1

    _, prev = _expected_chain(states, nonces)
    assert torch.equal(native.chain["prev_k"], prev), "restarted row reused a stale chain"
    assert not torch.equal(native.chain["prev_k"], late) or torch.equal(prev, late)


# ------------------------------------------------------------------ tier 2 guard
@pytest.mark.parametrize("fn", [
    rt.slice_sampling_metadata,
])
def test_row_index_uploads_are_not_blocking(fn):
    """`torch.tensor(list, device=cuda)` synchronises the compute stream; the repo
    ships pinned_to_device precisely to avoid it, and these run every step."""
    import inspect
    src = inspect.getsource(fn)
    assert "pinned_to_device" in src, f"{fn.__name__} builds its index eagerly"
    assert "device=device" not in src.replace("pinned_to_device", ""), \
        f"{fn.__name__} still has a blocking device= tensor construction"
