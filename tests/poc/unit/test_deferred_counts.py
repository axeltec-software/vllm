"""Deferring the decode-PoC mismatch/nan reductions to emit must be byte-identical
to the old per-step accumulation (mixed_decode.py). The count is the consensus
artifact (n_sphere_mismatches), so this pins the equivalence: same k, margin,
reference, tau -> same totals, whether summed per step or once at emit.
"""
import torch

TAU = 0.02


def _per_step(k, margin, ref):
    """The OLD behavior: accumulate per step (ref-aligned, finite, confident)."""
    mm = 0
    for t in range(k.shape[0]):
        if ref is not None and t < ref.shape[0]:
            mm += int((k[t] != ref[t]) and (k[t] >= 0) and (margin[t] >= TAU))
    n_nan = int((k == -1).sum())
    return (mm if ref is not None else -1), n_nan


def _deferred(k, margin, ref):
    """The NEW behavior: one batched reduction at emit (mixed_decode emit block)."""
    n_nan = int((k == -1).sum())
    if ref is not None:
        L = min(k.shape[0], ref.shape[0])
        mm = int(((k[:L] != ref[:L]) & (k[:L] >= 0) & (margin[:L] >= TAU)).sum())
    else:
        mm = -1
    return mm, n_nan


def test_deferred_equals_per_step_random():
    for seed in range(300):
        g = torch.Generator().manual_seed(seed)
        T = int(torch.randint(1, 60, (1,), generator=g))
        R = int(torch.randint(1, 70, (1,), generator=g))     # ref longer AND shorter than T
        k = torch.randint(-1, 16, (T,), generator=g)         # -1 = non-finite (snap marks it)
        margin = torch.rand(T, generator=g) * 0.05           # straddles TAU
        ref = torch.randint(0, 16, (R,), generator=g)
        assert _deferred(k, margin, ref) == _per_step(k, margin, ref), seed


def test_no_reference_is_minus_one():
    k = torch.randint(0, 16, (10,)); margin = torch.rand(10)
    assert _deferred(k, margin, None) == (-1, 0)


def test_nan_counted_from_k_equals_minus_one():
    k = torch.tensor([0, -1, 5, -1, -1, 3]); margin = torch.zeros(6)
    ref = torch.zeros(6, dtype=torch.int64)
    assert _deferred(k, margin, ref)[1] == 3        # three k == -1
