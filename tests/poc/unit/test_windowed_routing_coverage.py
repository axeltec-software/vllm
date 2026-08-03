"""Windowed seeded routing must still sweep EVERY expert over a trajectory.

The MoE scatter fix restricts each token's seeded top_k pick to a step-rotating window of
_ROUTE_WINDOW experts (so the grouped GEMM batches them). Its correctness claim is that the
window SLIDES each step, so across the full 256-step trajectory every expert is still
exercised -- that is what preserves the fraud bound (a prover cannot skip experts).

This asserts the claim directly, per (n_experts, top_k) shape, instead of trusting the
comment. Pure integer math -> CPU-runnable, no GPU."""
import pytest
import torch

from vllm.poc.gpu_random import _ROUTE_WINDOW, _forced_logits_windowed

TRAJ = 256          # production trajectory length (k0 + 256 decode steps)


def _experts_hit(n_experts, top_k, steps, window=_ROUTE_WINDOW, seed_val=12345):
    """Union of experts selected across `steps` decode steps for one nonce."""
    dev = torch.device("cpu")
    seen = set()
    for s in range(steps):
        seed = torch.full((1, 1), seed_val, dtype=torch.int64, device=dev)
        st = torch.full((1, 1), s, dtype=torch.int64, device=dev)
        logits = _forced_logits_windowed(seed, st, n_experts, top_k, window, dev)
        # selected experts are the ones scattered above the -1e4 floor
        seen.update(torch.nonzero(logits[0] > -1.0e3).flatten().tolist())
    return seen


@pytest.mark.parametrize("n_experts,top_k", [
    (64, 8),      # OLMoE-1B-7B (the local MoE)
    (128, 8),     # Qwen3-MoE class
    (256, 8),     # MiniMax-M2 class (production)
])
def test_full_trajectory_sweeps_every_expert(n_experts, top_k):
    seen = _experts_hit(n_experts, top_k, TRAJ)
    missing = set(range(n_experts)) - seen
    assert not missing, (
        f"n_experts={n_experts}: {len(missing)} experts NEVER selected over a "
        f"{TRAJ}-step trajectory -> fraud bound weakened (missing e.g. {sorted(missing)[:8]})")


@pytest.mark.parametrize("n_experts", [64, 256])
def test_window_limits_experts_per_step(n_experts):
    """The point of the window: a single step must activate ~window experts, not all of
    them (that is what makes the grouped GEMM fast)."""
    per_step = _experts_hit(n_experts, 8, steps=1)
    assert len(per_step) <= _ROUTE_WINDOW, (
        f"one step activated {len(per_step)} experts, window is {_ROUTE_WINDOW} "
        f"-> the scatter fix is not limiting the batch")


def test_selection_is_deterministic():
    """Same (seed, step) -> same experts. Seeded routing must be reproducible."""
    a = _experts_hit(64, 8, steps=4)
    b = _experts_hit(64, 8, steps=4)
    assert a == b
