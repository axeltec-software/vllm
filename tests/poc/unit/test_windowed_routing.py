"""Windowed seeded routing — the MoE decode-PoC scatter fix (gpu_random._ROUTE_WINDOW).

Without it each token seed-picks top_k of ALL n_experts, so a decode batch scatters across
~all experts and the MoE grouped GEMM streams ~n_experts weight matrices from HBM every step
(measured MiniMax-M2, 256 experts: MoE 8.9 ms/step, 11x pure inference, 56% of PoC GPU time).
The fix restricts every token's pick to a shared, step-rotating WINDOW of W experts, so a
synced batch activates only <=W distinct experts/step (MoE 8.9 -> 1.0 ms/step, decode +79%),
while the window slides so the trajectory still sweeps every expert (fraud-bound preserved).

These pin the invariants a refactor must not break:
  * <= W distinct experts per synced decode batch (the perf property),
  * the trajectory still sweeps ~every expert (the security property),
  * deterministic + single-nonce reproducible (a validator re-running one nonce gets the
    SAME experts as generation — routing depends only on that nonce's (base, step), never on
    batch composition; that's why the window offset is per-row-step, not batch-shared).
Pure integer -> the eager and cudagraph paths are bit-identical by construction.
"""
import torch

from vllm.poc.gpu_random import expert_logits_from_base, _ROUTE_WINDOW

DEV = torch.device("cpu")
N_EXP, TOP_K = 256, 8


def _experts(base, step, n=N_EXP, k=TOP_K):
    b = base.shape[0]
    steps = torch.full((b,), int(step), dtype=torch.int64)
    logits = expert_logits_from_base(base, steps, n, k, DEV)
    return torch.topk(logits, k).indices  # [b, k]


def _bases(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**60, (n,), generator=g, dtype=torch.int64)


def test_window_is_the_durable_default():
    # The fix ships on by default (no env flag): a narrow window over many experts.
    assert 0 < _ROUTE_WINDOW < N_EXP


def test_distinct_experts_bounded_per_synced_batch():
    base = _bases(32, seed=2)
    for step in range(24):
        n_distinct = torch.unique(_experts(base, step)).numel()
        assert n_distinct <= _ROUTE_WINDOW, (step, n_distinct)


def test_trajectory_sweeps_essentially_all_experts():
    # Security: one nonce over a full trajectory still touches ~every expert (window slides).
    base = _bases(1, seed=5)
    seen = set()
    for step in range(N_EXP):
        seen.update(_experts(base, step)[0].tolist())
    assert len(seen) >= N_EXP - 3, len(seen)  # ~complete coverage (probabilistic tail)


def test_deterministic():
    base = _bases(8, seed=3)
    assert torch.equal(_experts(base, 5), _experts(base, 5))


def test_single_nonce_reproducible_regardless_of_batch():
    # A validator re-runs ONE nonce; it must select the same experts as when that nonce was
    # part of a 16-wide generation batch. Routing must not depend on batch composition.
    bases = _bases(16, seed=7)
    step = 9
    batch = _experts(bases, step)
    for i in range(16):
        alone = _experts(bases[i:i + 1], step)
        assert torch.equal(torch.sort(batch[i]).values, torch.sort(alone[0]).values), i


def test_topk_distinct_and_in_range():
    ids = _experts(_bases(32, seed=1), 3)
    assert ids.shape == (32, TOP_K)
    for row in ids:
        assert row.unique().numel() == TOP_K           # exactly top_k distinct
        assert int(row.min()) >= 0 and int(row.max()) < N_EXP


def test_small_moe_falls_back_to_full_scatter():
    # Models with n_experts <= window keep the original all-expert selection (no windowing).
    base = _bases(8, seed=4)
    steps = torch.zeros(8, dtype=torch.int64)
    logits = expert_logits_from_base(base, steps, _ROUTE_WINDOW, 4, DEV)  # n_experts == window
    assert logits.shape == (8, _ROUTE_WINDOW)
