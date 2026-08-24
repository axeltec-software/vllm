# SPDX-License-Identifier: Apache-2.0
"""The seeded expert pick: contiguous run (start = seed % n, k consecutive).

Replaced the Fisher-Yates / windowed / grouped formulas (git history has
them): with the selection override the engine never re-derives the set, so
the pick only needs distinctness + determinism + full-expert coverage —
the consensus security lives in seeded embeds, per-layer reflections and
the chained snap. One arithmetic op, any n_experts (incl. 384), trivial to
reimplement in the chain-side validator.
"""
import torch

from vllm.poc import gpu_random as g

DEV = torch.device("cpu")


def _ids(seeds, n_experts, top_k):
    base = torch.tensor(seeds, dtype=torch.int64)
    steps = torch.zeros(len(seeds), dtype=torch.int64)
    logits = g.expert_logits_from_base(base, steps, n_experts, top_k, DEV)
    return torch.topk(logits, top_k).indices


def test_pick_is_distinct_and_contiguous():
    for n in (64, 256, 384):          # OLMoE / MiniMax+V3 / Kimi-K2
        ids = _ids(range(200), n, 8)
        for row in ids:
            assert len(set(row.tolist())) == 8            # distinct
            srt = sorted(row.tolist())
            span = (max(srt) - min(srt)) % n
            assert span == 7 or (n - 1 - span) < 8        # consecutive mod n


def test_pick_deterministic_and_seed_sensitive():
    a = _ids([123, 456], 256, 8)
    b = _ids([123, 456], 256, 8)
    assert torch.equal(a, b)
    assert not torch.equal(a[0], a[1])


def test_pick_covers_every_expert_across_seeds():
    """Uniform reachability: the prover must hold ALL experts."""
    n = 64
    seen = set()
    for chunk in range(0, 4096, 512):
        seen.update(_ids(range(chunk, chunk + 512), n, 8).flatten().tolist())
    assert seen == set(range(n))


def test_pick_formula_snapshot_consensus_freeze():
    """Any drift in the derivation flips this — a consensus change."""
    ids = _ids([0, 1, 999999], 64, 6)
    base = torch.tensor([0, 1, 999999], dtype=torch.int64)
    steps = torch.zeros(3, dtype=torch.int64)
    seed = g._batched_murmur3_32(steps.view(-1, 1).to(torch.int32),
                                 base.view(-1, 1))
    start = torch.remainder(seed.view(-1), 64)
    for r in range(3):
        expect = [(int(start[r]) + i) % 64 for i in range(6)]
        assert ids[r].tolist() == expect


def test_ladder_softmax_weights_are_frozen_constants():
    """The PoC expert mixing weights are softmax over the rank ladder
    (top_k..1) — protocol constants, identical for every model and backend."""
    for top_k in (6, 8):
        ladder = torch.arange(top_k, 0, -1, dtype=torch.float32)
        w = torch.softmax(ladder, dim=-1)
        assert abs(float(w.sum()) - 1.0) < 1e-6
        assert torch.all(w[:-1] > w[1:])
        assert float(w[-1]) > 5e-4
    w8 = torch.softmax(torch.arange(8, 0, -1, dtype=torch.float32), -1)
    assert abs(float(w8[0]) - 0.6318) < 1e-3


# ------------------------------ engine selection recovers the seeded set
def test_engine_selection_recovers_seeded_set_real_selector():
    """With the override retired, the ENGINE's selection over the forced
    ladder is the production path again: it must recover exactly the seeded
    set for every real model configuration (sigmoid ± realistic bias,
    softmax without bias). On CUDA the bias branch runs the production
    fused kernel."""
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk)
    import pytest
    dev = torch.device("cuda") if torch.cuda.is_available() else DEV
    n_experts, top_k = 64, 6
    base = torch.tensor([11, 222, 3333, 44444], dtype=torch.int64, device=dev)
    steps = torch.tensor([0, 1, 7, 255], dtype=torch.int64, device=dev)
    forced = g.expert_logits_from_base(base, steps, n_experts, top_k, dev)
    seeded = torch.topk(forced, top_k).indices.sort().values
    hidden = torch.randn(4, 16, device=dev)

    def sel(scoring, bias, n_group=1, topk_group=1):
        _, ids = grouped_topk(hidden, forced, top_k, False, n_group,
                              topk_group, scoring, 1.0, bias)
        return ids.to(torch.int64).sort().values

    assert torch.equal(sel("softmax", None), seeded)      # V2-family
    assert torch.equal(sel("sigmoid", None), seeded)      # MiniMax-family
    if dev.type == "cuda":
        gen = torch.Generator().manual_seed(7)
        bias = (torch.randn(n_experts, generator=gen) * 0.1).to(dev)
        assert torch.equal(sel("sigmoid", bias), seeded)  # V3/Kimi regime
        # beyond the sigmoid gap the bias overrides the pick — the LIVE
        # Kimi-K2 limitation pinned in test_bias_bound
        adv = torch.zeros(n_experts, device=dev); adv[0] = 0.8
        assert not torch.equal(sel("sigmoid", adv), seeded)


# --------------- selection override (class RETAINED, NOT INSTALLED — PR #2)
def test_select_override_discards_engine_selection_for_poc_rows():
    from vllm.poc.native import PoCSelectOverride, _install_poc_select_patch

    dev = torch.device("cuda") if torch.cuda.is_available() else DEV
    n_experts, top_k, B = 64, 6, 4
    base = torch.tensor([11, 222, 3333, 44444], dtype=torch.int64, device=dev)
    steps = torch.tensor([0, 1, 7, 255], dtype=torch.int64, device=dev)
    mask = torch.zeros(B, dtype=torch.bool, device=dev)
    forced = g.expert_logits_from_base(base, steps, n_experts, top_k, dev)
    seeded = torch.topk(forced, top_k).indices.sort().values

    class FakeRouter:
        def select_experts(self, hidden_states, router_logits, **kw):
            n = hidden_states.shape[0]
            ids = torch.arange(top_k, device=dev).unsqueeze(0).repeat(n, 1)
            return torch.full((n, top_k), 0.5, device=dev), ids.to(torch.int32)

    router = FakeRouter()
    ov = PoCSelectOverride(base, steps, n_experts, top_k, mask)
    _install_poc_select_patch(router, ov)
    mask[1] = True
    mask[3] = True
    h = torch.randn(B, 16, device=dev)
    logits = torch.where(mask.unsqueeze(-1), forced,
                         torch.randn(B, n_experts, device=dev))
    w, ids = router.select_experts(hidden_states=h, router_logits=logits)
    got = ids.to(torch.int64).sort().values
    assert torch.equal(got[1], seeded[1]) and torch.equal(got[3], seeded[3])
    assert torch.equal(got[0], torch.arange(top_k, device=dev))   # chat row
    assert torch.all(w[0] == 0.5)                                 # chat weights
    assert torch.allclose(w[1].sum(), torch.tensor(1.0, device=dev), atol=1e-3)


def test_select_override_immune_to_engine_selection_numerics():
    """Whatever the engine's selection produced — any scoring function, any
    e_score_correction_bias, any tie-break — PoC rows get the seeded set."""
    from vllm.poc.native import PoCSelectOverride, _install_poc_select_patch
    from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
        grouped_topk)
    import pytest
    dev = torch.device("cuda") if torch.cuda.is_available() else DEV
    if dev.type != "cuda":
        pytest.skip("bias branch needs the fused CUDA kernel")

    n_experts, top_k, B = 64, 6, 4
    base = torch.tensor([11, 222, 3333, 44444], dtype=torch.int64, device=dev)
    steps = torch.tensor([0, 1, 7, 255], dtype=torch.int64, device=dev)
    mask = torch.ones(B, dtype=torch.bool, device=dev)
    forced = g.expert_logits_from_base(base, steps, n_experts, top_k, dev)
    seeded = torch.topk(forced, top_k).indices.sort().values
    adversarial = torch.zeros(n_experts, device=dev)
    adversarial[0] = 0.8
    hidden = torch.randn(B, 16, device=dev)

    class EngineRouter:
        def select_experts(self, hidden_states, router_logits, **kw):
            return grouped_topk(hidden_states, router_logits, top_k, False,
                                8, 4, "sigmoid", 1.0, adversarial)

    router = EngineRouter()
    ov = PoCSelectOverride(base, steps, n_experts, top_k, mask)
    _install_poc_select_patch(router, ov)
    _, ids = router.select_experts(hidden_states=hidden, router_logits=forced)
    assert torch.equal(ids.to(torch.int64).sort().values, seeded)
