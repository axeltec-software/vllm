"""expert_logits_from_base is per-ROW independent: murmur/Fisher-Yates/scatter touch
each row alone, so stacking many layers' seeds into one batch is BYTE-IDENTICAL to a
per-layer loop. This is the invariant the in-graph router move relies on — each MoE
layer's PoCRouterWrapper computes its own rows in-forward, yet every row matches what a
single fused call would produce (no topk, so no cross-row tie coupling)."""
import pytest
import torch

from vllm.poc.gpu_random import expert_logits_from_base


@pytest.mark.parametrize("L,B,n_exp,top_k", [(4, 6, 64, 8), (16, 32, 64, 8), (2, 1, 8, 2)])
def test_batched_equals_per_layer(L, B, n_exp, top_k):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bases = [torch.randint(0, 2**31, (B,), dtype=torch.int64, device=dev) for _ in range(L)]
    steps = torch.arange(1, B + 1, dtype=torch.int64, device=dev)
    per = [expert_logits_from_base(bases[i], steps, n_exp, top_k, dev) for i in range(L)]
    base_all = torch.stack(bases).reshape(-1)
    batched = expert_logits_from_base(base_all, steps.repeat(L), n_exp, top_k, dev).view(L, B, n_exp)
    for i in range(L):
        assert torch.equal(batched[i], per[i]), f"layer {i} differs"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
