#!/usr/bin/env python3
"""FA-vs-FI cross-backend investigation (parked): why honest cross-backend = ~17% on 7B
but ~4.5% on 235B, and how to make the metric backend-robust WITHOUT pinning.

Pure simulation (no GPU/model). The sphere snap is `nearest of SPHERE_POINTS codebook
points in SPHERE_DIM-dim`. A real hidden projects to a near-random query -> tiny
top1-top2 margin -> the snap is a hair-trigger amplifier: mismatch rate = f(relative
hidden perturbation ε). Each config change perturbs the hidden by some ε:
  ε≈0.4% engine(cg↔eager) -> ~5% · ε≈1.2% FA↔FI(7B) -> ~17% · ε≈2.5% fraud -> ~34%
235B FA↔FI is only ~4.5% because FA/FI perturbs the bigger FP8 model's hidden less (~0.3%).

FIX when backend can't be pinned: TOP-k tolerance (match if prover's sphere_k ∈ validator's
top-2 nearest). Absorbs adjacent-cell backend flips, keeps fraud separation:
  hard snap:  engine 6% · FA↔FI 18% · fraud 34%   (thin margin)
  top-2:      engine 0.7% · FA↔FI 4% · fraud 15%   (clean ~3.5×, p_mismatch~8-10%)  ★
  top-3:      engine 0.1% · FA↔FI 1.2% · fraud 6.7%
Top-k tolerance == enforced-sampling-with-k-candidates, but on the cheap hidden snap
(no LM head). Artifact unchanged for top-1-stored + top-2-checked; or store top-2 (~2 ints/step).
"""
import torch
from vllm.poc.sphere import (build_equidistant_codebook, project_to_sphere,
                             nearest_sphere_index, SPHERE_DIM, SPHERE_POINTS)

def mismatch(cb, eps, tol=1, N=40000, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = project_to_sphere(torch.randn(N, cb.shape[1], generator=g))
    base = nearest_sphere_index(q, cb)
    p = project_to_sphere(q + eps * torch.randn(N, cb.shape[1], generator=g))
    topk = (p.float() @ cb.float().T).topk(max(tol, 1), dim=1).indices
    return (topk == base.unsqueeze(1)).any(1).logical_not().float().mean().item() * 100

if __name__ == "__main__":
    cb = build_equidistant_codebook(SPHERE_POINTS, SPHERE_DIM)
    EPS = {"engine 0.4%": 0.004, "FA↔FI honest 1.2%": 0.012, "fraud 2.5%": 0.025}
    for tol in (1, 2, 3):
        print(f"-- top-{tol} tolerance ({SPHERE_POINTS} pts / {SPHERE_DIM}-dim) --")
        for n, e in EPS.items():
            print(f"   {n:20} -> {mismatch(cb, e, tol):5.1f}%")
