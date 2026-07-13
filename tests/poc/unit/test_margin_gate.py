"""Unit tests for the validator-side margin gate (vllm/poc/sphere.snap_with_margin).

The gate counts a teacher-forced disagreement as a mismatch only if the validator's
own snap margin (top1-top2 cosine gap) >= tau. A low-margin disagreement is boundary
jitter (cross-HW/backend fp noise flipping a near-tie snap), not fraud. Pure CPU math.
"""
import torch

from vllm.poc.sphere import (
    SPHERE_DIM,
    get_sphere_codebook,
    project_to_sphere,
    snap_with_guard,
    snap_with_margin,
)


def _gate_mismatch(k, ref, margin, tau):
    """The exact acceptance-gate expression used in mixed_decode.py."""
    return ((k != ref) & (k >= 0) & (margin >= tau)).to(torch.int64)


def test_k_matches_snap_with_guard():
    # snap_with_margin must pick the SAME nearest point as the ungated snap — the
    # gate changes only whether a disagreement counts, never the trajectory.
    cb = get_sphere_codebook()
    q = project_to_sphere(torch.randn(64, SPHERE_DIM))
    k_guard, _ = snap_with_guard(q, cb)
    k_margin, _, margin = snap_with_margin(q, cb)
    assert torch.equal(k_guard, k_margin)
    assert (margin >= 0).all()                     # margin is a gap, never negative


def test_non_finite_row_is_fault_not_fraud():
    cb = get_sphere_codebook()
    q = project_to_sphere(torch.randn(3, SPHERE_DIM))
    q[1, 0] = float("nan")
    k, bad, margin = snap_with_margin(q, cb)
    assert bad[1] and k[1].item() == -1            # sentinel, excluded by (k >= 0)
    assert margin[1].item() == 0.0
    assert not bad[0] and not bad[2]


def test_margin_larger_near_a_point_than_on_a_boundary():
    # 2-point orthonormal codebook: e0, e1.
    cb = torch.zeros(2, 4)
    cb[0, 0] = 1.0
    cb[1, 1] = 1.0
    near = project_to_sphere(torch.tensor([[1.0, 0.05, 0.0, 0.0]]))   # deep in cell 0
    boundary = project_to_sphere(torch.tensor([[1.0, 1.0, 0.0, 0.0]]))  # on the 0|1 border
    _, _, m_near = snap_with_margin(near, cb)
    _, _, m_boundary = snap_with_margin(boundary, cb)
    assert m_near.item() > 0.9                      # decisive pick
    assert m_boundary.item() < 1e-6                 # a coin-flip
    assert m_near.item() > m_boundary.item()


def test_gate_forgives_boundary_disagreement_keeps_confident_one():
    cb = torch.zeros(2, 4)
    cb[0, 0] = 1.0
    cb[1, 1] = 1.0
    # both queries snap to cell 0 and DISAGREE with a reference of 1; only the
    # confident one should count once tau sits between the two margins.
    q = project_to_sphere(torch.tensor([
        [1.0, 0.05, 0.0, 0.0],   # near point 0  -> big margin  -> real disagreement
        [1.0, 1.0, 0.0, 0.0],    # on the border -> ~0 margin   -> jitter, forgive
    ]))
    k, _, margin = snap_with_margin(q, cb)
    ref = torch.tensor([1, 1])
    tau = 0.5
    counted = _gate_mismatch(k, ref, margin, tau)
    assert counted.tolist() == [1, 0]               # confident counted, border forgiven


def test_tau_zero_is_a_noop():
    # tau=0 must reproduce the ungated count exactly (margin >= 0 always true), so
    # the feature is off by default and artifacts/verdicts are unchanged.
    cb = get_sphere_codebook()
    q = project_to_sphere(torch.randn(128, SPHERE_DIM))
    k, _, margin = snap_with_margin(q, cb)
    ref = torch.randint(0, 16, (128,))
    ungated = ((k != ref) & (k >= 0)).to(torch.int64)
    gated0 = _gate_mismatch(k, ref, margin, 0.0)
    assert torch.equal(ungated, gated0)
    # a positive tau can only REDUCE the count (forgive), never increase it.
    gated_hi = _gate_mismatch(k, ref, margin, 0.05)
    assert gated_hi.sum().item() <= gated0.sum().item()
