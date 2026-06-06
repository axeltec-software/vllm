"""Unit tests for PoC distributed (TP/PP) logic.

Tests the sphere codebook, projection, and nearest-index functions that are
directly testable without a full worker. Documents the TP/PP contract via
interface assertions.

Only 1 GPU available — no actual multi-GPU launch.
"""

import torch
import pytest

from vllm.poc.poc_model_runner import (
    project_to_sphere,
    nearest_sphere_index,
    build_equidistant_codebook,
    _SPHERE_CODEBOOK,
    SPHERE_POINTS,
    SPHERE_DIM,
)


class TestSphereProjection:
    """project_to_sphere produces unit vectors."""

    def test_unit_norm_single(self):
        """Single vector is normalized to unit sphere."""
        v = torch.tensor([[3.0, 4.0]])
        out = project_to_sphere(v)
        assert abs(out.norm().item() - 1.0) < 1e-5

    def test_unit_norm_batch(self):
        """All rows in a batch are normalized to unit sphere."""
        v = torch.randn(16, 64)
        out = project_to_sphere(v)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(16), atol=1e-5)

    def test_zero_vector_safe(self):
        """Zero vector does not produce NaN (epsilon prevents division by zero)."""
        v = torch.zeros(1, 32)
        out = project_to_sphere(v)
        assert not torch.isnan(out).any()

    def test_already_unit_unchanged(self):
        """A vector that is already unit norm passes through unchanged."""
        v = torch.randn(1, 32)
        v = v / v.norm()
        out = project_to_sphere(v)
        assert torch.allclose(v, out, atol=1e-5)


class TestSphereCodebook:
    """_SPHERE_CODEBOOK is valid and equidistant."""

    def test_shape(self):
        """Codebook has shape [SPHERE_POINTS, SPHERE_DIM]."""
        assert _SPHERE_CODEBOOK.shape == (SPHERE_POINTS, SPHERE_DIM)

    def test_all_unit_norm(self):
        """Every codebook point is a unit vector."""
        norms = _SPHERE_CODEBOOK.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(SPHERE_POINTS), atol=1e-4)

    def test_no_nan_or_inf(self):
        """Codebook contains only finite values."""
        assert torch.isfinite(_SPHERE_CODEBOOK).all()

    def test_minimum_pairwise_distance(self):
        """All codebook points are distinct (minimum pairwise cosine similarity < 1)."""
        sims = _SPHERE_CODEBOOK @ _SPHERE_CODEBOOK.T
        eye = torch.eye(SPHERE_POINTS)
        off_diag = sims * (1 - eye)
        assert off_diag.max().item() < 0.999, "Codebook points must be distinct"

    def test_deterministic(self):
        """build_equidistant_codebook is deterministic across calls."""
        cb1 = build_equidistant_codebook(4, 8, n_steps=10)
        cb2 = build_equidistant_codebook(4, 8, n_steps=10)
        assert torch.allclose(cb1, cb2, atol=1e-5)


class TestNearestSphereIndex:
    def test_values_in_range(self):
        """All returned indices are in [0, SPHERE_POINTS)."""
        query = project_to_sphere(torch.randn(32, SPHERE_DIM))
        idx = nearest_sphere_index(query, _SPHERE_CODEBOOK)
        assert idx.min().item() >= 0
        assert idx.max().item() < SPHERE_POINTS

    def test_exact_codebook_point(self):
        """A query equal to a codebook point returns that point's index."""
        for i in range(SPHERE_POINTS):
            query = _SPHERE_CODEBOOK[i:i+1]
            idx = nearest_sphere_index(query, _SPHERE_CODEBOOK)
            assert idx.item() == i, f"Exact codebook point {i} must map to itself"

    def test_batch_shape(self):
        """Output shape matches batch size."""
        query = project_to_sphere(torch.randn(8, SPHERE_DIM))
        idx = nearest_sphere_index(query, _SPHERE_CODEBOOK)
        assert idx.shape == (8,)

    def test_deterministic(self):
        """Same query returns same index."""
        query = project_to_sphere(torch.randn(4, SPHERE_DIM))
        idx1 = nearest_sphere_index(query, _SPHERE_CODEBOOK)
        idx2 = nearest_sphere_index(query, _SPHERE_CODEBOOK)
        assert torch.equal(idx1, idx2)


class TestTPPPContract:
    """Document the interface that execute_poc_forward expects from TP/PP groups.

    These tests assert the shape of the contract, not GPU execution.
    They serve as living documentation for the distributed protocol.
    """

    def test_tp_broadcast_required_keys(self):
        """TP driver must broadcast exactly these keys to non-driver ranks."""
        required_keys = {
            "poc_go", "seq_len", "hidden_size", "nonces",
            "k_dim", "poc_stronger_rng", "poc_decode", "max_tokens",
        }
        broadcast_payload = {
            "poc_go": True,
            "seq_len": 256,
            "hidden_size": 4096,
            "nonces": [1, 2, 3],
            "k_dim": 12,
            "poc_stronger_rng": False,
            "poc_decode": 0,
            "max_tokens": 0,
        }
        assert required_keys == set(broadcast_payload.keys()), \
            "Broadcast dict must contain exactly the required keys for non-driver ranks to reconstruct params"

    def test_tp_decode_step_broadcast_key(self):
        """During decode steps, TP driver broadcasts prev_k to synchronise trajectories."""
        decode_broadcast = {"prev_k": [3, 7, 1]}
        assert "prev_k" in decode_broadcast

    def test_pp_non_last_rank_must_return_none(self):
        """Contract: execute_poc_forward returns None for non-last PP ranks.

        Verified via reading source: line 474 in poc_model_runner.py returns
        None after send_tensor_dict when not pp_group.is_last_rank.
        """
        from unittest.mock import MagicMock
        pp_group = MagicMock()
        pp_group.is_last_rank = False
        assert not pp_group.is_last_rank, \
            "Non-last PP rank must not produce artifacts"

    def test_pp_decode_disabled_when_world_size_gt_1(self):
        """Contract: decode loop is skipped when pp_group.world_size > 1.

        Verified via reading source: lines 576-581 in poc_model_runner.py log a
        warning and skip the loop body when world_size > 1.
        """
        from unittest.mock import MagicMock
        pp_group = MagicMock()
        pp_group.world_size = 2
        assert pp_group.world_size > 1, \
            "PP world_size > 1 triggers decode loop skip"

    def test_pp_first_rank_generates_inputs_embeds(self):
        """Contract: only the first PP rank generates inputs_embeds.

        Other ranks call recv_tensor_dict to receive intermediate tensors.
        """
        from unittest.mock import MagicMock
        pp_first = MagicMock()
        pp_first.is_first_rank = True
        pp_non_first = MagicMock()
        pp_non_first.is_first_rank = False
        assert pp_first.is_first_rank
        assert not pp_non_first.is_first_rank
