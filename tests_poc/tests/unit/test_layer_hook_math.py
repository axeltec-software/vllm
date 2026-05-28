"""Unit tests for Householder reflection math in PoC layer hooks.

No server required. Tests import directly from vllm.poc.
"""

import torch
import torch.nn as nn
import pytest

from vllm.poc.gpu_random import generate_householder_vector, apply_householder
from vllm.poc.layer_hooks import (
    LayerHouseholderHook,
    poc_forward_context,
    poc_forward_context_with_mask,
    is_poc_forward_active,
    get_poc_position_mask,
)

HIDDEN_SIZE = 64
DEVICE = torch.device("cpu")


def _random_unit_vector(size: int) -> torch.Tensor:
    v = torch.randn(size)
    return v / v.norm()


def _random_hidden(batch: int, hidden: int) -> torch.Tensor:
    return torch.randn(batch, hidden)


class SimpleLayer(nn.Module):
    """Minimal layer that returns (hidden, residual) tuple like vLLM decoder layers."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, residual=None):
        out = self.proj(x)
        if residual is not None:
            return (out, residual)
        return out


class SimpleModel(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.model = nn.ModuleDict({
            "layers": nn.ModuleList(
                [SimpleLayer(hidden_size) for _ in range(num_layers)]
            )
        })

    def forward(self, x):
        h = x
        for layer in self.model["layers"]:
            h = layer(h)
        return h


class TestHouseholderMath:
    def test_self_inverse(self):
        """H(H(x)) == x for any x and unit vector v."""
        v = _random_unit_vector(HIDDEN_SIZE)
        x = _random_hidden(8, HIDDEN_SIZE)
        hx = apply_householder(x, v)
        hhx = apply_householder(hx, v)
        assert torch.allclose(x, hhx, atol=1e-5), "Householder must be self-inverse"

    def test_norm_preservation(self):
        """||H(x)|| == ||x|| — reflections are isometries."""
        v = _random_unit_vector(HIDDEN_SIZE)
        x = _random_hidden(16, HIDDEN_SIZE)
        hx = apply_householder(x, v)
        norms_x = x.norm(dim=-1)
        norms_hx = hx.norm(dim=-1)
        assert torch.allclose(norms_x, norms_hx, atol=1e-5), "Norm must be preserved"

    def test_reflection_of_v(self):
        """H(v) == -v (reflection flips the vector parallel to v)."""
        v = _random_unit_vector(HIDDEN_SIZE)
        hv = apply_householder(v.unsqueeze(0), v).squeeze(0)
        assert torch.allclose(hv, -v, atol=1e-5), "H(v) must equal -v"

    def test_orthogonal_to_v_unchanged(self):
        """Vectors orthogonal to v must pass through unchanged."""
        v = _random_unit_vector(HIDDEN_SIZE)
        x = torch.randn(HIDDEN_SIZE)
        x = x - (x @ v) * v
        hx = apply_householder(x.unsqueeze(0), v).squeeze(0)
        assert torch.allclose(x, hx, atol=1e-5), "Vectors orthogonal to v must be unchanged"

    def test_deterministic_from_seed(self):
        """generate_householder_vector is deterministic for the same seed."""
        v1 = generate_householder_vector("test_seed_abc", HIDDEN_SIZE, DEVICE)
        v2 = generate_householder_vector("test_seed_abc", HIDDEN_SIZE, DEVICE)
        assert torch.equal(v1, v2), "Same seed must produce same vector"

    def test_different_seeds_differ(self):
        """Different seeds produce different Householder vectors."""
        v1 = generate_householder_vector("seed_A", HIDDEN_SIZE, DEVICE)
        v2 = generate_householder_vector("seed_B", HIDDEN_SIZE, DEVICE)
        assert not torch.equal(v1, v2), "Different seeds should produce different vectors"

    def test_unit_norm(self):
        """Generated Householder vector must be a unit vector."""
        v = generate_householder_vector("norm_check", HIDDEN_SIZE, DEVICE)
        assert abs(v.norm().item() - 1.0) < 1e-5, "Householder vector must be unit norm"


class TestHookContext:
    def test_no_context_flag_false(self):
        """is_poc_forward_active() is False outside any context."""
        assert not is_poc_forward_active()

    def test_binary_context_sets_flag(self):
        """poc_forward_context() sets and clears the active flag."""
        with poc_forward_context():
            assert is_poc_forward_active()
        assert not is_poc_forward_active()

    def test_mask_context_sets_mask(self):
        """poc_forward_context_with_mask() sets and clears the global mask."""
        mask = torch.ones(10, dtype=torch.bool)
        with poc_forward_context_with_mask(mask):
            retrieved = get_poc_position_mask()
            assert retrieved is not None
            assert torch.equal(retrieved, mask)
        assert get_poc_position_mask() is None

    def test_context_restores_on_exception(self):
        """Context managers restore state even if the body raises."""
        try:
            with poc_forward_context():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert not is_poc_forward_active()


class TestHookTransformation:
    def _make_model_and_hook(self, num_layers=3):
        model = SimpleModel(num_layers, HIDDEN_SIZE)
        hook = LayerHouseholderHook(model, "test_block_hash", DEVICE, HIDDEN_SIZE)
        return model, hook

    def test_hook_count_matches_layers(self):
        """One hook per transformer layer."""
        num_layers = 4
        model, hook = self._make_model_and_hook(num_layers)
        assert hook.num_layers == num_layers
        hook.detach()

    def test_detach_removes_all_hooks(self):
        """After detach(), num_layers is 0."""
        _, hook = self._make_model_and_hook()
        hook.detach()
        assert hook.num_layers == 0

    def test_no_transform_without_context(self):
        """Without any PoC context, layer output is unchanged."""
        model = SimpleModel(2, HIDDEN_SIZE)
        hook = LayerHouseholderHook(model, "hash_no_ctx", DEVICE, HIDDEN_SIZE)
        x = torch.randn(4, HIDDEN_SIZE)
        out_with_hook = model(x)
        hook.detach()
        out_without_hook = model(x)
        assert torch.allclose(out_with_hook, out_without_hook, atol=1e-6)

    def test_binary_context_transforms_all(self):
        """With poc_forward_context, ALL positions differ from baseline."""
        model = SimpleModel(2, HIDDEN_SIZE)
        hook = LayerHouseholderHook(model, "hash_binary", DEVICE, HIDDEN_SIZE)
        x = torch.randn(4, HIDDEN_SIZE)
        baseline = model(x).detach().clone()
        with poc_forward_context():
            transformed = model(x).detach()
        hook.detach()
        assert not torch.allclose(baseline, transformed, atol=1e-4), \
            "Binary mode must transform all positions"

    def test_mask_mode_transforms_only_poc_positions(self):
        """With a mask, only selected positions differ from baseline."""
        model = SimpleModel(2, HIDDEN_SIZE)
        hook = LayerHouseholderHook(model, "hash_mask", DEVICE, HIDDEN_SIZE)
        n_chat, n_poc = 4, 4
        total = n_chat + n_poc
        x = torch.randn(total, HIDDEN_SIZE)
        baseline = model(x).detach().clone()
        poc_mask = torch.cat([
            torch.zeros(n_chat, dtype=torch.bool),
            torch.ones(n_poc, dtype=torch.bool),
        ])
        with poc_forward_context_with_mask(poc_mask):
            transformed = model(x).detach()
        hook.detach()
        chat_unchanged = torch.allclose(baseline[:n_chat], transformed[:n_chat], atol=1e-5)
        poc_changed = not torch.allclose(baseline[n_chat:], transformed[n_chat:], atol=1e-4)
        assert chat_unchanged, "Chat positions must be unchanged"
        assert poc_changed, "PoC positions must be transformed"

    def test_multiple_attach_detach_cycles(self):
        """Repeated attach/detach cycles leave no dangling hooks."""
        model = SimpleModel(3, HIDDEN_SIZE)
        for _ in range(3):
            hook = LayerHouseholderHook(model, "cycle_hash", DEVICE, HIDDEN_SIZE)
            assert hook.num_layers == 3
            hook.detach()
            assert hook.num_layers == 0
        x = torch.randn(2, HIDDEN_SIZE)
        out = model(x)
        assert out.shape == (2, HIDDEN_SIZE)

    def test_tuple_output_both_components_transformed(self):
        """Both hidden and residual components of a tuple output are transformed."""
        layer = SimpleLayer(HIDDEN_SIZE)
        hook = LayerHouseholderHook.__new__(LayerHouseholderHook)
        hook.hooks = []
        hook.reflection_vectors = []
        hook.block_hash = "tuple_test"
        v = generate_householder_vector("tuple_test_layer_0", HIDDEN_SIZE, DEVICE)
        hook.reflection_vectors.append(v)
        h = layer.register_forward_hook(hook._create_hook(0))
        hook.hooks = [h]

        x = torch.randn(4, HIDDEN_SIZE)
        residual = torch.randn(4, HIDDEN_SIZE)
        with poc_forward_context():
            out = layer(x, residual)

        assert isinstance(out, tuple) and len(out) == 2
        hidden_out, residual_out = out
        base_hidden = layer.proj(x).detach()
        assert not torch.allclose(base_hidden, hidden_out, atol=1e-4), \
            "Hidden must be transformed"
        assert not torch.allclose(residual, residual_out, atol=1e-4), \
            "Residual must be transformed"
        hook.detach()
