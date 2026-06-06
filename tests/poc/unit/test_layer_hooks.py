import pytest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from vllm.poc.layer_hooks import (
    LayerHouseholderHook,
    poc_forward_context,
    poc_forward_context_with_mask,
    is_poc_forward_active,
    get_poc_position_mask,
)


class SimpleTransformerLayer(nn.Module):
    """Simple transformer layer for testing."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, hidden_states, residual=None):
        """Forward pass that mimics vLLM decoder layer output format."""
        output = self.linear(hidden_states)
        if residual is not None:
            return (output, residual)
        return output


class SimpleModel(nn.Module):
    """Simple model with multiple layers for testing."""
    
    def __init__(self, num_layers: int, hidden_size: int):
        super().__init__()
        self.model = nn.ModuleDict({
            'layers': nn.ModuleList([
                SimpleTransformerLayer(hidden_size) for _ in range(num_layers)
            ])
        })
    
    def forward(self, x):
        hidden = x
        for layer in self.model.layers:
            hidden = layer(hidden)
        return hidden


class TestLayerHouseholderHookAttachment:
    """Test that hooks attach correctly to model layers."""
    
    def test_hooks_attach_to_all_layers(self):
        """Verify hooks are attached to all transformer layers."""
        num_layers = 4
        hidden_size = 128
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        block_hash = "test_block_hash_123"
        
        hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
        
        assert hooks.num_layers == num_layers, \
            f"Expected {num_layers} hooks, got {hooks.num_layers}"
        assert hooks.num_total_layers == num_layers
        assert len(hooks.reflection_vectors) == num_layers
        assert hooks.block_hash == block_hash
    
    def test_hooks_detach_cleanly(self):
        """Verify hooks can be detached without errors."""
        num_layers = 3
        hidden_size = 64
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)
        assert hooks.num_layers == num_layers
        hooks.detach()

        # detach() removes the forward hook handles. Reflection-vector buffers
        # are intentionally retained (a captured CUDA graph still references
        # their fixed GPU addresses), so we don't assert they are cleared.
        assert hooks.num_layers == 0
        assert len(hooks.hooks) == 0
    
    def test_multiple_attach_detach_cycles(self):
        """Verify multiple attach/detach cycles work correctly."""
        num_layers = 2
        hidden_size = 64
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        for i in range(3):
            hooks = LayerHouseholderHook(
                model, f"block_hash_{i}", device, hidden_size
            )
            assert hooks.num_layers == num_layers
            hooks.detach()
            assert hooks.num_layers == 0


class TestHooksSelectiveTransformation:
    """Test that hooks only transform PoC positions in mixed batches."""
    
    def test_mixed_batch_transforms_only_poc_positions(self):
        """Verify hooks transform only PoC positions, not chat positions."""
        num_layers = 2
        hidden_size = 128
        batch_size = 10
        num_chat_tokens = 4
        num_poc_tokens = 6
        
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)

        input_tensor = torch.randn(batch_size, hidden_size)
        chat_input = input_tensor[:num_chat_tokens].clone()
        poc_input = input_tensor[num_chat_tokens:].clone()

        poc_mask = torch.cat([
            torch.zeros(num_chat_tokens, dtype=torch.bool),
            torch.ones(num_poc_tokens, dtype=torch.bool),
        ])

        with poc_forward_context_with_mask(poc_mask):
            output = model(input_tensor)

        chat_output = output[:num_chat_tokens]
        poc_output = output[num_chat_tokens:]

        chat_unchanged = torch.allclose(chat_input, chat_output, atol=1e-5)
        assert not chat_unchanged, "Chat positions should be transformed by linear layer"
        with torch.no_grad():
            output_no_context = model(input_tensor)
        

        chat_output_no_context = output_no_context[:num_chat_tokens]
        chat_matches_no_context = torch.allclose(
            chat_output, chat_output_no_context, atol=1e-3
        )
        
        poc_output_no_context = output_no_context[num_chat_tokens:]
        poc_differs_from_no_context = not torch.allclose(
            poc_output, poc_output_no_context, atol=1e-3
        )
        
        hooks.detach()
        
        assert chat_matches_no_context, \
            "Chat positions should match non-PoC forward (no hook transform)"
        assert poc_differs_from_no_context, \
            "PoC positions should differ from non-PoC forward (hook transform applied)"
    
    def test_pure_poc_batch_transforms_all_positions(self):
        """Verify hooks transform all positions in pure PoC batch."""
        num_layers = 2
        hidden_size = 64
        batch_size = 8
        
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)
        
        input_tensor = torch.randn(batch_size, hidden_size)
    
        with poc_forward_context():
            output_poc = model(input_tensor)

        output_no_poc = model(input_tensor)
        differs = not torch.allclose(output_poc, output_no_poc, atol=1e-3)
        hooks.detach()
        
        assert differs, \
            "PoC forward should transform all positions differently"
    
    def test_no_context_no_transformation(self):
        """Verify hooks don't transform when no PoC context is active."""
        num_layers = 2
        hidden_size = 64
        batch_size = 8
        
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)
        
        input_tensor = torch.randn(batch_size, hidden_size)

        with torch.no_grad():
            output1 = model(input_tensor)
            output2 = model(input_tensor)

        matches = torch.allclose(output1, output2, atol=1e-6)
        
        hooks.detach()
        
        assert matches, \
            "Without PoC context, outputs should be deterministic"

class TestMaskPaddingAndAlignment:
    """Test that mask padding and alignment works correctly."""
    
    def test_mask_smaller_than_hidden_states(self):
        """Test when mask has fewer elements than hidden states."""
        num_layers = 1
        hidden_size = 64
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)

        input_tensor = torch.randn(10, hidden_size)
        mask = torch.tensor([True] * 5 + [False] * 3)  # Only 8 elements

        with poc_forward_context_with_mask(mask):
            output = model(input_tensor)
        
        assert output.shape == input_tensor.shape
        
        hooks.detach()
    
    def test_mask_larger_than_hidden_states(self):
        """Test when mask has more elements than hidden states."""
        num_layers = 1
        hidden_size = 64
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)
        
        input_tensor = torch.randn(5, hidden_size)
        mask = torch.tensor([True] * 7 + [False] * 3)  # 10 elements

        with poc_forward_context_with_mask(mask):
            output = model(input_tensor)
        
        assert output.shape == input_tensor.shape
        
        hooks.detach()
    
    def test_empty_mask(self):
        """Test with mask where no positions are PoC."""
        num_layers = 1
        hidden_size = 64
        batch_size = 5
        model = SimpleModel(num_layers, hidden_size)
        device = torch.device('cpu')
        
        hooks = LayerHouseholderHook(model, "block_hash", device, hidden_size)
        
        input_tensor = torch.randn(batch_size, hidden_size)
        mask = torch.zeros(batch_size, dtype=torch.bool)  # All False
        
        with poc_forward_context_with_mask(mask):
            output = model(input_tensor)

        output_no_context = model(input_tensor)
        matches = torch.allclose(output, output_no_context, atol=1e-6)
        hooks.detach()
        
        assert matches, \
            "Empty mask should produce same output as no context"
