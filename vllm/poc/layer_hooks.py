"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model's learned output structure.

Supports two modes:
1. Binary mode (legacy): All positions are PoC or all are chat
2. Mask mode (mixed batching): Position mask specifies which positions are PoC
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List, Optional

import torch

from .gpu_random import generate_householder_vector, apply_householder

# Context variable for conditional hook activation
# Default False means hooks pass through unchanged (for inference)
_poc_forward_active: ContextVar[bool] = ContextVar('poc_forward_active', default=False)

# Context variable for position-based hook activation (mixed batching)
# When set, only positions where mask is True are transformed
# Shape: [total_tokens] where True = PoC position, False = chat position
_poc_position_mask: ContextVar[Optional[torch.Tensor]] = ContextVar('poc_position_mask', default=None)


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes (legacy binary mode).

    Hooks transform ALL hidden states when this context is active.
    This is for backward compatibility with pure-PoC batches.

    Usage:
        with poc_forward_context():
            hidden_states = model(...)  # All positions transformed
    """
    token = _poc_forward_active.set(True)
    try:
        yield
    finally:
        _poc_forward_active.reset(token)


@contextmanager
def poc_forward_context_with_mask(poc_mask: torch.Tensor):
    """Context manager for mixed batching with position mask.

    Hooks selectively transform only PoC positions based on the mask.
    Chat positions pass through unchanged.

    Args:
        poc_mask: Boolean tensor [total_tokens] where True = PoC position

    Usage:
        # For mixed batch: chat (50 tokens) + poc (256 tokens)
        poc_mask = torch.cat([
            torch.zeros(50, dtype=torch.bool),   # chat: no transform
            torch.ones(256, dtype=torch.bool),   # poc: transform
        ])
        with poc_forward_context_with_mask(poc_mask):
            hidden_states = model(...)  # Only PoC positions transformed
    """
    token = _poc_position_mask.set(poc_mask)
    try:
        yield
    finally:
        _poc_position_mask.reset(token)


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active (binary mode)."""
    return _poc_forward_active.get()


def get_poc_position_mask() -> Optional[torch.Tensor]:
    """Get the PoC position mask for selective transformation."""
    return _poc_position_mask.get()


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers.
    
    These hooks apply the same transform to all nonces in a round (determined
    by block_hash). Combined with per-nonce hidden state transforms, this
    provides strong structure breaking.
    
    Usage:
        # At round init
        hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
        
        # Run forward passes...
        
        # At round end
        hooks.detach()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.reflection_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash
        self._setup(model, block_hash, device, hidden_size)
    
    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        # Try common patterns for different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama, Qwen, Mistral style
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return list(model.transformer.h)
        elif hasattr(model, 'layers'):
            # Direct layers attribute
            return list(model.layers)
        return []
    
    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)
        
        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)
            
            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)
    
    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies Householder reflection.

        Supports two modes:
        1. Binary mode: Transform ALL positions when poc_forward_context() active
        2. Mask mode: Transform only PoC positions when poc_forward_context_with_mask() active

        vLLM decoder layers typically return (hidden_states, residual).
        We must transform BOTH to prevent residual connections from
        preserving untransformed values.

        EXPERIMENT: Also normalize to unit sphere at each layer to break
        magnitude-based structure accumulation.
        """
        def hook(module, input, output):
            # Check for position mask mode (mixed batching)
            poc_mask = get_poc_position_mask()

            if poc_mask is not None:
                # Mask mode: selective transformation
                v = self.reflection_vectors[layer_idx]
                return self._apply_selective_transform(output, poc_mask, v)

            # Check for binary mode (legacy)
            if not is_poc_forward_active():
                return output

            # Binary mode: transform all positions
            v = self.reflection_vectors[layer_idx]

            def normalize_and_transform(x):
                # First normalize to unit sphere (break magnitude structure)
                x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
                # Then apply Householder reflection
                return apply_householder(x_norm, v.to(x_norm.dtype))

            if isinstance(output, tuple):
                if len(output) >= 2:
                    # (hidden_states, residual, ...) format - transform both
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    transformed_hidden = normalize_and_transform(hidden)
                    transformed_residual = normalize_and_transform(residual)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    # Single element tuple
                    hidden = output[0]
                    transformed = normalize_and_transform(hidden)
                    return (transformed,)
            else:
                transformed = normalize_and_transform(output)
                return transformed

        return hook

    def _apply_selective_transform(
        self,
        output,
        poc_mask: torch.Tensor,
        v: torch.Tensor,
    ):
        """Apply transformation only to PoC positions (mask=True).

        Chat positions pass through unchanged.

        Args:
            output: Layer output (tensor or tuple)
            poc_mask: Boolean tensor [total_tokens] where True = PoC position
            v: Householder reflection vector
        """
        def selective_transform(x):
            # x shape: [total_tokens, hidden_size] or [batch, seq, hidden]
            original_shape = x.shape
            if x.dim() == 3:
                # Flatten to [total_tokens, hidden_size]
                x = x.view(-1, x.shape[-1])

            # Ensure mask is on same device and has correct shape
            mask = poc_mask.to(x.device)
            if mask.shape[0] != x.shape[0]:
                # Handle potential shape mismatch (e.g., due to padding)
                if mask.shape[0] < x.shape[0]:
                    # Pad mask with False (treat extra positions as chat)
                    pad_size = x.shape[0] - mask.shape[0]
                    mask = torch.cat([
                        mask,
                        torch.zeros(pad_size, dtype=torch.bool, device=x.device)
                    ])
                else:
                    # Truncate mask
                    mask = mask[:x.shape[0]]

            # Get PoC positions
            poc_indices = mask.nonzero(as_tuple=True)[0]

            if poc_indices.numel() == 0:
                # No PoC positions - pass through
                return x.view(original_shape) if len(original_shape) == 3 else x

            # Extract PoC positions
            poc_hidden = x[poc_indices]

            # Normalize and transform PoC positions
            poc_norm = poc_hidden / (poc_hidden.norm(dim=-1, keepdim=True) + 1e-8)
            poc_transformed = apply_householder(poc_norm, v.to(poc_norm.dtype))

            # Create output with original chat positions unchanged
            result = x.clone()
            result[poc_indices] = poc_transformed

            return result.view(original_shape) if len(original_shape) == 3 else result

        if isinstance(output, tuple):
            if len(output) >= 2:
                # (hidden_states, residual, ...) format - transform both
                hidden = output[0]
                residual = output[1]
                rest = output[2:] if len(output) > 2 else ()
                return (selective_transform(hidden), selective_transform(residual)) + rest
            else:
                # Single element tuple
                return (selective_transform(output[0]),)
        else:
            return selective_transform(output)
    
    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []
    
    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)
